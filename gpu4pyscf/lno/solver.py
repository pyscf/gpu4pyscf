#!/usr/bin/env python3
"""GPU molecular restricted fragment CCSD with precomputed MO integral blocks.

The CCSD working tensors, intermediates, and DIIS history are CuPy arrays in
GPU mode. No AO integrations or AO->MO transformation occurs in this program.
Only tiny scalar convergence/DIIS overlap information crosses GPU->CPU.

The RCCSD amplitude equations and intermediates were translated to an array-
backend implementation from PySCF v2.13.0 (Apache-2.0), specifically
pyscf.cc.rccsd and pyscf.cc.rintermediates; the equations are NOT taken from
GPU4PySCF.cc.ccsd_incore (which computes AO integrals during iterations).
https://github.com/pyscf/pyscf/blob/v2.13.0/pyscf/cc/rccsd.py
https://github.com/pyscf/pyscf/blob/v2.13.0/pyscf/cc/rintermediates.py

Restrictions: real RHF, precomputed MO integral blocks, optional packed
DF vvL factors for the four-virtual contractions. No open shell, gradients,
lambda, or CCSD(T).
The NumPy backend is for correctness debugging only, not production speed.

Original PySCF contributions copyright 2014-2024 The PySCF Developers,
licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class FixedLAS:
    nocc: int
    nvir: int
    fock: np.ndarray
    mo_energy: np.ndarray
    eris: dict
    uocc_loc: np.ndarray
    t1_mp2: np.ndarray
    t2_mp2: np.ndarray
    t1_ref: np.ndarray
    t2_ref: np.ndarray
    e_solver_ref: float
    e_projected_ref: float
    e_mp2_ref: float
    source: str
    vvL: np.ndarray | None = None


def _pair_indices(n):
    """PySCF lib.pack_tril convention: row-major lower triangular pairs."""
    lower = np.tril_indices(n)
    pairs = np.empty((n, n), dtype=np.int64)
    pairs[lower] = np.arange(n*(n+1)//2)
    pairs[(lower[1],lower[0])] = pairs[lower]
    return pairs


def read_fixed_las(path: str | Path) -> FixedLAS:
    with h5py.File(path, 'r') as h5:
        groups = [(name, group) for name, group in h5.items()
                  if isinstance(group, h5py.Group)]
        if len(groups) != 1:
            raise ValueError('Expected exactly one fragment in the HDF5 file')
        name, g = groups[0]
        nocc = int(g.attrs['nocc_active'])
        nmo = int(g.attrs['nmo_active'])
        nvir = nmo - nocc
        fock = g['fock_active'][()]
        mo_energy = g['mo_energy_active'][()]
        eri = {k: v[()] for k,v in g['eri'].items()}
        uocc_loc = g['uocc_loc'][()]
        t1_mp2 = g['t1_mp2'][()]
        t2_mp2 = g['t2_mp2'][()]
        t1_ref = g['t1_ccsd'][()]
        t2_ref = g['t2_ccsd'][()]
        e_solver_ref = float(g.attrs['e_corr_solver_ccsd_Ha'])
        e_projected_ref = float(g.attrs['e_fragment_ccsd_Ha'])
        e_mp2_ref = float(g.attrs['e_corr_solver_mp2_Ha'])

    if not all(np.isrealobj(x) for x in (fock, mo_energy, *eri.values(),t1_ref,t2_ref)):
        raise NotImplementedError('Only real RHF is supported')
    if fock.shape != (nmo,nmo) or mo_energy.shape != (nmo,):
        raise ValueError('Inconsistent Fock/MO energy shape')
    if t1_ref.shape != (nocc,nvir) or t2_ref.shape != (nocc,nocc,nvir,nvir):
        raise ValueError('Inconsistent CCSD amplitudes shape')
    if uocc_loc.shape[0] != nocc:
        raise ValueError('Incorrect fragment projection matrix')
    if not np.allclose(np.diag(fock), mo_energy, atol=1e-9):
        raise ValueError('Fock and orbital energy diagonals differ')
    if not np.allclose(eri['ovov'], eri['ovvo'].transpose(0,1,3,2), atol=1e-9):
        raise ValueError('ovov/ovvo permutation mismatch')
    nvir_pair = nvir*(nvir+1)//2
    if eri['ovvv'].shape == (nocc,nvir,nvir_pair):
        ij = _pair_indices(nvir)
        eri['ovvv'] = eri['ovvv'][:,:,ij].copy()
    elif eri['ovvv'].shape != (nocc,nvir,nvir,nvir):
        raise ValueError('Unrecognized packed ovvv dimensions')
    if eri['vvvv'].shape == (nvir_pair,nvir_pair):
        ij = _pair_indices(nvir)
        eri['vvvv'] = eri['vvvv'][ij[:,:,None,None], ij[None,None,:,:]].copy()
    elif eri['vvvv'].shape != (nvir,nvir,nvir,nvir):
        raise ValueError('Unrecognized packed vvvv dimensions')

    shapes = dict(oooo=(nocc,)*4, ovoo=(nocc,nvir,nocc,nocc),
                  oovv=(nocc,nocc,nvir,nvir),ovov=(nocc,nvir,nocc,nvir),
                  ovvo=(nocc,nvir,nvir,nocc),ovvv=(nocc,nvir,nvir,nvir),
                  vvvv=(nvir,)*4)
    for k, sh in shapes.items():
        if eri[k].shape != sh:
            raise ValueError(f'Wrong shape of {k}: {eri[k].shape} != {sh}')
    return FixedLAS(nocc,nvir,fock,mo_energy,eri,uocc_loc,t1_mp2,t2_mp2,
                    t1_ref,t2_ref,e_solver_ref,e_projected_ref,e_mp2_ref,name)


class RestrictedCCSD:
    """PySCF rccsd/rintermediates equations; either NumPy or CuPy execution."""
    def __init__(self, fixed: FixedLAS, xp, level_shift=0.0):
        self.xp = xp
        self.fixed = fixed
        self.level_shift = level_shift
        self.no, self.nv = fixed.nocc, fixed.nvir
        self.fock = xp.asarray(fixed.fock, dtype=xp.float64)
        self.mo_energy = xp.asarray(fixed.mo_energy, dtype=xp.float64)
        self.eris = {key: xp.asarray(val, dtype=xp.float64)
                     for key, val in fixed.eris.items() if val is not None}
        npair = self.nv*(self.nv+1)//2
        pair = xp.asarray(_pair_indices(self.nv))
        if self.eris['ovvv'].shape == (self.no,self.nv,npair):
            self.eris['ovvv'] = self.eris['ovvv'][:,:,pair].copy()
        if 'vvvv' in self.eris and self.eris['vvvv'].shape == (npair,npair):
            v4=self.eris.pop('vvvv')
            self.eris['vvvv'] = v4[pair[:,:,None,None],pair[None,None,:,:]].copy()
            del v4
        self.vvL = None
        if getattr(fixed, 'vvL', None) is not None:
            # PySCF DFCCSD stores packed virtual pairs as (npair,naux).
            # Expand only the *three-center* factors; never form nvir**4 ERIs.
            packed = xp.asarray(fixed.vvL, dtype=xp.float64)
            if packed.shape[0] != self.nv*(self.nv+1)//2:
                raise ValueError('Unexpected DF vvL packed shape')
            self.vvL = packed[pair].transpose(2,0,1).copy()
        if self.vvL is None and 'vvvv' not in self.eris:
            raise ValueError('Neither four-index vvvv nor DF vvL provided')
        # DF blocked-ERI GEMM: reconstruct only a *small virtual slab* and
        # contract it against all (or unique) occupied pairs. Never persist the
        # full four-virtual tensor. Retain old auxiliary-batched contraction as
        # an independent numerical reference, not as the production default.
        self.df_tmp_limit_bytes = int(getattr(fixed, 'df_tmp_limit_bytes', 384*2**20))
        self.df_contract_algorithm = getattr(fixed, 'df_contract_algorithm', 'blocked-eri')
        self.df_virtual_block_cap = int(getattr(fixed, 'df_virtual_block_cap', 16))
        if self.df_contract_algorithm not in ('blocked-eri','aux-batched'):
            raise ValueError('df_contract_algorithm must be blocked-eri or aux-batched')
        if self.df_tmp_limit_bytes < 1024 or self.df_virtual_block_cap < 1:
            raise ValueError('Invalid DF buffer budget or virtual block size')
        self.df_diagnostics = dict(algorithm=self.df_contract_algorithm,
                                   virtual_block=None, occupied_pairs=None,
                                   n_blocks=0, temporary_peak_estimate_bytes=0)
        self.gpu_pool_used_sampled_peak_bytes = None
        self._sample_gpu_pool_used()
        self.projector = xp.asarray(fixed.uocc_loc) @ xp.asarray(fixed.uocc_loc).T

    def _sample_gpu_pool_used(self):
        """Sample *active* CuPy allocations; unlike pool.total_bytes(), this
        excludes cached free blocks. Samples are not a certified device peak.
        """
        if hasattr(self.xp, 'get_default_memory_pool'):
            used=int(self.xp.get_default_memory_pool().used_bytes())
            if (self.gpu_pool_used_sampled_peak_bytes is None or
                    used > self.gpu_pool_used_sampled_peak_bytes):
                self.gpu_pool_used_sampled_peak_bytes=used

    def es(self, pattern, *args):
        return self.xp.einsum(pattern, *args, optimize=True)

    # PySCF 2.13.0 rintermediates.cc_Foo / cc_Fvv / cc_Fov
    def Foo(self, t1, t2):
        e=self.es; v=self.eris['ovov']; no=self.no
        z=2*e('kcld,ilcd->ki',v,t2)-e('kdlc,ilcd->ki',v,t2)
        z+=2*e('kcld,ic,ld->ki',v,t1,t1)-e('kdlc,ic,ld->ki',v,t1,t1)
        return z+self.fock[:no,:no]

    def Fvv(self,t1,t2):
        e=self.es; v=self.eris['ovov']; no=self.no
        z=-2*e('kcld,klad->ac',v,t2)+e('kdlc,klad->ac',v,t2)
        z+=-2*e('kcld,ka,ld->ac',v,t1,t1)+e('kdlc,ka,ld->ac',v,t1,t1)
        return z+self.fock[no:,no:]

    def Fov(self,t1,t2):
        e=self.es; v=self.eris['ovov']; no=self.no
        z=2*e('kcld,ld->kc',v,t1)-e('kdlc,ld->kc',v,t1)
        return z+self.fock[:no,no:]

    def Loo(self,t1,t2):
        e=self.es; o=self.eris['ovoo']; no=self.no
        z=self.Foo(t1,t2)+e('kc,ic->ki',self.fock[:no,no:],t1)
        z+=2*e('lcki,lc->ki',o,t1)-e('kcli,lc->ki',o,t1)
        return z

    def Lvv(self,t1,t2):
        e=self.es; v=self.eris['ovvv']; no=self.no
        z=self.Fvv(t1,t2)-e('kc,ka->ac',self.fock[:no,no:],t1)
        z+=2*e('kdac,kd->ac',v,t1)-e('kcad,kd->ac',v,t1)
        return z

    def Woooo(self,t1,t2):
        e=self.es; er=self.eris; ov=er['ovov']; oo=er['ovoo']
        z=e('lcki,jc->klij',oo,t1)+e('kclj,ic->klij',oo,t1)
        z+=e('kcld,ijcd->klij',ov,t2)+e('kcld,ic,jd->klij',ov,t1,t1)
        return z+er['oooo'].transpose(0,2,1,3)

    def Wvvvv(self,t1,t2):
        e=self.es; er=self.eris; v=er['ovvv']
        z=e('kdac,kb->abcd',v,-t1)-e('kcbd,ka->abcd',v,t1)
        return z+er['vvvv'].transpose(0,2,1,3)

    def _df_aux_batched(self, mats):
        """Legacy auxiliary-batched matmul, retained for numerical A/B checks."""
        xp = self.xp
        nmat,nv,_ = mats.shape
        naux = self.vvL.shape[0]
        per_aux = max(1, 8*nmat*nv*nv*3)  # account for temporary and output
        batch = min(naux, max(1, self.df_tmp_limit_bytes//per_aux))
        out = xp.zeros_like(mats)
        for p0 in range(0,naux,batch):
            l = self.vvL[p0:p0+batch]
            tmp = xp.matmul(l[:,None,:,:], mats[None,:,:,:])
            tmp = xp.matmul(tmp, l[:,None,:,:].transpose(0,1,3,2))
            self._sample_gpu_pool_used()
            out += tmp.sum(axis=0)
        self.df_diagnostics.update(virtual_block=0,occupied_pairs=nmat,
            n_blocks=(naux+batch-1)//batch,
            temporary_peak_estimate_bytes=3*8*batch*nmat*nv*nv)
        return out

    def _df_blocked_eri(self, mats):
        """Contract DF (ac|bd) with tau in two large GEMMs per virtual slab.

        For a in [a0:a1], compute
          V[a,c,b,d] = sum_P L[P,a,c] L[P,b,d]
          out[k,a,b] = sum_cd tau[k,c,d] V[a,c,b,d]
        without allocating or storing the complete nvir**4 integral tensor.
        Arithmetic is O(naux*nvir**4 + nmat*nvir**4), rather than the
        original O(naux*nmat*nvir**3) auxiliary-batched GEMMs. This matters
        when naux greatly exceeds nvir, as in the 510-auxiliary n-butane LAS.
        """
        xp=self.xp
        nmat,nv,_=mats.shape
        naux=self.vvL.shape[0]
        # Approximate 3 materialized slab tensors (GEMM output, transpose,
        # potential contiguous copy) plus the output. For large LAS choose
        # a bounded virtual dimension, independent of GPU total capacity.
        bytes_per_a=8*nv**3
        block=min(self.df_virtual_block_cap, max(1,nv-1),
                  max(1,self.df_tmp_limit_bytes//max(1,3*bytes_per_a)))
        if nv>1 and 3*bytes_per_a>self.df_tmp_limit_bytes:
            # A single virtual slab is too large; fallback preserves DF form.
            return self._df_aux_batched(mats)
        # L[P,a,c] transposed to GEMM-friendly (a*c,P), one contiguous copy.
        left=xp.ascontiguousarray(self.vvL.transpose(1,2,0).reshape(nv*nv,naux))
        right=xp.ascontiguousarray(self.vvL.reshape(naux,nv*nv))
        tmat=xp.ascontiguousarray(mats.reshape(nmat,nv*nv))
        out=xp.empty((nmat,nv,nv),dtype=xp.float64)
        for a0 in range(0,nv,block):
            a1=min(a0+block,nv)
            a_size=a1-a0
            # (a,c|b,d) in the slab; never build the entire (a,c,b,d).
            slab=left[a0*nv:a1*nv,:] @ right
            # (a,c,b,d) -> (c,d,a,b), then flatten cd and ab.
            self._sample_gpu_pool_used()
            slab_cd_ab=xp.ascontiguousarray(
                slab.reshape(a_size,nv,nv,nv).transpose(1,3,0,2)
                .reshape(nv*nv,a_size*nv))
            out[:,a0:a1,:]=(tmat @ slab_cd_ab).reshape(nmat,a_size,nv)
            self._sample_gpu_pool_used()
            del slab_cd_ab,slab
        self.df_diagnostics.update(virtual_block=block,occupied_pairs=nmat,
            n_blocks=(nv+block-1)//block,
            temporary_peak_estimate_bytes=int(3*bytes_per_a*block +
             8*(2*nv*nv*naux+nmat*nv*nv)))
        return out

    def contract_Wvvvv_tau(self, t1, tau, assume_pair_symmetry=False):
        """Wvvvv_abcd * tau_ijcd with DF GEMM, without full nvir**4 vvvv.

        When restricted tau_ijcd == tau_jidc, only i>=j pairs are needed;
        the other half is recovered by transposition in a,b. Callers must
        certify this symmetry (update_amps does). Generic inputs use all ij.
        The legacy auxiliary-batched kernel remains selectable for checking.
        """
        xp=self.xp; e=self.es; nv=self.nv; no=self.no
        if assume_pair_symmetry:
            ii,jj=xp.tril_indices(no)
            mats=tau[ii,jj]  # (nocc*(nocc+1)/2, nv, nv)
        else:
            mats=tau.reshape(no*no,nv,nv)
        # t1-dependent Wvvvv terms never require four virtual dimensions.
        ovvv=self.eris['ovvv']
        x1=e('kdac,mcd->mak',ovvv,mats)
        x2=e('kcbd,mcd->mbk',ovvv,mats)
        corr=-e('mak,kb->mab',x1,t1)-e('mbk,ka->mab',x2,t1)
        if self.vvL is None:
            v4=self.eris['vvvv']
            vv=e('acbd,mcd->mab',v4,mats)
        elif self.df_contract_algorithm=='blocked-eri':
            vv=self._df_blocked_eri(mats)
        else:
            vv=self._df_aux_batched(mats)
        reduced=corr+vv
        if assume_pair_symmetry:
            full=xp.empty((no,no,nv,nv),dtype=xp.float64)
            full[ii,jj,:,:]=reduced
            full[jj,ii,:,:]=reduced.transpose(0,2,1)
            return full
        return reduced.reshape(no,no,nv,nv)

    def Wvoov(self,t1,t2):
        e=self.es; er=self.eris; v=er['ovvv']; o=er['ovoo']; ov=er['ovov']
        z=e('kcad,id->akic',v,t1)-e('kcli,la->akic',o,t1)
        z+=er['ovvo'].transpose(2,0,3,1)
        z-=0.5*e('ldkc,ilda->akic',ov,t2)
        z-=0.5*e('lckd,ilad->akic',ov,t2)
        z-=e('ldkc,id,la->akic',ov,t1,t1)
        z+=e('ldkc,ilad->akic',ov,t2)
        return z

    def Wvovo(self,t1,t2):
        e=self.es; er=self.eris; v=er['ovvv']; o=er['ovoo']; ov=er['ovov']
        z=e('kdac,id->akci',v,t1)-e('lcki,la->akci',o,t1)
        z+=er['oovv'].transpose(2,0,3,1)
        z-=0.5*e('lckd,ilda->akci',ov,t2)
        z-=e('lckd,id,la->akci',ov,t1,t1)
        return z

    def energy(self,t1,t2):
        e=self.es; ov=self.eris['ovov']; no=self.no
        tau=t2+e('ia,jb->ijab',t1,t1)
        z=2*e('ia,ia->',self.fock[:no,no:],t1)
        z+=2*e('ijab,iajb->',tau,ov)-e('ijab,ibja->',tau,ov)
        return z

    def fragment_energy(self,t1,t2):
        e=self.es; project=self.eris['ovov'].transpose(0,2,1,3)
        tau=t2+e('ia,jb->ijab',t1,t1)
        z=2*e('ijab,kjab,ik->',tau,project,self.projector)
        z-=e('ijab,kjba,ik->',tau,project,self.projector)
        return z

    def update_amps(self,t1,t2):
        """PySCF 2.13 rccsd.update_amps, non-CC2 branch, CuPy-compatible."""
        xp=self.xp; e=self.es; no,nv=self.no,self.nv; er=self.eris
        fock=self.fock
        mo_o=self.mo_energy[:no]
        mo_v=self.mo_energy[no:]+self.level_shift
        fov=fock[:no,no:].copy()
        Foo=self.Foo(t1,t2)
        Fvv=self.Fvv(t1,t2)
        Fov=self.Fov(t1,t2)
        Foo[xp.arange(no),xp.arange(no)]-=mo_o
        Fvv[xp.arange(nv),xp.arange(nv)]-=mo_v

        # PySCF restricted singles residual
        t1new=-2*e('kc,ka,ic->ia',fov,t1,t1)
        t1new+=e('ac,ic->ia',Fvv,t1)-e('ki,ka->ia',Foo,t1)
        t1new+=2*e('kc,kica->ia',Fov,t2)-e('kc,ikca->ia',Fov,t2)
        t1new+=e('kc,ic,ka->ia',Fov,t1,t1)+fov.conj()
        t1new+=2*e('kcai,kc->ia',er['ovvo'],t1)-e('kiac,kc->ia',er['oovv'],t1)
        ovvv=er['ovvv']; ovoo=er['ovoo']
        t1new+=2*e('kdac,ikcd->ia',ovvv,t2)-e('kcad,ikcd->ia',ovvv,t2)
        t1new+=2*e('kdac,kd,ic->ia',ovvv,t1,t1)-e('kcad,kd,ic->ia',ovvv,t1,t1)
        t1new+=-2*e('lcki,klac->ia',ovoo,t2)+e('kcli,klac->ia',ovoo,t2)
        t1new+=-2*e('lcki,lc,ka->ia',ovoo,t1,t1)+e('kcli,lc,ka->ia',ovoo,t1,t1)

        # PySCF restricted doubles residual, CCSD (not CC2)
        tmp2=e('kibc,ka->abic',er['oovv'],-t1)+ovvv.conj().transpose(1,3,0,2)
        tmp=e('abic,jc->ijab',tmp2,t1)
        t2new=tmp+tmp.transpose(1,0,3,2)
        tmp2=e('kcai,jc->akij',er['ovvo'],t1)+ovoo.transpose(1,3,0,2).conj()
        tmp=e('akij,kb->ijab',tmp2,t1)
        t2new-=tmp+tmp.transpose(1,0,3,2)
        t2new+=er['ovov'].conj().transpose(0,2,1,3)

        Loo=self.Loo(t1,t2)
        Lvv=self.Lvv(t1,t2)
        Loo[xp.arange(no),xp.arange(no)]-=mo_o
        Lvv[xp.arange(nv),xp.arange(nv)]-=mo_v
        Woooo=self.Woooo(t1,t2)
        Wvoov=self.Wvoov(t1,t2)
        Wvovo=self.Wvovo(t1,t2)
        tau=t2+e('ia,jb->ijab',t1,t1)
        t2new+=e('klij,klab->ijab',Woooo,tau)+self.contract_Wvvvv_tau(t1,tau,assume_pair_symmetry=True)
        tmp=e('ac,ijcb->ijab',Lvv,t2)
        t2new+=tmp+tmp.transpose(1,0,3,2)
        tmp=e('ki,kjab->ijab',Loo,t2)
        t2new-=tmp+tmp.transpose(1,0,3,2)
        tmp=2*e('akic,kjcb->ijab',Wvoov,t2)-e('akci,kjcb->ijab',Wvovo,t2)
        t2new+=tmp+tmp.transpose(1,0,3,2)
        tmp=e('akic,kjbc->ijab',Wvoov,t2)
        t2new-=tmp+tmp.transpose(1,0,3,2)
        tmp=e('bkci,kjac->ijab',Wvovo,t2)
        t2new-=tmp+tmp.transpose(1,0,3,2)

        eia=mo_o[:,None]-mo_v[None,:]
        t1new/=eia
        t2new/=(eia[:,None,:,None]+eia[None,:,None,:])
        return t1new,t2new


# Backward-compatible alias. Batched DIIS is now the default.
from .diis import ResidentDIIS as GPUResidentDIIS


def _as_float(x):
    return float(x.item())


def solve(fixed: FixedLAS, backend='gpu', max_cycle=100,
          conv_tol=1e-9, conv_tol_normt=1e-7, diis_space=6,
          diis_start_cycle=1, level_shift=0.0, verbose=True):
    if backend=='gpu':
        try:
            import cupy as xp
        except ImportError as exc:
            raise RuntimeError('GPU mode requires CuPy and an NVIDIA CUDA driver') from exc
        if xp.cuda.runtime.getDeviceCount()<1:
            raise RuntimeError('No CUDA GPU was detected')
    elif backend=='numpy':
        xp=np
    else:
        raise ValueError('backend must be gpu or numpy')
    cc=RestrictedCCSD(fixed,xp,level_shift=level_shift)
    t1=xp.asarray(fixed.t1_mp2,dtype=xp.float64)
    t2=xp.asarray(fixed.t2_mp2,dtype=xp.float64)
    if backend=='gpu': xp.cuda.get_current_stream().synchronize()
    tstart=time.perf_counter()
    init_mp2=_as_float(cc.energy(t1,t2))
    diis=GPUResidentDIIS(xp,space=diis_space)
    converged=False
    old_e=init_mp2
    history=[]
    for cycle in range(1,max_cycle+1):
        updated1,updated2=cc.update_amps(t1,t2)
        # No transfer of amplitudes to CPU; DIIS keeps working vectors on GPU.
        vec=xp.concatenate((updated1.ravel(),updated2.ravel()))
        prev=xp.concatenate((t1.ravel(),t2.ravel()))
        err=vec-prev
        if cycle>=diis_start_cycle and diis_space>1:
            vec=diis.extrapolate(vec,err)
        t1=vec[:cc.no*cc.nv].reshape(cc.no,cc.nv)
        t2=vec[cc.no*cc.nv:].reshape(cc.no,cc.no,cc.nv,cc.nv)
        # Match symmetry of spin-adapted restricted doubles amplitudes.
        t2=(t2+t2.transpose(1,0,3,2))*.5
        energy=_as_float(cc.energy(t1,t2))
        norm=_as_float(xp.linalg.norm(err))
        de=energy-old_e
        history.append(dict(iteration=cycle,energy_Ha=energy,delta_energy_Ha=de,
                            delta_amplitudes_norm=norm))
        if verbose:
            print(f'CCSD iter {cycle:3d}  E_corr={energy:+.14f}  dE={de:+.3e}  |dT|={norm:.3e}',flush=True)
        if not np.isfinite(energy) or not np.isfinite(norm):
            raise FloatingPointError('CCSD divergence or NaN')
        if abs(de)<conv_tol and norm<conv_tol_normt:
            converged=True
            break
        old_e=energy
    if backend=='gpu': xp.cuda.get_current_stream().synchronize()
    elapsed=time.perf_counter()-tstart
    final_e=_as_float(cc.energy(t1,t2))
    proj_e=_as_float(cc.fragment_energy(t1,t2))
    amp_t1_diff=_as_float(xp.max(xp.abs(t1-xp.asarray(fixed.t1_ref))))
    amp_t2_diff=_as_float(xp.max(xp.abs(t2-xp.asarray(fixed.t2_ref))))
    results={
      'backend':backend,'nocc':fixed.nocc,'nvir':fixed.nvir,
      'converged':converged,'iterations':len(history),
      'elapsed_solver_s':elapsed,'init_mp2_energy_Ha':init_mp2,
      'mp2_snapshot_reference_Ha':fixed.e_mp2_ref,
      'correlation_energy_Ha':final_e,
      'solver_reference_Ha':fixed.e_solver_ref,
      'solver_energy_diff_Ha':final_e-fixed.e_solver_ref,
      'fragment_energy_Ha':proj_e,
      'fragment_reference_Ha':fixed.e_projected_ref,
      'fragment_energy_diff_Ha':proj_e-fixed.e_projected_ref,
      'max_t1_difference':amp_t1_diff,
      'max_t2_difference':amp_t2_diff,
      'history':history,
    }
    if verbose:
        print('\n--- Fixed-LAS RCCSD ---')
        for k,v in results.items():
            if k!='history': print(f'{k}: {v}')
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('h5file', type=Path)
    parser.add_argument('--backend',choices=('gpu','numpy'),default='gpu')
    parser.add_argument('--df-algorithm', choices=('blocked-eri','aux-batched'),default='blocked-eri')
    parser.add_argument('--df-tmp-mib',type=int,default=384)
    parser.add_argument('--df-virtual-block',type=int,default=16)
    parser.add_argument('--max-cycle',type=int,default=100)
    parser.add_argument('--conv-tol',type=float,default=1e-9)
    parser.add_argument('--conv-tol-normt',type=float,default=1e-7)
    parser.add_argument('--diis-space',type=int,default=6)
    parser.add_argument('--diis-start-cycle',type=int,default=1)
    parser.add_argument('--level-shift',type=float,default=0.)
    parser.add_argument('--json',type=Path)
    parser.add_argument('--energy-tol',type=float,default=2e-7,
                        help='maximum accepted difference to CPU CCSD (Ha)')
    args=parser.parse_args()
    fixed=read_fixed_las(args.h5file)
    fixed.df_contract_algorithm=args.df_algorithm
    fixed.df_tmp_limit_bytes=args.df_tmp_mib*2**20
    fixed.df_virtual_block_cap=args.df_virtual_block
    out=solve(fixed,args.backend,args.max_cycle,args.conv_tol,args.conv_tol_normt,
              args.diis_space,args.diis_start_cycle,args.level_shift)
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(out,indent=2)+'\n')
    if not out['converged']:
        raise SystemExit('FAIL: CCSD did not converge')
    if (abs(out['solver_energy_diff_Ha'])>args.energy_tol or
        abs(out['fragment_energy_diff_Ha'])>args.energy_tol):
        raise SystemExit('FAIL: CCSD energy differs from CPU snapshot')
    print('PASS: independent fixed-LAS CCSD agrees with CPU snapshot')

if __name__=='__main__':
    main()
