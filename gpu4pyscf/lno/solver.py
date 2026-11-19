"""GPU restricted CCSD solver for a single LNO local active space.

The fragment driver lives in :mod:`gpu4pyscf.lno.bridge`.  The solver
receives precomputed real RHF MO integrals, initial MP2 amplitudes, and
returns the converged amplitudes to the PySCF-forge fragment projector.
No command-line interface or molecule-wide CCSD driver is defined here.
"""
from __future__ import annotations

import time
import sys
from dataclasses import dataclass

import cupy as cp
import numpy as np
from pyscf.lib import logger
from gpu4pyscf.lib.diis import DIIS


@dataclass
class FixedLAS:
    """Real RHF data for a single fragment's active occupied and virtual MOs."""
    nocc: int
    nvir: int
    fock: np.ndarray
    mo_energy: np.ndarray
    eris: dict
    vvL: np.ndarray | None = None
    df_contract_algorithm: str = 'blocked-eri'
    df_tmp_limit_bytes: int = 384 * 2**20
    df_virtual_block_cap: int = 16


def _pair_indices(n):
    """PySCF lower-triangle order, including both permutations."""
    lower = np.tril_indices(n)
    pairs = np.empty((n, n), dtype=np.int64)
    pairs[lower] = np.arange(n * (n + 1) // 2)
    pairs[(lower[1], lower[0])] = pairs[lower]
    return pairs


class RestrictedCCSD:
    """Restricted CCSD equations for one LNO fragment's active LAS.

    This is a subsystem solver, not the full-molecule LNO-CCSD driver.
    The orbital and integral tensors refer exclusively to the active LAS.
    The driver in bridge.LNOCCSD constructs fragments and projects their
    correlation contributions onto the localized occupied orbitals.
    """
    def __init__(self, fixed: FixedLAS, level_shift=0.0):
        self.fixed = fixed
        self.level_shift = level_shift
        self.no, self.nv = fixed.nocc, fixed.nvir
        self.fock = cp.asarray(fixed.fock, dtype=cp.float64)
        self.mo_energy = cp.asarray(fixed.mo_energy, dtype=cp.float64)
        self.eris = {key: cp.asarray(val, dtype=cp.float64)
                     for key, val in fixed.eris.items() if val is not None}
        npair = self.nv*(self.nv+1)//2
        pair = cp.asarray(_pair_indices(self.nv))
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
            packed = cp.asarray(fixed.vvL, dtype=cp.float64)
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

    def _sample_gpu_pool_used(self):
        """Sample *active* CuPy allocations; unlike pool.total_bytes(), this
        excludes cached free blocks. Samples are not a certified device peak.
        """
        used = int(cp.get_default_memory_pool().used_bytes())
        if (self.gpu_pool_used_sampled_peak_bytes is None or
                used > self.gpu_pool_used_sampled_peak_bytes):
            self.gpu_pool_used_sampled_peak_bytes = used

    def es(self, pattern, *args):
        return cp.einsum(pattern, *args, optimize=True)

    # PySCF 2.13.0 rintermediates.cc_Foo / cc_Fvv / cc_Fov
    def Foo(self, t1, t2):
        e=self.es
        v=self.eris['ovov']
        no=self.no
        z=2*e('kcld,ilcd->ki',v,t2)-e('kdlc,ilcd->ki',v,t2)
        z+=2*e('kcld,ic,ld->ki',v,t1,t1)-e('kdlc,ic,ld->ki',v,t1,t1)
        return z+self.fock[:no,:no]

    def Fvv(self,t1,t2):
        e=self.es
        v=self.eris['ovov']
        no=self.no
        z=-2*e('kcld,klad->ac',v,t2)+e('kdlc,klad->ac',v,t2)
        z+=-2*e('kcld,ka,ld->ac',v,t1,t1)+e('kdlc,ka,ld->ac',v,t1,t1)
        return z+self.fock[no:,no:]

    def Fov(self,t1,t2):
        e=self.es
        v=self.eris['ovov']
        no=self.no
        z=2*e('kcld,ld->kc',v,t1)-e('kdlc,ld->kc',v,t1)
        return z+self.fock[:no,no:]

    def Loo(self,t1,t2):
        e=self.es
        o=self.eris['ovoo']
        no=self.no
        z=self.Foo(t1,t2)+e('kc,ic->ki',self.fock[:no,no:],t1)
        z+=2*e('lcki,lc->ki',o,t1)-e('kcli,lc->ki',o,t1)
        return z

    def Lvv(self,t1,t2):
        e=self.es
        v=self.eris['ovvv']
        no=self.no
        z=self.Fvv(t1,t2)-e('kc,ka->ac',self.fock[:no,no:],t1)
        z+=2*e('kdac,kd->ac',v,t1)-e('kcad,kd->ac',v,t1)
        return z

    def Woooo(self,t1,t2):
        e=self.es
        er=self.eris
        ov=er['ovov']
        oo=er['ovoo']
        z=e('lcki,jc->klij',oo,t1)+e('kclj,ic->klij',oo,t1)
        z+=e('kcld,ijcd->klij',ov,t2)+e('kcld,ic,jd->klij',ov,t1,t1)
        return z+er['oooo'].transpose(0,2,1,3)

    def Wvvvv(self,t1,t2):
        e=self.es
        er=self.eris
        v=er['ovvv']
        z=e('kdac,kb->abcd',v,-t1)-e('kcbd,ka->abcd',v,t1)
        return z+er['vvvv'].transpose(0,2,1,3)

    def _df_aux_batched(self, mats):
        """Legacy auxiliary-batched matmul, retained for numerical A/B checks."""
        nmat,nv,_ = mats.shape
        naux = self.vvL.shape[0]
        per_aux = max(1, 8*nmat*nv*nv*3)  # account for temporary and output
        batch = min(naux, max(1, self.df_tmp_limit_bytes//per_aux))
        out = cp.zeros_like(mats)
        for p0 in range(0,naux,batch):
            l = self.vvL[p0:p0+batch]
            tmp = cp.matmul(l[:,None,:,:], mats[None,:,:,:])
            tmp = cp.matmul(tmp, l[:,None,:,:].transpose(0,1,3,2))
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
        left=cp.ascontiguousarray(self.vvL.transpose(1,2,0).reshape(nv*nv,naux))
        right=cp.ascontiguousarray(self.vvL.reshape(naux,nv*nv))
        tmat=cp.ascontiguousarray(mats.reshape(nmat,nv*nv))
        out=cp.empty((nmat,nv,nv),dtype=cp.float64)
        for a0 in range(0,nv,block):
            a1=min(a0+block,nv)
            a_size=a1-a0
            # (a,c|b,d) in the slab; never build the entire (a,c,b,d).
            slab=left[a0*nv:a1*nv,:] @ right
            # (a,c,b,d) -> (c,d,a,b), then flatten cd and ab.
            self._sample_gpu_pool_used()
            slab_cd_ab=cp.ascontiguousarray(
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
        e=self.es
        nv=self.nv
        no=self.no
        if assume_pair_symmetry:
            ii,jj=cp.tril_indices(no)
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
            full=cp.empty((no,no,nv,nv),dtype=cp.float64)
            full[ii,jj,:,:]=reduced
            full[jj,ii,:,:]=reduced.transpose(0,2,1)
            return full
        return reduced.reshape(no,no,nv,nv)

    def Wvoov(self,t1,t2):
        e=self.es
        er=self.eris
        v=er['ovvv']
        o=er['ovoo']
        ov=er['ovov']
        z=e('kcad,id->akic',v,t1)-e('kcli,la->akic',o,t1)
        z+=er['ovvo'].transpose(2,0,3,1)
        z-=0.5*e('ldkc,ilda->akic',ov,t2)
        z-=0.5*e('lckd,ilad->akic',ov,t2)
        z-=e('ldkc,id,la->akic',ov,t1,t1)
        z+=e('ldkc,ilad->akic',ov,t2)
        return z

    def Wvovo(self,t1,t2):
        e=self.es
        er=self.eris
        v=er['ovvv']
        o=er['ovoo']
        ov=er['ovov']
        z=e('kdac,id->akci',v,t1)-e('lcki,la->akci',o,t1)
        z+=er['oovv'].transpose(2,0,3,1)
        z-=0.5*e('lckd,ilda->akci',ov,t2)
        z-=e('lckd,id,la->akci',ov,t1,t1)
        return z

    def energy(self,t1,t2):
        e=self.es
        ov=self.eris['ovov']
        no=self.no
        tau=t2+e('ia,jb->ijab',t1,t1)
        z=2*e('ia,ia->',self.fock[:no,no:],t1)
        z+=2*e('ijab,iajb->',tau,ov)-e('ijab,ibja->',tau,ov)
        return z

    def update_amps(self,t1,t2):
        """PySCF 2.13 rccsd.update_amps, non-CC2 branch, CuPy-compatible."""
        e=self.es
        no,nv=self.no,self.nv
        er=self.eris
        fock=self.fock
        mo_o=self.mo_energy[:no]
        mo_v=self.mo_energy[no:]+self.level_shift
        fov=fock[:no,no:].copy()
        Foo=self.Foo(t1,t2)
        Fvv=self.Fvv(t1,t2)
        Fov=self.Fov(t1,t2)
        Foo[cp.arange(no),cp.arange(no)]-=mo_o
        Fvv[cp.arange(nv),cp.arange(nv)]-=mo_v

        # PySCF restricted singles residual
        t1new=-2*e('kc,ka,ic->ia',fov,t1,t1)
        t1new+=e('ac,ic->ia',Fvv,t1)-e('ki,ka->ia',Foo,t1)
        t1new+=2*e('kc,kica->ia',Fov,t2)-e('kc,ikca->ia',Fov,t2)
        t1new+=e('kc,ic,ka->ia',Fov,t1,t1)+fov.conj()
        t1new+=2*e('kcai,kc->ia',er['ovvo'],t1)-e('kiac,kc->ia',er['oovv'],t1)
        ovvv=er['ovvv']
        ovoo=er['ovoo']
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
        Loo[cp.arange(no),cp.arange(no)]-=mo_o
        Lvv[cp.arange(nv),cp.arange(nv)]-=mo_v
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

def solve(fixed: FixedLAS, t1_init, t2_init, *, conv_tol=1e-9,
          conv_tol_normt=1e-7, max_cycle=100, diis_space=6,
          diis_start_cycle=1, level_shift=0.0, verbose=0, log=None):
    """Solve one LAS RCCSD problem using CuPy arrays.

    Args:
        fixed: Active-LAS Fock, energies, and MO integral blocks.
        t1_init, t2_init: Initial MP2 amplitudes in the active LAS (NumPy).
        conv_tol, conv_tol_normt: Energy and amplitude convergence criteria.

    Returns:
        (e_corr, t1_host, t2_host, diagnostics).  ``e_corr`` is the
        *unprojected* CCSD energy within this LAS; the LNO fragment energy
        projection is performed in bridge.LNOCCSD.impurity_solve.
    """
    no, nv = fixed.nocc, fixed.nvir
    if np.shape(t1_init) != (no, nv) or np.shape(t2_init) != (no, no, nv, nv):
        raise ValueError('Initial MP2 amplitudes do not match the LAS dimensions')
    tic = time.perf_counter()
    cc = RestrictedCCSD(fixed, level_shift=level_shift)
    t1 = cp.asarray(t1_init, dtype=cp.float64).copy()
    t2 = cp.asarray(t2_init, dtype=cp.float64).copy()
    cp.cuda.get_current_stream().synchronize()
    setup_s = time.perf_counter() - tic

    diis = DIIS(incore=True)
    diis.space = max(1, int(diis_space))
    energy = float(cc.energy(t1, t2).item())
    converged = False
    iterations = 0
    t0 = time.perf_counter()
    log = logger.Logger(sys.stdout, verbose) if log is None else log
    for cycle in range(1, max_cycle + 1):
        iterations = cycle
        t1new, t2new = cc.update_amps(t1, t2)
        vec = cp.concatenate((t1new.ravel(), t2new.ravel()))
        prev = cp.concatenate((t1.ravel(), t2.ravel()))
        error = vec - prev
        err_norm = float(cp.linalg.norm(error).item())
        if cycle >= diis_start_cycle and diis_space > 1:
            vec = diis.update(vec, error)
        t1 = vec[:no*nv].reshape(no, nv)
        t2 = vec[no*nv:].reshape(no, no, nv, nv)
        t2 = 0.5 * (t2 + t2.transpose(1, 0, 3, 2))
        new_energy = float(cc.energy(t1, t2).item())
        de = new_energy - energy
        log.debug('LNO fragment CCSD cycle %d: E=%+.14f  dE=%+.3e  dT=%.3e',
                  cycle, new_energy, de, err_norm)
        if not np.isfinite(new_energy) or not np.isfinite(err_norm):
            raise FloatingPointError('Nonfinite LNO fragment CCSD amplitudes or energy')
        energy = new_energy
        if abs(de) < conv_tol and err_norm < conv_tol_normt:
            converged = True
            break
    cp.cuda.get_current_stream().synchronize()
    iteration_s = time.perf_counter() - t0
    t1_host = cp.asnumpy(t1)
    t2_host = cp.asnumpy(t2)
    return energy, t1_host, t2_host, {
        'converged': converged,
        'iterations': iterations,
        'nocc': no,
        'nvir': nv,
        'energy_Ha': energy,
        'prepare_s': setup_s,
        'iterate_s': iteration_s,
        'eri_representation': 'df-vvL' if fixed.vvL is not None else 'full-vvvv',
        'df_contraction_diagnostics': dict(cc.df_diagnostics) if fixed.vvL is not None else None,
        'gpu_memory_pool_used_sampled_peak_bytes': cc.gpu_pool_used_sampled_peak_bytes,
    }
