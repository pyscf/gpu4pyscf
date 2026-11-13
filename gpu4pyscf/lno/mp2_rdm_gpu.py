#!/usr/bin/env python3
"""03: GPU MP2 one-hole LNO density matrices (molecular real RHF).

Faithfully reproduces the 1h occupied and virtual density definitions in
pyscf-forge/pyscf/lno/make_lno_rdm1.py, with a common T(i,K,a,b) tensor.

This module computes the MP2 *LNO selection densities*, not the fragment
post-CCSD RDM or the projected fragment correlation energy.

The caller still diagonalizes/truncates LNOs using the original PySCF-forge
natorb_select/subspace_eigh machinery. No changes to frozen-core semantics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class OneHoleResult:
    dm_occ: np.ndarray
    dm_vir: np.ndarray
    info: dict


def _subspace_eigh_occ(moeocc, u):
    """Use the same semicanonicalization as the original forge subspace_eigh.

    For a one-dimensional subspace the original *does not rotate* the vector.
    Keep this convention to avoid changing the LNO selection projector.
    """
    u=np.asarray(u)
    energy=np.asarray(moeocc)
    if energy.ndim!=1 or u.ndim!=2 or u.shape[0]!=len(energy):
        raise ValueError('Invalid occupied projection/energies')
    if u.shape[1]<1:
        raise ValueError('At least one localized occupied orbital is required')
    f=(u.T * energy[None,:]) @ u
    if u.shape[1]==1:
        return np.asarray([f[0,0]]), np.array(u,copy=True)
    eps, rot=np.linalg.eigh(f)
    return eps, u@rot


def _get_backend(backend):
    if backend=='numpy':
        return np
    if backend=='gpu':
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount()<1:
            raise RuntimeError('03 GPU RDM: no CUDA device available')
        return cp
    raise ValueError('MP2 RDM backend must be gpu or numpy')


def make_one_hole_density_pair(eris, moeocc, moevir, uocc, *, backend='gpu',
                               workspace_mib=512, occupied_block=None):
    """Compute BOTH 1h MP2 LNO densities from the same MP2 amplitudes.

    Input eris must be the original forge global DF-ovL object, with
    `get_occ_blk(i0,i1)` -> [i,a,P]. uocc is the localized occupied
    subspace projection [i,K] before its semicanonicalization.

    This routine uses canonical denominator eps_i+eps_K-eps_a-eps_b just as
    forge's two `1h` variants do, and returns CPU NumPy arrays for the
    *unchanged* LAS selection code. MP2 amplitudes stay on device.

    Memory is bounded in localized hole K and occupied i. The contraction
    forms no full-system MP2 T2[nocc,nocc,nvir,nvir]. For very large systems
    a predictable budget exception is raised rather than a silent CPU run.
    """
    if workspace_mib<1:
        raise ValueError('workspace_mib must be >=1')
    eo=np.asarray(moeocc)
    ev=np.asarray(moevir)
    if eo.ndim!=1 or ev.ndim!=1 or len(eo)!=int(eris.nocc) or len(ev)!=int(eris.nvir):
        raise ValueError('Global correlated occupied/virtual energy shape mismatch')
    if not (np.isrealobj(eo) and np.isrealobj(ev) and np.isrealobj(uocc)):
        raise NotImplementedError('03 currently supports real closed-shell RHF only')
    if not (np.all(np.isfinite(eo)) and np.all(np.isfinite(ev))):
        raise ValueError('Non-finite orbital energies')
    nocc,nvir,naux=int(eris.nocc),int(eris.nvir),int(eris.naux)
    eps_k,u=_subspace_eigh_occ(eo,uocc)
    nk=len(eps_k)
    xp=_get_backend(backend)
    to_host=(xp.asnumpy if backend=='gpu' else np.asarray)
    t0=time.perf_counter()

    # At most one K at a time; T(i,K,a,b) occupies nocc*nvir**2 entries.
    # Additional matrices in the two RDM GEMMs are budgeted explicitly.
    tbytes=8*nocc*nvir*nvir
    lbytes=8*nocc*nvir*naux
    # Two independent occupied/virtual density workspaces plus factors, T,
    # swapped T, dense denominator, and GEMM output. Conservative bound.
    estimate=int(lbytes + 5*tbytes + 8*(2*nocc*nocc+4*nvir*nvir)
                 + 8*nvir*naux + 8*nocc*nvir)
    if estimate>int(workspace_mib)*2**20:
        raise MemoryError(
            f'03 MP2 RDM estimated workspace {estimate/2**20:.1f} MiB exceeds '
            f'--mp2-rdm-workspace-mib={workspace_mib}. Increase the budget; '
            'this path never silently falls back to CPU.')

    # Forge eris are either RAM or HDF5 backed. get_occ_blk handles both.
    # Original MP2 RDM reads these CPU-hosted global DF factors; Task 04
    # will address the global DF integral construction.
    all_factors=np.asarray(eris.get_occ_blk(0,nocc))
    if all_factors.shape!=(nocc,nvir,naux):
        raise ValueError(f'Expected global ovL[{nocc},{nvir},{naux}], '
                         f'got {all_factors.shape}')
    if not np.isrealobj(all_factors) or not np.all(np.isfinite(all_factors)):
        raise NotImplementedError('Only real finite DF ovL factors supported')
    fac=xp.asarray(np.ascontiguousarray(all_factors,dtype=np.float64))
    del all_factors
    udev=xp.asarray(np.asarray(u,dtype=np.float64))
    eov=xp.asarray(eo[:,None]-ev[None,:],dtype=xp.float64)
    ekv=xp.asarray(eps_k[:,None]-ev[None,:],dtype=xp.float64)
    occ=xp.zeros((nocc,nocc),dtype=xp.float64)
    vir=xp.zeros((nvir,nvir),dtype=xp.float64)

    # Factor contraction is BLAS-3, no nocc**2*nvir**2 amplitude array.
    # L[i,a,P] and K-factor L_K[b,P] share the same DF auxiliary index.
    flat_fac=fac.reshape(nocc*nvir,naux)
    for k in range(nk):
        # K[b,P] = sum_j U[j,K] L[j,b,P]
        loc=xp.tensordot(udev[:,k],fac,axes=(0,0))
        numerator=(flat_fac @ loc.T).reshape(nocc,nvir,nvir)
        denom=eov[:,:,None]+ekv[k][None,None,:]
        if backend=='numpy' and np.any(np.abs(denom)<1e-12):
            raise ZeroDivisionError('Near-zero MP2 denominator in 03')
        t=numerator/denom     # [i,a,b], identical to forge t2iKvv[:,K]
        # Occupied 1h: -4 <iKab|jKab> + 2 <iKab|jKba>.
        a=t.reshape(nocc,-1)
        b=t.transpose(0,2,1).reshape(nocc,-1)
        occ+=-4*(a @ a.T)+2*(a @ b.T)

        # Virtual 1h: T[I,j,a,b] = T[j,I,b,a]. For each localized hole K:
        # +2 (ac|bc) - (ac|cb) - (ca|bc) +2 (ca|cb), in amplitude products.
        # T_vir[j,a,b] == T_occ[j,b,a].
        tv=t.transpose(0,2,1)
        A=tv.transpose(1,0,2).reshape(nvir,-1)
        B=tv.transpose(2,0,1).reshape(nvir,-1)
        vir+=2*(A @ A.T) - (A @ B.T) - (B @ A.T) + 2*(B @ B.T)
        del loc,numerator,denom,t,a,b,tv,A,B

    if backend=='gpu':
        xp.cuda.get_current_stream().synchronize()
    elapsed=time.perf_counter()-t0
    dmo=np.array(to_host(occ),copy=True)
    dmv=np.array(to_host(vir),copy=True)
    # Numerical hermiticity: forge's result is real symmetric to roundoff.
    # Do not symmetrize: verification should observe raw differences.
    info=dict(backend=backend,algorithm='shared-1h-gemm',
              nocc=nocc,nvir=nvir,naux=naux,n_local_holes=nk,
              elapsed_s=float(elapsed),estimated_workspace_bytes=estimate,
              amplitude_tensor_peak_bytes=tbytes,
              occ_asymmetry_max_abs=float(np.max(np.abs(dmo-dmo.T))),
              vir_asymmetry_max_abs=float(np.max(np.abs(dmv-dmv.T))))
    return OneHoleResult(dmo,dmv,info)
