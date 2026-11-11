"""04 — global molecular DF (occupied virtual | auxiliary) GPU transformation.

The PySCF-forge LNO interface requires a CPU/HDF5-backed tensor with shape
(ncorrelated_occupied, nvirtual, naux).  The *contraction* is done on the
GPU; DF AO integrals are still generated/streamed by PySCF on the host.
Only real molecular RHF is supported; other systems raise explicitly.
"""
from __future__ import annotations
import time
import numpy as np


def _backend(name):
    if name == 'numpy':
        return np
    if name == 'gpu':
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError('04 global GPU AO2MO requires CuPy') from exc
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('04 global GPU AO2MO requires an NVIDIA GPU')
        return cp
    raise ValueError('Expected gpu or numpy backend')


def _as_host(x, xp):
    return x if xp is np else xp.asnumpy(x)


def _sync(xp):
    if xp is not np:
        xp.cuda.get_current_stream().synchronize()


def transform_ovl(
    with_df, coeff_occ, coeff_vir, out, *, backend='gpu',
    workspace_mib=256, host_scratch_mib=256,
):
    """Stream DF factors and write the (ov|P) matrix into NumPy or HDF5.

    `with_df.loop(blksize)` returns auxiliary-major AO factors in PySCF's
    packed lower-triangle `s2` layout (real RHF); full AO-square layout is
    supported as well.  `out` is NumPy array or h5py dataset.  Avoid any
    persistent O(nAO^4) or O(nAux*nAO^2) intermediates.
    """
    xp = _backend(backend)
    co = np.asarray(coeff_occ)
    cv = np.asarray(coeff_vir)
    if co.dtype != np.float64 or cv.dtype != np.float64:
        raise NotImplementedError('04 supports only real float64 molecular RHF')
    if co.ndim != 2 or cv.ndim != 2 or co.shape[0] != cv.shape[0]:
        raise ValueError('MO coefficient matrices must have matching AO dimensions')
    nao, no = co.shape
    nv = cv.shape[1]
    naux = int(with_df.get_naoaux())
    if out.shape != (no, nv, naux):
        raise ValueError(f'Unexpected ovL storage shape {out.shape}, expected {(no,nv,naux)}')
    if np.dtype(out.dtype) != np.dtype(np.float64):
        raise ValueError('ovL output must be float64')
    if workspace_mib < 1 or host_scratch_mib < 1:
        raise ValueError('Workspace and host-scratch budgets must be positive')
    # GPU transient: unpacked AO L, intermediate L@Cvir, transformed block,
    # plus packed input and a conservative 2x margin for BLAS workspaces.
    bytes_per_aux = 8*(nao*nao + nao*nv + no*nv + nao*(nao+1)//2)
    gpu_budget = workspace_mib * 2**20
    aux_block = min(naux, max(1, int(gpu_budget // (2*bytes_per_aux))))
    if 2*bytes_per_aux > gpu_budget:
        raise MemoryError('04 GPU workspace cannot hold a single unpacked AO factor; '
                          'increase --global-ovl-workspace-mib or use CPU backend')
    # With HDF5, write complete occupied slabs, not millions of thin HDF5
    # hyperslabs.  Full NumPy storage can be written auxiliary-slab-wise.
    is_hdf5 = not isinstance(out, np.ndarray)
    if is_hdf5:
        occ_block = min(no, max(1, (host_scratch_mib*2**20)//max(1, 8*nv*naux)))
        if 8*nv*naux > host_scratch_mib*2**20:
            raise MemoryError('Host scratch too small for one occupied HDF5 slab')
    else:
        occ_block = no
    ti, tj = np.tril_indices(nao)
    ii = xp.asarray(ti)
    jj = xp.asarray(tj)
    cvir = xp.asarray(cv, order='C')
    # Keep occupied coefficients on the device and index only the current
    # occupied slab; this limits transferred coefficient data and GPU memory.
    cocc = xp.asarray(co, order='C')
    max_block_seen = 0
    n_df_passes = 0
    n_df_chunks = 0
    start = time.perf_counter()
    for i0 in range(0,no,occ_block):
        i1 = min(no,i0+occ_block)
        if is_hdf5:
            host_slab = np.empty((i1-i0,nv,naux),dtype=np.float64)
        p = 0
        n_df_passes += 1
        # with_df.loop is the official PySCF DF factor iterator; do not use
        # an AO shell integral assumption or assume a built _cderi ndarray.
        for raw in with_df.loop(blksize=aux_block):
            raw = np.asarray(raw)
            if raw.ndim != 2:
                raise ValueError('DF blocks must be two-dimensional auxiliary-major arrays')
            if raw.dtype != np.float64:
                raise NotImplementedError('Complex/signed periodic DF blocks are unsupported')
            for k0 in range(0,raw.shape[0],aux_block):
                slab = raw[k0:k0+aux_block]
                b = slab.shape[0]
                if p+b > naux:
                    raise RuntimeError('DF loop returned more auxiliary functions than get_naoaux')
                src = xp.asarray(slab)
                if src.shape[1] == nao*(nao+1)//2:
                    # Expand the real symmetric packed AO factor on device.
                    aofac = xp.empty((b,nao,nao),dtype=xp.float64)
                    aofac[:,ii,jj] = src
                    aofac[:,jj,ii] = src
                elif src.shape[1] == nao*nao:
                    aofac = src.reshape(b,nao,nao)
                else:
                    raise ValueError(f'Unexpected DF AO factor width {src.shape[1]}')
                # Two GEMMs: L^P_{mu,nu} C_{nu,a}, then C_{mu,i} L^P_{mu,a}
                # -> (naux_block, nocc_block, nvir).  Real RHF, no conjugation.
                partial = xp.matmul(aofac,cvir)
                transformed = xp.matmul(cocc[:,i0:i1].T[None,:,:],partial)
                block_host = _as_host(transformed.transpose(1,2,0),xp)
                if is_hdf5:
                    host_slab[:,:,p:p+b] = block_host
                else:
                    out[i0:i1,:,p:p+b] = block_host
                p += b
                n_df_chunks += 1
                max_block_seen = max(max_block_seen,b)
                del src,aofac,partial,transformed,block_host
        if p != naux:
            raise RuntimeError(f'DF loop returned {p} of expected {naux} auxiliary functions')
        if is_hdf5:
            out[i0:i1,:,:] = host_slab
            del host_slab
    _sync(xp)
    return {
        'backend':backend,
        'nao':int(nao),'nocc':int(no),'nvir':int(nv),'naux':int(naux),
        'output_storage':'hdf5' if is_hdf5 else 'ram',
        'aux_block_requested':int(aux_block),
        'aux_block_max_seen':int(max_block_seen),
        'occupied_block':int(occ_block),
        'df_stream_passes':int(n_df_passes),
        'df_chunks_processed':int(n_df_chunks),
        'workspace_mib':int(workspace_mib),
        'host_scratch_mib':int(host_scratch_mib),
        'elapsed_transform_s':float(time.perf_counter()-start),
        'ovL_bytes':int(no*nv*naux*8),
    }


def build_global_eris(mylno, *, backend='gpu', workspace_mib=256,
                      host_scratch_mib=256):
    """Build PySCF-forge's original global LNO ERI interface with GPU ovL.

    The original class implements get_occ_blk, xform_occ, etc.; constructing
    it and only replacing its `build` operation guarantees compatibility
    with subsequent LNO MP2 density construction.
    """
    from pyscf.lno import lno as lno_impl
    from pyscf import lib
    import h5py
    if mylno.with_df is None:
        raise RuntimeError('04 requires an RHF with density fitting enabled')
    occ,vir = mylno.split_mo_coeff()[1:3]
    occ=np.asarray(occ); vir=np.asarray(vir)
    if occ.dtype != np.float64 or vir.dtype != np.float64:
        raise NotImplementedError('04 requires real molecular RHF')
    naux=int(mylno.with_df.get_naoaux())
    nocc,nvir=occ.shape[1],vir.shape[1]
    ovl_bytes=nocc*nvir*naux*8
    mem_now=mylno.max_memory-lib.current_memory()[0]
    existing=getattr(mylno,'_ovL',None)
    if existing is not None:
        # Reuse/restart is an I/O operation; the original implementation
        # checks the shape and opens the stored file without transforming.
        if not isinstance(existing,str):
            raise NotImplementedError('04 only supports string paths for precomputed ovL')
        obj=lno_impl._LNODFOUTCOREERIS(
            mylno.with_df,occ,vir,mylno.max_memory,ovL=existing,
            verbose=mylno.verbose,stdout=mylno.stdout)
        obj.build()
        return obj,{
            'backend':'reused-existing','nocc':int(nocc),'nvir':int(nvir),
            'naux':naux,'ovL_bytes':int(ovl_bytes),
            'output_storage':'hdf5','elapsed_transform_s':0.0,
        }
    outcore=bool(getattr(mylno,'force_outcore_ao2mo',False)
                 or getattr(mylno,'_ovL_to_save',None) is not None
                 or ovl_bytes/(2**20) > mem_now*.5)
    if outcore:
        obj=lno_impl._LNODFOUTCOREERIS(
            mylno.with_df,occ,vir,mylno.max_memory,
            ovL=None,ovL_to_save=mylno._ovL_to_save,
            verbose=mylno.verbose,stdout=mylno.stdout)
        target=getattr(mylno,'_ovL_to_save',None)
        obj.feri=h5py.File(target,'w') if isinstance(target,str) else lib.H5TmpFile()
        obj.ovL=obj.feri.create_dataset('ovL',(nocc,nvir,naux),
                                        dtype=np.float64,chunks=(1,nvir,naux))
    else:
        obj=lno_impl._LNODFINCOREERIS(
            mylno.with_df,occ,vir,mylno.max_memory,
            verbose=mylno.verbose,stdout=mylno.stdout)
        obj.ovL=np.empty((nocc,nvir,naux),dtype=np.float64)
    try:
        info=transform_ovl(mylno.with_df,occ,vir,obj.ovL,
                           backend=backend,workspace_mib=workspace_mib,
                           host_scratch_mib=host_scratch_mib)
    except BaseException:
        if outcore:
            obj.feri.close()
        raise
    obj._gpu_global_ovl_info=info
    return obj,info


def compare_global_eris(gpu, reference, *, occ_block=2):
    """Maximum absolute elementwise ovL deviation with bounded host scratch."""
    if (gpu.nocc,gpu.nvir,gpu.naux)!=(reference.nocc,reference.nvir,reference.naux):
        raise AssertionError('CPU/GPU global ovL dimensions differ')
    max_abs=0.0
    for i0 in range(0,gpu.nocc,occ_block):
        i1=min(gpu.nocc,i0+occ_block)
        delta=np.asarray(gpu.get_occ_blk(i0,i1))-np.asarray(reference.get_occ_blk(i0,i1))
        if delta.size:
            max_abs=max(max_abs,float(np.max(np.abs(delta))))
    return max_abs
