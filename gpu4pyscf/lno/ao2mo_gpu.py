"""02: GPU AO-DF three-center -> molecular fragment LAS MO integrals.

Exact target: the *real RHF molecular* PySCF-forge modified CCSD and
DFCCSD ERI structures.  One-electron terms and frozen-core treatment are
performed by the original _ChemistsERIs._common_init_ and are not replaced.

Algorithm
---------
1. Stream PySCF with_df.loop() AO DF Cholesky factors (AO s2 packed).
2. On GPU, expand AO s2 and apply C^T L[P,AO,AO] C using batched GEMMs.
3. Store the transformed Loo, Lov and packed Lvv factors in bounded host
   scratch (NumPy or HDF5), keeping only AO blocks on device.
4. Contract the LAS integral sectors using chunked GPU GEMMs, writing
   output to PySCF-compatible HDF5 datasets.  Full creates packed vvvv;
   DF writes only vvL and NEVER assembles full vvvv.

The ``numpy`` backend is available for deterministic equation tests but
is NOT a CUDA performance path.  The bridge selects the GPU backend.
"""
from __future__ import annotations

import time
import numpy as np


def _arrays(backend):
    if backend == 'numpy':
        return np
    if backend == 'gpu':
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA device detected for fragment AO2MO')
        return cp
    raise ValueError("ao2mo backend must be 'numpy' or 'gpu'")


def _synchronize(xp):
    if xp is not np:
        xp.cuda.get_current_stream().synchronize()


def _to_cpu(xp, a):
    return np.asarray(a) if xp is np else xp.asnumpy(a)


def transform_ao_df_block(ao_factors, mo_coeff, *, xp=np, packed=True):
    """Return (P,nmo,nmo) real DF 3-center integrals in an AO->MO basis.

    PySCF's AO `s2` order is lower-triangular row-major order.
    The same function is used by synthetic tests on NumPy and by CUDA.
    """
    c = xp.asarray(mo_coeff, dtype=xp.float64)
    z = xp.asarray(ao_factors, dtype=xp.float64)
    if c.ndim != 2 or z.ndim != 2 and z.ndim != 3:
        raise ValueError('Expected MO matrix (nao,nmo) and AO factors')
    nao, _ = c.shape
    npair = nao*(nao+1)//2
    if packed:
        if z.ndim != 2 or z.shape[1] != npair:
            raise ValueError(f'Expected AO s2 shape (naux_block,{npair}), got {z.shape}')
        ind = xp.tril_indices(nao)
        full = xp.zeros((z.shape[0],nao,nao),dtype=xp.float64)
        full[:,ind[0],ind[1]] = z
        full[:,ind[1],ind[0]] = z
    else:
        if z.shape[1:] != (nao,nao):
            raise ValueError('Expected AO s1 factors (naux_block,nao,nao)')
        full = z
    # CuPy batched DGEMM; neither the AO nor the MO pair dimensions are
    # collapsed to create a four-index intermediate.
    intermediate = xp.matmul(full,c)
    result = xp.matmul(c.T[None,:,:],intermediate)
    return result


def _packed_virtual(xp, transformed, no):
    nv = transformed.shape[1]-no
    ind = xp.tril_indices(nv)
    return transformed[:,no+ind[0],no+ind[1]]


def _allocate_factor_scratch(naux,no,nv,host_scratch_mib):
    from pyscf import lib
    dims={'oo':no*no,'ov':no*nv,'vo':no*nv,'vv':nv*(nv+1)//2}
    nbytes=8*naux*sum(dims.values())
    scratch=None
    if nbytes <= int(host_scratch_mib)*2**20:
        out={key:np.empty((naux,n),dtype=np.float64) for key,n in dims.items()}
    else:
        scratch=lib.H5TmpFile()
        # Factor scratch is one-time; HDF5 protects against unbounded host RAM.
        out={key:scratch.create_dataset(key,(naux,n),'f8') for key,n in dims.items()}
    return out,scratch,nbytes


def _contract_rows(xp,A,B,destination,*,workspace_bytes, aux_block_cap=None):
    """Stream (A.T @ B) in output-row / auxiliary blocks to a host dataset.

    A,B shapes: (naux,nleft), (naux,nright). Both may be HDF5 datasets.
    Output may be large, but GPU buffers are bounded by workspace_bytes.
    """
    naux,nleft=A.shape
    if B.shape[0]!=naux or destination.shape!=(nleft,B.shape[1]):
        raise ValueError('Bad GEMM tensor shapes')
    nright=B.shape[1]
    budget=max(8*1024*1024,int(workspace_bytes))
    # An output row can be large; avoid allocating full output on device.
    row_block=max(1,min(nleft,int((budget//2)//max(8,8*nright))))
    for i0 in range(0,nleft,row_block):
        i1=min(nleft,i0+row_block)
        out=xp.zeros((i1-i0,nright),dtype=xp.float64)
        # Remaining budget shared by A(aux,row_block) and B(aux,nright).
        pcap=max(1,int((budget-out.nbytes)//max(8,8*((i1-i0)+nright))))
        if aux_block_cap is not None:
            pcap=min(pcap,int(aux_block_cap))
        for p0 in range(0,naux,pcap):
            p1=min(naux,p0+pcap)
            # Slice is read on CPU, where HDF5 may be the scratch backend.
            aa=xp.asarray(np.asarray(A[p0:p1,i0:i1]),dtype=xp.float64)
            bb=xp.asarray(np.asarray(B[p0:p1,:]),dtype=xp.float64)
            out += xp.matmul(aa.T,bb)
        destination[i0:i1,:]=_to_cpu(xp,out)
    _synchronize(xp)


def _write_mo_eris(eris,fac,*,xp,mode,workspace_bytes,max_full_gib):
    from pyscf import lib
    no=eris.nocc
    nv=eris.fock.shape[0]-no
    p=nv*(nv+1)//2
    file_obj=lib.H5TmpFile()
    eris.feri = file_obj
    eris.oooo=file_obj.create_dataset('oooo',(no,no,no,no),'f8')
    eris.ovoo=file_obj.create_dataset('ovoo',(no,nv,no,no),'f8')
    eris.ovov=file_obj.create_dataset('ovov',(no,nv,no,nv),'f8')
    eris.ovvo=file_obj.create_dataset('ovvo',(no,nv,nv,no),'f8')
    eris.oovv=file_obj.create_dataset('oovv',(no,no,nv,nv),'f8')
    eris.ovvv=file_obj.create_dataset('ovvv',(no,nv,p),'f8')
    if mode == 'full':
        full_bytes=8*p*p
        if full_bytes>float(max_full_gib)*2**30:
            raise MemoryError(f'Packed vvvv would require {full_bytes/2**30:.2f} GiB of scratch; '
                              'use --eri-mode df or explicitly increase --ao2mo-max-full-gib')
        eris.vvvv=file_obj.create_dataset('vvvv',(p,p),'f8')
    else:
        naux=fac['vv'].shape[0]
        eris.vvL=file_obj.create_dataset('vvL',(p,naux),'f8')
        # Read/write row blocks so factor scratch can be disk backed.
        for p0 in range(0,p,512):
            eris.vvL[p0:p0+512,:]=np.asarray(fac['vv'][:,p0:p0+512]).T
    # Explicit chemists' ordering; matches PySCF-forge's definitions:
    # (ov|oo), (ov|ov), (ov|vo), (oo|vv), (ov|vv) and packed (vv|vv).
    sectors=[('oooo','oo','oo',(no,no,no,no)),
             ('ovoo','ov','oo',(no,nv,no,no)),
             ('ovov','ov','ov',(no,nv,no,nv)),
             ('ovvo','ov','vo',(no,nv,nv,no)),
             ('ovvv','ov','vv',(no,nv,p))]
    for key,xa,xb,shape in sectors:
        dest=getattr(eris,key)
        left,right=fac[xa].shape[1],fac[xb].shape[1]
        flatrow=max(1,min(left,int((workspace_bytes//2)//max(8,8*right))))
        A,B=fac[xa],fac[xb]
        # Write each row slab directly to HDF5 using the original tensor
        # axes; never materialize the entire ovvv or ovoo sector on host.
        stride=shape[1]
        inner=shape[2:]
        for r0 in range(0,left,flatrow):
            r1=min(left,r0+flatrow)
            out=xp.zeros((r1-r0,right),dtype=xp.float64)
            pcap=max(1,int((workspace_bytes-out.nbytes)//max(8,8*((r1-r0)+right))))
            for k0 in range(0,A.shape[0],pcap):
                k1=min(A.shape[0],k0+pcap)
                ag=xp.asarray(np.asarray(A[k0:k1,r0:r1]))
                bg=xp.asarray(np.asarray(B[k0:k1,:]))
                out += ag.T @ bg
            slab=_to_cpu(xp,out)
            pos=r0
            while pos<r1:
                first,second=divmod(pos,stride)
                count=min(r1-pos,stride-second)
                dest[first,second:second+count]=slab[pos-r0:pos-r0+count].reshape((count,)+inner)
                pos+=count
    # oovv is (oo|vv); build a packed-virtual intermediate in RAM or
    # HDF5 depending on the budget, then unpack one occupied pair at a
    # time. This avoids a full host-side (no**2,nv,nv) scratch allocation.
    nocc_pair_rows=no*no
    packed_bytes=8*nocc_pair_rows*p
    if packed_bytes<=workspace_bytes:
        temp=np.empty((nocc_pair_rows,p),dtype=np.float64)
        tmp_on_disk=False
    else:
        temp=file_obj.create_dataset('_oovv_tril_scratch',(nocc_pair_rows,p),'f8')
        tmp_on_disk=True
    _contract_rows(xp,fac['oo'],fac['vv'],temp,workspace_bytes=workspace_bytes)
    tri=np.tril_indices(nv)
    for i in range(no):
        for j in range(no):
            packed=np.asarray(temp[i*no+j,:])
            sq=np.empty((nv,nv),dtype=np.float64)
            sq[tri[0],tri[1]]=packed
            sq[tri[1],tri[0]]=packed
            eris.oovv[i,j]=sq
    del temp
    if tmp_on_disk:
        del file_obj['_oovv_tril_scratch']
    if mode=='full':
        _contract_rows(xp,fac['vv'],fac['vv'],eris.vvvv,workspace_bytes=workspace_bytes)
    _synchronize(xp)
    return eris


def fragment_ao2mo(mcc,mo_coeff=None,*,eri_mode='full',backend='gpu',
                    workspace_mib=256,ao_block_mib=96,
                    host_scratch_mib=1024,max_full_gib=16):
    """Construct PySCF-forge-compatible LAS ERIs using GPU GEMMs.

    Preserves original PySCF-forge `_common_init_` for effective Fock,
    including all frozen-occupied/SCF environmental terms.  Requires an
    RHF molecular SCF with DF AO factors. Does not support complex/PBC.
    """
    if eri_mode not in ('full','df'):
        raise ValueError('eri_mode must be full or df (never auto)')
    if min(workspace_mib,ao_block_mib,host_scratch_mib,max_full_gib)<=0:
        raise ValueError('All integral workspace budgets must be positive')
    from pyscf.lno import lnoccsd as impl
    if hasattr(mcc._scf,'kpt') or hasattr(mcc._scf,'kpts'):
        raise NotImplementedError('02 supports molecular RHF only, not PBC')
    df=getattr(mcc._scf,'with_df',None)
    if df is None:
        raise ValueError('02 requires DF-RHF AO factors (mf.with_df)')
    xp=_arrays(backend)
    if eri_mode=='df':
        eris=impl._DFChemistsERIs()
    else:
        eris=impl._ChemistsERIs()
    # This calls the exact original Fock/HF initialization and removes
    # frozen MOs via _mo_without_core. Do NOT form Fock from LAS occ only.
    eris._common_init_(mcc,mo_coeff)
    coeff=np.asarray(eris.mo_coeff)
    if coeff.dtype!=np.float64 or not np.isrealobj(coeff):
        raise NotImplementedError('02 supports real float64 molecular RHF only')
    nao,nmo=coeff.shape
    no=eris.nocc
    nv=nmo-no
    if no<=0 or nv<=0:
        raise ValueError('Empty LAS occupied or virtual subspace')
    naux=int(df.get_naoaux())
    if eri_mode=='full':
        npair=nv*(nv+1)//2
        if (8*npair*npair)>float(max_full_gib)*2**30:
            raise MemoryError(
                f'Packed vvvv exceeds {max_full_gib:g} GiB storage budget; '
                'request --eri-mode df or raise --ao2mo-max-full-gib explicitly')
    factors,scratch,host_bytes=_allocate_factor_scratch(naux,no,nv,host_scratch_mib)
    t0=time.perf_counter()
    # Determine number of AO factors by workspace. PySCF may choose a larger
    # default with_df.loop() block; split the yielded block without copy.
    bytes_per_factor=8*(nao*nao + nao*nmo + nmo*nmo + nao*(nao+1)//2)
    block_cap=max(1,int(ao_block_mib*2**20)//bytes_per_factor)
    tri=xp.tril_indices(nv)
    aux_offset=0
    for ao_block in df.loop():
        raw=np.asarray(ao_block)
        if raw.ndim!=2 or raw.shape[1]!=nao*(nao+1)//2 or raw.dtype!=np.float64:
            raise ValueError(f'PySCF DF AO blocks must be float64 s2 packed; got {raw.shape}, {raw.dtype}')
        for start in range(0,raw.shape[0],block_cap):
            stop=min(raw.shape[0],start+block_cap)
            npart=stop-start
            mo=transform_ao_df_block(raw[start:stop],coeff,xp=xp,packed=True)
            factors['oo'][aux_offset:aux_offset+npart,:]=_to_cpu(xp,mo[:,:no,:no].reshape(npart,-1))
            factors['ov'][aux_offset:aux_offset+npart,:]=_to_cpu(xp,mo[:,:no,no:].reshape(npart,-1))
            factors['vo'][aux_offset:aux_offset+npart,:]=_to_cpu(xp,mo[:,no:,:no].reshape(npart,-1))
            factors['vv'][aux_offset:aux_offset+npart,:]=_to_cpu(xp,mo[:,no+tri[0],no+tri[1]])
            aux_offset+=npart
            del mo
    if aux_offset!=naux:
        raise RuntimeError(f'DF loop produced {aux_offset} aux vectors; with_df reports {naux}')
    ao2mo_s=time.perf_counter()-t0
    result=_write_mo_eris(eris,factors,xp=xp,mode=eri_mode,
                         workspace_bytes=int(workspace_mib)*2**20,
                         max_full_gib=max_full_gib)
    _synchronize(xp)
    result._gpu_ao2mo_info={
        'backend':backend,'eri_mode':eri_mode,'nao':nao,'nmo':nmo,
        'naux':naux,'ao_df_mo_transform_s':ao2mo_s,
        'total_gpu_ao2mo_s':time.perf_counter()-t0,
        'factor_scratch_bytes':host_bytes,
        'scratch_storage':'hdf5' if scratch is not None else 'ram',
        'gpu_workspace_mib':int(workspace_mib),
        'ao_block_mib':int(ao_block_mib),
    }
    # HDF5 scratch must remain accessible until all block products finished.
    if scratch is not None:
        scratch.close()
    return result


def compare_eri_objects(gpu,reference,*,eri_mode):
    """Return per-tensor max absolute errors for the ORIGINAL CPU AO2MO.

    This compares every sector, not merely CCSD energies or LAS overlap.
    The frozen-core/environmental Fock and MO denominators are included.
    """
    keys=['fock','mo_energy','oooo','ovoo','ovov','ovvo','oovv','ovvv']
    keys.append('vvvv' if eri_mode=='full' else 'vvL')
    errs={}
    for k in keys:
        ga=np.asarray(getattr(gpu,k))
        ref=np.asarray(getattr(reference,k))
        if ga.shape!=ref.shape:
            raise AssertionError(f'AO2MO mismatch in {k}: {ga.shape} != {ref.shape}')
        errs[k]=float(np.max(np.abs(ga-ref))) if ga.size else 0.0
    return errs
