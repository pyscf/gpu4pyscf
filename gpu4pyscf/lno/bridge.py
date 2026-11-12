#!/usr/bin/env python3
"""Connect GPU LNO components to a molecular RHF PySCF-forge calculation.

PySCF-forge retains LNO selection and fragment-energy projection on the CPU.
The bridge temporarily replaces selected instance methods and restores them.
"""
from __future__ import annotations

import time
import inspect
from types import MethodType
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .solver import RestrictedCCSD, _pair_indices
from .diis import ResidentDIIS, ScalarResidentDIIS


_REQUIRED_BLOCKS = ('oooo','ovoo','ovov','ovvo','oovv','ovvv')


def _numpy_eri(eris, key):
    """Read h5py datasets and NumPy arrays without retaining file handles."""
    x = getattr(eris, key, None)
    if x is None:
        raise ValueError(f'Missing required MO integral block: {key}')
    return np.array(x, copy=True)


def _prepared_eri_from_eris(eris, nocc, nvir):
    """Read the PySCF fragment's MO ERI blocks without any AO integration.

    Keep packed ovvv/vvvv and packed DF vvL as supplied; unpacking takes
    place on device in the GPU solver. Never assemble four-virtual integrals
    when DF factors are provided.
    Returns (block dict, packed DF factors or None).
    """
    out = {name: _numpy_eri(eris,name) for name in _REQUIRED_BLOCKS}
    p = nvir*(nvir+1)//2
    if out['ovvv'].shape not in ((nocc,nvir,p), (nocc,nvir,nvir,nvir)):
        raise ValueError(f'Unexpected ovvv shape {out["ovvv"].shape}')
    vvL_obj = getattr(eris,'vvL',None)
    vvvv_obj = getattr(eris,'vvvv',None)
    if vvL_obj is not None and vvvv_obj is None:
        vvL = np.array(vvL_obj,copy=True)
        if vvL.shape[0] != p or vvL.ndim!=2:
            raise ValueError(f'Expected vvL[{p},naux], got {vvL.shape}')
        out['vvvv'] = None
    elif vvvv_obj is not None:
        vvL = None
        out['vvvv']=np.array(vvvv_obj,copy=True)
        if out['vvvv'].shape not in ((p,p),(nvir,nvir,nvir,nvir)):
            raise ValueError(f'Unexpected vvvv shape {out["vvvv"].shape}')
    else:
        raise ValueError('Neither DF vvL nor vvvv present in fragment ERIs')
    shapes=dict(oooo=(nocc,)*4,ovoo=(nocc,nvir,nocc,nocc),
                ovov=(nocc,nvir,nocc,nvir),ovvo=(nocc,nvir,nvir,nocc),
                oovv=(nocc,nocc,nvir,nvir))
    for key, shape in shapes.items():
        if out[key].shape != shape:
            raise ValueError(f'Unexpected shape of {key}: {out[key].shape}, expected {shape}')
    for key, val in out.items():
        if val is not None and (not np.isrealobj(val) or val.dtype!=np.float64):
            raise NotImplementedError(f'{key}: only float64 real MO integrals supported')
    if vvL is not None and (not np.isrealobj(vvL) or vvL.dtype!=np.float64):
        raise NotImplementedError('Only float64 real DF factors supported')
    return out,vvL


def _fixed_from_live_eris(eris, uocc_loc):
    """Create the duck-typed input expected by RestrictedCCSD."""
    no = int(eris.nocc)
    fock = np.asarray(eris.fock)
    nv = fock.shape[0]-no
    if no <= 0 or nv <= 0 or fock.shape != (no+nv,no+nv):
        raise ValueError('Invalid number of active occupied/virtual orbitals')
    ene = np.asarray(eris.mo_energy)
    if ene.shape != (no+nv,) or not np.allclose(np.diag(fock),ene,atol=1e-8):
        raise ValueError('Invalid fragment orbital energies')
    if not np.isrealobj(fock) or not np.isrealobj(ene) or not np.isrealobj(uocc_loc):
        raise NotImplementedError('Only real restricted closed-shell orbitals supported')
    if uocc_loc.shape[0] != no:
        raise ValueError('Fragment projector has incompatible dimensions')
    blocks, vvL = _prepared_eri_from_eris(eris,no,nv)
    return SimpleNamespace(nocc=no,nvir=nv, fock=np.array(fock,copy=True),
                           mo_energy=np.array(ene,copy=True), eris=blocks,
                           vvL=vvL, uocc_loc=np.array(uocc_loc,copy=True))


def estimate_gpu_working_bytes(no,nv,df_naux=None,df_tmp_mib=384,
                               df_virtual_block=16):
    """Conservative size heuristic, NOT an observed CUDA memory peak.

    DF includes both unpacked L[P,a,c] and two/three transient GEMM slabs.
    The bound is intentionally pessimistic for fallback decisions and does
    not constrain CuPy's caching allocator or vendor library workspaces.
    """
    base = 8 * (32*no*no*nv*nv + 16*no**4 + 12*no*nv**3 + 16*no**3*nv)
    if df_naux is None:
        return int(base + 8*16*nv**4)
    slab_limit=int(df_tmp_mib)*2**20
    bytes_per_a=8*nv**3
    if 3*bytes_per_a>slab_limit:
        # Auxiliary-batched fallback may still require a substantial workspace.
        additional=max(3*bytes_per_a,3*8*no*no*nv*nv)
    else:
        a_block=min(max(1,nv-1),int(df_virtual_block),
                    max(1,slab_limit//(3*bytes_per_a)))
        additional=3*bytes_per_a*a_block
    return int(base + 2*8*nv*nv*df_naux + additional)


def run_live_fragment_ccsd(eris,t1_init,t2_init,uocc_loc,
                           backend='gpu',conv_tol=1e-9,
                           conv_tol_normt=1e-7,max_cycle=100,
                           diis_space=6,diis_start_cycle=1,level_shift=0.0,
                           verbose=False,diis_mode='batched',
                           df_algorithm='blocked-eri',df_tmp_mib=384,
                           df_virtual_block=16):
    """Run PySCF-style CCSD equations on an existing active-space MO ERI set.

    Returns (e_corr, t1_numpy, t2_numpy, diagnostics).
    Input amplitudes are never modified. Output arrays are on CPU because
    the unchanged LNO fragment projection operates on NumPy arrays.
    """
    if backend == 'gpu':
        import cupy as xp
        if xp.cuda.runtime.getDeviceCount()<1:
            raise RuntimeError('No CUDA device found')
    elif backend == 'numpy':
        xp=np
    else:
        raise ValueError('backend must be gpu or numpy')

    pool=(xp.get_default_memory_pool() if backend=='gpu' else None)
    pool_reserved_before=(int(pool.total_bytes()) if pool is not None else None)
    pool_reserved_observed_max=pool_reserved_before
    t_prepare=time.perf_counter()
    fixed=_fixed_from_live_eris(eris,np.asarray(uocc_loc))
    no,nv=fixed.nocc,fixed.nvir
    if np.asarray(t1_init).shape != (no,nv) or np.asarray(t2_init).shape != (no,no,nv,nv):
        raise ValueError('MP2 amplitude shapes incompatible with active LAS')
    fixed.df_contract_algorithm=df_algorithm
    fixed.df_tmp_limit_bytes=int(df_tmp_mib)*2**20
    fixed.df_virtual_block_cap=int(df_virtual_block)
    cc=RestrictedCCSD(fixed,xp,level_shift=level_shift)
    t1=xp.asarray(t1_init,dtype=xp.float64).copy()
    t2=xp.asarray(t2_init,dtype=xp.float64).copy()
    if backend=='gpu':
        xp.cuda.get_current_stream().synchronize()
        pool_reserved_observed_max=max(pool_reserved_observed_max,int(pool.total_bytes()))
    prepare_s=time.perf_counter()-t_prepare

    if diis_mode not in ('scalar', 'batched'):
        raise ValueError('diis_mode must be scalar or batched')
    # Preserve both DIIS implementations for cross-checks.
    if diis_mode=='scalar':
        diis=ScalarResidentDIIS(xp,space=diis_space)
    else:
        diis=ResidentDIIS(xp,space=diis_space)
    old_e=float(cc.energy(t1,t2).item())
    converged=False
    start=time.perf_counter()
    for cycle in range(1,max_cycle+1):
        t1next,t2next=cc.update_amps(t1,t2)
        vec=xp.concatenate((t1next.ravel(),t2next.ravel()))
        prev=xp.concatenate((t1.ravel(),t2.ravel()))
        error=vec-prev
        err_norm=float(xp.linalg.norm(error).item())
        if cycle>=diis_start_cycle and diis_space>1:
            vec=diis.extrapolate(vec,error)
        t1=vec[:no*nv].reshape(no,nv)
        t2=vec[no*nv:].reshape(no,no,nv,nv)
        t2=(t2+t2.transpose(1,0,3,2))*.5
        energy=float(cc.energy(t1,t2).item())
        de=energy-old_e
        if verbose:
            print(f'GPU-LNO RCCSD iter {cycle:3d}: E={energy:.14f} dE={de:+.2e} |dT|={err_norm:.2e}',flush=True)
        if pool is not None:
            # MemoryPool.total_bytes tracks cached and active allocations.
            # A high-water *sample* is useful for A/B comparisons but is not
            # an authoritative CUDA device peak allocation.
            pool_reserved_observed_max=max(pool_reserved_observed_max,int(pool.total_bytes()))
        if not np.isfinite(energy) or not np.isfinite(err_norm):
            raise FloatingPointError('Non-finite CCSD iterate encountered')
        if abs(de)<conv_tol and err_norm<conv_tol_normt:
            converged=True
            break
        old_e=energy
    if backend=='gpu': xp.cuda.get_current_stream().synchronize()
    iterate_s=time.perf_counter()-start
    t_copy=time.perf_counter()
    to_cpu=(xp.asnumpy if backend=='gpu' else np.asarray)
    t1_cpu=np.array(to_cpu(t1),copy=True)
    t2_cpu=np.array(to_cpu(t2),copy=True)
    copy_s=time.perf_counter()-t_copy
    return energy,t1_cpu,t2_cpu,dict(nocc=no,nvir=nv,nmo=no+nv,
           iterations=cycle,converged=converged,
           energy_Ha=energy,prepare_s=prepare_s,iterate_s=iterate_s,
           copy_s=copy_s,diis_mode=diis_mode,
           eri_representation=('df-vvL' if fixed.vvL is not None else 'full-vvvv'),
           estimated_working_bytes=estimate_gpu_working_bytes(no,nv,
               None if fixed.vvL is None else fixed.vvL.shape[1],
               df_tmp_mib=df_tmp_mib,df_virtual_block=df_virtual_block),
           gpu_memory_pool_reserved_before_bytes=pool_reserved_before,
           gpu_memory_pool_reserved_max_observed_bytes=pool_reserved_observed_max,
           gpu_memory_pool_active_sampled_peak_bytes=cc.gpu_pool_used_sampled_peak_bytes,
           df_contract_algorithm=(cc.df_contract_algorithm if fixed.vvL is not None else None),
           df_contraction_diagnostics=(dict(cc.df_diagnostics) if fixed.vvL is not None else None))


@dataclass
class FragmentRun:
    fragment: int
    nmo: int
    backend: str
    total_s: float
    e_mp2_Ha: float
    e_ccsd_Ha: float
    solver_info: dict | None = None


class LNOCCSDGPUBridge:
    """Temporarily substitute the CCSD kernel within mylno.impurity_solve.

    This wrapper is intentionally non-invasive: restoring the original bound
    method on exit is enough to restore the standard CPU behavior.
    Use one context per complete serial molecular RHF LNOCCSD.kernel call.
    """
    def __init__(self,mylno,backend='gpu',min_gpu_nmo=20,
                 device_memory_fraction=0.65,strict_convergence=False,
                 verbose=False,diis_mode='batched',eri_mode='full',
                 allow_memory_fallback=True,
                 df_algorithm='blocked-eri',df_tmp_mib=384,
                 df_virtual_block=16,
                 ao2mo_backend='cpu', verify_ao2mo=False,
                 ao2mo_workspace_mib=256, ao2mo_ao_block_mib=96,
                 ao2mo_host_scratch_mib=1024, ao2mo_max_full_gib=16,
                 mp2_rdm_backend='cpu',verify_mp2_rdm=False,
                 mp2_rdm_workspace_mib=512,
                 global_ovl_backend='cpu', verify_global_ovl=False,
                 global_ovl_workspace_mib=256, global_ovl_host_scratch_mib=256):
        if diis_mode not in ('scalar', 'batched'):
            raise ValueError('diis_mode must be scalar or batched')
        if df_algorithm not in ('blocked-eri','aux-batched'):
            raise ValueError('df_algorithm must be blocked-eri or aux-batched')
        if int(df_tmp_mib)<1 or int(df_virtual_block)<1:
            raise ValueError('DF buffer and block sizes must be positive')
        self.df_algorithm=df_algorithm
        self.df_tmp_mib=int(df_tmp_mib)
        self.df_virtual_block=int(df_virtual_block)
        self.diis_mode=diis_mode
        if backend not in ('gpu','numpy','cpu'):
            raise ValueError('backend must be cpu, gpu, or numpy')
        if eri_mode not in ('full','df'):
            raise ValueError('eri_mode must be full or df; full is the default')
        if ao2mo_backend not in ('cpu','gpu','numpy'):
            raise ValueError('ao2mo_backend must be cpu, gpu or numpy')
        if ao2mo_backend=='gpu' and backend!='gpu':
            raise ValueError('GPU AO2MO requires GPU CCSD backend')
        if min(ao2mo_workspace_mib,ao2mo_ao_block_mib,
               ao2mo_host_scratch_mib,ao2mo_max_full_gib)<=0:
            raise ValueError('AO2MO workspace budgets must be positive')
        self.ao2mo_backend=ao2mo_backend
        self.verify_ao2mo=bool(verify_ao2mo)
        self.ao2mo_workspace_mib=int(ao2mo_workspace_mib)
        self.ao2mo_ao_block_mib=int(ao2mo_ao_block_mib)
        self.ao2mo_host_scratch_mib=int(ao2mo_host_scratch_mib)
        self.ao2mo_max_full_gib=float(ao2mo_max_full_gib)
        if mp2_rdm_backend not in ('cpu','gpu','numpy'):
            raise ValueError('MP2 RDM backend must be cpu, gpu, or numpy')
        if mp2_rdm_backend=='gpu' and backend!='gpu':
            raise ValueError('GPU MP2 RDM requires GPU backend')
        if int(mp2_rdm_workspace_mib)<1:
            raise ValueError('MP2 RDM workspace must be positive')
        self.mp2_rdm_backend=mp2_rdm_backend
        self.verify_mp2_rdm=bool(verify_mp2_rdm)
        self.mp2_rdm_workspace_mib=int(mp2_rdm_workspace_mib)
        if global_ovl_backend not in ('cpu','gpu','numpy'):
            raise ValueError('Global ovL backend must be cpu, gpu, or numpy')
        if global_ovl_backend=='gpu' and backend!='gpu':
            raise ValueError('GPU global ovL transformation requires GPU backend')
        if int(global_ovl_workspace_mib)<1 or int(global_ovl_host_scratch_mib)<1:
            raise ValueError('04 global ovL workspace budgets must be positive')
        self.global_ovl_backend=global_ovl_backend
        self.verify_global_ovl=bool(verify_global_ovl)
        self.global_ovl_workspace_mib=int(global_ovl_workspace_mib)
        self.global_ovl_host_scratch_mib=int(global_ovl_host_scratch_mib)
        self.global_ovl_info=None
        self._orig_global_ao2mo=None
        self.mp2_rdm_records=[]
        self._mp2_rdm_pending=None
        self._orig_mp2_rdm_occ=None
        self._orig_mp2_rdm_vir=None
        self.eri_mode=eri_mode
        self.allow_memory_fallback=bool(allow_memory_fallback)
        self.mylno=mylno
        self.backend=backend
        self.min_gpu_nmo=int(min_gpu_nmo)
        self.device_memory_fraction=float(device_memory_fraction)
        self.strict_convergence=bool(strict_convergence)
        self.verbose=bool(verbose)
        self.records=[]
        self.active_projectors=[]   # AO-basis projectors for subspace comparison
        self._orig_impurity=None
        # Public, backend-independent wall-time statistics (seconds and calls).
        self.stage_wall_s={key:0.0 for key in ('04','03','02','01')}
        self.stage_calls={key:0 for key in ('04','03','02','01')}
        self.bridge_wall_s=None
        self._entered_at=None

    @staticmethod
    def _synchronize(backend):
        """Finish asynchronous CUDA work at a measurement boundary."""
        if backend=='gpu':
            import cupy as cp
            cp.cuda.get_current_stream().synchronize()

    def _record_stage(self,stage,seconds):
        self.stage_wall_s[stage]+=float(seconds)
        self.stage_calls[stage]+=1

    def timing_summary(self):
        """Return accumulated 04/03/02/01 wall times and call counts.

        Stage 03 includes BOTH occupied and virtual RDM hooks, but GPU
        computes the pair in the first hook; the second is only a cached read.
        Stage 01 spans the CCSD kernel (including host/device transfers).
        """
        result={key:{'wall_s':self.stage_wall_s[key],
                     'calls':self.stage_calls[key]}
                for key in ('04','03','02','01')}
        result['bridge_wall_s']=self.bridge_wall_s
        result['other_wall_s']=(None if self.bridge_wall_s is None else
            max(0.0,self.bridge_wall_s-sum(self.stage_wall_s.values())))
        return result

    def timing_report(self):
        """Multi-line report suitable for PySCF's standard output file."""
        labels=(('04','Global AO-to-MO'),
                ('03','MP2 1h 1-RDM (occ+vir)'),
                ('02','Fragment AO-to-LAS'),
                ('01','Fragment CCSD kernel'))
        lines=['========== LNO-CCSD Stage Wall Times ==========']
        for key,label in labels:
            lines.append(f'{key} {label:<27s}: '
                         f'{self.stage_wall_s[key]:12.3f} s  '
                         f'({self.stage_calls[key]} calls)')
        if self.bridge_wall_s is not None:
            other=max(0.0,self.bridge_wall_s-sum(self.stage_wall_s.values()))
            lines.append(f'Other LNO operations             : {other:12.3f} s')
            lines.append(f'LNO bridge wall time             : {self.bridge_wall_s:12.3f} s')
        lines.append('===============================================')
        return '\n'.join(lines)

    def __enter__(self):
        if self._orig_impurity is not None:
            raise RuntimeError('LNOCCSDGPUBridge is not reentrant')
        if getattr(self.mylno,'ccsd_t',False):
            raise NotImplementedError('GPU-LNO bridge supports CCSD only, not CCSD(T)')
        if getattr(self.mylno,'prune_lno_basis',False):
            raise NotImplementedError('Pruned AO-domain LNO has not been checked')
        if self.backend=='gpu':
            import cupy
            if cupy.cuda.runtime.getDeviceCount()<1:
                raise RuntimeError('CUDA GPU unavailable')
        try:
            # Wrap BOTH CPU originals and GPU implementations at their
            # public interfaces so the measured boundaries are identical.
            if not hasattr(self.mylno,'ao2mo'):
                raise RuntimeError('LNO object lacks global ao2mo')
            self._orig_global_ao2mo=self.mylno.ao2mo

            def timed_global_ao2mo(*args,**kwargs):
                impl=(self._orig_global_ao2mo if
                      self.global_ovl_backend=='cpu' else self._global_ao2mo)
                self._synchronize(self.global_ovl_backend)
                t0=time.perf_counter()
                try:
                    return impl(*args,**kwargs)
                finally:
                    self._synchronize(self.global_ovl_backend)
                    self._record_stage('04',time.perf_counter()-t0)

            self.mylno.ao2mo=timed_global_ao2mo
            self._orig_impurity=self.mylno.impurity_solve
            if not (hasattr(self.mylno,'make_lo_rdm1_occ') and
                    hasattr(self.mylno,'make_lo_rdm1_vir')):
                raise RuntimeError('LNO object lacks one-hole MP2 RDM hooks')
            self._orig_mp2_rdm_occ=self.mylno.make_lo_rdm1_occ
            self._orig_mp2_rdm_vir=self.mylno.make_lo_rdm1_vir

            def timed_mp2_occ(*args,**kwargs):
                impl=(self._orig_mp2_rdm_occ if
                      self.mp2_rdm_backend=='cpu' else self._mp2_rdm_occ)
                self._synchronize(self.mp2_rdm_backend)
                t0=time.perf_counter()
                try:
                    return impl(*args,**kwargs)
                finally:
                    self._synchronize(self.mp2_rdm_backend)
                    self._record_stage('03',time.perf_counter()-t0)

            def timed_mp2_vir(*args,**kwargs):
                impl=(self._orig_mp2_rdm_vir if
                      self.mp2_rdm_backend=='cpu' else self._mp2_rdm_vir)
                self._synchronize(self.mp2_rdm_backend)
                t0=time.perf_counter()
                try:
                    return impl(*args,**kwargs)
                finally:
                    self._synchronize(self.mp2_rdm_backend)
                    self._record_stage('03',time.perf_counter()-t0)

            self.mylno.make_lo_rdm1_occ=timed_mp2_occ
            self.mylno.make_lo_rdm1_vir=timed_mp2_vir
            self.mylno.impurity_solve=self._wrapped_impurity
            self._entered_at=time.perf_counter()
        except BaseException:
            self.__exit__(None,None,None)
            raise
        return self

    def __exit__(self,exc_type,exc,tb):
        if self._entered_at is not None:
            self._synchronize(self.backend)
            self.bridge_wall_s=time.perf_counter()-self._entered_at
            self._entered_at=None
        if self._orig_impurity is not None:
            self.mylno.impurity_solve=self._orig_impurity
        if self._orig_global_ao2mo is not None:
            self.mylno.ao2mo=self._orig_global_ao2mo
        if self._orig_mp2_rdm_occ is not None:
            self.mylno.make_lo_rdm1_occ=self._orig_mp2_rdm_occ
        if self._orig_mp2_rdm_vir is not None:
            self.mylno.make_lo_rdm1_vir=self._orig_mp2_rdm_vir
        self._orig_global_ao2mo=None
        self._orig_mp2_rdm_occ=None
        self._orig_mp2_rdm_vir=None
        self._mp2_rdm_pending=None
        self._orig_impurity=None
        return False

    def _global_ao2mo(self):
        from .global_ovl_gpu import build_global_eris, compare_global_eris
        if self.verify_global_ovl and getattr(self.mylno,'_ovL_to_save',None) is not None:
            raise ValueError('04 --verify-global-ovl cannot overwrite a user ovL save path')
        if self.verify_global_ovl and getattr(self.mylno,'_ovL',None) is not None:
            raise ValueError('04 verification requires newly computed ovL, not a previously saved file')
        t0=time.perf_counter()
        eris,info=build_global_eris(
            self.mylno,backend=self.global_ovl_backend,
            workspace_mib=self.global_ovl_workspace_mib,
            host_scratch_mib=self.global_ovl_host_scratch_mib)
        info=dict(info)
        info['total_global_ao2mo_s']=float(time.perf_counter()-t0)
        info['cpu_reference_max_abs']=None
        if self.verify_global_ovl:
            reference=self._orig_global_ao2mo()
            try:
                err=compare_global_eris(eris,reference)
            finally:
                file_handle=getattr(reference,'feri',None)
                if file_handle is not None:
                    file_handle.close()
            info['cpu_reference_max_abs']=float(err)
            print(f'[04 GLOBAL ovL VERIFY] nocc={eris.nocc} nvir={eris.nvir} '
                  f'naux={eris.naux} max_abs={err:.3e}',flush=True)
            if not np.isfinite(err) or err>1e-9:
                raise AssertionError(f'04 GPU global ovL vs CPU mismatch: {err}')
        self.global_ovl_info=info
        print(f'[04 GLOBAL ovL] backend={info["backend"]} '
              f'storage={info["output_storage"]} '
              f'time={info["total_global_ao2mo_s"]:.3f}s',flush=True)
        return eris

    def _mp2_rdm_occ(self, eris, moeocc, moevir, uocc, uvir, dm_type):
        if dm_type!='1h':
            raise NotImplementedError(
                f'03 GPU MP2 RDM supports occupied 1h only; got {dm_type!r}. '
                'Select --mp2-rdm-backend cpu for other LNO density flavors.')
        from .mp2_rdm_gpu import make_one_hole_density_pair
        res=make_one_hole_density_pair(
            eris,moeocc,moevir,uocc,backend=self.mp2_rdm_backend,
            workspace_mib=self.mp2_rdm_workspace_mib)
        rec=dict(res.info)
        rec['occ_type']='1h'
        rec['vir_type']=None
        rec['reference_max_abs']={}
        if self.verify_mp2_rdm:
            ref=self._orig_mp2_rdm_occ(eris,moeocc,moevir,uocc,uvir,'1h')
            err=float(np.max(np.abs(res.dm_occ-np.asarray(ref))))
            rec['reference_max_abs']['occ']=err
            print(f'[03 MP2 RDM VERIFY] fragment={len(self.mp2_rdm_records)+1} '
                  f'occ_1h max_abs={err:.3e}',flush=True)
            if not np.isfinite(err) or err>1e-9:
                raise AssertionError(f'03 occupied 1h MP2 RDM GPU vs CPU mismatch: {err}')
        self.mp2_rdm_records.append(rec)
        self._mp2_rdm_pending=(eris,np.array(uocc,copy=True),res.dm_vir)
        return res.dm_occ

    def _mp2_rdm_vir(self, eris, moeocc, moevir, uocc, uvir, dm_type):
        if dm_type!='1h':
            raise NotImplementedError(
                f'03 GPU MP2 RDM supports virtual 1h only; got {dm_type!r}. '
                'Select --mp2-rdm-backend cpu for other LNO density flavors.')
        pending=self._mp2_rdm_pending
        if pending is None or pending[0] is not eris or not np.array_equal(pending[1],uocc):
            raise RuntimeError('03 1h MP2 density cache invalid: occupied/virtual '
                               'RDM hooks must be called consecutively per fragment')
        self._mp2_rdm_pending=None
        rec=self.mp2_rdm_records[-1]
        rec['vir_type']='1h'
        dm=pending[2]
        if self.verify_mp2_rdm:
            ref=self._orig_mp2_rdm_vir(eris,moeocc,moevir,uocc,uvir,'1h')
            err=float(np.max(np.abs(dm-np.asarray(ref))))
            rec['reference_max_abs']['vir']=err
            print(f'[03 MP2 RDM VERIFY] fragment={len(self.mp2_rdm_records)} '
                  f'vir_1h max_abs={err:.3e}',flush=True)
            if not np.isfinite(err) or err>1e-9:
                raise AssertionError(f'03 virtual 1h MP2 RDM GPU vs CPU mismatch: {err}')
        return dm

    def _wrapped_impurity(self,mf,mo_coeff,uocc_loc,eris,frozen=None,log=None):
        # This duplicates only the light CCSD object constructor of lnoccsd.py;
        # the actual integral build, MP2, fragment energy and post-processing
        # stay in the original pyscf.lno.lnoccsd.impurity_solve function.
        from pyscf.lno import lnoccsd as impl
        tic=time.perf_counter()
        index=len(self.records)+1
        mo_occ=self.mylno.mo_occ
        frozen,maskact=impl.get_maskact(frozen,mo_occ.size)
        nocc=int(np.count_nonzero(maskact & (mo_occ>1e-10)))
        nvir=int(np.count_nonzero(maskact & (mo_occ<=1e-10)))
        nmo=nocc+nvir
        # Every run must use exactly the same active LAS subspace. Comparing
        # C_active @ C_active.T is invariant to orbital signs/rotations and
        # avoids fragile orbital-coefficient hashes. It is a diagnostic only:
        # this AO-metric representation is NOT itself a density matrix.
        coeff_ao=np.asarray(mo_coeff)
        if coeff_ao.ndim!=2 or coeff_ao.shape[1] != maskact.size:
            raise ValueError('Fragment LAS MO coefficients do not match frozen mask')
        active_coeff=coeff_ao[:,maskact]
        self.active_projectors.append(active_coeff @ active_coeff.conj().T)
        if hasattr(mf, 'kpt') or hasattr(mf, 'kpts'):
            raise NotImplementedError('PBC/k-point LNO is not supported by this molecular GPU bridge')
        # 02: select the *producer* of fragment MO integrals before ao2mo.
        # Selecting a representation only inside the GPU adapter is too late:
        # the parent LNO factory may have already assembled packed full vvvv.
        if self.eri_mode == 'df':
            if getattr(mf, 'with_df', None) is None:
                raise ValueError('Forced df-vvL requires a density-fitted RHF object')
            if not hasattr(impl, 'MODIFIED_DFCCSD'):
                raise RuntimeError('This PySCF-forge version lacks MODIFIED_DFCCSD')
            factory = impl.MODIFIED_DFCCSD
        elif self.eri_mode == 'full':
            if not hasattr(impl, 'MODIFIED_CCSD'):
                raise RuntimeError('This PySCF-forge version lacks MODIFIED_CCSD')
            factory = impl.MODIFIED_CCSD
        else:
            factory = impl.CCSD
        mcc=factory(mf,mo_coeff=mo_coeff,frozen=frozen).set(verbose=self.mylno.verbose_imp)
        if getattr(mcc, 'cc2', False):
            raise NotImplementedError('CC2 is not supported by this GPU RCCSD bridge')
        mcc._s1e=self.mylno._s1e
        mcc._h1e=self.mylno._h1e
        mcc._vhf=self.mylno._vhf
        if self.mylno.kwargs_imp is not None:
            mcc=mcc.set(**self.mylno.kwargs_imp)
        if self.strict_convergence:
            # Apply the same tighter stopping thresholds to the CPU and GPU
            # solvers; otherwise comparisons can reflect different stopping
            # conditions rather than different integral representations.
            mcc.conv_tol=min(mcc.conv_tol,1e-10)
            mcc.conv_tol_normt=min(mcc.conv_tol_normt,1e-8)
        cpu_kernel=mcc.kernel
        record=dict(fragment=index,nocc=nocc,nvir=nvir,nmo=nmo,
                    backend=('cpu' if self.backend=='cpu' or nmo<self.min_gpu_nmo else self.backend),
                    requested_eri_mode=self.eri_mode, solver_class=type(mcc).__name__,
                    eri_representation=None, df_naux=None,
                    fragment_ao2mo_s=None, solver_info=None,
                    ao2mo_backend=self.ao2mo_backend,
                    ao2mo_info=None,ao2mo_cpu_reference_max_abs=None,
                    mp2_rdm_info=(self.mp2_rdm_records[index-1]
                                  if self.mp2_rdm_backend!='cpu' else None))
        self.records.append(record)
        # This wrapped method is invoked by the unmodified PySCF-forge
        # impurity_solve, before the MP2 initial amplitudes are created.
        original_ao2mo=mcc.ao2mo
        def recorded_ao2mo(*args,**kwargs):
            self._synchronize(self.ao2mo_backend)
            t0=time.perf_counter()
            if self.ao2mo_backend=='cpu':
                eri=original_ao2mo(*args,**kwargs)
            else:
                from .ao2mo_gpu import fragment_ao2mo
                eri=fragment_ao2mo(
                    mcc,mo_coeff=kwargs.get('mo_coeff',args[0] if args else None),
                    eri_mode=self.eri_mode,backend=self.ao2mo_backend,
                    workspace_mib=self.ao2mo_workspace_mib,
                    ao_block_mib=self.ao2mo_ao_block_mib,
                    host_scratch_mib=self.ao2mo_host_scratch_mib,
                    max_full_gib=self.ao2mo_max_full_gib)
                record['ao2mo_info']=getattr(eri,'_gpu_ao2mo_info',None)
            self._synchronize(self.ao2mo_backend)
            record['fragment_ao2mo_s']=time.perf_counter()-t0
            self._record_stage('02',record['fragment_ao2mo_s'])
            if self.verify_ao2mo and self.ao2mo_backend!='cpu':
                reference=original_ao2mo(*args,**kwargs)
                from .ao2mo_gpu import compare_eri_objects
                errors=compare_eri_objects(eri,reference,eri_mode=self.eri_mode)
                record['ao2mo_cpu_reference_max_abs']=errors
                max_error=max(errors.values(),default=0.0)
                summary='  '.join(f'{key}={value:.3e}' for key,value in errors.items())
                print(f'[02 AO2MO VERIFY] fragment={index} mode={self.eri_mode} '
                      f'max_abs={max_error:.3e}  {summary}',flush=True)
                if max_error>1e-9:
                    raise AssertionError(
                        f'02 GPU AO2MO vs PySCF CPU reference mismatch: {errors}')
            has_df=(getattr(eri,'vvL',None) is not None)
            has_full=(getattr(eri,'vvvv',None) is not None)
            if has_df == has_full:
                raise RuntimeError('Invalid fragment ERIs: expected exactly one of vvL/vvvv')
            observed='df-vvL' if has_df else 'full-vvvv'
            if observed != ('df-vvL' if self.eri_mode=='df' else 'full-vvvv'):
                raise RuntimeError(f'Forced {self.eri_mode} yielded {observed}')
            record['eri_representation']=observed
            if has_df:
                record['df_naux']=int(eri.vvL.shape[1])
                if (self.backend=='cpu' or nmo<self.min_gpu_nmo
                        or self.allow_memory_fallback):
                    # Older PySCF-forge _DFChemistsERIs calls the underlying
                    # dfccsd._contract_vvvv_t2 with one vvL argument. PySCF
                    # 2.13 requires two (vvL,VVL) for a general 2-factor
                    # contraction. Use identical vvL factors for RHF.
                    # This is a *local instance* shim, never a site-package
                    # patch or a modification of the GPU contraction.
                    try:
                        from pyscf.cc import dfccsd as pyscf_dfccsd
                    except ImportError:
                        # Unit-test fake PySCF has no cc module.
                        pass
                    else:
                        count=len(inspect.signature(pyscf_dfccsd._contract_vvvv_t2).parameters)
                        if count not in (6,7):
                            raise RuntimeError(f'Unknown PySCF DF contraction signature with {count} parameters')
                        if count == 7:
                            def two_factor_contract(eri_self, cc_self, t2, direct=False, out=None, verbose=None):
                                if direct:
                                    raise NotImplementedError('DF contraction does not support AO-direct mode')
                                return pyscf_dfccsd._contract_vvvv_t2(
                                    cc_self, eri_self.mol, eri_self.vvL, eri_self.vvL, t2, out, verbose)
                            eri._contract_vvvv_t2=MethodType(two_factor_contract,eri)
                            record['df_cpu_contract_compat']='pyscf-2.13-two-factor'
                        else:
                            record['df_cpu_contract_compat']='legacy-one-factor'
            # Size of the two *alternative* inputs, not total working memory.
            record['full_vvvv_packed_bytes']=8*(nvir*(nvir+1)//2)**2
            if has_df:
                record['df_vvL_bytes']=8*int(eri.vvL.shape[0])*int(eri.vvL.shape[1])
            return eri
        mcc.ao2mo=recorded_ao2mo

        def adapted_kernel(*args,**kwargs):
            if self.backend=='cpu' or nmo<self.min_gpu_nmo:
                return cpu_kernel(*args,**kwargs)
            eri_obj=kwargs.get('eris',args[0] if args else None)
            t1=kwargs.get('t1',None)
            t2=kwargs.get('t2',None)
            if eri_obj is None or t1 is None or t2 is None:
                raise ValueError('Bridge expects mcc.kernel(eris=..., t1=..., t2=...)')
            if self.backend=='gpu':
                import cupy as cp
                freemem,_=cp.cuda.runtime.memGetInfo()
                naux=(int(getattr(eri_obj,'vvL').shape[1]) if
                      getattr(eri_obj,'vvvv',None) is None and
                      getattr(eri_obj,'vvL',None) is not None else None)
                needed=estimate_gpu_working_bytes(nocc,nvir,naux,
                    df_tmp_mib=self.df_tmp_mib,df_virtual_block=self.df_virtual_block)
                if needed > freemem*self.device_memory_fraction:
                    if not self.allow_memory_fallback:
                        raise MemoryError(
                            f'Fragment {index}: estimated {needed/2**20:.1f} MiB exceeds '
                            f'GPU budget {freemem*self.device_memory_fraction/2**20:.1f} MiB; '
                            'CPU fallback disabled by --no-gpu-fallback')
                    print(f'[GPU-LNO] fragment {index}: estimated {needed/2**20:.1f} MiB exceeds GPU budget; use CPU')
                    record['backend']='cpu-memory-fallback'
                    return cpu_kernel(*args,**kwargs)
            e,t1out,t2out,info=run_live_fragment_ccsd(
                eri_obj,t1,t2,uocc_loc,backend=self.backend,
                conv_tol=(min(mcc.conv_tol,1e-10) if self.strict_convergence else mcc.conv_tol),
                conv_tol_normt=(min(mcc.conv_tol_normt,1e-8) if self.strict_convergence else mcc.conv_tol_normt),
                max_cycle=mcc.max_cycle,
                diis_space=(mcc.diis_space if mcc.diis else 1),
                diis_start_cycle=getattr(mcc,'diis_start_cycle',1),
                level_shift=mcc.level_shift,verbose=self.verbose,
                diis_mode=self.diis_mode,
                df_algorithm=self.df_algorithm,
                df_tmp_mib=self.df_tmp_mib,
                df_virtual_block=self.df_virtual_block)
            mcc.e_corr=e
            mcc.converged=bool(info['converged'])
            mcc.t1=t1out.copy()
            mcc.t2=t2out.copy()
            record['solver_info']=info
            if not info['converged']:
                raise RuntimeError(f'GPU CCSD did not converge for fragment {index}')
            return e,t1out,t2out
        def timed_ccsd_kernel(*args,**kwargs):
            # This measures the same kernel boundary on CPU/GPU.  On GPU
            # it includes memory checks, host->device preparation, all CCSD
            # iterations, DIIS and device->host return of amplitudes.
            self._synchronize(self.backend)
            t0=time.perf_counter()
            try:
                if self.backend=='cpu':
                    return cpu_kernel(*args,**kwargs)
                return adapted_kernel(*args,**kwargs)
            finally:
                self._synchronize(self.backend)
                elapsed=time.perf_counter()-t0
                record['ccsd_kernel_wall_s']=elapsed
                self._record_stage('01',elapsed)

        mcc.kernel=timed_ccsd_kernel
        res,msg=impl.impurity_solve(mcc,mo_coeff,uocc_loc,mo_occ,maskact,eris,
             ccsd_t=self.mylno.ccsd_t,log=log,
             verbose_imp=self.mylno.verbose_imp,
             max_las_size_ccsd=self.mylno._max_las_size_ccsd,
             max_las_size_ccsd_t=self.mylno._max_las_size_ccsd_t)
        record['total_s']=time.perf_counter()-tic
        record['e_mp2_Ha']=float(res[0])
        record['e_ccsd_Ha']=float(res[1])
        if self.verbose:
            print(f'[GPU-LNO] fragment {index:2d}/{self.mylno.nfrag}: {record["backend"]:>5s}, nmo={nmo:3d}, E_frag={float(res[1]):+.12f} Ha, time={record["total_s"]:.3f}s',flush=True)
        return res,msg
