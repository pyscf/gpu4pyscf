"""GPU LNO-CCSD driver for molecular real restricted Hartree-Fock references.

This class subclasses PySCF-forge's LNOCCSD and overrides the four GPU
computational stages. It does not patch the original PySCF or PySCF-forge
objects. LNO orbital selection and fragment-energy projection remain those
of PySCF-forge; all computational stages here use GPU CuPy kernels.
"""
from __future__ import annotations

import time

import cupy as cp
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.lno import lnoccsd as forge_lnoccsd

from .solver import FixedLAS, solve
from .ao2mo import fragment_ao2mo
from .global_ovl import build_global_eris
from .mp2_rdm import make_one_hole_density_pair


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
    """Construct the active-LAS input for the fragment GPU solver."""
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
    return FixedLAS(nocc=no,nvir=nv,fock=np.array(fock,copy=True),
                    mo_energy=np.array(ene,copy=True),eris=blocks,vvL=vvL)


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




class LNOCCSD(forge_lnoccsd.LNOCCSD):
    """GPU local-natural-orbital CCSD for real molecular RHF.

    Parameters:
        mf: Converged density-fitted PySCF RHF reference.
        lo_coeff: AO coefficients of the localized occupied orbitals.
        frag_lolist: Orbital-index lists defining the LNO fragments.
        frozen: Molecular frozen core, as in PySCF-forge LNOCCSD.
        eri_mode: 'full' stores packed LAS vvvv; 'df' keeps LAS vvL.

    The object runs the complete molecular LNO-CCSD workflow with kernel().
    Each active fragment is solved by solver.RestrictedCCSD; the molecular
    result is the sum of PySCF-forge-projected fragment correlation energies.
    Only real, molecular, closed-shell RHF and CCSD (not CCSD(T)) are supported.
    """
    def __init__(self, mf, lo_coeff, frag_lolist, *,
                 lno_type=None, lno_thresh=None, frozen=None,
                 eri_mode='full', mp2_rdm_workspace_mib=512,
                 global_ovl_workspace_mib=256,
                 global_ovl_host_scratch_mib=256,
                 ao2mo_workspace_mib=256, ao2mo_ao_block_mib=96,
                 ao2mo_host_scratch_mib=1024, ao2mo_max_full_gib=16,
                 df_algorithm='blocked-eri', df_tmp_mib=384,
                 df_virtual_block=16, device_memory_fraction=0.65,
                 strict_convergence=False):
        super().__init__(mf, lo_coeff, frag_lolist, lno_type=lno_type,
                         lno_thresh=lno_thresh, frozen=frozen)
        if eri_mode not in ('full', 'df'):
            raise ValueError('eri_mode must be full or df')
        if df_algorithm not in ('blocked-eri', 'aux-batched'):
            raise ValueError('Unknown DF contraction algorithm')
        if not 0 < device_memory_fraction <= 1:
            raise ValueError('device_memory_fraction must be in (0, 1]')
        if min(mp2_rdm_workspace_mib, global_ovl_workspace_mib,
               global_ovl_host_scratch_mib, ao2mo_workspace_mib,
               ao2mo_ao_block_mib, ao2mo_host_scratch_mib,
               ao2mo_max_full_gib, df_tmp_mib, df_virtual_block) <= 0:
            raise ValueError('Workspace limits must be positive')
        self.eri_mode = eri_mode
        self.mp2_rdm_workspace_mib = int(mp2_rdm_workspace_mib)
        self.global_ovl_workspace_mib = int(global_ovl_workspace_mib)
        self.global_ovl_host_scratch_mib = int(global_ovl_host_scratch_mib)
        self.ao2mo_workspace_mib = int(ao2mo_workspace_mib)
        self.ao2mo_ao_block_mib = int(ao2mo_ao_block_mib)
        self.ao2mo_host_scratch_mib = int(ao2mo_host_scratch_mib)
        self.ao2mo_max_full_gib = float(ao2mo_max_full_gib)
        self.df_algorithm = df_algorithm
        self.df_tmp_mib = int(df_tmp_mib)
        self.df_virtual_block = int(df_virtual_block)
        self.device_memory_fraction = float(device_memory_fraction)
        self.strict_convergence = bool(strict_convergence)
        self._pending_density = None
        self.fragment_records = []
        self.stage_wall_s = {key: 0. for key in ('04', '03', '02', '01')}
        self.stage_calls = {key: 0 for key in self.stage_wall_s}
        self.lno_wall_s = None

    def _stage(self, key, start):
        cp.cuda.get_current_stream().synchronize()
        self.stage_wall_s[key] += time.perf_counter() - start
        self.stage_calls[key] += 1

    def kernel(self, eris=None):
        """Run the complete GPU LNO-CCSD calculation, without monkey-patching."""
        if self.ccsd_t:
            raise NotImplementedError('CCSD(T) is not implemented')
        if self.prune_lno_basis:
            raise NotImplementedError('Domain-pruned LNO orbitals are unsupported')
        if tuple(self.lno_type) != ('1h', '1h'):
            raise NotImplementedError('Only the occupied/virtual 1h LNO densities are implemented')
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('CUDA GPU is required for LNO-CCSD')
        self._pending_density = None
        self.fragment_records.clear()
        for key in self.stage_wall_s:
            self.stage_wall_s[key] = 0.
            self.stage_calls[key] = 0
        self.lno_wall_s = None
        start = time.perf_counter()
        try:
            return super().kernel(eris=eris)
        finally:
            cp.cuda.get_current_stream().synchronize()
            self.lno_wall_s = time.perf_counter() - start

    def ao2mo(self):
        """04: Construct global (occupied virtual | auxiliary) DF factors."""
        start = time.perf_counter()
        eris, self.global_ovl_info = build_global_eris(
            self, workspace_mib=self.global_ovl_workspace_mib,
            host_scratch_mib=self.global_ovl_host_scratch_mib)
        self._stage('04', start)
        logger.info(self, '04 Global AO-to-MO: %.3f s', self.stage_wall_s['04'])
        return eris

    def make_lo_rdm1_occ(self, eris, moeocc, moevir, uocc_loc, uvir_loc, occ_lno_type):
        """03: Compute both one-hole MP2 LNO densities in one GPU call."""
        if occ_lno_type != '1h':
            raise NotImplementedError('Only occupied 1h LNO density is implemented')
        start = time.perf_counter()
        result = make_one_hole_density_pair(
            eris, moeocc, moevir, uocc_loc,
            workspace_mib=self.mp2_rdm_workspace_mib)
        self._pending_density = (eris, np.array(uocc_loc, copy=True), result.dm_vir)
        self._stage('03', start)
        return result.dm_occ

    def make_lo_rdm1_vir(self, eris, moeocc, moevir, uocc_loc, uvir_loc, vir_lno_type):
        if vir_lno_type != '1h':
            raise NotImplementedError('Only virtual 1h LNO density is implemented')
        cached = self._pending_density
        if (cached is None or cached[0] is not eris or
                not np.array_equal(cached[1], uocc_loc)):
            raise RuntimeError('Occupied and virtual 1h density hooks must be consecutive')
        self._pending_density = None
        return cached[2]

    def impurity_solve(self, mf, mo_coeff, uocc_loc, eris, frozen=None, log=None):
        """02/01: Solve one fragment and project its MP2/CCSD energies.

        The fragment's MP2 initial amplitudes and the projected energies use
        the original PySCF-forge definitions. GPU kernels build the LAS ERIs
        and iterate the RCCSD amplitudes. No CPU fallback or runtime patch.
        """
        log = logger.new_logger(self) if log is None else log
        mo_occ = self.mo_occ
        frozen, maskact = forge_lnoccsd.get_maskact(frozen, mo_occ.size)
        nocc = int(np.count_nonzero(maskact & (mo_occ > 1e-10)))
        nvir = int(np.count_nonzero(maskact & (mo_occ <= 1e-10)))
        nmo = nocc + nvir
        zero = lib.tag_array(0., spin_comp=np.zeros(2))
        if nocc == 0 or nvir == 0 or nmo > self._max_las_size_ccsd:
            if nmo > self._max_las_size_ccsd:
                log.warn('Fragment LAS size %d exceeds the limit %d',
                         nmo, self._max_las_size_ccsd)
            return (zero, zero, 0.), 'Skipped empty or oversized LAS'
        if hasattr(mf, 'kpts') or hasattr(mf, 'kpt'):
            raise NotImplementedError('Only molecular RHF is supported')
        if self.eri_mode == 'df':
            if mf.with_df is None:
                raise ValueError('DF mode requires an RHF density-fitting object')
            factory = forge_lnoccsd.MODIFIED_DFCCSD
        else:
            factory = forge_lnoccsd.MODIFIED_CCSD
        mcc = factory(mf, mo_coeff=mo_coeff, frozen=frozen).set(verbose=self.verbose_imp)
        mcc._s1e = self._s1e
        mcc._h1e = self._h1e
        mcc._vhf = self._vhf
        if self.kwargs_imp is not None:
            mcc = mcc.set(**self.kwargs_imp)
        if getattr(mcc, 'cc2', False):
            raise NotImplementedError('CC2 is unsupported')
        start = time.perf_counter()
        fragment_eris = fragment_ao2mo(
            mcc, eri_mode=self.eri_mode,
            workspace_mib=self.ao2mo_workspace_mib,
            ao_block_mib=self.ao2mo_ao_block_mib,
            host_scratch_mib=self.ao2mo_host_scratch_mib,
            max_full_gib=self.ao2mo_max_full_gib)
        self._stage('02', start)
        ovov = np.asarray(fragment_eris.ovov)
        oovv = ovov.reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3)
        t1, t2 = mcc.init_amps(eris=fragment_eris)[1:]
        e_mp2 = forge_lnoccsd.get_fragment_energy(oovv, t2, uocc_loc)

        fixed = _fixed_from_live_eris(fragment_eris, np.asarray(uocc_loc))
        fixed.df_contract_algorithm = self.df_algorithm
        fixed.df_tmp_limit_bytes = self.df_tmp_mib * 2**20
        fixed.df_virtual_block_cap = self.df_virtual_block
        naux = fixed.vvL.shape[1] if fixed.vvL is not None else None
        estimate = estimate_gpu_working_bytes(
            nocc, nvir, naux, df_tmp_mib=self.df_tmp_mib,
            df_virtual_block=self.df_virtual_block)
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        if estimate > free_bytes * self.device_memory_fraction:
            raise MemoryError(
                f'Fragment {len(self.fragment_records)+1}: estimated GPU workspace '
                f'{estimate/2**20:.1f} MiB exceeds available memory budget '
                f'{free_bytes*self.device_memory_fraction/2**20:.1f} MiB')
        start = time.perf_counter()
        energy, t1, t2, info = solve(
            fixed, t1, t2,
            conv_tol=min(mcc.conv_tol, 1e-10) if self.strict_convergence else mcc.conv_tol,
            conv_tol_normt=(min(mcc.conv_tol_normt, 1e-8)
                            if self.strict_convergence else mcc.conv_tol_normt),
            max_cycle=mcc.max_cycle,
            diis_space=mcc.diis_space if mcc.diis else 1,
            diis_start_cycle=getattr(mcc, 'diis_start_cycle', 1),
            level_shift=mcc.level_shift,
            verbose=self.verbose_imp, log=log)
        self._stage('01', start)
        if not info['converged']:
            raise RuntimeError(f'CCSD did not converge for LAS with {nmo} orbitals')
        tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
        e_cc = forge_lnoccsd.get_fragment_energy(oovv, tau, uocc_loc)
        self.fragment_records.append({
            'fragment': len(self.fragment_records)+1,
            'nocc': nocc, 'nvir': nvir, 'nmo': nmo,
            'e_corr_solver_Ha': energy, 'e_corr_fragment_Ha': float(e_cc),
            'eri_mode': self.eri_mode, 'solver_info': info,
        })
        log.info('LNO fragment %d: LAS %d occ + %d vir; E(MP2)=%+.12f E(CCSD)=%+.12f',
                 len(self.fragment_records), nocc, nvir, float(e_mp2), float(e_cc))
        msg = f'E_corr(MP2) = {float(e_mp2):.15g}  E_corr(CCSD) = {float(e_cc):.15g}'
        return (e_mp2, e_cc, 0.), msg

    def timing_summary(self):
        result = {key: {'wall_s': self.stage_wall_s[key],
                        'calls': self.stage_calls[key]}
                  for key in ('04', '03', '02', '01')}
        result['lno_wall_s'] = self.lno_wall_s
        return result

    def timing_report(self):
        labels = (('04', 'Global AO-to-MO'), ('03', 'MP2 1h 1-RDM'),
                  ('02', 'Fragment AO-to-LAS'), ('01', 'Fragment CCSD'))
        lines = ['========== GPU LNO-CCSD Stage Wall Times ==========']
        for key, label in labels:
            lines.append(f'{key} {label:<27s}: {self.stage_wall_s[key]:12.3f} s '
                         f'({self.stage_calls[key]} calls)')
        if self.lno_wall_s is not None:
            other = max(0., self.lno_wall_s - sum(self.stage_wall_s.values()))
            lines.append(f'Other LNO operations             : {other:12.3f} s')
            lines.append(f'LNO total wall time              : {self.lno_wall_s:12.3f} s')
        lines.append('===============================================')
        return '\n'.join(lines)
