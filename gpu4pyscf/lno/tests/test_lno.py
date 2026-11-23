"""Small-molecule numerical comparisons for the four GPU LNO-CCSD stages."""

import numpy as np
import pytest

cp = pytest.importorskip('cupy')
pyscf = pytest.importorskip('pyscf')
pytest.importorskip('pyscf.lno')  # LNO-CCSD is provided by PySCF-forge.

from pyscf import gto, scf
from pyscf.lno import lnoccsd as forge_lnoccsd
from gpu4pyscf.lno.ao2mo import fragment_ao2mo, compare_eri_objects
from gpu4pyscf.lno.bridge import _fixed_from_live_eris
from gpu4pyscf.lno.global_ovl import build_global_eris, compare_global_eris
from gpu4pyscf.lno.mp2_rdm import make_one_hole_density_pair
from gpu4pyscf.lno.solver import solve


@pytest.fixture(scope='module')
def reference():
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('GPU required for numerical LNO tests')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA runtime unavailable')
    mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0.92 0 -0.24',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    nocc = int((mf.mo_occ > 0).sum())
    loc = mf.mo_coeff[:, :nocc]
    lno = forge_lnoccsd.LNOCCSD(mf, loc, [[i] for i in range(nocc)])
    return mf, lno


def test_global_ao_to_mo(reference):
    """Stage 04 must give the same DF ovL as PySCF-forge."""
    _, lno = reference
    gpu_eris, _ = build_global_eris(lno)
    cpu_eris = lno.ao2mo()
    assert compare_global_eris(gpu_eris, cpu_eris) < 1e-9


def test_one_hole_mp2_density(reference):
    """Stage 03 occupied and virtual 1h densities must match forge."""
    _, lno = reference
    global_eris = lno.ao2mo()
    eo, ev = lno.split_mo_energy()[1:3]
    u = np.eye(global_eris.nocc)[:, :1]
    uv = np.empty((global_eris.nvir, 0))
    gpu = make_one_hole_density_pair(global_eris, eo, ev, u,
                                     workspace_mib=128)
    cpu_occ = lno.make_lo_rdm1_occ(global_eris, eo, ev, u, uv, '1h')
    cpu_vir = lno.make_lo_rdm1_vir(global_eris, eo, ev, u, uv, '1h')
    np.testing.assert_allclose(gpu.dm_occ, cpu_occ, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(gpu.dm_vir, cpu_vir, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('eri_mode', ['full', 'df'])
def test_fragment_integral_blocks(reference, eri_mode):
    """Stage 02: compare the full LAS tensor blocks, including DF vvL."""
    mf, _ = reference
    factory = (forge_lnoccsd.MODIFIED_CCSD if eri_mode == 'full'
               else forge_lnoccsd.MODIFIED_DFCCSD)
    mcc = factory(mf, mo_coeff=mf.mo_coeff, frozen=0)
    gpu_eris = fragment_ao2mo(mcc, eri_mode=eri_mode,
                              workspace_mib=64, ao_block_mib=16,
                              host_scratch_mib=64)
    cpu_eris = mcc.ao2mo()
    errors = compare_eri_objects(gpu_eris, cpu_eris, eri_mode=eri_mode)
    assert max(errors.values()) < 1e-9


def test_fragment_ccsd_amplitudes(reference):
    """Stage 01: compare the GPU LAS CCSD energy with forge CPU CCSD."""
    mf, _ = reference
    mcc = forge_lnoccsd.MODIFIED_CCSD(mf, mo_coeff=mf.mo_coeff, frozen=0)
    eris = mcc.ao2mo()
    nocc = eris.nocc
    t1, t2 = mcc.init_amps(eris=eris)[1:]
    fixed = _fixed_from_live_eris(eris, np.eye(nocc)[:, :1])
    e_gpu, _, _, info = solve(fixed, t1, t2, max_cycle=100,
                              conv_tol=1e-9, conv_tol_normt=1e-7)
    e_cpu, _, _ = mcc.kernel(eris=eris, t1=t1.copy(), t2=t2.copy())
    assert info['converged']
    assert abs(e_gpu - e_cpu) < 1e-7
