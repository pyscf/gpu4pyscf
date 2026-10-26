"""Small periodic finite-field calculations; included in the slow test suite."""
import cupy as cp
import numpy as np
import pytest
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.tools import pbc
from gpu4pyscf.pbc.dft import KRKS
from gpu4pyscf.pbc.dft import numint
from gpu4pyscf.pbc.properties.external_field import apply
from gpu4pyscf.pbc.properties.external_field.optimizer import rotation

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def helium():
    cell = gto.M(a=np.eye(3)*5, atom='He 1 1 1', basis='cc-pvdz',
                 unit='Bohr', verbose=0, precision=1e-8)
    cell.mesh = [25]*3
    old_threads, old_nproc = lib.num_threads(), pbc.nproc
    lib.num_threads(2)
    pbc.nproc = 1
    try:
        mf = KRKS(cell, cell.make_kpts([2,2,2]), xc='LDA,VWN')
        mf.conv_tol = 1e-12
        mf.kernel()
        assert mf.converged
        yield mf
    finally:
        lib.num_threads(old_threads)
        pbc.nproc = old_nproc


def test_zero_field_reduces_to_krks(helium):
    ff = apply(helium, field=[0.,0.,0.]).run()
    assert ff.converged and isinstance(ff.mo_coeff, cp.ndarray)
    assert abs(ff.e_tot-helium.e_tot) < 1e-12
    cp.testing.assert_allclose(ff.make_rdm1(ff.mo_coeff, ff.mo_occ),
                               helium.make_rdm1(), atol=1e-12, rtol=0)
    assert abs(ff.e_field) < 1e-15


def _assert_random_direction_stability(ff):
    """Sample total-enthalpy curvature at fixed accepted Berry branch."""
    coeff = ff.mo_coeff
    branch = tuple(p.copy() for p in ff._field_state.branch)
    accepted = ff._field_state.result
    reference = ff.evaluate_orbitals(coeff, reference=branch)
    assert float(cp.linalg.norm(reference[5])) < ff.conv_tol_grad
    rng = np.random.default_rng(73)
    shape = (len(coeff), coeff.shape[-1]-ff.nocc, ff.nocc)
    assert shape[1] > 0
    steps = (2e-3, 1e-3)
    for sample in range(3):
        # Independent complex vo blocks at each k point; unit norm gives
        # comparable rotation amplitudes for all sampled directions.
        direction = rng.normal(size=shape) + 1j*rng.normal(size=shape)
        direction = cp.asarray(direction / np.linalg.norm(direction))
        curvatures = []
        for step in steps:
            plus, _ = rotation(coeff, direction, step, ff.nocc)
            minus, _ = rotation(coeff, direction, -step, ff.nocc)
            # Recompute both KS and Berry contributions without relaxing the
            # trial orbitals or committing their branch to the mean field.
            e_plus = ff.evaluate_orbitals(plus, reference=branch)[0]
            e_minus = ff.evaluate_orbitals(minus, reference=branch)[0]
            curvature = ((e_plus-reference[0]) +
                         (e_minus-reference[0])) / step**2
            assert np.isfinite(curvature)
            # Allow small negative values from energy cancellation, but reject
            # negative curvature that is resolved at either step size.
            noise = max(1e-5, 64*np.finfo(float).eps *
                        max(1., abs(reference[0])) / step**2)
            assert curvature >= -noise, (
                f'Negative enthalpy curvature: field={ff.field}, '
                f'direction={sample}, step={step}, curvature={curvature}')
            curvatures.append(curvature)
        np.testing.assert_allclose(curvatures[0], curvatures[1],
            rtol=5e-3, atol=1e-5,
            err_msg=f'Curvature depends on step size for direction {sample}')
        print(f'He stability field={ff.field.tolist()}, direction={sample}: '
              f'curvatures={curvatures}', flush=True)
    assert ff.mo_coeff is coeff
    assert ff._field_state.result is accepted
    for actual, expected in zip(ff._field_state.branch, branch):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('sign', [-1, 1])
def test_finite_field_scf_cg_agreement_and_restart(helium, sign, tmp_path):
    results = []
    for solver in ('scf', 'cg'):
        ff = apply(helium, field=[0.,0.,sign*1e-4], solver=solver)
        ff.conv_tol = 1e-10
        ff.conv_tol_grad = 1e-9
        ff.conv_tol_dm = 1e-8
        ff.max_cycle = 150
        ff.run()
        assert ff.converged and isinstance(ff.mo_coeff, cp.ndarray)
        assert ff.history[-1]['gradient_norm'] < ff.conv_tol_grad
        assert ff.history[-1]['density_norm'] < ff.conv_tol_dm
        assert abs(ff.history[-1]['delta_e']) < ff.conv_tol
        assert sign*ff.polarization()[2] > 0
        # The reported field energy must agree with the reported polarization.
        np.testing.assert_allclose(ff.e_field,
            -helium.cell.vol*np.dot(ff.field, ff.polarization()), atol=1e-14, rtol=0)
        results.append(ff)
    scf, cg = results
    np.testing.assert_allclose(scf.polarization(), cg.polarization(), atol=2e-10, rtol=0)
    cp.testing.assert_allclose(scf.make_rdm1(scf.mo_coeff, scf.mo_occ),
                               cg.make_rdm1(cg.mo_coeff, cg.mo_occ), atol=2e-8, rtol=0)
    assert abs(scf.e_tot-cg.e_tot) < 1e-12
    _assert_random_direction_stability(cg)
    path = tmp_path/'branch.json'
    cg.save_branch(path)
    restored = apply(helium, field=cg.field, solver='scf').load_branch(path)
    restored.conv_tol_grad = 1e-9
    restored.conv_tol_dm = 1e-8
    restored.kernel(mo_coeff=cg.mo_coeff, mo_occ=cg.mo_occ)
    assert restored.converged
    np.testing.assert_allclose(restored.polarization(), cg.polarization(), atol=2e-10, rtol=0)
