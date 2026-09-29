"""Regressions for shifted meshes: charges, XC, lattice sums and DF J/K."""
import numpy as np
import cupy as cp
import pytest
from pyscf.pbc import gto, dft, df
from gpu4pyscf.pbc.dft.multigrid_v3 import MultiGridNumInt
from gpu4pyscf.pbc.df.df import GDF
from gpu4pyscf.pbc.df.aft import AFTDF
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh, kpts_to_bvkmesh


@pytest.fixture
def cell():
    return gto.M(a=np.eye(3)*5, unit='Bohr',
                 atom='He 0 0 0; He 1.2 1.4 1.1',
                 basis=[[0, [1., 1.]], [1, [.8, 1.]]],
                 precision=1e-10, mesh=[35]*3, verbose=0)


def density(cell, kpts):
    s = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    # Tr(D_k S_k) = Ne at every k, with time-reversal symmetry preserved.
    return cell.nelectron / cell.nao * np.linalg.inv(s)


@pytest.mark.parametrize('mesh', [
    (2, 2, 2), (3, 3, 2), pytest.param((11, 11, 8), marks=pytest.mark.slow)])
@pytest.mark.parametrize('xc', ['lda,vwn', 'r2scan'])
def test_shifted_multigrid(cell, mesh, xc):
    kpts = cell.make_kpts(mesh, with_gamma_point=False)
    dm = density(cell, kpts)
    mf = dft.KRKS(cell, kpts, xc=xc)
    mf.grids = dft.gen_grid.UniformGrids(cell)
    mf.grids.build()
    ref = mf._numint.nr_rks(cell, mf.grids, xc, dm, kpts=kpts)
    ni = MultiGridNumInt(cell)
    ni.allow_mesh_reduction = False
    n, exc, vxc = ni.nr_rks(cell, None, xc, cp.asarray(dm), kpts=kpts, with_j=False)
    np.testing.assert_allclose(n, cell.nelectron, atol=2e-7, rtol=0)
    np.testing.assert_allclose(n, ref[0], atol=2e-7, rtol=0)
    np.testing.assert_allclose(exc, ref[1], atol=2e-7, rtol=0)
    np.testing.assert_allclose(vxc.get(), ref[2], atol=2e-7, rtol=0)
    # Also exercise spin-resolved density normalization.
    udm = np.array([dm*.6, dm*.4])
    uref = mf._numint.nr_uks(cell, mf.grids, xc, udm, kpts=kpts)
    un, ue, uv = ni.nr_uks(cell, None, xc, cp.asarray(udm), kpts=kpts, with_j=False)
    np.testing.assert_allclose(un, uref[0], atol=2e-7, rtol=0)
    np.testing.assert_allclose(ue, uref[1], atol=2e-7, rtol=0)
    np.testing.assert_allclose(uv.get(), uref[2], atol=2e-7, rtol=0)


@pytest.mark.parametrize('mesh', [(2, 2, 2), (3, 3, 2)])
@pytest.mark.parametrize('gamma', [True, False])
def test_lattice_sums_and_jk(cell, mesh, gamma):
    from gpu4pyscf.pbc.scf import rsjk

    kpts = cell.make_kpts(mesh, with_gamma_point=gamma)
    dm = density(cell, kpts)
    for name, fn in [('int1e_ovlp', int1e.int1e_ovlp),
                     ('int1e_kin', int1e.int1e_kin)]:
        ref = cell.pbc_intor(name, kpts=kpts)
        np.testing.assert_allclose(fn(cell, kpts).get(), ref, atol=2e-8, rtol=0)
    for cpu_cls, gpu_cls in [(df.GDF, GDF), (df.AFTDF, AFTDF)]:
        ref_df, gpu_df = cpu_cls(cell, kpts), gpu_cls(cell, kpts)
        if cpu_cls is df.GDF:
            ref_df.auxbasis = gpu_df.auxbasis = 'weigend'
        ref = ref_df.get_jk(dm, kpts=kpts, exxdiv=None)
        out = gpu_df.get_jk(cp.asarray(dm), kpts=kpts, exxdiv=None)
        for got, expected in zip(out, ref):
            np.testing.assert_allclose(got.get(), expected, atol=2e-6, rtol=0)
        if gpu_cls is GDF:
            np.testing.assert_array_equal(gpu_df.kmesh, kpts_to_kmesh(cell, kpts))
    # Direct range-separated J/K uses the same BvK and q grouping.
    vj = rsjk.get_j(cell, cp.asarray(dm), kpts=kpts)
    vk = rsjk.get_k(cell, cp.asarray(dm), kpts=kpts, exxdiv=None)
    np.testing.assert_allclose(vj.get(), ref[0], atol=2e-6, rtol=0)
    np.testing.assert_allclose(vk.get(), ref[1], atol=2e-6, rtol=0)


def _check_derivatives(cell, kpts, grad_sigma, energy, atol=2e-6):
    from gpu4pyscf.pbc.grad.rhf import _finite_diff_cells

    assert grad_sigma.shape == (cell.natm + 3, 3)
    scaled_kpts = cell.get_scaled_kpts(kpts)
    disp = 1e-4
    coords = cell.atom_coords()
    coords[0, 0] += disp
    cell1 = cell.set_geom_(coords, unit='Bohr', inplace=False)
    coords[0, 0] -= 2 * disp
    cell2 = cell.set_geom_(coords, unit='Bohr', inplace=False)
    cases = [(0, 0, cell1, cell2)]
    for i, j in [(0, 0), (0, 1)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp)
        cases.append((cell.natm + i, j, cell1, cell2))
    for row, col, cell1, cell2 in cases:
        e1 = energy(cell1, cell1.get_abs_kpts(scaled_kpts))
        e2 = energy(cell2, cell2.get_abs_kpts(scaled_kpts))
        np.testing.assert_allclose(
            grad_sigma[row, col], (e1 - e2) / (2 * disp), atol=atol, rtol=0)


@pytest.mark.parametrize('method', ['aft', 'gdf', 'rsjk'])
@pytest.mark.parametrize('unrestricted', [False, True])
def test_shifted_jk_derivatives(cell, method, unrestricted):
    from types import SimpleNamespace
    from gpu4pyscf.pbc.grad.rhf import _gdf_ejk_derivatives
    from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt

    kpts = cell.make_kpts([2, 1, 1], with_gamma_point=False)
    dm = density(cell, kpts)
    if unrestricted:
        dm = np.array([dm * .6, dm * .4])
    dm_sf = dm.sum(axis=0) if unrestricted else dm
    k_factor = 1 if unrestricted else .5
    dm_gpu = cp.asarray(dm)
    if method == 'aft':
        mydf = AFTDF(cell, kpts)
        grad_sigma = mydf.get_ej_derivatives(cp.asarray(dm_sf), kpts)
        grad_sigma -= k_factor * mydf.get_ek_derivatives(dm_gpu, kpts, exxdiv=None)
    elif method == 'gdf':
        mydf = GDF(cell, kpts)
        mydf.auxbasis = 'weigend'
        mf = SimpleNamespace(with_df=mydf, exxdiv=None)
        grad_sigma = _gdf_ejk_derivatives(mf, dm_gpu, kpts)
    else:
        opt = PBCJKMatrixOpt(cell).build()
        grad_sigma = opt._get_ejk_derivatives(dm_gpu, kpts, exxdiv=None)

    def energy(cell, kpts):
        mydf = df.GDF(cell, kpts) if method == 'gdf' else df.AFTDF(cell, kpts)
        if method == 'gdf':
            mydf.auxbasis = 'weigend'
        # Build K first: older PySCF versions do not rebuild J-only GDF
        # integrals to include the off-diagonal k-point pairs needed by K.
        _, vk = mydf.get_jk(dm, kpts=kpts, with_j=False, exxdiv=None)
        vj, _ = mydf.get_jk(dm_sf, kpts=kpts, with_k=False)
        ej = np.einsum('kij,kji->', dm_sf, vj).real
        ek = np.einsum('skij,skji->', dm.reshape(-1, len(kpts), cell.nao, cell.nao),
                       vk.reshape(-1, len(kpts), cell.nao, cell.nao)).real
        return .5 * (ej - k_factor * ek) / len(kpts)

    _check_derivatives(cell, kpts, grad_sigma, energy)


def test_shifted_nuclear_derivatives(cell):
    from gpu4pyscf.pbc.df.grad.krhf import get_nuc

    kpts = cell.make_kpts([2, 1, 1], with_gamma_point=False)
    dm = density(cell, kpts)
    grad_sigma = get_nuc(cell, cp.asarray(dm), kpts)

    def energy(cell, kpts):
        v = df.AFTDF(cell, kpts).get_nuc(kpts)
        return np.einsum('kij,kji->', dm, v).real / len(kpts)

    _check_derivatives(cell, kpts, grad_sigma, energy)


def test_shifted_ppnl_derivatives():
    from pyscf.pbc.gto.pseudo.pp_int import get_pp_nl
    from gpu4pyscf.pbc.grad.pp import ppnl_derivatives

    cell = gto.M(a=np.eye(3)*6, unit='Bohr',
                 atom='C 0 0 0; C 1.5 1.3 1.2',
                 basis='gth-szv', pseudo='gth-pade', precision=1e-10, verbose=0)
    kpts = cell.make_kpts([2, 1, 1], with_gamma_point=False)
    dm = density(cell, kpts)
    grad_sigma = ppnl_derivatives(cell, cp.asarray(dm), kpts)

    def energy(cell, kpts):
        return np.einsum('kij,kji->', dm, get_pp_nl(cell, kpts)).real / len(kpts)

    _check_derivatives(cell, kpts, grad_sigma, energy)
