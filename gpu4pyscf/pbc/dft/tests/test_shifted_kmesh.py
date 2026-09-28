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

