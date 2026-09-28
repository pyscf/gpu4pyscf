import numpy as np
import pytest
from pyscf.pbc import gto
from gpu4pyscf.pbc.tools.k2gamma import (
    kpts_to_kmesh, kpts_to_bvkmesh)


@pytest.fixture
def cell():
    return gto.M(a=[[3, .2, 0], [0, 4, .3], [.1, 0, 5]],
                 atom='He 0 0 0', basis=[[0, [1., 1.]]], verbose=0)


@pytest.mark.parametrize('mesh', [(2, 2, 2), (3, 3, 2), (11, 11, 8)])
@pytest.mark.parametrize('gamma', [True, False])
def test_sampling_and_bvk_mesh(cell, mesh, gamma):
    kpts = cell.make_kpts(mesh, with_gamma_point=gamma)
    np.testing.assert_array_equal(kpts_to_kmesh(cell, kpts), mesh)
    bvk = kpts_to_bvkmesh(cell, kpts, bound_by_supmol=False)
    scaled = cell.get_scaled_kpts(kpts)
    np.testing.assert_allclose(np.exp(2j*np.pi*scaled*bvk), 1, atol=1e-12)
    expected = np.array(mesh)
    if not gamma:
        expected *= np.where(expected % 2 == 0, 2, 1)
    np.testing.assert_array_equal(bvk, expected)

