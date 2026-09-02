# Copyright 2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import cupy as cp
import pyscf
from pyscf.pbc.dft import gen_grid as gen_grid_cpu
from gpu4pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import rks as rks_cpu
from gpu4pyscf.pbc.dft import rks
from pyscf.pbc.dft import krks as krks_cpu
from gpu4pyscf.pbc.dft import krks, kuks
from gpu4pyscf.pbc.dft.gen_grid import get_becke_weight_derivative
from gpu4pyscf.pbc.grad.krks import get_vxc_full_response, get_vxc
from gpu4pyscf.pbc.grad.kuks import get_vxc_full_response as unrestricted_get_vxc_full_response
from gpu4pyscf.pbc.grad.kuks import get_vxc as unrestricted_get_vxc
from gpu4pyscf.dft.tests.test_grids import find_matching_index_between_two_grids

def numerical_gradient_exc_becke(cell, xc, kpts, auxbasis, atom_grid, dm, unrestricted=False):
    def get_energy(cell):
        if unrestricted:
            mf = kuks.KUKS(cell, xc=xc, kpts=kpts)
        else:
            mf = krks.KRKS(cell, xc=xc, kpts=kpts)
        if auxbasis is not None:
            mf = mf.density_fit(auxbasis=auxbasis)
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = atom_grid

        mf.initialize_grids(cell, dm, kpts)
        if unrestricted:
            n, exc, vxc = mf._numint.nr_uks(cell, mf.grids, mf.xc, dm, 0, hermi=1, kpts=kpts, kpts_band=None)
        else:
            n, exc, vxc = mf._numint.nr_rks(cell, mf.grids, mf.xc, dm, 0, hermi=1, kpts=kpts, kpts_band=None)
        return exc

    numerical_gradient = np.zeros((cell.natm, 3))
    dx = 1e-4
    cell_copy = cell.copy()
    for i_atom in range(cell.natm):
        for i_xyz in range(3):
            xyz_p = cell.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            cell_copy.set_geom_(xyz_p, unit='Bohr')
            cell_copy.build()
            E_p = get_energy(cell_copy)

            xyz_m = cell.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            cell_copy.set_geom_(xyz_m, unit='Bohr')
            cell_copy.build()
            E_m = get_energy(cell_copy)

            numerical_gradient[i_atom, i_xyz] = (E_p - E_m) / (2 * dx)

    translation_invariance = np.sum(numerical_gradient, axis=0)
    assert np.max(np.abs(translation_invariance)) < 1e-8, "Bad numerical gradient"

    # np.set_printoptions(precision=16, suppress=True, linewidth=np.inf)
    # print(repr(numerical_gradient))
    return numerical_gradient

class KnownValues(unittest.TestCase):
    def test_argsort(self):
        cell = pyscf.M(atom='He 0 0 0', a=np.eye(3)*3)
        grids = gen_grid.UniformGrids(cell)
        grids.mesh = [19] * 3
        for tile in [3, 4, 6, 8]:
            idx = grids.argsort(tile=tile)
            self.assertEqual(len(np.unique(idx)), 19**3)

    def test_becke_grid_atom_grid(self):
        cell = pyscf.M(
            atom = """
                H 0 0 0
                F 1 0 0.1
            """,
            a = np.diag([2.5, 3, 4]),
            basis = "6-31g",
            # verbose = 4,
        )

        mf = rks_cpu.RKS(cell, xc = 'pbe0').density_fit()
        mf.conv_tol = 1e-9
        mf.grids = gen_grid_cpu.BeckeGrids(cell)
        mf.grids.atom_grid = (50,194)
        mf.grids.prune = None
        mf.small_rho_cutoff = 0
        ref_energy = mf.kernel()
        assert mf.converged

        ref_grid_coords = mf.grids.coords
        ref_grid_weights = mf.grids.weights

        mf = rks.RKS(cell, xc = 'pbe0').density_fit()
        mf.conv_tol = 1e-9
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = (50,194)
        mf.grids.prune = None
        mf.small_rho_cutoff = 0
        test_energy = mf.kernel()
        assert mf.converged

        test_grid_coords = mf.grids.coords.get()
        test_grid_weights = mf.grids.weights.get()

        idx1, idx2 = find_matching_index_between_two_grids(ref_grid_coords,  ref_grid_weights,  1.0,
                                                           test_grid_coords, test_grid_weights, 1.0,)

        assert np.abs(test_energy - ref_energy) < 1e-6
        assert np.max(np.abs(test_grid_coords[idx2] - ref_grid_coords[idx1])) < 1e-14
        assert np.max(np.abs(test_grid_weights[idx2] - ref_grid_weights[idx1])) < 1e-12

    def test_becke_grid_level(self):
        cell = pyscf.M(
            atom = """
                H 0 0 0
                F 1 0 0.1
            """,
            a = np.diag([2.5, 3, 3]),
            basis = "6-31g",
            # verbose = 4,
        )

        kpts = cell.make_kpts([3,1,1])
        mf = krks_cpu.KRKS(cell, xc = 'pbe0', kpts = kpts).density_fit()
        mf.conv_tol = 1e-9
        mf.grids = gen_grid_cpu.BeckeGrids(cell)
        mf.grids.level = 2
        mf.grids.prune = None
        mf.small_rho_cutoff = 0
        ref_energy = mf.kernel()
        assert mf.converged

        ref_grid_coords = mf.grids.coords
        ref_grid_weights = mf.grids.weights

        mf = krks.KRKS(cell, xc = 'pbe0', kpts = kpts).density_fit()
        mf.conv_tol = 1e-9
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.level = 2
        mf.grids.prune = None
        mf.small_rho_cutoff = 0
        test_energy = mf.kernel()
        assert mf.converged

        test_grid_coords = mf.grids.coords.get()
        test_grid_weights = mf.grids.weights.get()

        idx1, idx2 = find_matching_index_between_two_grids(ref_grid_coords,  ref_grid_weights,  1.0,
                                                           test_grid_coords, test_grid_weights, 1.0,)

        assert np.abs(test_energy - ref_energy) < 1e-6
        assert np.max(np.abs(test_grid_coords[idx2] - ref_grid_coords[idx1])) < 1e-14
        assert np.max(np.abs(test_grid_weights[idx2] - ref_grid_weights[idx1])) < 1e-12

    def test_becke_weight_derivative(self):
        cell = pyscf.M(
            a = np.eye(3) * 3.5668 * 1.01, # The additional factor of 1.01 guarantees no grid point is right at the -0.5 ~ 0.5 box cutoff
            atom = '''
                C     0.      0.      0.
                C     0.8917  0.8917  0.8917
                C     1.7834  1.7834  0.
                C     2.6751  2.6751  0.8917
                C     1.7834  0.      1.7834
                C     2.6751  0.8917  2.6751
                C     0.      1.7834  1.7834
                C     0.8917  2.6751  2.6751
            ''',
            basis = 'sto-6g',
        )
        grids = gen_grid.BeckeGrids(cell)
        grids.atom_grid = (10,14)
        grids.build()

        test_dw = get_becke_weight_derivative(grids, cell.natm)

        truncation_range = (3000, 5000) # Cross the 4096 boundary
        test_dw_truncated = get_becke_weight_derivative(grids, cell.natm, truncation_range)

        dx = 1e-5
        reference_dw = cp.empty([cell.natm, 3, grids.coords.shape[0]])
        cell_copy = cell.copy()
        for i_atom in range(cell.natm):
            for i_xyz in range(3):
                xyz_p = cell.atom_coords()
                xyz_p[i_atom, i_xyz] += dx
                cell_copy.set_geom_(xyz_p, unit='Bohr')
                cell_copy.build()
                grids.reset(cell_copy)
                grids.build()
                w_p = grids.weights.copy()

                xyz_m = cell.atom_coords()
                xyz_m[i_atom, i_xyz] -= dx
                cell_copy.set_geom_(xyz_m, unit='Bohr')
                cell_copy.build()
                grids.reset(cell_copy)
                grids.build()
                w_m = grids.weights.copy()

                reference_dw[i_atom, i_xyz, :] = (w_p - w_m) / (2 * dx)

        reference_dw_truncated = reference_dw[:, :, truncation_range[0] : truncation_range[1]]

        assert cp.max(cp.abs(test_dw - reference_dw)) < 2e-9
        assert cp.max(cp.abs(test_dw_truncated - reference_dw_truncated)) < 2e-9

    def test_xc_gradient_lda_with_response(self):
        cell = pyscf.M(
            a = np.eye(3) * 3.5668,
            atom = '''
                C     0.      0.      0.
                C     0.8917  0.8917  0.8917
                C     1.7834  1.7834  0.
                C     2.6751  2.6751  0.8917
                C     1.7834  0.      1.7834
                C     2.6751  0.8917  2.6751
                C     0.      1.7834  1.7834
                C     0.8917  2.6751  2.6751
            ''',
            basis = 'def2-svp',
            verbose = 0,
        )

        kpts = np.array([[0,0,0]])
        mf = krks.KRKS(cell, xc="LDA0", kpts=kpts).density_fit(auxbasis='def2-universal-jkfit')
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-10

        mf.kernel()

        dm = mf.make_rdm1()
        if dm.ndim == 2:
            dm = dm[None,:,:]
        test_gradient = get_vxc_full_response(mf._numint, cell, mf.grids, mf.xc, dm, kpts, hermi=1)

        # ref_gradient = numerical_gradient_exc_becke(cell, "LDA0", kpts, 'def2-universal-jkfit', (50,194), dm)
        ref_gradient = np.array([
            [ 0.0000000094857455,  0.0000000095390362,  0.0000000095212727],
            [-0.0002930449305438, -0.0002930448950167, -0.0002930448950167],
            [-0.0000000263611355, -0.0000000263611355,  0.0000000985167503],
            [ 0.0002930621789687,  0.0002930622144959, -0.0002931344589285],
            [-0.0000000261479727,  0.000000098570041 , -0.0000000263966626],
            [ 0.0002930622322594, -0.0002931345122192,  0.0002930622144959],
            [ 0.0000000986766224, -0.0000000262012634, -0.0000000262012634],
            [-0.0002931345122192,  0.0002930622144959,  0.0002930622144959],
        ])

        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-9

    def test_xc_gradient_gga_with_response(self):
        cell = pyscf.M(
            a = '''0.      1.7834  1.7834
                   1.7834  0.      1.7834
                   1.7834  1.7834  0.    ''',
            atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917',
            basis = 'gth-dzvp',
            pseudo = 'gth-pade',
            verbose = 0,
        )

        kpts = cell.make_kpts((1,2,3))
        mf = krks.KRKS(cell, xc="HSE06", kpts=kpts).density_fit(auxbasis='def2-universal-jkfit')
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-10

        mf.kernel()

        dm = mf.make_rdm1()
        if dm.ndim == 2:
            dm = dm[None,:,:]
        test_gradient = get_vxc_full_response(mf._numint, cell, mf.grids, mf.xc, dm, kpts, hermi=1)

        # dm is not very stable, and numerical gradient is super fast
        ref_gradient = numerical_gradient_exc_becke(cell, "HSE06", kpts, 'def2-universal-jkfit', (50,194), dm)

        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-9

    def test_xc_gradient_gga_without_response(self):
        cell = pyscf.M(
            a = np.eye(3) * 3.5668,
            atom = '''
                C     0.      0.      0.
                C     0.8917  0.9017  0.8917
                C     1.7834  1.7834  0.
                C     2.6751  2.6751  0.8917
                C     1.7834  0.      1.7834
                C     2.6751  0.8917  2.6751
                C     0.      1.7834  1.7834
                C     0.8917  2.6751  2.6751
            ''',
            basis = 'def2-svp',
            verbose = 0,
        )

        kpts = np.array([[0,0,0]])
        mf = krks.KRKS(cell, xc="PBE", kpts=kpts).density_fit(auxbasis='def2-universal-jkfit')
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-10

        mf.kernel()

        dm = mf.make_rdm1()
        if dm.ndim == 2:
            dm = dm[None,:,:]
        test_gradient = get_vxc(mf._numint, cell, mf.grids, mf.xc, dm, kpts, hermi=1)

        # ref_gradient = numerical_gradient_exc_becke(cell, "PBE", kpts, 'def2-universal-jkfit', (99,590), dm)
        ref_gradient = np.array([
            [ 0.000429164934701 , -0.0022409300370896,  0.0004291650412824],
            [-0.0000602672400873, -0.0009193305317012, -0.0000602672400873],
            [ 0.0004409704956743, -0.0019386623151263, -0.0004303180389797],
            [ 0.0000603361982598,  0.0037129129282221, -0.0000604878991339],
            [-0.0004055164026795, -0.0022406373290096, -0.0004055164382066],
            [ 0.0000604298833196,  0.0024061113990115,  0.0000604298477924],
            [-0.0004303180745069, -0.0019386622795992,  0.0004409704601471],
            [-0.0000604880057153,  0.0037129129637492,  0.0000603362693141],
        ])

        assert np.max(np.abs(test_gradient - ref_gradient)) < 3e-4

    def test_xc_gradient_mgga_with_response(self):
        cell = pyscf.M(
            a = '''0.      1.7834  1.7834
                   1.7834  0.      1.7834
                   1.7834  1.7834  0.    ''',
            atom = 'C 0.,  0.,  0.; C 0.9017,  0.8917,  0.8917',
            basis = 'def2-svp',
            verbose = 0,
        )

        kpts = cell.make_kpts((1,1,3))
        mf = krks.KRKS(cell, xc="r2scan", kpts=kpts).density_fit(auxbasis='def2-universal-jkfit')
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-10

        mf.kernel()

        dm = mf.make_rdm1()
        if dm.ndim == 2:
            dm = dm[None,:,:]
        test_gradient = get_vxc_full_response(mf._numint, cell, mf.grids, mf.xc, dm, kpts, hermi=1)

        # dm is not very stable, and numerical gradient is super fast
        ref_gradient = numerical_gradient_exc_becke(cell, "r2scan", kpts, 'def2-universal-jkfit', (50,194), dm)

        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-9

    def test_xc_gradient_mgga_without_response(self):
        cell = pyscf.M(
            a = '''0.      1.7834  1.7834
                   1.7834  0.      1.7834
                   1.7834  1.7834  0.    ''',
            atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917',
            basis = 'gth-dzvp',
            pseudo = 'gth-pade',
            verbose = 0,
        )

        kpts = cell.make_kpts((1,2,3))
        mf = krks.KRKS(cell, xc="r2scan0", kpts=kpts).density_fit(auxbasis='def2-universal-jkfit')
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = (120,590)
        mf.conv_tol = 1e-10

        mf.kernel()

        dm = mf.make_rdm1()
        if dm.ndim == 2:
            dm = dm[None,:,:]
        test_gradient = get_vxc(mf._numint, cell, mf.grids, mf.xc, dm, kpts, hermi=1)

        # dm is not very stable, and numerical gradient is super fast
        ref_gradient = numerical_gradient_exc_becke(cell, "r2scan0", kpts, 'def2-universal-jkfit', (120,590), dm)

        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-4

    def test_xc_gradient_unrestricted_no_k_without_response(self):
        cell = pyscf.M(
            a = '''0.      1.7834  1.7834
                   1.7834  0.      1.7834
                   1.7834  1.7834  0.    ''',
            atom = 'C 0.,  0.,  0.; C 0.8917,  0.9017,  0.8917',
            basis = 'def2-svp',
            verbose = 0,
        )

        kpts = cell.make_kpts((1,1,1))
        mf = kuks.KUKS(cell, xc="HSE06", kpts=kpts).density_fit(auxbasis='def2-universal-jkfit')
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-10

        mf.kernel()

        dm = mf.make_rdm1()
        if dm.ndim == 3:
            dm = dm[:,None,:,:]
        test_gradient = unrestricted_get_vxc(mf._numint, cell, mf.grids, mf.xc, dm, kpts, hermi=1)

        # ref_gradient = numerical_gradient_exc_becke(cell, "HSE06", kpts, 'def2-universal-jkfit', (99,590), dm, unrestricted=True)
        ref_gradient = np.array([
            [ 0.0000210273753964, -0.0175452356021566,  0.0000210258033206],
            [-0.0000210273665147,  0.0175452356021566, -0.0000210257944389],
        ])

        assert np.max(np.abs(test_gradient - ref_gradient)) < 2e-4

    def test_xc_gradient_unrestricted_no_k_with_response(self):
        cell = pyscf.M(
            a = np.eye(3) * 3.5668,
            atom = '''
                C     0.      0.      0.
                C     0.8917  0.8917  0.8917
                C     1.7834  1.7834  0.
                C     2.6751  2.6751  0.8917
                C     1.7834  0.      1.7834
                C     2.6751  0.8917  2.6751
                C     0.      1.7834  1.7834
                C     0.8917  2.6751  2.6751
            ''',
            basis = 'gth-tzvp',
            pseudo = 'gth-pade',
            verbose = 0,
        )

        kpts = cell.make_kpts((1,1,1))
        mf = kuks.KUKS(cell, xc="r2scan", kpts=kpts).density_fit(auxbasis='def2-universal-jkfit')
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-10

        mf.kernel()

        dm = mf.make_rdm1()
        if dm.ndim == 3:
            dm = dm[:,None,:,:]
        test_gradient = unrestricted_get_vxc_full_response(mf._numint, cell, mf.grids, mf.xc, dm, kpts, hermi=1)

        # ref_gradient = numerical_gradient_exc_becke(cell, "r2scan", kpts, 'def2-universal-jkfit', (50,194), dm, unrestricted=True)
        ref_gradient = np.array([
            [ 0.0000003321254383,  0.0000003318678665,  0.0000003322675468],
            [-0.0003303815532263, -0.0003303816153988, -0.0003303810025557],
            [-0.0000001970335006, -0.0000001969180374,  0.0000003920330727],
            [ 0.0003302472517674,  0.0003302459639087, -0.0003304415585603],
            [-0.000000197513117 ,  0.0000003920863634, -0.0000001963496032],
            [ 0.0003302466300426, -0.0003304415407968,  0.0003302464079979],
            [ 0.0000003911893032, -0.0000001970335006, -0.0000001977085162],
            [-0.0003304409634808,  0.0003302473672306,  0.0003302460438448],
        ])

        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-9

    def test_xc_gradient_unrestricted_k_with_response(self):
        cell = pyscf.M(
            a = '''0.      1.7934  1.7834
                   1.7834  0.      1.7834
                   1.7834  1.7834  0.    ''',
            atom = 'C 0.,  0.,  0.; Si 0.8917,  0.8917,  0.8917',
            basis = 'gth-tzvp',
            pseudo = 'gth-pade',
            verbose = 0,
        )

        kpts = cell.make_kpts((1,3,1))
        mf = kuks.KUKS(cell, xc="lda", kpts=kpts)
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = (40,194)
        mf.conv_tol = 1e-10

        mf.kernel()

        dm = mf.make_rdm1()
        if dm.ndim == 3:
            dm = dm[:,None,:,:]
        test_gradient = unrestricted_get_vxc_full_response(mf._numint, cell, mf.grids, mf.xc, dm, kpts, hermi=1)

        # dm is not very stable, and numerical gradient is super fast
        ref_gradient = numerical_gradient_exc_becke(cell, "lda", kpts, None, (40,194), dm, unrestricted=True)

        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-9

if __name__ == '__main__':
    print("Full Tests for pbc.dft.numint")
    unittest.main()
