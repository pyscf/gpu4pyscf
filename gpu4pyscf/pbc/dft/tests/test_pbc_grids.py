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
from gpu4pyscf.pbc.dft import rks, uks
from pyscf.pbc.dft import krks as krks_cpu
from gpu4pyscf.pbc.dft import krks, kuks
from gpu4pyscf.pbc.dft.gen_grid import get_becke_weight_derivative
from gpu4pyscf.pbc.grad.krks import get_vxc_full_response, get_vxc
from gpu4pyscf.pbc.grad.kuks import get_vxc_full_response as unrestricted_get_vxc_full_response
from gpu4pyscf.pbc.grad.kuks import get_vxc as unrestricted_get_vxc
from gpu4pyscf.dft.tests.test_grids import find_matching_index_between_two_grids
from gpu4pyscf.pbc.grad.rhf import _finite_diff_cells

def numerical_gradient_exc_becke(cell, xc, kmesh, auxbasis, atom_grid, dm, unrestricted=False, dx = 1e-4):
    assert np.array(kmesh).shape == (3,)
    def get_energy(cell):
        kpts = cell.make_kpts(kmesh)
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

    numerical_gradient = np.zeros((cell.natm + 3, 3))
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

    for i_xyz in range(3):
        for j_xyz in range(3):
            cell_p, cell_m = _finite_diff_cells(cell, i_xyz, j_xyz, disp = dx)
            E_p = get_energy(cell_p)
            E_m = get_energy(cell_m)

            numerical_gradient[cell.natm + i_xyz, j_xyz] = (E_p - E_m) / (2 * dx)

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
        reference_dw = cp.empty([cell.natm + 3, 3, grids.coords.shape[0]])
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

        for i_xyz in range(3):
            for j_xyz in range(3):
                cell_p, cell_m = _finite_diff_cells(cell, i_xyz, j_xyz, disp = dx)
                grids.reset(cell_p)
                grids.build()
                w_p = grids.weights.copy()

                grids.reset(cell_m)
                grids.build()
                w_m = grids.weights.copy()

                reference_dw[cell.natm + i_xyz, j_xyz] = (w_p - w_m) / (2 * dx)

        reference_dw_truncated = reference_dw[:, :, truncation_range[0] : truncation_range[1]]

        assert cp.max(cp.abs(test_dw[:-3] - reference_dw[:-3])) < 2e-9
        assert cp.max(cp.abs(test_dw_truncated[:-3] - reference_dw_truncated[:-3])) < 2e-9

        assert cp.max(cp.abs(test_dw[-3:] - reference_dw[-3:])) < 5e-9
        assert cp.max(cp.abs(test_dw_truncated[-3:] - reference_dw_truncated[-3:])) < 5e-9

    def test_xc_gradient_lda_with_response(self):
        cell = pyscf.M(
            a = np.array([
                [15.9069652593, 0, 0],
                [0, 15.9069652593, 0],
                [0, 6, 15.9069652593],
            ]),
            atom = """
                O 15.43509000 9.59549000 8.94968000
                H 15.05724000 9.21878000 9.73314000
                H 0.51550474 9.33856000 9.01857000
                He 2.51550474 9.33856000 9.01857000
            """,
            basis = {'default': 'def2-svp', 'He': """
            #BASIS SET: (2s) -> [1s]
            He    S
                0.2432879285E+01       0.4301284983E+00
                0.4330512863E+00       0.6789135305E+00
            He F
                1.0 1.0
            He G
                1.2 1.0
            END
            """},
            verbose = 0,
        )

        kpts = np.array([[0,0,0]])
        mf = rks.RKS(cell, xc="LDA0").density_fit(auxbasis='def2-universal-jkfit')
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-10
        mf.with_df.linear_dep_threshold = 1e-10 # The default in pyscf==2.8.0 is 1e-9, in pyscf==2.14.0 is 1e-10

        mf.kernel()

        dm = mf.make_rdm1()
        if dm.ndim == 2:
            dm = dm[None,:,:]
        test_gradient = get_vxc_full_response(mf._numint, cell, mf.grids, mf.xc, dm, kpts, hermi=1)

        ref_gradient = numerical_gradient_exc_becke(cell, "LDA0", [1,1,1], 'def2-universal-jkfit', (50,194), dm)

        assert np.max(np.abs(test_gradient - ref_gradient)) < 2e-8

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

        kmesh = (1,2,3)
        kpts = cell.make_kpts(kmesh)
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
        ref_gradient = numerical_gradient_exc_becke(cell, "HSE06", kmesh, 'def2-universal-jkfit', (50,194), dm)

        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-7

    def test_xc_gradient_gga_without_response(self):
        cell = pyscf.M(
            a = np.eye(3) * 3.6668,
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

        # ref_gradient = numerical_gradient_exc_becke(cell, "PBE", [1,1,1], 'def2-universal-jkfit', (99,590), dm)
        ref_gradient = np.array([
            [ 0.0057493621596905,  0.0040870682127547,  0.0057493626570704],
            [-0.0347991650784252, -0.0320027493927455, -0.0347991649363166],
            [ 0.0246147672555708,  0.0208271039525698,  0.0145351780389547],
            [-0.0077134854237215, -0.0061064405798561, -0.014724848114156 ],
            [ 0.0216872688696412,  0.0122791700718494,  0.0216872685854241],
            [-0.0093490771035931, -0.0138048160280846, -0.0093490768904303],
            [ 0.0145351779679004,  0.0208271048052211,  0.0246147670779351],
            [-0.014724848647063 , -0.0061064410417089, -0.0077134864895356],
            [ 2.679998687966645 ,  0.204392930385211 ,  0.179311419188366 ],
            [ 0.204415101734412 ,  2.6794902670701504,  0.204415101734412 ],
            [ 0.1793114174120092,  0.2043929328365834,  2.6799986901693273],
        ])

        assert np.max(np.abs(test_gradient - ref_gradient)) < 6e-4

    def test_xc_gradient_mgga_with_response(self):
        cell = pyscf.M(
            a = '''0.      1.7834  1.7834
                   1.7834  0.      1.7834
                   1.7834  1.7834  0.    ''',
            atom = 'C 0.,  0.,  0.; C 0.9017,  0.8917,  0.8917',
            basis = 'def2-svp',
            verbose = 0,
        )

        kmesh = (1,1,3)
        kpts = cell.make_kpts(kmesh)
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
        ref_gradient = numerical_gradient_exc_becke(cell, "r2scan", kmesh, 'def2-universal-jkfit', (50,194), dm)

        assert np.max(np.abs(test_gradient[:-3] - ref_gradient[:-3])) < 1e-9
        assert np.max(np.abs(test_gradient[-3:] - ref_gradient[-3:])) < 3e-8

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

        kmesh = (1,2,3)
        kpts = cell.make_kpts(kmesh)
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
        ref_gradient = numerical_gradient_exc_becke(cell, "r2scan0", kmesh, 'def2-universal-jkfit', (120,590), dm)

        assert np.max(np.abs(test_gradient[:-3] - ref_gradient[:-3])) < 1e-4
        assert np.max(np.abs(test_gradient[-3:] - ref_gradient[-3:])) < 2e-3

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
        mf = uks.UKS(cell, xc="HSE06").density_fit(auxbasis='def2-universal-jkfit')
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-10

        mf.kernel()

        dm = mf.make_rdm1()
        if dm.ndim == 3:
            dm = dm[:,None,:,:]
        test_gradient = unrestricted_get_vxc(mf._numint, cell, mf.grids, mf.xc, dm, kpts, hermi=1)

        # ref_gradient = numerical_gradient_exc_becke(cell, "HSE06", (1,1,1), 'def2-universal-jkfit', (99,590), dm, unrestricted=True, dx=1e-5)
        ref_gradient = np.array([
            [ 0.0000210217621088, -0.0175452376183216,  0.000021022206198 ],
            [-0.0000210217621088,  0.0175452377071394, -0.000021022206198 ],
            [-0.5849730015938803, -0.0000565129276708, -0.0047818454085302],
            [-0.0000557453638805, -0.5846170649803639, -0.0000557508705867],
            [-0.0047818451420767, -0.0000565183455592, -0.5849730222884375],
        ])

        assert np.max(np.abs(test_gradient[:-3] - ref_gradient[:-3])) < 2e-4
        assert np.max(np.abs(test_gradient[-3:] - ref_gradient[-3:])) < 5e-4

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
        mf.with_df.linear_dep_threshold = 1e-10 # The default in pyscf==2.8.0 is 1e-9, in pyscf==2.14.0 is 1e-10

        mf.kernel()

        dm = mf.make_rdm1()
        if dm.ndim == 3:
            dm = dm[:,None,:,:]
        test_gradient = unrestricted_get_vxc_full_response(mf._numint, cell, mf.grids, mf.xc, dm, kpts, hermi=1)

        # ref_gradient = numerical_gradient_exc_becke(cell, "r2scan", (1,1,1), 'def2-universal-jkfit', (50,194), dm, unrestricted=True)
        ref_gradient = np.array([
            [ 0.000000331477068 ,  0.0000003322497832,  0.0000003327649267],
            [-0.0003304407325544, -0.0003304415763239, -0.0003304411144711],
            [-0.0000001969979735, -0.0000001975841712,  0.0000003919886638],
            [ 0.0003303068574212,  0.0003303051521186, -0.0003305007467702],
            [-0.0000001974953534,  0.0000003921751812, -0.0000001968025742],
            [ 0.0003303066264948, -0.0003305007645338,  0.000330305631735 ],
            [ 0.0000003910827218, -0.0000001970779095, -0.0000001967759289],
            [-0.0003305006668342,  0.0003303075324368,  0.0003303051876458],
            [10.485175710082117 , -0.0000000322586402, -0.0000000312638804],
            [-0.0000000314681614, 10.48517571377694  , -0.0000000326494387],
            [-0.000000033377745 , -0.0000000317967874, 10.485175704362248 ],
        ])

        # It can match down to 1e-9, if the finite difference is computed using the same dm from SCF.
        # However if we save the finite difference result, it suffers from the numerical instability of dm, and the a 3e-7 error is observed.
        assert np.max(np.abs(test_gradient - ref_gradient)) < 5e-7

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

        kmesh = (1,4,1)
        kpts = cell.make_kpts(kmesh)
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
        ref_gradient = numerical_gradient_exc_becke(cell, "lda", kmesh, None, (40,194), dm, unrestricted=True, dx=1e-5)

        assert np.max(np.abs(test_gradient[:-3] - ref_gradient[:-3])) < 1e-9
        assert np.max(np.abs(test_gradient[-3:] - ref_gradient[-3:])) < 4e-9

if __name__ == '__main__':
    print("Full Tests for PBC Becke grids")
    unittest.main()
