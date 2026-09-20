# Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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
from gpu4pyscf.pbc.dft.rks import RKS
from gpu4pyscf.pbc.dft.krks import KRKS
from gpu4pyscf.pbc.scf.hf import RHF
from gpu4pyscf.pbc.scf.khf import KRHF
from gpu4pyscf.pbc.scf.uhf import UHF
from gpu4pyscf.pbc.dft.kuks import KUKS
from gpu4pyscf.pbc.dft.gen_grid import BeckeGrids
from gpu4pyscf.pbc.df.grad.krhf import get_nuc
from gpu4pyscf.pbc.gto.int1e import kin_derivatives
from gpu4pyscf.lib.multi_gpu import num_devices
from gpu4pyscf.pbc.grad.rhf import _finite_diff_cells

def numerical_hcore_gradient_and_stresstensor(cell, dm, kmesh):
    if dm.ndim == 2:
        dm = dm[None,:,:]

    def get_energy(cell):
        mf = KRHF(cell).density_fit(auxbasis="def2-universal-jkfit")
        kpts = cell.make_kpts(kmesh)
        hcore = mf.get_hcore(kpts = kpts)
        e = cp.einsum("kij,kji->", hcore, dm)
        e = e.real / kpts.shape[0]
        return e

    numerical_gradient = np.zeros((cell.natm + 3, 3))
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

    for i_xyz in range(3):
        for j_xyz in range(3):
            cell_p, cell_m = _finite_diff_cells(cell, i_xyz, j_xyz, disp = dx)
            E_p = get_energy(cell_p)
            E_m = get_energy(cell_m)

            numerical_gradient[cell.natm + i_xyz, j_xyz] = (E_p - E_m) / (2 * dx)

    # np.set_printoptions(precision=16, suppress=True, linewidth=np.inf)
    # print(repr(numerical_gradient))
    return numerical_gradient

def numerical_gradient_and_stresstensor(cell, get_energy):
    assert callable(get_energy)

    numerical_gradient = np.zeros((cell.natm + 3, 3))
    dx = 1e-4
    cell_copy = cell.copy()
    for i_atom in range(cell.natm):
        for i_xyz in range(3):
            xyz_p = cell.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            cell_copy.set_geom_(xyz_p, unit='Bohr')
            cell_copy.build()
            E_p = get_energy(cell_copy)
            if isinstance(E_p, tuple):
                E_p = E_p[0]

            xyz_m = cell.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            cell_copy.set_geom_(xyz_m, unit='Bohr')
            cell_copy.build()
            E_m = get_energy(cell_copy)
            if isinstance(E_m, tuple):
                E_m = E_m[0]

            numerical_gradient[i_atom, i_xyz] = (E_p - E_m) / (2 * dx)

    translation_invariance = np.sum(numerical_gradient, axis=0)
    assert np.max(np.abs(translation_invariance)) < 1e-7, "Bad numerical gradient"

    for i_xyz in range(3):
        for j_xyz in range(3):
            cell_p, cell_m = _finite_diff_cells(cell, i_xyz, j_xyz, disp = dx)
            E_p = get_energy(cell_p)
            E_m = get_energy(cell_m)
            if isinstance(E_p, tuple):
                E_p = E_p[0]
            if isinstance(E_m, tuple):
                E_m = E_m[0]

            numerical_gradient[cell.natm + i_xyz, j_xyz] = (E_p - E_m) / (2 * dx)

    np.set_printoptions(precision=16, suppress=True, linewidth=np.inf)
    print(f"ref_gradient = np.{repr(numerical_gradient)}")
    return numerical_gradient

class KnownValues(unittest.TestCase):
    def test_gdf_hcore_derivatives_rhf(self):
        cell = pyscf.M(
            atom = """
                O 15.43509000 9.59549000 8.94968000
                H 15.05724000 9.21878000 9.73314000
                H 0.51550474 9.33856000 9.01857000
                He 2.51550474 9.33856000 9.01857000
            """,
            a = np.eye(3) * (15.9069652593 / 3),
            unit = "Angstrom",
            basis = "def2-SVP",
            verbose = 0,
        )

        def get_energy(cell):
            mf = RHF(cell).density_fit(auxbasis="def2-universal-jkfit")
            mf.conv_tol = 1e-12
            e = mf.kernel()
            assert mf.converged
            return e, mf
        test_energy, mf = get_energy(cell)

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()
        test_derivatives = np.vstack((test_gradient, gobj.stress * cell.vol))

        # Energy check is consistency check
        ref_energy = -78.84270992380236
        # ref_derivatives = numerical_gradient_and_stresstensor(cell, get_energy)
        ref_derivatives = np.array([
            [-0.063013447402227 ,  0.0161632252826394, -0.0025874167874917],
            [-0.000704673723817 ,  0.001038964825284 , -0.0022375882480219],
            [ 0.0599051951155616, -0.0166196268480689,  0.0044088567108247],
            [ 0.0038129256552111, -0.0005825626203659,  0.0004161493194488],
            [ 0.1479508021162701, -0.0302303583055163,  0.0074059591526066],
            [-0.0302303594423847,  0.0338498824703493, -0.0000233173835795],
            [ 0.0074059608579091, -0.0000233153940599,  0.0259459928031447],
        ])

        assert np.abs(test_energy - ref_energy) < 1e-9
        assert np.max(np.abs(test_derivatives[:-3, :] - ref_derivatives[:-3, :])) < 5e-8
        assert np.max(np.abs(test_derivatives[-3:, :] - ref_derivatives[-3:, :])) < 1e-7

        dm = mf.make_rdm1()
        kmesh = np.array([1,1,1])
        kpts = cell.make_kpts(kmesh)

        test_hcore_derivatives = get_nuc(cell, dm, kpts)
        test_hcore_derivatives += kin_derivatives(cell, dm, kpts)
        assert test_hcore_derivatives.shape == (cell.natm + 3, 3)

        ref_hcore_derivatives = numerical_hcore_gradient_and_stresstensor(cell, dm, kmesh)

        assert np.max(np.abs(test_hcore_derivatives[:-3, :] - ref_hcore_derivatives[:-3, :])) < 3e-8
        assert np.max(np.abs(test_hcore_derivatives[-3:, :] - ref_hcore_derivatives[-3:, :])) < 3e-7

    def test_gdf_hcore_derivatives_rks(self):
        cell = pyscf.M(
            atom = """
                O 15.43509000 9.59549000 8.94968000
                H 15.05724000 9.21878000 9.73314000
                H 0.51550474 9.33856000 9.01857000
                He 2.51550474 9.33856000 9.01857000
            """,
            a = np.eye(3) * 15.9069652593,
            unit = "Angstrom",
            basis = "6-31g",
            verbose = 0,
        )

        def get_energy(cell):
            mf = RKS(cell, xc="r2scan").density_fit(auxbasis="def2-universal-jkfit")
            mf.grids = BeckeGrids(cell)
            mf.grids.atom_grid = (50, 194)
            mf.conv_tol = 1e-12
            e = mf.kernel()
            assert mf.converged
            return e, mf
        test_energy, mf = get_energy(cell)

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()
        test_derivatives = np.vstack((test_gradient, gobj.stress * cell.vol))

        # Energy check is consistency check
        ref_energy = -79.23783762162086
        # ref_derivatives = numerical_gradient_and_stresstensor(cell, get_energy)
        ref_derivatives = np.array([
            [-0.052554448615183 ,  0.0019232464154584,  0.0164716913531038],
            [ 0.0195730997631927,  0.0102995896611446, -0.0248131673430407],
            [ 0.0348620867640648, -0.0122673990432531,  0.0083400308170667],
            [-0.0018807354251749,  0.0000445629666501,  0.0000014450307617],
            [ 0.0405035486750194, -0.0299275689030765,  0.0332470406760876],
            [-0.0299700908357181, -0.001340544457662 ,  0.0136869616795821],
            [ 0.0332548719939041,  0.0136434722008971, -0.0355085206393824],
        ])

        assert np.abs(test_energy - ref_energy) < 1e-9
        assert np.max(np.abs(test_derivatives[:-3, :] - ref_derivatives[:-3, :])) < 3e-8
        # assert np.max(np.abs(test_derivatives[-3:, :] - ref_derivatives[-3:, :])) < 3e-7 # TODO: Support Becke grid stress tensor

        dm = mf.make_rdm1()
        kmesh = np.array([1,1,1])
        kpts = cell.make_kpts(kmesh)

        test_hcore_derivatives = get_nuc(cell, dm, kpts)
        test_hcore_derivatives += kin_derivatives(cell, dm, kpts)
        assert test_hcore_derivatives.shape == (cell.natm + 3, 3)

        ref_hcore_derivatives = numerical_hcore_gradient_and_stresstensor(cell, dm, kmesh)

        assert np.max(np.abs(test_hcore_derivatives[:-3, :] - ref_hcore_derivatives[:-3, :])) < 3e-8
        assert np.max(np.abs(test_hcore_derivatives[-3:, :] - ref_hcore_derivatives[-3:, :])) < 3e-7

    def test_gdf_hcore_derivatives_krhf(self):
        cell = pyscf.M(
            atom = """
                O 15.43509000 9.59549000 8.94968000
                H 15.05724000 9.21878000 9.73314000
                H 0.51550474 9.33856000 9.01857000
            """,
            a = np.eye(3) * (15.9069652593 / 3),
            unit = "Angstrom",
            basis = "sto-6g",
            verbose = 0,
        )

        kmesh = np.array([3,1,1])

        def get_energy(cell):
            kpts = cell.make_kpts(kmesh)
            mf = KRHF(cell, kpts=kpts).density_fit(auxbasis="def2-universal-jkfit")
            mf.conv_tol = 1e-11
            e = mf.kernel()
            assert mf.converged
            return e, mf
        test_energy, mf = get_energy(cell)

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()
        test_derivatives = np.vstack((test_gradient, gobj.stress * cell.vol))

        # Energy check is consistency check
        ref_energy = -75.68294188844864
        # ref_derivatives = numerical_gradient_and_stresstensor(cell, get_energy)
        ref_derivatives = np.array([
            [-0.0590151794455096, -0.0182923543690094,  0.0523572944644002],
            [ 0.0116237738012614,  0.0251060370715095, -0.0479677144227253],
            [ 0.047391405857411 , -0.0068136822761744, -0.004389579544295 ],
            [ 0.0891122930113397, -0.0303158925873959,  0.0254392091392219],
            [-0.0303158941505899, -0.0126309679870928,  0.0368723210897315],
            [ 0.0254392105603074,  0.0368723237187396, -0.068196215678995 ],
        ])

        assert np.abs(test_energy - ref_energy) < 1e-9
        assert np.max(np.abs(test_derivatives[:-3, :] - ref_derivatives[:-3, :])) < 5e-7
        assert np.max(np.abs(test_derivatives[-3:, :] - ref_derivatives[-3:, :])) < 5e-7

        dm = mf.make_rdm1()
        kpts = cell.make_kpts(kmesh)

        test_hcore_derivatives = get_nuc(cell, dm, kpts)
        test_hcore_derivatives += kin_derivatives(cell, dm, kpts)
        assert test_hcore_derivatives.shape == (cell.natm + 3, 3)

        ref_hcore_derivatives = numerical_hcore_gradient_and_stresstensor(cell, dm, kmesh)

        assert np.max(np.abs(test_hcore_derivatives[:-3, :] - ref_hcore_derivatives[:-3, :])) < 3e-8
        assert np.max(np.abs(test_hcore_derivatives[-3:, :] - ref_hcore_derivatives[-3:, :])) < 3e-7

    @unittest.skipIf(num_devices > 1, '')
    def test_gdf_hcore_derivatives_krks(self):
        cell = pyscf.M(
            a = '''0.      1.7834  1.7834
                   1.7834  0.      1.7834
                   1.7834  1.7834  0.    ''',
            atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917',
            basis = 'def2-svp',
            mesh = [10086] * 3,
            verbose = 0,
        )

        kmesh = np.array([3,1,1])

        def get_energy(cell):
            kpts = cell.make_kpts(kmesh)
            mf = KRKS(cell, xc="wB97X", kpts=kpts).density_fit(auxbasis="def2-universal-jkfit")
            mf.grids = BeckeGrids(cell)
            mf.grids.atom_grid = (50, 194)
            mf.conv_tol = 1e-12
            e = mf.kernel()
            assert mf.converged
            return e, mf
        test_energy, mf = get_energy(cell)

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()
        test_derivatives = np.vstack((test_gradient, gobj.stress * cell.vol))

        # Energy check is consistency check
        ref_energy = -75.65250113719456
        # ref_derivatives = numerical_gradient_and_stresstensor(cell, get_energy)
        ref_derivatives = np.array([
            [ 0.0207234415228186, -0.0205836620637001, -0.0205837245204066],
            [-0.0207234041482707,  0.0205836266786719,  0.0205836975908369],
            [-0.4214593518980791, -0.1513553901588693, -0.1513553918641719],
            [-0.1513088777471694, -0.4214285198855805,  0.1508046670295471],
            [-0.1513088915316985,  0.1508046269549368, -0.4214284877690488],
        ])

        assert np.abs(test_energy - ref_energy) < 1e-9
        assert np.max(np.abs(test_derivatives[:-3, :] - ref_derivatives[:-3, :])) < 5e-7
        # assert np.max(np.abs(test_derivatives[-3:, :] - ref_derivatives[-3:, :])) < 5e-6 # TODO: Support Becke grid stress tensor

        dm = mf.make_rdm1()
        kpts = cell.make_kpts(kmesh)

        test_hcore_derivatives = get_nuc(cell, dm, kpts)
        test_hcore_derivatives += kin_derivatives(cell, dm, kpts)
        assert test_hcore_derivatives.shape == (cell.natm + 3, 3)

        ref_hcore_derivatives = numerical_hcore_gradient_and_stresstensor(cell, dm, kmesh)

        assert np.max(np.abs(test_hcore_derivatives[:-3, :] - ref_hcore_derivatives[:-3, :])) < 3e-8
        assert np.max(np.abs(test_hcore_derivatives[-3:, :] - ref_hcore_derivatives[-3:, :])) < 3e-7

    def test_gdf_hcore_derivatives_uhf(self):
        cell = pyscf.M(
            atom = """
                O 15.43509000 9.59549000 8.94968000
                H 15.05724000 9.21878000 9.73314000
                H 0.51550474 9.33856000 9.01857000
                He 2.51550474 9.33856000 9.01857000
            """,
            a = np.eye(3) * 15.9069652593,
            unit = "Angstrom",
            charge = 1,
            spin = 1,
            basis = "6-31g",
            verbose = 0,
        )

        def get_energy(cell):
            mf = UHF(cell).density_fit(auxbasis="def2-universal-jkfit")
            mf.conv_tol = 1e-11
            e = mf.kernel()
            assert mf.converged
            return e, mf
        test_energy, mf = get_energy(cell)

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()
        test_derivatives = np.vstack((test_gradient, gobj.stress * cell.vol))

        # Energy check is consistency check
        ref_energy = -78.48457315211476
        # ref_derivatives = numerical_gradient_and_stresstensor(cell, get_energy)
        ref_derivatives = np.array([
            [-0.0555001189184168,  0.00711174671153  ,  0.0089337752484653],
            [ 0.0414229687351053,  0.0090893369275591, -0.0309807412435248],
            [ 0.0139675427135444, -0.0163045759649094,  0.0220698445474454],
            [ 0.0001096094592867,  0.0001034923968746, -0.0000228787655487],
            [ 0.0143442309763486, -0.0362892617999933,  0.0630823853953189],
            [-0.0362892622263189,  0.0190900627927704,  0.0113875727691948],
            [ 0.0630823866032415,  0.0113875747587144, -0.0252774496800612],
        ])

        assert np.abs(test_energy - ref_energy) < 1e-9
        assert np.max(np.abs(test_derivatives[:-3, :] - ref_derivatives[:-3, :])) < 1e-7
        assert np.max(np.abs(test_derivatives[-3:, :] - ref_derivatives[-3:, :])) < 1e-7

    def test_gdf_hcore_derivatives_kuks(self):
        cell = pyscf.M(
            a = '''0.      1.7834  1.8834
                   1.7834  0.      1.7834
                   1.7834  1.7834  0.    ''',
            atom = 'C 0.,  0.,  0.; C 0.8917,  0.9017,  0.8917',
            basis = """
            BASIS "ao basis" SPHERICAL PRINT
            #BASIS SET: (12s,6p) -> [2s,1p]
            C    S
                0.7427370491E+03       0.9163596281E-02
                0.1361800249E+03       0.4936149294E-01
                0.3809826352E+02       0.1685383049E+00
                0.1308778177E+02       0.3705627997E+00
                0.5082368648E+01       0.4164915298E+00
                0.2093200076E+01       0.1303340841E+00
            C    SP
                0.3049723950E+02      -0.1325278809E-01       0.3759696623E-02
                0.6036199601E+01      -0.4699171014E-01       0.3767936984E-01
                0.1876046337E+01      -0.3378537151E-01       0.1738967435E+00
                0.7217826470E+00       0.2502417861E+00       0.4180364347E+00
                # 0.3134706954E+00       0.5951172526E+00       0.4258595477E+00
                # 0.1436865550E+00       0.2407061763E+00       0.1017082955E+00
            END
            """, # Modified sto-6g
            mesh = [10086] * 3,
            precision = 1e-9,
            verbose = 0,
        )

        kmesh = np.array([3,1,1])

        def get_energy(cell):
            kpts = cell.make_kpts(kmesh)
            mf = KUKS(cell, xc="PBE0", kpts=kpts).density_fit(auxbasis="def2-universal-jkfit")
            mf.grids = BeckeGrids(cell)
            mf.grids.atom_grid = (99, 590)
            mf.conv_tol = 1e-11
            e = mf.kernel()
            assert mf.converged
            return e, mf
        test_energy, mf = get_energy(cell)

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()
        test_derivatives = np.vstack((test_gradient, gobj.stress * cell.vol))

        # Energy check is consistency check
        ref_energy = -73.96736967398559
        # ref_derivatives = numerical_gradient_and_stresstensor(cell, get_energy)
        ref_derivatives = np.array([
            [-0.0432393890292815,  0.0152970496714033,  0.0907656186655004],
            [ 0.0432393193250391, -0.0152969768407729, -0.0907655656590123],
            [ 1.2659054821284599, -0.2414305584608201, -0.2869095677482392],
            [-0.2414146290874442,  1.2612306482395752,  0.305671657088169 ],
            [-0.2868897047392238,  0.3056643988941232,  1.4221600833508319],
        ])

        assert np.abs(test_energy - ref_energy) < 1e-9
        assert np.max(np.abs(test_derivatives[:-3, :] - ref_derivatives[:-3, :])) < 1e-6
        # assert np.max(np.abs(test_derivatives[-3:, :] - ref_derivatives[-3:, :])) < 1e-6 # TODO: Support Becke grid stress tensor

if __name__ == '__main__':
    print("Full Tests for PBC GDF Hcore gradient and stress tensor")
    unittest.main()
