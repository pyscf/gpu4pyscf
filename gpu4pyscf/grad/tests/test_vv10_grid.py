# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

import numpy as np
import unittest
import pytest
import pyscf
from gpu4pyscf.dft import rks, uks
from gpu4pyscf.grad.rks import _get_denlc
from gpu4pyscf.lib.multi_gpu import num_devices

def setUpModule():
    global mol, mol_unrestricted, xc, atom_grid, nlc_atom_grid_loose, nlc_atom_grid_dense

    atom = '''
    O  0.0000  0.7375 -0.0528
    O  0.0000 -0.7375 -0.1528
    H  0.8190  0.8170  0.4220
    H -0.8190 -0.8170  1.4220
    '''
    basis = 'def2-svp'
    xc = "wb97x-v"

    atom_grid = (99,590)
    nlc_atom_grid_loose = (10,14)
    nlc_atom_grid_dense = (99,590)

    mol = pyscf.M(atom=atom, basis=basis, max_memory=32000,
                  output='/dev/null', verbose=1)

    atom_unrestricted = '''
        C     0.000000    0.000000    0.000000
        H     1.090000    0.000000    0.000000
        H    -0.845000    0.943190    0.000000
        H    -0.545000   -0.943190    0.000000
    '''

    mol_unrestricted = pyscf.M(atom=atom_unrestricted, basis=basis, charge=0, spin=1,
                               max_memory=32000, output='/dev/null', verbose=1)

def tearDownModule():
    global mol, mol_unrestricted
    mol.stdout.close()
    del mol
    mol_unrestricted.stdout.close()
    del mol_unrestricted

def make_mf(mol, nlc_atom_grid, restricted = True):
    if restricted:
        mf = rks.RKS(mol, xc = xc)
    else:
        mf = uks.UKS(mol, xc = xc)
    mf.conv_tol = 1e-10
    mf.grids.atom_grid = atom_grid
    mf.nlcgrids.atom_grid = nlc_atom_grid
    mf.kernel()
    assert mf.converged
    return mf

def numerical_denlc(mf, dm, denlc_only = True):
    # In order to use this function, go to gpu4pyscf/dft/rks.py,
    # find get_veff() function (imported below), and insert:
    # `ks.enlc = enlc`
    # right after where enlc is computed.
    if denlc_only:
        assert dm is not None
        if dm.ndim == 2:
            from gpu4pyscf.dft.rks import get_veff as get_veff_energy
        else:
            from gpu4pyscf.dft.uks import get_veff as get_veff_energy

    mol = mf.mol

    dx = 1e-5
    mol_copy = mol.copy()
    numerical_gradient = np.zeros([mol.natm, 3])
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            mol_copy.build()
            mf.reset(mol_copy)
            mf.nlcgrids.build()

            if denlc_only:
                get_veff_energy(mf, mol = mol_copy, dm = dm)
                energy_p = mf.enlc
            else:
                energy_p = mf.kernel()

            xyz_m = mol.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            mf.reset(mol_copy)
            mf.nlcgrids.build()

            if denlc_only:
                get_veff_energy(mf, mol = mol_copy, dm = dm)
                energy_m = mf.enlc
            else:
                energy_m = mf.kernel()

            numerical_gradient[i_atom, i_xyz] = (energy_p - energy_m) / (2 * dx)
    mf.reset(mol)
    mf.kernel()

    np.set_printoptions(linewidth = np.iinfo(np.int32).max, threshold = np.iinfo(np.int32).max, precision = 16, suppress = True)
    print(repr(numerical_gradient))
    return numerical_gradient

def analytical_denlc(grad_obj, dm):
    mol = grad_obj.mol
    denlc_orbital, denlc_grid = _get_denlc(grad_obj, mol, dm)
    denlc = 2 * denlc_orbital
    if grad_obj.grid_response:
        assert denlc_grid is not None
        denlc += denlc_grid
    return denlc

class KnownValues(unittest.TestCase):
    def test_nlc_loose_grid_with_response(self):
        mf = make_mf(mol, nlc_atom_grid_loose)
        dm = mf.make_rdm1()
        grad_obj = mf.Gradients()
        grad_obj.grid_response = True

        # reference_gradient = numerical_denlc(mf, dm)
        reference_gradient = np.array([
            [ 0.001143260325992 ,  0.0017358143505897,  0.0006100543972765],
            [-0.0003844551614562, -0.0016676535215254,  0.0003250853961023],
            [-0.0009018708738151,  0.000184766798389 , -0.0005937183092386],
            [ 0.0001430657092794, -0.0002529276267593, -0.000341421485528 ],
        ])
        test_gradient = analytical_denlc(grad_obj, dm)

        assert np.linalg.norm(test_gradient - reference_gradient) < 1e-8

    @pytest.mark.slow
    def test_nlc_dense_grid_with_response(self):
        mf = make_mf(mol, nlc_atom_grid_dense)
        dm = mf.make_rdm1()
        grad_obj = mf.Gradients()
        grad_obj.grid_response = True

        # reference_gradient = numerical_denlc(mf, dm)
        reference_gradient = np.array([
            [ 0.001318206092199 ,  0.0012072857220879,  0.000511811046322 ],
            [-0.0003562931241707, -0.0010925822732655,  0.0004861572269754],
            [-0.0011002806221683,  0.0001493420768994, -0.0006867203722338],
            [ 0.0001383676548339, -0.0002640455264158, -0.000311247898982 ],
        ])
        test_gradient = analytical_denlc(grad_obj, dm)

        assert np.linalg.norm(test_gradient - reference_gradient) < 1e-8

    @unittest.skipIf(num_devices > 1, '')
    def test_nlc_dense_grid_without_response(self):
        mf = make_mf(mol, nlc_atom_grid_dense)
        dm = mf.make_rdm1()
        grad_obj = mf.Gradients()
        assert grad_obj.grid_response is False

        # reference_gradient = numerical_denlc(mf, dm)
        reference_gradient = np.array([
            [ 0.001318206092199 ,  0.0012072857220879,  0.000511811046322 ],
            [-0.0003562931241707, -0.0010925822732655,  0.0004861572269754],
            [-0.0011002806221683,  0.0001493420768994, -0.0006867203722338],
            [ 0.0001383676548339, -0.0002640455264158, -0.000311247898982 ],
        ])
        test_gradient = analytical_denlc(grad_obj, dm)

        assert np.linalg.norm(test_gradient - reference_gradient) < 1e-8

    @unittest.skipIf(num_devices > 1, '')
    def test_wb97xv_loose_grid_with_response(self):
        mf = make_mf(mol, nlc_atom_grid_loose)
        grad_obj = mf.Gradients()
        grad_obj.grid_response = True

        # reference_gradient = numerical_denlc(mf, dm = None, denlc_only = False)
        reference_gradient = np.array([
            [ 0.0059973402244395,  0.0478981220908281,  0.0267982727564231],
            [ 0.0505556343455282, -0.052278126361216 , -0.095057491478201 ],
            [-0.0190222422702391, -0.0086736008597654, -0.0082766163700398],
            [-0.0375302562360957,  0.013052920166956 ,  0.0765366763744169],
        ])
        test_gradient = grad_obj.kernel()

        assert np.linalg.norm(test_gradient - reference_gradient) < 1e-5

    @pytest.mark.slow
    def test_wb97xv_dense_grid_with_response(self):
        mf = make_mf(mol, nlc_atom_grid_dense)
        grad_obj = mf.Gradients()
        grad_obj.grid_response = True

        # reference_gradient = numerical_denlc(mf, dm = None, denlc_only = False)
        reference_gradient = np.array([
            [ 0.0061438626630661,  0.0473716838200744,  0.0266715716179533],
            [ 0.050604123202902 , -0.0517109242537117, -0.0948872909134479],
            [-0.0192063581039292, -0.0087039268237277, -0.0083493503666432],
            [-0.0375411474351495,  0.0130424609778856,  0.0765658953127968],
        ])
        test_gradient = grad_obj.kernel()

        assert np.linalg.norm(test_gradient - reference_gradient) < 1e-5

    def test_unrestricted_nlc_loose_grid_with_response(self):
        mf = make_mf(mol_unrestricted, nlc_atom_grid_loose, restricted = False)
        dm = mf.make_rdm1()
        grad_obj = mf.Gradients()
        grad_obj.grid_response = True

        # reference_gradient = numerical_denlc(mf, dm)
        reference_gradient = np.array([
            [-0.0004755863279582, -0.0002109066751105,  0.                ],
            [-0.000964123844302 ,  0.0001098022513191,  0.                ],
            [ 0.000811048781954 , -0.0006588373811789,  0.                ],
            [ 0.0006286613899592,  0.0007599418053172,  0.                ],
        ])
        test_gradient = analytical_denlc(grad_obj, dm)

        assert np.linalg.norm(test_gradient - reference_gradient) < 1e-8

    @unittest.skipIf(num_devices > 1, '')
    def test_unrestricted_nlc_dense_grid_without_response(self):
        mf = make_mf(mol_unrestricted, nlc_atom_grid_dense, restricted = False)
        dm = mf.make_rdm1()
        grad_obj = mf.Gradients()
        assert grad_obj.grid_response is False

        # reference_gradient = numerical_denlc(mf, dm)
        reference_gradient = np.array([
            [-0.0003204622806702, -0.0002658539430267,  0.                ],
            [-0.0011826315343688,  0.00009877765203  ,  0.                ],
            [ 0.0008693887169203, -0.0006959881906909, -0.0000000000003469],
            [ 0.0006337050977717,  0.0008630644820345, -0.0000000000003469],
        ])
        test_gradient = analytical_denlc(grad_obj, dm)

        assert np.linalg.norm(test_gradient - reference_gradient) < 1e-8

    @unittest.skipIf(num_devices > 1, '')
    def test_unrestricted_wb97xv_loose_grid_with_response(self):
        mf = make_mf(mol_unrestricted, nlc_atom_grid_loose, restricted = False)
        grad_obj = mf.Gradients()
        grad_obj.grid_response = True

        # reference_gradient = numerical_denlc(mf, dm = None, denlc_only = False)
        reference_gradient = np.array([
            [ 0.066462332881656 , -0.0443292179852506, -0.0000001456612608],
            [ 0.0003404558412967, -0.0102625335784978, -0.0000000582645043],
            [-0.0626305372009028,  0.0470635349358872, -0.0000000042632564],
            [-0.0041723087207401,  0.0075280844669123, -0.0000001030286967],
        ])
        test_gradient = grad_obj.kernel()

        assert np.linalg.norm(test_gradient - reference_gradient) < 2e-6

    @unittest.skipIf(num_devices > 1, '')
    def test_unrestricted_wb97xv_dense_grid_without_response(self):
        mf = make_mf(mol_unrestricted, nlc_atom_grid_dense, restricted = False)
        grad_obj = mf.Gradients()
        assert grad_obj.grid_response is False

        # reference_gradient = numerical_denlc(mf, dm = None, denlc_only = False)
        reference_gradient = np.array([
            [ 0.06671206378428  , -0.0443454720056025, -0.0000001463718036],
            [ 0.0000733937355335, -0.0102775192800664, -0.0000000547117907],
            [-0.0625935602016625,  0.047039745965094 , -0.0000000049737992],
            [-0.0041919538062984,  0.0075831103174551, -0.0000001026734253],
        ])
        test_gradient = grad_obj.kernel()

        assert np.linalg.norm(test_gradient - reference_gradient) < 2e-6


if __name__ == "__main__":
    print("Full Tests for vv10 gradient, including grid response")
    unittest.main()
