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

import unittest
import numpy as np
import pyscf
from pyscf import  dft as cpu_dft
from pyscf import  scf as cpu_scf
from pyscf import qmmm as cpu_qmmm
from gpu4pyscf import  dft as gpu_dft
from gpu4pyscf import  scf as gpu_scf
from gpu4pyscf import qmmm as gpu_qmmm

def setUpModule():
    global mol
    mol = pyscf.M(atom='''
        O  0.000000  0.000000  0.000000
        H  0.758602  0.000000  0.504284
        H  0.958602  0.000000  -0.504284
    ''',
    basis='ccpvdz',
    verbose=1,
    output = '/dev/null')
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_energy_and_gradient(self):
        np.random.seed(10)

        mm_coords = (np.random.random((5, 3)) - 0.5) * 20
        mm_charges = (np.random.random(5) - 0.5) * 2

        cpu_mf = cpu_dft.RKS(mol, xc='pbe')
        cpu_mf.grids.atom_grid = (50,194)
        cpu_mf.conv_tol = 1e-12
        cpu_mf = cpu_qmmm.mm_charge(cpu_mf, mm_coords, mm_charges, unit = "Bohr")
        cpu_energy = cpu_mf.kernel()
        assert cpu_mf.converged

        # Get around with a bug in pyscf<=3.10.0 that get_zetas() return a float instead of a np.ndarray
        cpu_mf.mm_mol.get_zetas = lambda : cpu_mf.mm_mol.atom_charges() * 0 + 1e16

        cpu_gobj = cpu_mf.nuc_grad_method()
        cpu_gradient = cpu_gobj.kernel()

        cpu_dm = cpu_mf.make_rdm1()
        cpu_gradient_mm = cpu_gobj.grad_nuc_mm() + cpu_gobj.grad_hcore_mm(cpu_dm)

        cpu_dipole = cpu_mf.dip_moment()

        ###

        gpu_mf = gpu_dft.RKS(mol, xc = 'pbe')
        gpu_mf.grids.atom_grid = (50,194)
        gpu_mf.conv_tol = 1e-12
        gpu_mf = gpu_qmmm.mm_charge(gpu_mf, mm_coords, mm_charges, unit = "Bohr")
        gpu_energy = gpu_mf.kernel()
        assert gpu_mf.converged

        gpu_gobj = gpu_mf.nuc_grad_method()
        gpu_gradient = gpu_gobj.kernel()

        gpu_dm = gpu_mf.make_rdm1()
        gpu_gradient_mm = gpu_gobj.grad_nuc_mm() + gpu_gobj.grad_hcore_mm(gpu_dm)

        gpu_dipole = gpu_mf.dip_moment()

        assert np.max(np.abs(gpu_energy - cpu_energy)) < 1e-9
        assert np.max(np.abs(gpu_gradient - cpu_gradient)) < 1e-6
        assert np.max(np.abs(gpu_gradient_mm - cpu_gradient_mm)) < 1e-6
        assert np.max(np.abs(gpu_dipole - cpu_dipole)) < 1e-6

    def test_with_ecp(self):
        # Reference answer from CPU implementation
        mm_coords = np.array([[-5, 0, 0], [5, 0, 0]])
        mm_charges = np.array([-1, 1])

        mol_with_ecp = pyscf.M(atom = '''
            K  1.0 0.0 0.0
            H -0.2 0.1 0.0
        ''',
        basis = 'LANL2DZ',
        ecp = 'LANL2DZ',
        verbose = 0)

        mf = gpu_dft.RKS(mol_with_ecp, xc = "r2scan")
        mf.grids.level = 0
        mf.conv_tol = 1e-10
        mf = gpu_qmmm.mm_charge(mf, mm_coords, mm_charges)
        mf = mf.density_fit()
        energy = mf.kernel()
        assert mf.converged
        assert abs(energy - -28.422481074762427) < 1e-9

        gobj = mf.nuc_grad_method()
        gradient = gobj.kernel()
        assert np.max(np.abs(gradient - np.array([[-3.81321503e-01,  3.21138282e-02, 0],
                                                  [ 3.80450204e-01, -3.21486520e-02, 0]]))) < 1e-6

    def test_to_cpu(self):
        mm_coords = np.array([[-5, 0, 0], [5, 0, 0]])
        mm_charges = np.array([-1, 1])

        mf = gpu_dft.RKS(mol, xc = "b3lyp")
        mf.grids.level = 1
        mf.conv_tol = 1e-10
        mf = mf.density_fit()
        mf = gpu_qmmm.mm_charge(mf, mm_coords, mm_charges)
        gpu_energy = mf.kernel()
        assert mf.converged

        mf = mf.to_cpu()
        cpu_energy = mf.kernel()
        assert mf.converged

        assert isinstance(mf, cpu_dft.rks.RKS)
        assert isinstance(mf, cpu_qmmm.QMMM)
        assert abs(gpu_energy - cpu_energy) < 1e-9

    # TODO: Support to_gpu() in pyscf
    # def test_to_gpu(self):
    #     mm_coords = np.array([[-5, 0, 0], [5, 0, 0]])
    #     mm_charges = np.array([-1, 1])

    #     mf = cpu_dft.RKS(mol, xc = "b3lyp")
    #     mf.grids.level = 1
    #     mf.conv_tol = 1e-10
    #     mf = mf.density_fit()
    #     mf = cpu_qmmm.mm_charge(mf, mm_coords, mm_charges)
    #     cpu_energy = mf.kernel()
    #     assert mf.converged

    #     mf = mf.to_gpu()
    #     cpu_energy = mf.kernel()
    #     assert mf.converged

    #     assert isinstance(mf, gpu_dft.rks.RKS)
    #     assert isinstance(mf, gpu_qmmm.QMMM)
    #     assert abs(gpu_energy - cpu_energy) < 1e-9

    def test_undo_qmmm(self):
        # Reference answer from CPU implementation
        mm_coords = np.array([[-5, 0, 0], [5, 0, 0]])
        mm_charges = np.array([-1, 1])

        mf = gpu_scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = mf.density_fit()
        mf = gpu_qmmm.mm_charge(mf, mm_coords, mm_charges)
        energy_with_mm = mf.kernel()
        assert mf.converged
        assert abs(energy_with_mm - -75.93434480655407) < 1e-10

        mf = mf.undo_qmmm()
        energy_without_mm = mf.kernel()
        assert mf.converged
        assert abs(energy_without_mm - -75.95594732431334) < 1e-10

    def test_dipole_with_charge(self):
        # Reference answer from CPU implementation
        mm_coords = np.array([[-5, 0, 0], [5, 0, 0]])
        mm_charges = np.array([-1, 1])

        mol_charged = pyscf.M(atom='''
            O      0.199968    0.000000   4.00000
            H      1.174548   -0.000000   4.00000
            H     -0.287258    0.844506   4.00000
            H     -0.287258   -0.844506   4.00000
        ''',
        basis='6-31g',
        verbose=0,
        charge=1)

        mf = gpu_dft.RKS(mol_charged, xc = "wB97X-d3bj")
        mf.conv_tol = 1e-12
        mf = mf.density_fit()
        mf = gpu_qmmm.mm_charge(mf, mm_coords, mm_charges)
        energy = mf.kernel()
        assert mf.converged
        assert abs(energy - -76.68819909313646) < 1e-9

        dm = mf.make_rdm1()
        dipole = mf.dip_moment(unit='DEBYE', dm=dm, origin=mol_charged.atom_coords().mean(axis=0))
        assert np.max(np.abs(dipole - np.array([-1.27710722e-01, 0, 2.23694789e-03]))) < 1e-6


if __name__ == "__main__":
    print("Full Tests for QMMM MM charges")
    unittest.main()
