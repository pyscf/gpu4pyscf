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
import cupy as cp
import pyscf
from pyscf import lib
from pyscf.dft import rks as rks_cpu
from pyscf.dft import uks as uks_cpu
from gpu4pyscf.dft import rks, uks
from gpu4pyscf.properties import polarizability

try:
    from pyscf.prop import polarizability as polar
except Exception:
    polar = None

lib.num_threads(8)

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='def2tzvpp'
grids_level = 6

def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas, max_memory=32000)
    mol.output = '/dev/null'
    mol.build()
    mol.verbose = 1

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def run_dft_polarizability(xc):
    mf = rks.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    e_dft = mf.kernel()
    polar = polarizability.eval_polarizability(mf)
    return e_dft, polar

def run_dft_df_polarizability(xc):
    mf = rks.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level = grids_level
    e_dft = mf.kernel()
    polar = polarizability.eval_polarizability(mf)
    return e_dft, polar

def _vs_cpu_rks(xc):
    mf = rks.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    e_gpu = mf.kernel()
    polar_gpu = polarizability.eval_polarizability(mf)
    
    mf_cpu = rks_cpu.RKS(mol, xc=xc)
    mf_cpu.conv_tol = 1e-12
    e_cpu = mf_cpu.kernel()
    polar_cpu = polar.rhf.Polarizability(mf_cpu).polarizability()

    assert np.abs(e_gpu - e_cpu) < 1e-5
    assert np.linalg.norm(polar_cpu - polar_gpu) < 1e-3

def _vs_cpu_uks(xc):
    mf = uks.UKS(mol, xc=xc)
    mf.grids.level = grids_level
    e_gpu = mf.kernel()
    polar_gpu = polarizability.eval_polarizability(mf)
    
    mf_cpu = uks_cpu.UKS(mol, xc=xc)
    mf_cpu.conv_tol = 1e-12
    e_cpu = mf_cpu.kernel()
    polar_cpu = polar.rhf.Polarizability(mf_cpu).polarizability()

    assert np.abs(e_gpu - e_cpu) < 1e-5
    assert np.linalg.norm(polar_cpu - polar_gpu) < 1e-3

def numerical_polarizability(mf, delta_E):
    mol = mf.mol
    Hcore = mf.get_hcore()
    dipole_integral = cp.asarray(mol.intor('cint1e_r_sph', comp=3))
    def apply_electric_field(mf, E):
        E = cp.asarray(E)
        delta_Hcore = cp.einsum('d,dij->ij', E, dipole_integral)

        mf.get_hcore = lambda *args: Hcore + delta_Hcore
        mf.kernel()
        dipole = mf.dip_moment(unit = "au", verbose = 0)

        return dipole

    polarizability_numerical = np.zeros((3,3))
    for i_xyz in range(3):
        E_1p = np.zeros(3)
        E_1p[i_xyz] = delta_E
        d_1p = apply_electric_field(mf, E_1p)

        E_1m = np.zeros(3)
        E_1m[i_xyz] = -delta_E
        d_1m = apply_electric_field(mf, E_1m)

        polarizability_numerical[i_xyz, :] = (d_1p - d_1m) / (2 * delta_E)

    mf.get_hcore = lambda *args: Hcore

    return polarizability_numerical

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by Q-Chem

    $rem
    MEM_STATIC    2000
    JOBTYPE       POLARIZABILITY
    METHOD        b3lyp
    RESPONSE_POLAR 0
    BASIS         def2-tzvpp
    SCF_CONVERGENCE     12
    XC_GRID 000099000590
    SYMMETRY FALSE
    SYM_IGNORE = TRUE
    $end

     -----------------------------------------------------------------------------
    Polarizability tensor      [a.u.]
      8.5899463     -0.0000000     -0.0000000
     -0.0000000      6.0162267     -0.0000000
     -0.0000000     -0.0000000      7.5683123

    $rem
    MEM_STATIC    2000
    JOBTYPE       POLARIZABILITY
    METHOD        b3lyp
    RESPONSE_POLAR -1
    BASIS         def2-tzvpp
    SCF_CONVERGENCE     12
    XC_GRID 000099000590
    SYMMETRY FALSE
    SYM_IGNORE = TRUE
    ri_j        True
    ri_k        True
    aux_basis     RIJK-def2-tzvpp
    THRESH        14
    $end

     <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Polarizability tensor      [a.u.]
        8.5902540     -0.0000000     -0.0000000
        0.0000000      6.0167648      0.0000000
       -0.0000000     -0.0000000      7.5688173
    
    '''
    def test_rks_b3lyp(self):
        print('-------- RKS B3LYP -------------')
        e_tot, polar = run_dft_polarizability('B3LYP')
        assert np.allclose(e_tot, -76.4666494276)
        qchem_polar = np.array([ [ 8.5899463,    -0.0000000,     -0.0000000],
                                 [-0.0000000,     6.0162267,     -0.0000000],
                                 [-0.0000000,    -0.0000000,      7.5683123]])
        assert np.allclose(polar, qchem_polar)

    def test_rks_b3lyp_df(self):
        print('-------- RKS density fitting B3LYP -------------')
        e_tot, polar = run_dft_df_polarizability('B3LYP')
        assert np.allclose(e_tot, -76.4666819553)
        qchem_polar = np.array([ [  8.5902540,    -0.0000000,     -0.0000000],
                                 [  0.0000000,     6.0167648,     -0.0000000],
                                 [ -0.0000000,    -0.0000000,      7.5688173]])
        assert np.allclose(polar, qchem_polar)

    # Since QChem 6.1 doesn't have vv10 response, we obtain reference result from numerical polarizability
    def test_rks_pbe_with_vv10(self):
        mf = rks.RKS(mol, xc = "pbe")
        mf.grids.atom_grid = (99,590)
        mf.nlc = 'vv10'
        mf.nlcgrids.atom_grid = (50,194)
        mf.conv_tol = 1e-16
        mf.direct_scf_tol = 1e-16
        mf.verbose = 0
        mf.kernel()
        test_polarizability = polarizability.eval_polarizability(mf)

        # ref_polarizability = numerical_polarizability(mf, 5e-4)
        ref_polarizability = np.array([
            [ 8.75655240e+00,  1.32813366e-12, -2.05643502e-07],
            [-1.22378276e-06,  6.24880993e+00, -7.88354937e-08],
            [-2.12646469e-11,  8.26925452e-07,  7.80851474e+00],
        ])

        assert np.linalg.norm(test_polarizability - ref_polarizability) < 2e-5

    def test_rks_pbe_with_vv10_df(self):
        mf = rks.RKS(mol, xc = "pbe")
        mf.grids.atom_grid = (99,590)
        mf.nlc = 'vv10'
        mf.nlcgrids.atom_grid = (50,194)
        mf.conv_tol = 1e-16
        mf.verbose = 0
        mf = mf.density_fit(auxbasis = "def2-universal-jkfit")
        mf.kernel()
        test_polarizability = polarizability.eval_polarizability(mf)

        # ref_polarizability = numerical_polarizability(mf, 5e-4)
        ref_polarizability = np.array([
             [ 8.75713162e+00, -2.93331970e-13, -2.30659936e-08],
             [ 1.12938014e-06,  6.24976678e+00,  6.23723295e-08],
             [-1.25358979e-10, -9.70652871e-08,  7.80953455e+00],
        ])

        assert np.linalg.norm(test_polarizability - ref_polarizability) < 2e-5

    def test_rks_wb97xv(self):
        mf = rks.RKS(mol, xc = "wb97x-v")
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-16
        mf.direct_scf_tol = 1e-16
        mf.verbose = 0
        mf.kernel()
        test_polarizability = polarizability.eval_polarizability(mf)

        # ref_polarizability = numerical_polarizability(mf, 5e-4)
        ref_polarizability = np.array([
            [ 8.48051130e+00,  1.26620050e-12, -5.76743098e-08],
            [ 1.36110971e-07,  5.99968451e+00, -5.26197974e-08],
            [-7.13398253e-12, -1.05076611e-05,  7.47143620e+00],
        ])

        assert np.linalg.norm(test_polarizability - ref_polarizability) < 4e-5

    def test_rks_wb97xv_df(self):
        mf = rks.RKS(mol, xc = "wb97x-v")
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-16
        mf.verbose = 0
        mf = mf.density_fit(auxbasis = "def2-universal-jkfit")
        mf.kernel()
        test_polarizability = polarizability.eval_polarizability(mf)

        # ref_polarizability = numerical_polarizability(mf, 5e-4)
        ref_polarizability = np.array([
            [ 8.48085798e+00, -6.46046743e-13,  3.67918096e-06],
            [-1.45786335e-06,  6.00033985e+00, -9.99222927e-07],
            [-3.60933614e-12, -1.04525515e-05,  7.47207291e+00],
        ])

        assert np.linalg.norm(test_polarizability - ref_polarizability) < 4e-5

    def test_rks_wb97xv_df_response_without_nlc(self):
        mf = rks.RKS(mol, xc = "wb97x-v")
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-16
        mf.verbose = 0
        mf = mf.density_fit(auxbasis = "def2-universal-jkfit")
        mf.kernel()
        test_polarizability = polarizability.eval_polarizability(mf, with_nlc = False)
        print(test_polarizability)

        # This is a consistency test, the reference value is not validated from independent source
        ref_polarizability = np.array([
            [ 8.48485899e+00,  4.40307440e-15, -3.26162370e-14],
            [ 4.40307440e-15,  6.00459819e+00,  3.39953163e-15],
            [-3.26162370e-14,  3.39953163e-15,  7.47621491e+00],
        ])

        assert np.linalg.norm(test_polarizability - ref_polarizability) < 1e-5

    @unittest.skipIf(polar is None, "Skipping test if pyscf.properties is not installed")
    def test_cpu_rks(self):
        _vs_cpu_rks('b3lyp')

    """
    # UKS is not supported yet
    @unittest.skipIf(polar is None, "Skipping test if pyscf.properties is not installed")
    def test_cpu_uks(self):
        _vs_cpu_uks('b3lyp')
    """
    
if __name__ == "__main__":
    print("Full Tests for polarizabillity")
    unittest.main()
