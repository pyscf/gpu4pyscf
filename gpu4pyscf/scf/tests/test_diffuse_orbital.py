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
import gpu4pyscf
from gpu4pyscf import scf, dft

def setUpModule():
    global mol #, ref_mol
    mol = pyscf.M(
        atom = """
            H 0 0 0
            Li 1.0 0.1 0
        """,
        basis = """
            X    S
                0.3425250914E+01       0.1543289673E+00
                0.6239137298E+00       0.5353281423E+00
                0.1688554040E+00       0.4446345422E+00
            X    S
                0.1611957475E+02       0.1543289673E+00
                0.2936200663E+01       0.5353281423E+00
                0.7946504870E+00       0.4446345422E+00
            X    P
                0.010000 1.0
            X    P
                0.010001 1.0
        """,
        verbose = 5,
        output = '/dev/null',
    )

    # ref_mol = pyscf.M(
    #     atom = """
    #         H 0 0 0
    #         Li 1.0 0.1 0
    #     """,
    #     basis = """
    #         X    S
    #             0.3425250914E+01       0.1543289673E+00
    #             0.6239137298E+00       0.5353281423E+00
    #             0.1688554040E+00       0.4446345422E+00
    #         X    S
    #             0.1611957475E+02       0.1543289673E+00
    #             0.2936200663E+01       0.5353281423E+00
    #             0.7946504870E+00       0.4446345422E+00
    #         X    P
    #             0.010000 1.0
    #         # X    P
    #         #     0.010001 1.0
    #     """,
    #     verbose = 5,
    # )

def tearDownModule():
    global mol #, ref_mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert gpu4pyscf.scf.hf.remove_overlap_zero_eigenvalue is False
        gpu4pyscf.scf.hf.remove_overlap_zero_eigenvalue = True

    @classmethod
    def tearDownClass(cls):
        gpu4pyscf.scf.hf.remove_overlap_zero_eigenvalue = False

    def test_rhf(self):
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        energy = mf.kernel()
        assert mf.converged
        assert np.abs(energy - -7.670162135801041) < 1e-5

        gobj = mf.Gradients()
        gradient = gobj.kernel()
        assert np.max(np.abs(gradient - np.array([
            [ 2.53027311e-01,  2.53027311e-02,  1.78111017e-19],
            [-2.53027311e-01, -2.53027311e-02, -1.78111017e-19],
        ]))) < 1e-5

        dipole = mf.dip_moment()
        assert np.max(np.abs(dipole - np.array([4.26375987e+00, 4.26375987e-01, 1.86659164e-16]))) < 1e-4

    def test_rhf_soscf(self):
        mf = dft.RKS(mol, xc = "wB97M-d3bj")
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-10
        mf = mf.newton()
        energy = mf.kernel()
        assert mf.converged
        assert np.abs(energy - -7.773544875779531) < 1e-5

        gobj = mf.Gradients()
        gradient = gobj.kernel()
        assert np.max(np.abs(gradient - np.array([
            [ 2.44614610e-01,  2.44653881e-02, -3.14001231e-18],
            [-2.44641034e-01, -2.44569088e-02, -6.41480825e-18],
        ]))) < 1e-5

    def test_uhf(self):
        mf = dft.RKS(mol, xc = "PBE")
        mf.grids.atom_grid = (50,194)
        mf = mf.density_fit(auxbasis = "def2-universal-jkfit")
        mf.conv_tol = 1e-10
        energy = mf.kernel()
        assert mf.converged
        assert np.abs(energy - -7.748763949503415) < 1e-5

        gobj = mf.Gradients()
        gobj.grid_response = True
        gradient = gobj.kernel()
        assert np.max(np.abs(gradient - np.array([
            [ 2.44275992e-01,  2.44757818e-02,  3.22713915e-19],
            [-2.44275992e-01, -2.44757818e-02, -3.17524970e-19],
        ]))) < 1e-5

    def test_rohf(self):
        mf = dft.ROKS(mol, xc = "PBE0")
        mf.grids.level = 3
        mf.conv_tol = 1e-10
        energy = mf.kernel()
        assert mf.converged
        assert np.abs(energy - -7.749335934429277) < 1e-5

        # TODO: There seems to be some problem with ROHF gradient, there's no class grad.rohf.Gradients,
        # and the gradient class falls back onto grad.rhf.Gradients

    def test_rhf_hessian(self):
        mf = dft.RKS(mol, xc = "PBE0")
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-10
        energy = mf.kernel()
        assert mf.converged
        assert np.abs(energy - -7.749450458759258) < 1e-5

        hobj = mf.Hessian()
        hessian = hobj.kernel()
        assert np.max(np.abs(hessian - np.array(
       [[[[ 5.47269479e-01,  6.77947776e-02,  1.36415823e-17],
         [ 6.77947776e-02, -1.23184348e-01, -2.77752324e-17],
         [ 1.36415823e-17, -2.77752324e-17, -1.29751072e-01]],

        [[-5.47077502e-01, -6.76941081e-02, -3.30772790e-18],
         [-6.76918425e-02,  1.23166473e-01,  1.95849919e-17],
         [-6.41033433e-18,  1.90019347e-17,  1.29935820e-01]]],


       [[[-5.47077502e-01, -6.76918425e-02, -6.41033433e-18],
         [-6.76941081e-02,  1.23166473e-01,  1.90019347e-17],
         [-3.30772790e-18,  1.95849919e-17,  1.29935820e-01]],

        [[ 5.47043561e-01,  6.77385721e-02,  1.73115648e-16],
         [ 6.77385721e-02, -1.23177949e-01,  7.29238276e-17],
         [ 1.73115648e-16,  7.29238276e-17, -1.29908851e-01]]]]
        ))) < 1e-5

    def test_uhf_solvent(self):
        mf = dft.UKS(mol, xc = "PBE0")
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-10
        mf = mf.PCM()
        mf.with_solvent.method = "IEF-PCM"
        energy = mf.kernel()
        assert mf.converged
        assert np.abs(energy - -7.770917908597051) < 1e-5

        gobj = mf.Gradients()
        gobj.grid_response = True
        gradient = gobj.kernel()
        assert np.max(np.abs(gradient - np.array([
            [ 2.52149477e-01,  2.52230067e-02,  5.40896707e-18],
            [-2.52149477e-01, -2.52230067e-02, -5.44037802e-18],
        ]))) < 1e-5

    def test_rhf_lowmem(self):
        mf = scf.hf_lowmem.RHF(mol)
        mf.conv_tol = 1e-10
        energy = mf.kernel()
        assert mf.converged
        assert np.abs(energy - -7.670162135801045) < 1e-5

        gobj = mf.Gradients()
        gradient = gobj.kernel()
        assert np.max(np.abs(gradient - np.array([
            [ 2.53027311e-01,  2.53027311e-02, -6.39723698e-19],
            [-2.53027311e-01, -2.53027311e-02,  6.39723698e-19],
        ]))) < 1e-5

    def test_rks_lowmem(self):
        mf = dft.rks_lowmem.RKS(mol, xc = "wB97M-V")
        mf.grids.atom_grid = (99,590)
        mf.nlcgrids.atom_grid = (50,194)
        mf.conv_tol = 1e-10
        energy = mf.kernel()
        assert mf.converged
        assert np.abs(energy - -7.755312446937159) < 1e-5

        gobj = mf.Gradients()
        gradient = gobj.kernel()
        assert np.max(np.abs(gradient - np.array([
            [ 2.44591704e-01,  2.44630215e-02,  1.52039894e-19],
            [-2.44604955e-01, -2.44532125e-02,  3.37936483e-17],
        ]))) < 1e-5

if __name__ == "__main__":
    print("Tests for System with Diffuse Orbitals (Ill-conditioned Overlap Matrices)")
    unittest.main()
