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
import pytest
import numpy as np
import pyscf
import gpu4pyscf
from gpu4pyscf import scf, dft
from gpu4pyscf.lib.cutensor import contract
try:
    from gpu4pyscf.dispersion import dftd3
except ImportError:
    dftd3 = None

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
        assert gpu4pyscf.scf.hf.remove_overlap_zero_eigenvalue is True
        cls.overlap_zero_eigenvalue_threshold = gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold

    @classmethod
    def tearDownClass(cls):
        assert gpu4pyscf.scf.hf.remove_overlap_zero_eigenvalue is True
        assert cls.overlap_zero_eigenvalue_threshold == gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold

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

        e, c = mf.canonicalize(mf.mo_coeff, mf.mo_occ)
        assert abs(e - mf.mo_energy).max() < 5e-7
        f = mf.get_fock()
        e1 = contract('qi,qi->i', contract('pi,pq->qi', c.conj(), f), c)
        assert abs(e - e1).max() < 1e-12

    @unittest.skipIf(dftd3 is None, "dftd3 not available")
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

    def test_rhf_df_against_qchem(self):
        mol = pyscf.M(
            atom = """
                C 0.000000 0.000000 0.000000
                C 0.881905 0.881905 0.881905
                C -0.881905 -0.881905 0.881905
                C 0.881905 -0.881905 -0.881905
                C -0.881905 0.881905 -0.881905
                H -1.524077 0.276170 -1.524077
                H 1.524077 1.524077 0.276170
                H 1.524077 -0.276170 -1.524077
                H 1.524077 0.276170 1.524077
                H -1.524077 -0.276170 1.524077
                H 1.524077 -1.524077 -0.276170
                H -0.276170 1.524077 -1.524077
                H 0.276170 1.524077 1.524077
                H 0.276170 -1.524077 -1.524077
                H -0.276170 -1.524077 1.524077
                H -1.524077 1.524077 -0.276170
                H -1.724077 -1.524077 0.276170
            """, # https://github.com/f3rmion/molpol135/blob/main/structures/neopentane/inp.xyz
            basis = "d-aug-cc-pvdz",
            charge = 0,
            verbose = 0,
        )

        assert gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold == 1e-6

        mf = mol.RHF().density_fit(auxbasis = "aug-cc-pvdz-rifit").to_gpu()
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        ### Q-Chem input
        # $rem
        # JOBTYPE force
        # METHOD HF
        # BASIS GEN
        # SYMMETRY      FALSE
        # SYM_IGNORE    TRUE
        # XC_GRID       000099000590
        # NL_GRID       000050000194
        # MAX_SCF_CYCLES 100
        # PURECART 1111
        # ri_j        True
        # ri_k        True
        # aux_basis     RIMP2-aug-cc-pVDZ
        # SCF_CONVERGENCE 10
        # THRESH        14
        # BASIS_LIN_DEP_THRESH 6
        # $end

        # $basis
        # H     0
        # S   4   1.00
        #      13.0100000              0.0196850
        #       1.9620000              0.1379770
        #       0.4446000              0.4781480
        #       0.1220000              0.5012400
        # S   1   1.00
        #       0.1220000              1.0000000
        # S   1   1.00
        #       0.0297400              1.0000000
        # S   1   1.00
        #       0.00725                1.000000
        # P   1   1.00
        #       0.7270000              1.0000000
        # P   1   1.00
        #       0.1410000              1.0000000
        # P   1   1.00
        #       0.02730                1.000000
        # ****
        # C     0
        # S   9   1.00
        #    6665.0000000              0.0006920
        #    1000.0000000              0.0053290
        #     228.0000000              0.0270770
        #      64.7100000              0.1017180
        #      21.0600000              0.2747400
        #       7.4950000              0.4485640
        #       2.7970000              0.2850740
        #       0.5215000              0.0152040
        #       0.1596000             -0.0031910
        # S   9   1.00
        #    6665.0000000             -0.0001460
        #    1000.0000000             -0.0011540
        #     228.0000000             -0.0057250
        #      64.7100000             -0.0233120
        #      21.0600000             -0.0639550
        #       7.4950000             -0.1499810
        #       2.7970000             -0.1272620
        #       0.5215000              0.5445290
        #       0.1596000              0.5804960
        # S   1   1.00
        #       0.1596000              1.0000000
        # S   1   1.00
        #       0.0469000              1.0000000
        # S   1   1.00
        #       0.0138                 1.000000
        # P   4   1.00
        #       9.4390000              0.0381090
        #       2.0020000              0.2094800
        #       0.5456000              0.5085570
        #       0.1517000              0.4688420
        # P   1   1.00
        #       0.1517000              1.0000000
        # P   1   1.00
        #       0.0404100              1.0000000
        # P   1   1.00
        #       0.0108                 1.000000
        # D   1   1.00
        #       0.5500000              1.0000000
        # D   1   1.00
        #       0.1510000              1.0000000
        # D   1   1.00
        #       0.0415                 1.000000
        # ****
        # $end
        ref_energy = -196.3486813198
        ref_gradient = np.array([
            [ 0.0043276, -0.0018665,  0.0500541, -0.0027863,  0.0025617, -0.0003143,
              0.0002619, -0.0001293, -0.0001536, -0.0048013,  0.0000031,  0.0004436,
             -0.0006010, -0.0004740,  0.0045722, -0.0000044, -0.0510934,],
            [ 0.0035242, -0.0019008,  0.0235071,  0.0027695, -0.0029402, -0.0005481,
              0.0004975,  0.0004053, -0.0005925, -0.0031218,  0.0000255, -0.0001117,
              0.0000284, -0.0001678,  0.0043918, -0.0000160, -0.0257504,],
            [ 0.0054562, -0.0043409,  0.0175991,  0.0032079,  0.0029882, -0.0002183,
             -0.0001964, -0.0000674, -0.0000995, -0.0036001,  0.0004863, -0.0000933,
              0.0000693, -0.0003414,  0.0034830,  0.0004414, -0.0247742,],
        ]).T

        assert np.abs(test_energy - ref_energy) < 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) < 2e-6

        gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold = 1e-20

        mf.reset()
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold = 1e-6

        ref_energy = -196.3487478362
        ref_gradient = np.array([
            [ 0.0043441, -0.0018739,  0.0500476, -0.0027931,  0.0025567, -0.0003096,
              0.0002595, -0.0001335, -0.0001550, -0.0047961, -0.0000016,  0.0004484,
             -0.0006034, -0.0004733,  0.0045720, -0.0000002, -0.0510887, ],
            [ 0.0035316, -0.0019137,  0.0235196,  0.0027775, -0.0029549, -0.0005400,
              0.0005011,  0.0004001, -0.0005870, -0.0031347,  0.0000221, -0.0001105,
              0.0000319, -0.0001685,  0.0043916, -0.0000142, -0.0257522, ],
            [ 0.0054686, -0.0043530,  0.0175716,  0.0032195,  0.0029954, -0.0002145,
             -0.0001932, -0.0000687, -0.0001003, -0.0036006,  0.0004825, -0.0000904,
              0.0000654, -0.0003386,  0.0034802,  0.0004376, -0.0247615, ],
        ]).T

        assert np.abs(test_energy - ref_energy) < 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

    @pytest.mark.slow
    def test_rhf_direct_frequency_against_qchem(self):
        mol = pyscf.M(
            atom = """
                F      0.5288      0.1610      0.9359
                C      0.0000      0.0000      0.0000
                H      0.2051      0.8240     -0.6786
                H      0.3345     -0.9314     -0.4496
                H     -1.0685     -0.0537      0.1921
            """,
            basis = "d-aug-cc-pvdz",
            charge = 0,
            verbose = 0,
        )

        assert gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold == 1e-6
        gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold = 1e-3

        mf = mol.RHF().to_gpu()
        test_energy = mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hessian = hobj.kernel()
        from pyscf.hessian import thermo
        freq_info = thermo.harmonic_analysis(mol, hessian)
        test_frequency = freq_info["freq_wavenumber"]

        gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold = 1e-6

        ### Q-Chem input
        ### Why not using density fitting? Because Q-Chem 6.1 has a bug with DF Hessian, I got a frequency of "********"
        # $rem
        # JOBTYPE freq
        # METHOD HF
        # BASIS GEN
        # SYMMETRY      FALSE
        # SYM_IGNORE    TRUE
        # XC_GRID       000099000590
        # NL_GRID       000050000194
        # MAX_SCF_CYCLES 100
        # PURECART 1111
        # SCF_CONVERGENCE 10
        # THRESH        14
        # BASIS_LIN_DEP_THRESH 3
        # $end

        # $basis
        # H     0
        # S   4   1.00
        #      13.0100000              0.0196850
        #       1.9620000              0.1379770
        #       0.4446000              0.4781480
        #       0.1220000              0.5012400
        # S   1   1.00
        #       0.1220000              1.0000000
        # S   1   1.00
        #       0.0297400              1.0000000
        # S   1   1.00
        #       0.00725                1.000000
        # P   1   1.00
        #       0.7270000              1.0000000
        # P   1   1.00
        #       0.1410000              1.0000000
        # P   1   1.00
        #       0.02730                1.000000
        # ****
        # C     0
        # S   9   1.00
        #    6665.0000000              0.0006920
        #    1000.0000000              0.0053290
        #     228.0000000              0.0270770
        #      64.7100000              0.1017180
        #      21.0600000              0.2747400
        #       7.4950000              0.4485640
        #       2.7970000              0.2850740
        #       0.5215000              0.0152040
        #       0.1596000             -0.0031910
        # S   9   1.00
        #    6665.0000000             -0.0001460
        #    1000.0000000             -0.0011540
        #     228.0000000             -0.0057250
        #      64.7100000             -0.0233120
        #      21.0600000             -0.0639550
        #       7.4950000             -0.1499810
        #       2.7970000             -0.1272620
        #       0.5215000              0.5445290
        #       0.1596000              0.5804960
        # S   1   1.00
        #       0.1596000              1.0000000
        # S   1   1.00
        #       0.0469000              1.0000000
        # S   1   1.00
        #       0.0138                 1.000000
        # P   4   1.00
        #       9.4390000              0.0381090
        #       2.0020000              0.2094800
        #       0.5456000              0.5085570
        #       0.1517000              0.4688420
        # P   1   1.00
        #       0.1517000              1.0000000
        # P   1   1.00
        #       0.0404100              1.0000000
        # P   1   1.00
        #       0.0108                 1.000000
        # D   1   1.00
        #       0.5500000              1.0000000
        # D   1   1.00
        #       0.1510000              1.0000000
        # D   1   1.00
        #       0.0415                 1.000000
        # ****
        # F     0
        # S   9   1.00
        #   14710.0000000              0.0007210
        #    2207.0000000              0.0055530
        #     502.8000000              0.0282670
        #     142.6000000              0.1064440
        #      46.4700000              0.2868140
        #      16.7000000              0.4486410
        #       6.3560000              0.2647610
        #       1.3160000              0.0153330
        #       0.3897000             -0.0023320
        # S   9   1.00
        #   14710.0000000             -0.0001650
        #    2207.0000000             -0.0013080
        #     502.8000000             -0.0064950
        #     142.6000000             -0.0266910
        #      46.4700000             -0.0736900
        #      16.7000000             -0.1707760
        #       6.3560000             -0.1123270
        #       1.3160000              0.5628140
        #       0.3897000              0.5687780
        # S   1   1.00
        #       0.3897000              1.0000000
        # S   1   1.00
        #       0.0986300              1.0000000
        # S   1   1.00
        #       0.0250                 1.000000
        # P   4   1.00
        #      22.6700000              0.0448780
        #       4.9770000              0.2357180
        #       1.3470000              0.5085210
        #       0.3471000              0.4581200
        # P   1   1.00
        #       0.3471000              1.0000000
        # P   1   1.00
        #       0.0850200              1.0000000
        # P   1   1.00
        #       0.0208                 1.000000
        # D   1   1.00
        #       1.6400000              1.0000000
        # D   1   1.00
        #       0.4640000              1.0000000
        # D   1   1.00
        #       0.1310                 1.000000
        # ****
        # $end
        ref_energy = -138.9480169968
        ref_frequency = np.array([
            1440.19, 1440.22, 1621.61, 1621.65, 1703.27, 2618.03, 3199.64, 3221.43, 3221.61,
        ])

        assert np.abs(test_energy - ref_energy) < 1e-9
        assert np.max(np.abs(test_frequency - ref_frequency)) < 1 # In cm^-1

    def test_rks_df_against_qchem(self):
        mol = pyscf.M(
            atom = """
                        C 0.000000 0.000000 0.000000
                        C 0.881905 0.881905 0.881905
                        C -0.881905 -0.881905 0.881905
                        C 0.881905 -0.881905 -0.881905
                        C -0.881905 0.881905 -0.881905
                        H -1.524077 0.276170 -1.524077
                        H 1.524077 1.524077 0.276170
                        H 1.524077 -0.276170 -1.524077
                        H 1.524077 0.276170 1.524077
                        H -1.524077 -0.276170 1.524077
                        H 1.524077 -1.524077 -0.276170
                        H -0.276170 1.524077 -1.524077
                        H 0.276170 1.524077 1.524077
                        H 0.276170 -1.524077 -1.524077
                        H -0.276170 -1.524077 1.524077
                        H -1.524077 1.524077 -0.276170
                        H -1.724077 -1.524077 0.276170
            """,
            basis = "d-aug-cc-pvdz",
            charge = 0,
            verbose = 0,
        )

        assert gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold == 1e-6
        gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold = 1e-10

        mf = mol.RKS(xc = "PBE0").density_fit(auxbasis = "aug-cc-pvdz-rifit").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.grids.radi_method = gpu4pyscf.dft.radi.euler_macLaurin
        mf.grids.prune = None
        mf.grids.radii_adjust = None
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold = 1e-6

        ### Q-Chem input
        # $rem
        # JOBTYPE force
        # METHOD PBE0
        # BASIS GEN
        # SYMMETRY      FALSE
        # SYM_IGNORE    TRUE
        # XC_GRID       000099000590
        # BECKE_SHIFT UNSHIFTED
        # MAX_SCF_CYCLES 100
        # PURECART 1111
        # ri_j        True
        # ri_k        True
        # aux_basis     RIMP2-aug-cc-pVDZ
        # SCF_CONVERGENCE 10
        # THRESH        14
        # BASIS_LIN_DEP_THRESH 10
        # $end

        # $basis
        # H     0
        # S   4   1.00
        #      13.0100000              0.0196850
        #       1.9620000              0.1379770
        #       0.4446000              0.4781480
        #       0.1220000              0.5012400
        # S   1   1.00
        #       0.1220000              1.0000000
        # S   1   1.00
        #       0.0297400              1.0000000
        # S   1   1.00
        #       0.00725                1.000000
        # P   1   1.00
        #       0.7270000              1.0000000
        # P   1   1.00
        #       0.1410000              1.0000000
        # P   1   1.00
        #       0.02730                1.000000
        # ****
        # C     0
        # S   9   1.00
        #    6665.0000000              0.0006920
        #    1000.0000000              0.0053290
        #     228.0000000              0.0270770
        #      64.7100000              0.1017180
        #      21.0600000              0.2747400
        #       7.4950000              0.4485640
        #       2.7970000              0.2850740
        #       0.5215000              0.0152040
        #       0.1596000             -0.0031910
        # S   9   1.00
        #    6665.0000000             -0.0001460
        #    1000.0000000             -0.0011540
        #     228.0000000             -0.0057250
        #      64.7100000             -0.0233120
        #      21.0600000             -0.0639550
        #       7.4950000             -0.1499810
        #       2.7970000             -0.1272620
        #       0.5215000              0.5445290
        #       0.1596000              0.5804960
        # S   1   1.00
        #       0.1596000              1.0000000
        # S   1   1.00
        #       0.0469000              1.0000000
        # S   1   1.00
        #       0.0138                 1.000000
        # P   4   1.00
        #       9.4390000              0.0381090
        #       2.0020000              0.2094800
        #       0.5456000              0.5085570
        #       0.1517000              0.4688420
        # P   1   1.00
        #       0.1517000              1.0000000
        # P   1   1.00
        #       0.0404100              1.0000000
        # P   1   1.00
        #       0.0108                 1.000000
        # D   1   1.00
        #       0.5500000              1.0000000
        # D   1   1.00
        #       0.1510000              1.0000000
        # D   1   1.00
        #       0.0415                 1.000000
        # ****
        # $end
        ref_energy = -197.5357459569
        ref_gradient = np.array([
            [ 0.0041903,  0.0026785,  0.0421734,  0.0018135, -0.0021308,  0.0027382,
             -0.0027446, -0.0030728, -0.0030889, -0.0007840, -0.0029324, -0.0027843,
              0.0026993,  0.0028003,  0.0007709,  0.0029511, -0.0452779, ],
            [ 0.0031990,  0.0026618,  0.0198599, -0.0018612,  0.0016578,  0.0028120,
             -0.0025151, -0.0028517,  0.0026840, -0.0061754,  0.0029681, -0.0030445,
             -0.0029231,  0.0028169,  0.0069759, -0.0029604, -0.0233040, ],
            [ 0.0044435,  0.0005075,  0.0239076, -0.0014812, -0.0017465,  0.0027792,
              0.0030461,  0.0028961, -0.0030349, -0.0062857, -0.0027628,  0.0028914,
             -0.0029074,  0.0026484, -0.0001640, -0.0028164, -0.0219207, ],
        ]).T

        assert np.abs(test_energy - ref_energy) < 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

        mf.reset()
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        ref_energy = -197.5356677514
        ref_gradient = np.array([
            [ 0.0041733,  0.0026833,  0.0421856,  0.0018169, -0.0021210,   0.0027379,
             -0.0027464, -0.0030718, -0.0030916, -0.0007836, -0.0029309,  -0.0027890,
              0.0027009,  0.0027981,  0.0007689,  0.0029499, -0.0452806, ],
            [ 0.0031926,  0.0026653,  0.0198554, -0.0018612,  0.0016651,   0.0028058,
             -0.0025222, -0.0028491,  0.0026820, -0.0061629,  0.0029739,  -0.0030467,
             -0.0029293,  0.0028217,  0.0069794, -0.0029656, -0.0233040, ],
            [ 0.0044356,  0.0005120,  0.0239271, -0.0014869, -0.0017475,   0.0027818,
              0.0030418,  0.0029019, -0.0030372, -0.0062914, -0.0027579,   0.0028909,
             -0.0029058,  0.0026472, -0.0001672, -0.0028140, -0.0219303, ],
        ]).T

        assert np.abs(test_energy - ref_energy) < 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6


if __name__ == "__main__":
    print("Tests for System with Diffuse Orbitals (Ill-conditioned Overlap Matrices)")
    unittest.main()
