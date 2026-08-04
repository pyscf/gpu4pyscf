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
import numpy
import pyscf
import pytest
import cupy
from pyscf import gto
from gpu4pyscf import scf, dft
from gpu4pyscf.solvent import pcm

def setUpModule():
    global mol, epsilon, lebedev_order
    mol = gto.Mole()
    mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
    mol.basis = 'sto3g'
    mol.output = '/dev/null'
    mol.build(verbose=0)
    mol.nelectron = mol.nao * 2
    epsilon = 35.9
    lebedev_order = 3

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _energy_with_solvent(mf, method):
    cm = pcm.PCM(mol)
    cm.eps = epsilon
    cm.verbose = 0
    cm.lebedev_order = 29
    cm.method = method
    mf = mf.PCM(cm)
    e_tot = mf.kernel()
    return e_tot

@unittest.skipIf(pcm.libsolvent is None, "solvent extension not compiled")
class KnownValues(unittest.TestCase):
    def test_D_S(self):
        cm = pcm.PCM(mol)
        cm.lebedev_order = 3
        cm.method = 'IEF-PCM'
        cm.build()

        D0, S0 = pcm.get_D_S_slow(cm.surface, with_S=True, with_D=True)
        D1, S1 = pcm.get_D_S(cm.surface, with_S=True, with_D=True)

        assert cupy.linalg.norm(D0 - D1) < 1e-8
        assert cupy.linalg.norm(S0 - S1) < 1e-8

    def test_CPCM(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'C-PCM')
        print(f"Energy error in RHF with C-PCM: {numpy.abs(e_tot - -71.19244927767662)}")
        assert numpy.abs(e_tot - -71.19244927767662) < 1e-5

    def test_COSMO(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'COSMO')
        print(f"Energy error in RHF with COSMO: {numpy.abs(e_tot - -71.16259314943571)}")
        assert numpy.abs(e_tot - -71.16259314943571) < 1e-5

    def test_IEFPCM(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'IEF-PCM')
        print(f"Energy error in RHF with IEF-PCM: {numpy.abs(e_tot - -71.19244457024647)}")
        assert numpy.abs(e_tot - -71.19244457024647) < 1e-5

    def test_SSVPE(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'SS(V)PE')
        print(f"Energy error in RHF with SS(V)PE: {numpy.abs(e_tot - -71.13576912425178)}")
        assert numpy.abs(e_tot - -71.13576912425178) < 1e-5

    def test_uhf(self):
        e_tot = _energy_with_solvent(scf.UHF(mol), 'IEF-PCM')
        print(f"Energy error in UHF with IEF-PCM: {numpy.abs(e_tot - -71.19244457024645)}")
        assert numpy.abs(e_tot - -71.19244457024645) < 1e-5

    def test_rks(self):
        e_tot = _energy_with_solvent(dft.RKS(mol, xc='b3lyp'), 'IEF-PCM')
        print(f"Energy error in RKS with IEF-PCM: {numpy.abs(e_tot - -71.67007402042326)}")
        assert numpy.abs(e_tot - -71.67007402042326) < 1e-5

    def test_uks(self):
        e_tot = _energy_with_solvent(dft.UKS(mol, xc='b3lyp'), 'IEF-PCM')
        print(f"Energy error in UKS with IEF-PCM: {numpy.abs(e_tot - -71.67007402042326)}")
        assert numpy.abs(e_tot - -71.67007402042326) < 1e-5

    def test_dfrks(self):
        e_tot = _energy_with_solvent(dft.RKS(mol, xc='b3lyp').density_fit(), 'IEF-PCM')
        print(f"Energy error in DFRKS with IEF-PCM: {numpy.abs(e_tot - -71.67135250643568)}")
        assert numpy.abs(e_tot - -71.67135250643568) < 1e-5

    def test_dfuks(self):
        e_tot = _energy_with_solvent(dft.UKS(mol, xc='b3lyp').density_fit(), 'IEF-PCM')
        print(f"Energy error in DFUKS with IEF-PCM: {numpy.abs(e_tot - -71.67135250643567)}")
        assert numpy.abs(e_tot - -71.67135250643567) < 1e-5

    def test_to_gpu(self):
        mf = mol.RKS(xc='b3lyp').PCM()
        e_cpu = mf.kernel()
        mf = mf.to_gpu()
        e_gpu = mf.kernel()
        assert abs(e_cpu - e_gpu) < 1e-8
        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert abs(e_cpu - e_gpu) < 1e-8

        mf = mol.RKS(xc='b3lyp').density_fit().PCM()
        e_cpu = mf.kernel()
        mf = mf.to_gpu()
        e_gpu = mf.kernel()
        assert abs(e_cpu - e_gpu) < 1e-8
        chg = mf.analyze()[0][1]
        mf_cpu = mf.to_cpu()
        e_cpu = mf_cpu.kernel()
        assert abs(e_cpu - e_gpu) < 1e-8
        chg_ref = mf_cpu.analyze()[0][1]
        assert abs(chg - chg_ref).max() < 1e-5

    def test_df_and_pcm(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1')
        mf = mol.RHF().to_gpu().PCM()
        with self.assertRaises(RuntimeError):
            mf.density_fit()
        # call approx_hessian after applying PCM is allowed
        mf.newton().density_fit()

    def test_eps_inf(self):
        mol = gto.M(atom='''
C    0.000000    0.000000   -0.542500
O    0.000000    0.000000    0.677500
H    0.000000    0.935307   -1.082500
H    0.000000   -0.935307   -1.082500
''', basis='sto3g')
        mf = mol.RKS(xc='b3lyp').to_gpu().PCM()
        mf.with_solvent.eps = float('inf')
        mf.with_solvent.method = 'C-PCM'
        mf.with_solvent.lebedev_order = 29
        mf.run()
        assert abs(mf.e_tot - -112.95304419865343) < 1e-8

    def test_with_ecp(self):
        mol = gto.M(
            atom = """
                Al      0.000000    0.000000    0.000000
                F       1.650000    0.000000    0.000000
                F      -0.825000    1.428941    0.000000
                F      -0.825000   -1.428941    0.000000
            """,
            basis = "CRENBL",
            ecp = "CRENBL",
            verbose = 0,
        )

        mf = mol.RHF().PCM().to_gpu()
        mf.with_solvent.method = 'C-PCM'
        mf.with_solvent.lebedev_order = 3
        mf.with_solvent.eps = 78
        mf.conv_tol = 1e-10

        test_energy = mf.kernel()
        test_gradient = mf.Gradients().kernel()

        ### Q-Chem reference input
        # $rem
        # JOBTYPE force
        # METHOD HF
        # BASIS CRENBL
        # ECP CRENBL
        # ECP_FIT false
        # ECP_QUAD true
        # solvent_method          pcm
        # SYMMETRY      FALSE
        # SYM_IGNORE    TRUE
        # MAX_SCF_CYCLES 100
        # PURECART 1111
        # SCF_CONVERGENCE 10
        # THRESH        14
        # $end

        # $pcm
        # Theory CPCM
        # Method SWIG
        # Solver INVERSION
        # HeavyPoints 6
        # $end

        # $solvent
        # Dielectric 78
        # $end
        ref_energy = -73.8339216743
        ref_gradient = numpy.array([
            [ -0.006814,     -0.000000,     -0.000000],
            [ -0.006077,      0.000000,      0.000000],
            [  0.006446,     -0.013978,      0.000000],
            [  0.006446,      0.013978,      0.000000],
        ])

        assert abs(test_energy - ref_energy) < 2e-6
        assert numpy.max(numpy.abs(test_gradient - ref_gradient)) < 1e-5

    def test_iswig(self):
        mol = gto.M(
            atom = """
                O      0.000000    0.000000    0.000000
                H      0.957200    0.000000    0.000000
                H     -0.239987    0.926627    0.000000
                Ne 0 0 3.44686045
            """, # The Ne position overlaps with a grid on O
            basis = "6-31g",
            verbose = 0,
        )

        mf = mol.RHF().PCM().to_gpu()
        mf.with_solvent.method = 'C-PCM'
        mf.with_solvent.lebedev_order = 7
        mf.with_solvent.eps = 78
        mf.with_solvent.surface_discretization_method = "iswig"
        mf.conv_tol = 1e-10

        test_energy = mf.kernel()
        assert mf.converged

        ### Q-Chem reference input
        # $rem
        # JOBTYPE sp
        # METHOD HF
        # BASIS 6-31g
        # solvent_method          pcm
        # SYMMETRY      FALSE
        # SYM_IGNORE    TRUE
        # MAX_SCF_CYCLES 100
        # PURECART 1111
        # SCF_CONVERGENCE 10
        # THRESH        14
        # $end

        # $pcm
        # Theory CPCM
        # Method iSWIG
        # Solver INVERSION
        # HeavyPoints 26
        # HPoints 26
        # $end

        # $solvent
        # Dielectric 78
        # $end

        ### SWIG instead of iSWIG: -204.4719960138
        ref_energy = -204.4722728963

        assert abs(test_energy - ref_energy) < 1e-8

    def test_ghost_atom(self):
        mol = pyscf.M(
            atom = """
                ghost:O      -0.0456482954     0.3276635601     0.1150712045
                ghost:H       0.8839023862     0.1379773801     0.1537324880
                ghost:H      -0.4544197433    -0.3524420751     0.6270142868
                O       2.8650684263    -0.1847028876     0.0859519520
                H       3.1198758203    -0.2809098055    -0.8201356074
                H       3.3367304058     0.5708588279     0.4048656762
                ghost:Ne     100 0 0
            """,
            basis = "def2-SVP",
            verbose = 0,
        )
        mf = scf.RHF(mol)
        mf = mf.PCM()
        mf.with_solvent.method = "IEF-PCM"
        mf.with_solvent.eps = 80.0
        mf.conv_tol = 1e-12

        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        ### Q-Chem reference input
        # $molecule
        # 0 1
        #         @O      -0.0456482954     0.3276635601     0.1150712045
        #         @H       0.8839023862     0.1379773801     0.1537324880
        #         @H      -0.4544197433    -0.3524420751     0.6270142868
        #         O       2.8650684263    -0.1847028876     0.0859519520
        #         H       3.1198758203    -0.2809098055    -0.8201356074
        #         H       3.3367304058     0.5708588279     0.4048656762
        # $end

        # $rem
        # JOBTYPE freq
        # METHOD HF
        # BASIS def2-svp
        # SCF_CONVERGENCE 12
        # BASIS_LIN_DEP_THRESH 6
        # SOLVENT_METHOD PCM
        # SYMMETRY      FALSE
        # SYM_IGNORE    TRUE
        # MAX_SCF_CYCLES 100
        # PURECART 1111
        # THRESH        14
        # $end
        ref_energy = -75.9747732340

        ref_gradient = numpy.array([
            [-0.000112,      -0.000008,      -0.000007],
            [-0.000936,      -0.000017,       0.000188],
            [-0.000051,      -0.000021,       0.000016],
            [ 0.003634,       0.003660,      -0.003067],
            [-0.001186,      -0.001495,       0.001833],
            [-0.001350,      -0.002120,       0.001037],
            [0, 0, 0],
        ])

        assert abs(test_energy - ref_energy) < 1e-9
        assert numpy.max(numpy.abs(test_gradient - ref_gradient)) < 2e-6

if __name__ == "__main__":
    print("Full Tests for PCMs")
    unittest.main()
