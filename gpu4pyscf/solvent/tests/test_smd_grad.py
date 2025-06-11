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
from pyscf import gto
from gpu4pyscf import scf, dft
from gpu4pyscf.solvent.grad import smd as smd_grad
from gpu4pyscf.solvent import smd
from packaging import version

pyscf_25 = version.parse(pyscf.__version__) <= version.parse('2.5.0')

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
    mol.basis = 'sto3g'
    mol.output = '/dev/null'
    mol.build(verbose=0)

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _check_grad(atom, solvent='water'):
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = 'sto3g'
    mol.output = '/dev/null'
    mol.build(verbose=0)
    natm = mol.natm
    fd_cds = numpy.zeros([natm,3])
    eps = 1e-4
    for ia in range(mol.natm):
        for j in range(3):
            coords = mol.atom_coords(unit='B')
            coords[ia,j] += eps
            mol.set_geom_(coords, unit='B')
            mol.build()

            smdobj = smd.SMD(mol)
            smdobj.solvent = solvent
            e0_cds = smdobj.get_cds()

            coords[ia,j] -= 2.0*eps
            mol.set_geom_(coords, unit='B')
            mol.build()

            smdobj = smd.SMD(mol)
            smdobj.solvent = solvent
            e1_cds = smdobj.get_cds()

            coords[ia,j] += eps
            mol.set_geom_(coords, unit='B')
            fd_cds[ia,j] = (e0_cds - e1_cds) / (2.0 * eps)

    smdobj = smd.SMD(mol)
    smdobj.solvent = solvent
    grad_cds = smd.get_cds_legacy(smdobj)[1]
    mol.stdout.close()
    assert numpy.linalg.norm(fd_cds - grad_cds) < 1e-8

@unittest.skipIf(smd.libsolvent is None, "solvent extension not compiled")
class KnownValues(unittest.TestCase):
    def test_grad_water(self):
        mf = dft.rks.RKS(mol, xc='b3lyp').SMD()
        mf.grids.atom_grid = (99,590)
        mf.with_solvent.solvent = 'water'
        mf.with_solvent.sasa_ng = 590
        mf.kernel()
        g = mf.nuc_grad_method().kernel()
        g_ref = numpy.array(
            [[0.000000,       0.000000,      -0.101523],
            [0.043933,      -0.000000,       0.050761],
            [-0.043933,      -0.000000,       0.050761]]
            )
        assert numpy.linalg.norm(g - g_ref) < 1e-4

        mf = dft.uks.UKS(mol, xc='b3lyp').SMD()
        mf.grids.atom_grid = (99,590)
        mf.with_solvent.solvent = 'water'
        mf.with_solvent.sasa_ng = 590
        mf.kernel()
        g = mf.nuc_grad_method().kernel()
        assert numpy.linalg.norm(g - g_ref) < 1e-4

    def test_grad_solvent(self):
        mf = dft.rks.RKS(mol, xc='b3lyp').SMD()
        mf.grids.atom_grid = (99,590)
        mf.with_solvent.solvent = 'toluene'
        mf.with_solvent.sasa_ng = 590
        mf.kernel()
        g = mf.nuc_grad_method().kernel()
        g_ref = numpy.array(
            [[-0.000000,       0.000000,      -0.106849],
            [0.047191,      -0.000000,       0.053424],
            [-0.047191,       0.000000,       0.053424]]
            )
        assert numpy.linalg.norm(g - g_ref) < 1e-4

        mf = dft.uks.UKS(mol, xc='b3lyp').SMD()
        mf.grids.atom_grid = (99,590)
        mf.with_solvent.solvent = 'toluene'
        mf.with_solvent.sasa_ng = 590
        mf.kernel()
        g = mf.nuc_grad_method().kernel()
        assert numpy.linalg.norm(g - g_ref) < 1e-4

    def test_CN(self):
        atom = '''
C       0.000000     0.000000     0.000000
N       0.000000     0.000000     1.500000
H       0.000000     1.000000    -0.500000
H       0.866025    -0.500000    -0.500000
H      -0.866025    -0.500000    -0.500000
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_CC(self):
        atom = '''
C 0.000 0.000 0.000
C 1.339 0.000 0.000
H -0.507 0.927 0.000
H -0.507 -0.927 0.000
H 1.846 0.927 0.000
H 1.846 -0.927 0.000
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_OO(self):
        atom = '''
O 0.000 0.000 0.000
O 1.207 0.000 0.000
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_ON(self):
        atom = '''
N 0.000 0.000 0.000
O 1.159 0.000 0.000
H -0.360 0.000 0.000
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_OP(self):
        atom = '''
P 0.000 0.000 0.000
O 1.480 0.000 0.000
H -0.932 0.932 0.000
H -0.932 -0.932 0.000
H 0.368 0.000 0.933
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_OC(self):
        atom = '''
C 0.000 0.000 0.000
O 1.208 0.000 0.000
H -0.603 0.928 0.000
H -0.603 -0.928 0.000
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_F(self):
        atom = '''
C 0.000 0.000 0.000
F 1.380 0.000 0.000
H -0.520 0.920 -0.400
H -0.520 -0.920 -0.400
H -0.520 0.000 1.000
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_Si(self):
        atom = '''
Si 0.000 0.000 0.000
H 0.875 0.875 0.875
H -0.875 -0.875 0.875
H 0.875 -0.875 -0.875
H -0.875 0.875 -0.875
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_S(self):
        atom = '''
S 0.000 0.000 0.000
H 0.962 0.280 0.000
H -0.962 0.280 0.000
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_Cl(self):
        atom = '''
C 0.000 0.000 0.000
Cl 1.784 0.000 0.000
H -0.595 0.952 0.000
H -0.595 -0.476 0.824
H -0.595 -0.476 -0.824
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    def test_Br(self):
        atom = '''
C 0.000 0.000 0.000
Br 1.939 0.000 0.000
H -0.646 0.929 0.000
H -0.646 -0.464 0.804
H -0.646 -0.464 -0.804
    '''
        _check_grad(atom, solvent='water')
        _check_grad(atom, solvent='toluene')

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_to_gpu(self):
        import pyscf
        mf = pyscf.dft.RKS(mol, xc='b3lyp').SMD()
        mf.conv_tol = 1e-12
        mf.kernel()
        gradobj = mf.nuc_grad_method()
        g_cpu = gradobj.kernel()
        gradobj = gradobj.to_gpu()
        g_gpu = gradobj.kernel()
        assert numpy.linalg.norm(g_cpu - g_gpu) < 1e-5

        mf = pyscf.dft.RKS(mol, xc='b3lyp').density_fit().SMD()
        mf.conv_tol = 1e-12
        mf.kernel()
        gradobj = mf.nuc_grad_method()
        g_cpu = gradobj.kernel()
        gradobj = gradobj.to_gpu()
        g_gpu = gradobj.kernel()
        assert numpy.linalg.norm(g_cpu - g_gpu) < 1e-5

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_to_cpu(self):
        mf = dft.RKS(mol, xc='b3lyp').SMD()
        mf.conv_tol = 1e-12
        mf.kernel()
        gradobj = mf.nuc_grad_method()
        g_gpu = gradobj.kernel()
        gradobj = gradobj.to_cpu()
        g_cpu = gradobj.kernel()
        assert numpy.linalg.norm(g_cpu - g_gpu) < 1e-5

        mf = dft.RKS(mol, xc='b3lyp').density_fit().SMD()
        mf.conv_tol = 1e-12
        mf.kernel()
        gradobj = mf.nuc_grad_method()
        g_gpu = gradobj.kernel()
        gradobj = gradobj.to_cpu()
        g_cpu = gradobj.kernel()
        assert numpy.linalg.norm(g_cpu - g_gpu) < 1e-5

if __name__ == "__main__":
    print("Full Tests for Gradient of SMD")
    unittest.main()
