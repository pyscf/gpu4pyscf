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
from gpu4pyscf import scf, dft, lib
from gpu4pyscf.solvent.hessian import smd as smd_hess
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

def _check_hess(atom, solvent='water'):
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = 'sto-3g'
    mol.output = '/dev/null'
    mol.build(verbose=0)
    smdobj = smd.SMD(mol)
    smdobj.solvent = solvent
    hess_cds = smd_hess.get_cds(smdobj)

    eps = 1e-4
    coords = mol.atom_coords()
    v = numpy.zeros_like(coords)
    v[0,0] = eps
    mol.set_geom_(coords + v, unit='Bohr')
    mol.build()
    smdobj = smd.SMD(mol)
    smdobj.solvent = solvent
    g0 = smd_grad.get_cds(smdobj)

    mol.set_geom_(coords - v, unit='Bohr')
    mol.build()
    smdobj = smd.SMD(mol)
    smdobj.solvent = solvent
    g1 = smd_grad.get_cds(smdobj)
    h_fd = (g0 - g1)/2.0/eps
    mol.stdout.close()
    assert(numpy.linalg.norm(hess_cds[0,:,0,:] - h_fd) < 1e-3)

@unittest.skipIf(smd.libsolvent is None, "solvent extension not compiled")
class KnownValues(unittest.TestCase):
    def test_h2o(self):
        h2o = gto.Mole()
        h2o.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
        h2o.basis = 'sto3g'
        h2o.output = '/dev/null'
        h2o.build()

        mf = dft.RKS(h2o, xc='b3lyp').density_fit()
        mf.grids.atom_grid = (99,590)
        mf = mf.SMD()
        mf.with_solvent.solvent = 'toluene'
        mf.with_solvent.sasa_ng = 590
        mf.with_solvent.lebedev_order = 29

        mf.kernel()
        h = mf.Hessian().kernel()

        assert abs(h[0,0,0,0] - 0.9199776)  < 1e-3
        assert abs(h[0,0,1,1] - -0.0963789) < 1e-3
        assert abs(h[0,0,2,2] - 0.5852264)  < 1e-3
        assert abs(h[1,0,0,0] - -0.4599888) < 1e-3
        h2o.stdout.close()

    def test_CN(self):
        atom = '''
C  0.0  0.0  0.0
H  1.09  0.0  0.0
H  -0.545  0.944  0.0
H  -0.545  -0.944  0.0
N  0.0  0.0  1.16
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_CC(self):
        atom = '''
C 0.000 0.000 0.000
C 1.339 0.000 0.000
H -0.507 0.927 0.000
H -0.507 -0.927 0.000
H 1.846 0.927 0.000
H 1.846 -0.927 0.000
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_OO(self):
        atom = '''
O 0.000 0.000 0.000
O 1.207 0.000 0.000
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_ON(self):
        atom = '''
N 0.000 0.000 0.000
O 1.159 0.000 0.000
H -0.360 0.000 0.000
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_OP(self):
        atom = '''
P 0.000 0.000 0.000
O 1.480 0.000 0.000
H -0.932 0.932 0.000
H -0.932 -0.932 0.000
H 0.368 0.000 0.933
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_OC(self):
        atom = '''
C 0.000 0.000 0.000
O 1.208 0.000 0.000
H -0.603 0.928 0.000
H -0.603 -0.928 0.000
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_F(self):
        atom = '''
C 0.000 0.000 0.000
F 1.380 0.000 0.000
H -0.520 0.920 -0.400
H -0.520 -0.920 -0.400
H -0.520 0.000 1.000
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_Si(self):
        atom = '''
Si 0.000 0.000 0.000
H 0.875 0.875 0.875
H -0.875 -0.875 0.875
H 0.875 -0.875 -0.875
H -0.875 0.875 -0.875
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_S(self):
        atom = '''
S 0.000 0.000 0.000
H 0.962 0.280 0.000
H -0.962 0.280 0.000
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_Cl(self):
        atom = '''
C 0.000 0.000 0.000
Cl 1.784 0.000 0.000
H -0.595 0.952 0.000
H -0.595 -0.476 0.824
H -0.595 -0.476 -0.824
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    def test_Br(self):
        atom = '''
C 0.000 0.000 0.000
Br 1.939 0.000 0.000
H -0.646 0.929 0.000
H -0.646 -0.464 0.804
H -0.646 -0.464 -0.804
    '''
        _check_hess(atom, solvent='water')
        _check_hess(atom, solvent='toluene')

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_to_gpu(self):
        import pyscf
        mf = pyscf.dft.RKS(mol, xc='b3lyp').SMD()
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-7
        mf.kernel()
        hessobj = mf.Hessian()
        hess_cpu = hessobj.kernel()
        hessobj = hessobj.to_gpu()
        hess_gpu = hessobj.kernel()
        assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5
        
        mf = pyscf.dft.RKS(mol, xc='b3lyp').density_fit().SMD()
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-7
        mf.kernel()
        hessobj = mf.Hessian()
        hess_cpu = hessobj.kernel()
        hessobj = hessobj.to_gpu()
        hess_gpu = hessobj.kernel()
        assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_to_cpu(self):
        mf = dft.RKS(mol, xc='b3lyp').SMD()
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-7
        mf.kernel()
        hessobj = mf.Hessian()
        hess_gpu = hessobj.kernel()
        hessobj = hessobj.to_cpu()
        hess_cpu = hessobj.kernel()
        assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5
        
        mf = dft.RKS(mol, xc='b3lyp').density_fit().SMD()
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-7
        mf.kernel()
        hessobj = mf.Hessian()
        hess_gpu = hessobj.kernel()
        hessobj = hessobj.to_cpu()
        hess_cpu = hessobj.kernel()
        assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5

if __name__ == "__main__":
    print("Full Tests for Hessian of SMD")
    unittest.main()
