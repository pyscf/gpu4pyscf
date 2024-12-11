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
import pytest
from pyscf import gto
from gpu4pyscf.solvent import pcm
from gpu4pyscf import scf, dft
from packaging import version

pyscf_25 = version.parse(pyscf.__version__) <= version.parse('2.5.0')

def setUpModule():
    global mol, epsilon, lebedev_order, eps, xc, tol
    mol = gto.Mole()
    mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
    mol.basis = 'def2-tzvpp'
    mol.output = '/dev/null'
    mol.build(verbose=0)
    epsilon = 78.3553
    lebedev_order = 29
    eps = 1e-3
    xc = 'B3LYP'
    tol = 1e-3

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _make_mf(method='C-PCM', restricted=True, density_fit=True):
    if restricted:
        mf = dft.rks.RKS(mol, xc=xc)
    else:
        mf = dft.uks.UKS(mol, xc=xc)
    
    if density_fit:
        mf = mf.density_fit()
    mf = mf.PCM()
    mf.with_solvent.method = method
    mf.with_solvent.eps = epsilon
    mf.with_solvent.lebedev_order = lebedev_order
    mf.conv_tol = 1e-12
    mf.conv_tol_cpscf = 1e-7
    mf.grids.atom_grid = (99,590)
    mf.verbose = 0
    mf.kernel()
    return mf

def _check_hessian(mf, h, ix=0, iy=0):
    pmol = mf.mol.copy()
    pmol.build()

    g = mf.nuc_grad_method()
    g.auxbasis_response = True
    g.kernel()
    g_scanner = g.as_scanner()

    coords = pmol.atom_coords()
    v = np.zeros_like(coords)
    v[ix,iy] = eps
    pmol.set_geom_(coords + v, unit='Bohr')
    pmol.build()
    _, g0 = g_scanner(pmol)

    pmol.set_geom_(coords - v, unit='Bohr')
    pmol.build()
    _, g1 = g_scanner(pmol)

    h_fd = (g0 - g1)/2.0/eps

    print(f'Norm of H({ix},{iy}) diff, {np.linalg.norm(h[ix,:,iy,:] - h_fd)}')
    assert(np.linalg.norm(h[ix,:,iy,:] - h_fd) < tol)

@unittest.skipIf(pcm.libsolvent is None, "solvent extension not compiled")
class KnownValues(unittest.TestCase):
    def test_df_hess_cpcm(self):
        print('testing C-PCM Hessian with DF-RKS')
        mf = _make_mf(method='C-PCM')
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_hess_cpcm(self):
        print('testing C-PCM Hessian with RKS')
        mf = _make_mf(method='C-PCM', density_fit=False)
        hobj = mf.Hessian()
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_df_hess_iefpcm(self):
        print("testing IEF-PCM hessian with DF-RKS")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_hess_iefpcm(self):
        print("testing IEF-PCM hessian with RKS")
        mf = _make_mf(method='IEF-PCM', density_fit=False)
        hobj = mf.Hessian()
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_df_uks_hess_iefpcm(self):
        print("testing IEF-PCM hessian with DF-UKS")
        mf = _make_mf(method='IEF-PCM', restricted=False, density_fit=True)
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_uks_hess_iefpcm(self):
        print("testing IEF-PCM hessian with UHF")
        mf = _make_mf(method='IEF-PCM', restricted=False, density_fit=False)
        hobj = mf.Hessian()
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_to_gpu(self):
        import pyscf
        mol = gto.Mole()
        mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
        mol.basis = 'sto-3g'
        mol.output = '/dev/null'
        mol.build(verbose=0)
        mf = pyscf.dft.RKS(mol, xc='b3lyp').PCM()
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-7
        mf.grids.atom_grid = (50,194)
        mf.kernel()
        hessobj = mf.Hessian()
        hess_cpu = hessobj.kernel()
        hessobj = hessobj.to_gpu()
        hess_gpu = hessobj.kernel()
        assert np.linalg.norm(hess_cpu - hess_gpu) < 1e-5

        mf = pyscf.dft.RKS(mol, xc='b3lyp').density_fit().PCM()
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-7
        mf.grids.atom_grid = (50,194)
        mf.kernel()
        hessobj = mf.Hessian()
        hess_cpu = hessobj.kernel()
        hessobj = hessobj.to_gpu()
        hess_gpu = hessobj.kernel()
        assert np.linalg.norm(hess_cpu - hess_gpu) < 1e-5

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_to_cpu(self):
        mol = gto.Mole()
        mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
        mol.basis = 'sto-3g'
        mol.output = '/dev/null'
        mol.build(verbose=0)
        
        mf = dft.RKS(mol, xc='b3lyp').PCM()
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-7
        mf.grids.atom_grid = (50,194)
        mf.kernel()
        hessobj = mf.Hessian()
        hess_gpu = hessobj.kernel()
        hessobj = hessobj.to_cpu()
        hess_cpu = hessobj.kernel()
        assert np.linalg.norm(hess_cpu - hess_gpu) < 1e-5

        mf = dft.RKS(mol, xc='b3lyp').density_fit().PCM()
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-7
        mf.grids.atom_grid = (50,194)
        mf.kernel()
        hessobj = mf.Hessian()
        hess_gpu = hessobj.kernel()
        hessobj = hessobj.to_cpu()
        hess_cpu = hessobj.kernel()
        assert np.linalg.norm(hess_cpu - hess_gpu) < 1e-5
if __name__ == "__main__":
    print("Full Tests for Hessian of PCMs")
    unittest.main()
