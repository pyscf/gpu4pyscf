# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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

def _make_mf(method='C-PCM', restricted=True):
    if restricted:
        mf = dft.rks.RKS(mol, xc=xc).density_fit().PCM()
    else:
        mf = dft.uks.UKS(mol, xc=xc).density_fit().PCM()
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
    def test_hess_cpcm(self):
        print('testing C-PCM Hessian with DF-RKS')
        mf = _make_mf(method='C-PCM')
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_hess_iefpcm(self):
        print("testing IEF-PCM hessian with DF-RKS")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_uhf_hess_iefpcm(self):
        print("testing IEF-PCM hessian with DF-UKS")
        mf = _make_mf(method='IEF-PCM', restricted=False)
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_to_gpu(self):
        import pyscf
        # Not implemented yet
        '''
        mf = pyscf.dft.RKS(mol, xc='b3lyp').SMD()
        mf.kernel()
        hessobj = mf.Hessian()
        hess_cpu = hessobj.kernel()
        hessobj = hessobj.to_gpu()
        hess_gpu = hessobj.kernel()
        assert np.linalg.norm(hess_cpu - hess_gpu) < 1e-8
        '''
        mol = gto.Mole()
        mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
        mol.basis = 'sto-3g'
        mol.output = '/dev/null'
        mol.build(verbose=0)
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
        # Not implemented yet
        '''
        mf = dft.RKS(mol, xc='b3lyp').SMD()
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert abs(e_cpu - e_gpu) < 1e-8
        '''
        mol = gto.Mole()
        mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
        mol.basis = 'sto-3g'
        mol.output = '/dev/null'
        mol.build(verbose=0)
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
