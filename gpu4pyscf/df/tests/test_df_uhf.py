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
import cupy
import pyscf
from pyscf import lib
from pyscf import scf as cpu_scf
from pyscf.df import df_jk as cpu_df_jk
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf.df import df_jk as gpu_df_jk

lib.num_threads(8)

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='def2tzvpp'

def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas, charge=1, spin=1, max_memory=32000)
    mol.output = '/dev/null'
    mol.build()
    mol.verbose = 1

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _check_grad(tol=1e-5):
    mf = gpu_scf.uhf.UHF(mol)#.density_fit()
    mf.conv_tol = 1e-10
    mf.verbose = 1
    mf.kernel()

    g = mf.nuc_grad_method()
    g.auxbasis_response = True

    g_scanner = g.as_scanner()
    g_analy = g_scanner(mol)[1]
    print('analytical gradient:')
    print(g_analy)

    f_scanner = mf.as_scanner()
    coords = mol.atom_coords()
    grad_fd = np.zeros_like(coords)
    eps = 1.0/1024
    for i in range(len(coords)):
        for j in range(3):
            coords = mol.atom_coords()
            coords[i,j] += eps
            mol.set_geom_(coords, unit='Bohr')
            mol.build()
            e0 = f_scanner(mol)

            coords[i,j] -= 2.0 * eps
            mol.set_geom_(coords, unit='Bohr')
            mol.build()
            e1 = f_scanner(mol)

            coords[i,j] += eps
            mol.set_geom_(coords, unit='Bohr')
            grad_fd[i,j] = (e0-e1)/2.0/eps
    grad_fd = np.array(grad_fd).reshape(-1,3)
    print('finite difference gradient:')
    print(grad_fd)
    print('difference between analytical and finite difference gradient:', cupy.linalg.norm(g_analy - grad_fd))
    assert(cupy.linalg.norm(g_analy - grad_fd) < tol)

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by PySCF
    '''
    def test_uhf(self):
        print('------- UHF -----------------')
        mf = gpu_scf.UHF(mol).density_fit(auxbasis='def2-tzvpp-jkfit')
        e_tot = mf.kernel()
        e_pyscf = -75.6599919479438
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.allclose(e_tot, e_pyscf)

    def test_grad_uhf(self):
        _check_grad(tol=1e-5)

    def test_to_cpu(self):
        mf = gpu_scf.UHF(mol).density_fit()
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert isinstance(mf, cpu_df_jk._DFHF)
        assert np.allclose(e_gpu, e_cpu)

    def test_to_gpu(self):
        mf = cpu_scf.UHF(mol).density_fit()
        e_cpu = mf.kernel()
        mf = mf.to_gpu()
        e_gpu = mf.kernel()
        assert isinstance(mf, gpu_df_jk._DFHF)
        assert np.allclose(e_gpu, e_cpu)

if __name__ == "__main__":
    print("Full Tests for unrestricted Hartree-Fock")
    unittest.main()
