# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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
#
# Author: Chenghan Li <lch004218@gmail.com>

import unittest
import numpy as np
import pyscf
from gpu4pyscf.dft import rks
from gpu4pyscf.qmmm.pbc import itrf
from gpu4pyscf.qmmm.pbc.tools import estimate_error

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='3-21g'
auxbasis='cc-pvdz-jkfit'
scf_tol = 1e-10
max_scf_cycles = 50
screen_tol = 1e-14
grids_level = 3

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

def compute_octupole_error(xc):
    mf = rks.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf = itrf.add_mm_charges(
        mf, [[1,2,-1],[3,4,5]], np.eye(3)*15, [-5,5], [0.8,1.2], rcut_ewald=8, rcut_hcore=6)
    mf.conv_tol = scf_tol
    mf.max_cycle = max_scf_cycles
    mf.direct_scf_tol = screen_tol
    mf.grids.level = grids_level
    e_dft = mf.kernel()
    e_oct = estimate_error(mf.mol, mf.mm_mol.atom_coords(), mf.mm_mol.a, mf.mm_mol.atom_charges(),
                    20, mf.make_rdm1(), unit='Bohr', precision=1e-6)
    return e_dft, e_oct

def run_dft(xc):
    mf = rks.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf = itrf.add_mm_charges(
        mf, [[1,2,-1],[3,4,5]], np.eye(3)*15, [-5,5], [0.8,1.2], rcut_ewald=8, rcut_hcore=6)
    mf.conv_tol = scf_tol
    mf.max_cycle = max_scf_cycles
    mf.direct_scf_tol = screen_tol
    mf.grids.level = grids_level
    e_dft = mf.kernel()

    g = mf.nuc_grad_method()
    g.max_memory = 32000
    g.auxbasis_response = True
    g_qm = g.kernel()

    g_mm = g.grad_nuc_mm() + g.grad_hcore_mm(mf.make_rdm1()) + g.de_ewald_mm
    return e_dft, g_qm, g_mm

class KnownValues(unittest.TestCase):
    def test_estimate_error(self):
        print('-------- Octupole Error -------')
        e_tot, e_octupole = compute_octupole_error('PBE')
        assert np.allclose(e_octupole, 2.4943047809143978e-05)

    def test_rks_pbe0(self):
        print('-------- RKS PBE0 -------------')
        e_tot, g_qm, g_mm = run_dft('PBE0')
        assert np.allclose(e_tot, -76.00178807)
        assert np.allclose(g_qm, np.array([[ 0.03002572,  0.13947702, -0.09234864],
                                           [-0.00462601, -0.04602809,  0.02750759],
                                           [-0.01821532, -0.18473378,  0.04189843]]))
        assert np.allclose(g_mm, np.array([[-0.00914559,  0.08992359,  0.02114633],
                                           [ 0.00196155,  0.00136132,  0.00179565]]))

if __name__ == "__main__":
    print("Full Tests for QMMM PBC")
    unittest.main()
