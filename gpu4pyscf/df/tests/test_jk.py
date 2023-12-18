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
import dftd3.pyscf as disp
from pyscf import lib, df
from pyscf.geomopt.geometric_solver import optimize
from gpu4pyscf import scf
from gpu4pyscf.dft import rks
from gpu4pyscf.df import int3c2e, df_jk

lib.num_threads(8)

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='ccpvdz'

def setUpModule():
    global mol, auxmol
    mol = pyscf.M(atom=atom, basis=bas, max_memory=32000)
    mol.output = '/dev/null'
    mol.build()
    mol.verbose = 1
    auxmol = df.addons.make_auxmol(mol, auxbasis='sto3g')

def tearDownModule():
    global mol, auxmol
    mol.stdout.close()
    del mol, auxmol

class KnownValues(unittest.TestCase):

    def test_vj_incore(self):
        int3c_gpu = int3c2e.get_int3c2e(mol, auxmol, aosym=True, direct_scf_tol=1e-14)
        intopt = int3c2e.VHFOpt(mol, auxmol, 'int2e')
        intopt.build(1e-14, diag_block_with_triu=False, aosym=True)
        cupy.random.seed(1)
        nao = len(intopt.sph_ao_idx)
        dm = cupy.random.rand(nao, nao)
        dm = dm + dm.T

        # pass 1
        rhoj_outcore = cupy.einsum('ijL,ij->L', int3c_gpu, dm)
        rhoj_incore = 2.0*int3c2e.get_j_int3c2e_pass1(intopt, dm)
        assert cupy.linalg.norm(rhoj_outcore - rhoj_incore) < 1e-9

        # pass 2
        vj_outcore = cupy.einsum('ijL,L->ij', int3c_gpu, rhoj_outcore)
        vj_incore = int3c2e.get_j_int3c2e_pass2(intopt, rhoj_incore)
        assert cupy.linalg.norm(vj_outcore - vj_incore) < 1e-9

    def test_j_outcore(self):
        cupy.random.seed(1)
        nao = mol.nao
        dm = cupy.random.rand(nao, nao)
        dm = dm + dm.T
        mf = scf.RHF(mol).density_fit(auxbasis='sto3g')
        mf.kernel()
        vj0, _ = mf.get_jk(dm=dm, with_j=True, with_k=False)
        vj = df_jk.get_j(mf.with_df, dm)
        assert cupy.linalg.norm(vj - vj0) < 1e-9


if __name__ == "__main__":
    print("Full Tests for DF JK")
    unittest.main()