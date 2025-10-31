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
import cupy
from pyscf import lib, scf
from pyscf.dft.numint import NumInt as pyscf_numint
from gpu4pyscf.dft import Grids
from gpu4pyscf.dft import numint
from gpu4pyscf.dft.numint import NumInt
from gpu4pyscf import dft
from gpu4pyscf.dft import gen_grid

def setUpModule():
    global mol, grids_cpu, grids_gpu, dm, dm0, dm1, mo_occ, mo_coeff
    mol = pyscf.M(
        atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
        basis = 'ccpvdz',
        charge = 1,
        spin = 1,  # = 2S = spin_up - spin_down
        output = '/dev/null')

    np.random.seed(2)
    mf = scf.UHF(mol)
    mf.kernel()
    dm1 = mf.make_rdm1().copy()
    dm = dm1
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    dm0 = (mo_coeff[0]*mo_occ[0]).dot(mo_coeff[0].T)

    grids_gpu = Grids(mol)
    grids_gpu.level = 1
    grids_gpu.build()

    grids_cpu = grids_gpu.to_cpu()
    grids_cpu.weights = cupy.asnumpy(grids_gpu.weights)
    grids_cpu.coords = cupy.asnumpy(grids_gpu.coords)

def tearDownModule():
    global mol, grids_cpu, grids_gpu
    mol.stdout.close()
    del mol, grids_cpu, grids_gpu

LDA = 'LDA_C_VWN'
GGA_PBE = 'GGA_C_PBE'
MGGA_M06 = 'MGGA_C_M06'

class KnownValues(unittest.TestCase):

    def test_eval_rho(self):
        np.random.seed(1)
        dm = np.random.random(dm0.shape)
        ni_gpu = NumInt()
        ni_cpu = pyscf_numint()
        for xctype in ('LDA', 'GGA', 'MGGA'):
            deriv = 1
            if xctype == 'LDA':
                deriv = 0
            ao_gpu = ni_gpu.eval_ao(mol, grids_gpu.coords, deriv=deriv, transpose=False)
            ao_cpu = ni_cpu.eval_ao(mol, grids_cpu.coords, deriv=deriv)
            
            rho = ni_gpu.eval_rho(mol, ao_gpu, dm, xctype=xctype, hermi=0, with_lapl=False)
            ref = ni_cpu.eval_rho(mol, ao_cpu, dm, xctype=xctype, hermi=0, with_lapl=False)
            self.assertAlmostEqual(abs(rho[...,:grids_cpu.size].get() - ref).max(), 0, 10)

            rho = ni_gpu.eval_rho(mol, ao_gpu, dm0, xctype=xctype, hermi=1, with_lapl=False)
            ref = ni_cpu.eval_rho(mol, ao_cpu, dm0, xctype=xctype, hermi=1, with_lapl=False)
            self.assertAlmostEqual(abs(rho[...,:grids_cpu.size].get() - ref).max(), 0, 10)

    def test_sparse_index(self):
        mol = pyscf.M(atom='''
O   0.     0.    0.
H   5.5    0.3   2.8
H   1.7   -2.0   0.4''',
                      basis='def2-tzvpp')
        with lib.temporary_env(numint, MIN_BLK_SIZE=128**2):
            grids = gen_grid.Grids(mol).set(atom_grid=(200, 1454)).build()
            ni = NumInt()
            ni.build(mol, grids.coords)
            opt = ni.gdftopt
            opt.l_ctr_offsets
            ao_loc = opt._sorted_mol.ao_loc
            ngrids = grids.size
            dat = grids.get_non0ao_idx(opt)
            assert lib.fp(cupy.hstack([x[1] for x in dat]).get()) == 103.60117204957997
            assert lib.fp(cupy.hstack([x[2] for x in dat]).get()) == 5.616197331343498
            assert lib.fp(cupy.hstack([x[3] for x in dat]).get()) == -22.394314323727
            assert lib.fp(cupy.hstack([x[4] for x in dat]).get()) == 351.2385939586691
            assert [i.size for x in dat for i in x[1:]] == [
                46, 18, 9, 19, 50, 20, 9, 21, 28, 12, 9, 13, 49, 19, 9, 20, 45, 17,
                9, 18, 45, 17, 9, 18, 45, 17, 9, 18, 45, 17, 9, 18, 50, 20, 9, 21,
                55, 21, 9, 22, 53, 19, 9, 20, 48, 18, 9, 19, 53, 19, 9, 20, 48, 18,
                9, 19, 43, 17, 9, 18, 40, 16, 9, 17, 48, 18, 9, 19, 48, 18, 9, 19,
                48, 18, 9, 19, 57, 21, 9, 22, 23, 11, 9, 12, 28, 12, 9, 13, 37, 15,
                9, 16, 33, 13, 9, 14, 32, 14, 9, 15, 20, 10, 9, 11, 23, 11, 9, 12,
                23, 11, 9, 12, 23, 11, 9, 12, 37, 15, 9, 16]
            assert all(x.dtype == np.int32 for x in dat[0][1:])

            if hasattr(numint, '_sparse_index'):
                for i, i0 in enumerate(range(0, ngrids, numint.MIN_BLK_SIZE)):
                    i1 = min(i0+numint.MIN_BLK_SIZE, ngrids)
                    ref = numint._sparse_index(
                        opt._sorted_mol, grids.coords[i0:i1], opt.l_ctr_offsets, ao_loc, opt)
                    assert all(np.array_equal(r, x) for r, x in zip(ref[1:], dat[i][1:]))

    def test_scale_ao(self):
        ao = cupy.random.rand(1, 3, 256)
        wv = cupy.random.rand(1, 256)
        out = cupy.ones(6 * 256)
        ref = cupy.einsum('nip,np->ip', ao, wv)
        assert abs(ref - numint._scale_ao(ao, wv)).max() < 1e-12
        assert abs(ref - numint._scale_ao(ao, wv, out=out)).max() < 1e-12
        assert abs(ref - numint._scale_ao(ao+0j, wv, out=out)).max() < 1e-12
        assert abs(ref - numint._scale_ao(ao[0]+0j, wv[0], out=out)).max() < 1e-12

        ao = ao.transpose(1, 0, 2).copy(order='C').transpose(1, 0, 2)
        assert abs(ref - numint._scale_ao(ao, wv)).max() < 1e-12

        assert abs(ref - numint._scale_ao(ao[0], wv[0])).max() < 1e-12

if __name__ == "__main__":
    print("Full Tests for dft numint")
    unittest.main()
