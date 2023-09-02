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
import numpy
import cupy
from gpu4pyscf.lib.cupy_helper import *

class KnownValues(unittest.TestCase):
    def test_take_last2d(self):
        n = 3
        count = 4
        indices = numpy.arange(n)
        numpy.random.shuffle(indices)
        a = cupy.random.rand(count,n,n)
        b = take_last2d(a, indices)
        assert(cupy.linalg.norm(a[:,indices][:,:,indices] - b) < 1e-10)    
    
    def test_transpose_sum(self):
        n = 3
        count = 4
        a = cupy.random.rand(count,n,n)
        b = a + a.transpose(0,2,1)
        transpose_sum(a)
        assert(cupy.linalg.norm(a - b) < 1e-10)
    
    def test_krylov(self):
        a = cupy.random.random((10,10)) * 1e-2
        b = cupy.random.random(10)

        def aop(x):
            return cupy.dot(a, x.T).T
        x = krylov(aop, b)
        cupy.allclose(cupy.dot(a,x)+x, b)
    
    def test_solve_triangular(self):
        L = cupy.random.random((10,10))
        a = cupy.dot(L, L.T)
        b = cupy.random.random((10,5))
        low = cupy.linalg.cholesky(a)
        x = cupy.linalg.solve(low, b)
        solve_triangular(low, b)
        assert(cupy.linalg.norm(x-b) < 1e-10)

        b = cupy.random.random((10,1))
        low = cupy.linalg.cholesky(a)
        x = cupy.linalg.solve(low, b)
        solve_triangular(low, b)
        assert(cupy.linalg.norm(x-b) < 1e-10)


    def test_cderi_sparse(self):
        naux = 4
        nao = 3
        cderi = cupy.random.rand(nao,nao,naux)
        cderi = cderi + cderi.transpose([1,0,2])

        row, col = cupy.tril_indices(nao)
        cderi_sparse = cderi[row,col,:]
        p0 = 1; p1 = 3
        out = unpack_sparse(cderi_sparse, row, col, p0, p1, nao)
        assert cupy.linalg.norm(out - cderi[:,:,p0:p1]) < 1e-10

    def test_cutensor(self):
        a = cupy.random.rand(10,9,11)
        b = cupy.random.rand(11,7,13)
        c_einsum = cupy.einsum('ijk,ikl->jl', a[3:9,:,4:10], b[3:9,:6, 7:13])
        c_contract = contract('ijk,ikl->jl', a[3:9,:,4:10], b[3:9,:6, 7:13])
        assert cupy.linalg.norm(c_einsum - c_contract) < 1e-10

        a = cupy.random.rand(10,10,10,10)
        b = cupy.random.rand(20,20)
        c_einsum = cupy.einsum('lkji,jl->ik', a, b[10:20,10:20])
        c_contract = contract('lkji,jl->ik', a, b[10:20,10:20])
        assert cupy.linalg.norm(c_einsum - c_contract) < 1e-10

        a = cupy.random.rand(10,10,10,10)
        b = cupy.random.rand(20,20)
        c_einsum = cupy.einsum('lkji,jk->il', a, b[10:20,10:20])
        c_contract = contract('lkji,jk->il', a, b[10:20,10:20])
        assert cupy.linalg.norm(c_einsum - c_contract) < 1e-10

    def test_sparse(self):
        a = cupy.random.rand(20, 20)
        b = cupy.random.rand(5,5)
        indices = cupy.array([3,4,8,10,12]).astype(np.int32)
        a0 = a.copy()
        a0[cupy.ix_(indices, indices)] += b
        add_sparse(a, b, indices)
        assert cupy.linalg.norm(a - a0) < 1e-10

if __name__ == "__main__":
    print("Full tests for cupy helper module")
    unittest.main()