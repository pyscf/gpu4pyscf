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
from gpu4pyscf.lib.cupy_helper import (
    take_last2d, transpose_sum, krylov, unpack_sparse,
    add_sparse, takebak, empty_mapped, dist_matrix,
    grouped_dot, grouped_gemm, cond, cart2sph_cutensor, cart2sph)

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
        n = 31
        count = 127
        a = cupy.random.rand(count,n,n)
        b = a + a.transpose(0,2,1)
        transpose_sum(a)
        assert(cupy.linalg.norm(a - b) < 1e-10)

    def test_krylov(self):
        a = cupy.random.random((10,10)) * 1e-2
        b = cupy.random.random((3,10))

        def aop(x):
            return cupy.dot(a, x.T).T
        x = krylov(aop, b)

        assert cupy.allclose(cupy.dot(a,x.T)+x.T, b.T)

        a = cupy.random.random((10,10)) * 1e-2
        b = cupy.random.random((10))

        def aop(x):
            return cupy.dot(a, x.T).T
        x = krylov(aop, b)
        assert cupy.allclose(cupy.dot(a,x)+x, b)

    def test_cderi_sparse(self):
        naux = 4
        nao = 3
        cderi = cupy.random.rand(nao,nao,naux)
        cderi = cderi + cderi.transpose([1,0,2])

        row, col = cupy.tril_indices(nao)
        cderi_sparse = cderi[row,col,:]
        p0 = 1
        p1 = 3
        out = unpack_sparse(cderi_sparse, row, col, p0, p1, nao)
        assert cupy.linalg.norm(out - cderi[:,:,p0:p1]) < 1e-10

    def test_sparse(self):
        a = cupy.random.rand(20, 20)
        b = cupy.random.rand(5,5)
        indices = cupy.array([3,4,8,10,12]).astype(numpy.int32)
        a0 = a.copy()
        a0[cupy.ix_(indices, indices)] += b
        add_sparse(a, b, indices)
        assert cupy.linalg.norm(a - a0) < 1e-10

    def test_dist_matrix(self):
        a = cupy.random.rand(4, 3)
        rij = cupy.sum((a[:,None,:] - a[None,:,:])**2, axis=2)**0.5
        rij0 = dist_matrix(a, a)
        assert cupy.linalg.norm(rij - rij0) < 1e-10

    def test_takebak(self):
        a = empty_mapped((5, 8))
        a[:] = 1.
        idx = numpy.arange(8) * 2
        out = cupy.zeros((5, 16))
        takebak(out, a, idx)
        out[:,idx] -= 1.
        assert abs(out).sum() == 0.

    def test_cond(self):
        a = cupy.random.rand(5,5)
        cond_cpu = numpy.linalg.cond(a.get())
        cond_gpu = cond(a)
        assert abs(cond_cpu - cond_gpu) < 1e-5

    def test_grouped_dot(self):
        dtype = cupy.float64
        def initialize(dtype, M, N, K):
            sizes = [(M, K), (N, K), (M, N)]
            return [cupy.random.random(size).astype(dtype) for size in sizes]

        def generate_problems(problems):
            valid_sizes = [31]
            As, Bs, Cs = [], [], []
            for _ in range(problems):
                M = numpy.random.choice(valid_sizes)
                N = M
                K = 63
                A, B, C = initialize(dtype, M, N, K)
                As.append(A)
                Bs.append(B)
                Cs.append(C)
            return As, Bs, Cs

        groups = 20
        As, Bs, Cs = generate_problems(groups)
        res_Cs = Cs

        for i in range(groups):
            Cs[i] = cupy.dot(As[i].T, Bs[i])

        grouped_dot(As, Bs, res_Cs)
        res_Cs_2 = grouped_dot(As, Bs)

        res_Cs = cupy.concatenate(res_Cs, axis=None)
        res_Cs_2 = cupy.concatenate(res_Cs, axis=None)
        ans_Cs = cupy.concatenate(Cs, axis=None)
        assert(cupy.linalg.norm(res_Cs - ans_Cs) < 1e-8)
        assert(cupy.linalg.norm(res_Cs_2 - ans_Cs) < 1e-8)

    def test_grouped_gemm(self):
        dtype = cupy.float64
        def initialize(dtype, M, N, K):
            sizes = [(M, K), (M, N), (K, N)]
            return [cupy.random.random(size).astype(dtype) for size in sizes]

        def generate_problems(problems):
            valid_sizes = [31]
            As, Bs, Cs = [], [], []
            for _ in range(problems):
                M = numpy.random.choice(valid_sizes)
                N = M
                K = 63
                A, B, C = initialize(dtype, M, N, K)
                As.append(A)
                Bs.append(B)
                Cs.append(C)
            return As, Bs, Cs

        groups = 20
        As, Bs, Cs = generate_problems(groups)
        res_Cs = Cs

        for i in range(groups):
            Cs[i] = cupy.dot(As[i].T, Bs[i])

        grouped_gemm(As, Bs, res_Cs)
        res_Cs_2 = grouped_gemm(As, Bs)

        res_Cs = cupy.concatenate(res_Cs, axis=None)
        res_Cs_2 = cupy.concatenate(res_Cs, axis=None)
        ans_Cs = cupy.concatenate(Cs, axis=None)
        assert(cupy.linalg.norm(res_Cs - ans_Cs) < 1e-8)
        assert(cupy.linalg.norm(res_Cs_2 - ans_Cs) < 1e-8)

    def test_cart2sph(self):
        a_cart = cupy.random.rand(10,6,11)
        a_sph0 = cart2sph_cutensor(a_cart, axis=1, ang=2)
        a_sph1 = cart2sph(a_cart, axis=1, ang=2)
        assert cupy.linalg.norm(a_sph0 - a_sph1) < 1e-8

        a_cart = cupy.random.rand(10,10,11)
        a_sph0 = cart2sph_cutensor(a_cart, axis=1, ang=3)
        a_sph1 = cart2sph(a_cart, axis=1, ang=3)
        assert cupy.linalg.norm(a_sph0 - a_sph1) < 1e-8

        a_cart = cupy.random.rand(10,15,11)
        a_sph0 = cart2sph_cutensor(a_cart, axis=1, ang=4)
        a_sph1 = cart2sph(a_cart, axis=1, ang=4)
        assert cupy.linalg.norm(a_sph0 - a_sph1) < 1e-8

        a_cart = cupy.random.rand(10,21,11)
        a_sph0 = cart2sph_cutensor(a_cart, axis=1, ang=5)
        a_sph1 = cart2sph(a_cart, axis=1, ang=5)
        assert cupy.linalg.norm(a_sph0 - a_sph1) < 1e-8

        a_cart = cupy.random.rand(10,28,11)
        a_sph0 = cart2sph_cutensor(a_cart, axis=1, ang=6)
        a_sph1 = cart2sph(a_cart, axis=1, ang=6)
        assert cupy.linalg.norm(a_sph0 - a_sph1) < 1e-8

        '''
        a_cart = cupy.random.rand(10,36,11)
        a_sph0 = cart2sph_cutensor(a_cart, axis=1, ang=7)
        a_sph1 = cart2sph(a_cart, axis=1, ang=7)
        assert cupy.linalg.norm(a_sph0 - a_sph1) < 1e-8
        '''
if __name__ == "__main__":
    print("Full tests for cupy helper module")
    unittest.main()
