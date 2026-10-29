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
import numpy
import cupy
import cupyx.scipy.linalg
import scipy.linalg
from gpu4pyscf.lib import cupy_helper
from gpu4pyscf.lib.cupy_helper import (
    take_last2d, transpose_sum, krylov, unpack_sparse,
    add_sparse, takebak, empty_mapped, dist_matrix,
    grouped_dot, grouped_gemm, cond, cart2sph,
    copy_array, contract)
from gpu4pyscf.lib import cusolver

def cart2sph_cutensor(t, axis=0, ang=1, out=None):
    '''
    transform 'axis' of a tensor from cartesian basis into spherical basis with cutensor
    '''
    from gpu4pyscf.gto import mole
    if(ang <= 1):
        if(out is not None): out[:] = t
        return t
    size = list(t.shape)
    c2s = mole.cart2sph_by_l(ang)
    if(not t.flags['C_CONTIGUOUS']): t = cupy.asarray(t, order='C')
    li_size = c2s.shape
    nli = size[axis] // li_size[0]
    i0 = max(1, np.prod(size[:axis]))
    i3 = max(1, np.prod(size[axis+1:]))
    out_shape = size[:axis] + [nli*li_size[1]] + size[axis+1:]

    t_cart = t.reshape([i0*nli, li_size[0], i3])
    if(out is not None):
        out = out.reshape([i0*nli, li_size[1], i3])
    t_sph = contract('min,ip->mpn', t_cart, c2s, out=out)
    return t_sph.reshape(out_shape)

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

        a = cupy.random.rand(count,n,n)
        b = a - a.transpose(0,2,1)
        transpose_sum(a, hermi=2)
        assert(cupy.linalg.norm(a - b) < 1e-10)

        a = cupy.random.rand(count,n,n) + cupy.random.rand(count,n,n) * 1j
        b = a + a.transpose(0,2,1).conj()
        transpose_sum(a)
        assert(cupy.linalg.norm(a - b) < 1e-10)

        a = cupy.random.rand(count,n,n) + cupy.random.rand(count,n,n) * 1j
        b = a - a.transpose(0,2,1).conj()
        transpose_sum(a, hermi=2)
        assert(cupy.linalg.norm(a - b) < 1e-10)

    def test_krylov(self):
        a = cupy.random.random((10,10)) * 1e-2
        b = cupy.random.random((3,10))

        def aop(x):
            return cupy.dot(a, x.T).T
        x = krylov(aop, b)
        assert cupy.allclose(cupy.dot(a,x.T)+x.T, b.T, atol=1e-5)

        a = cupy.random.random((10,10)) * 1e-2
        b = cupy.random.random((10))

        def aop(x):
            return cupy.dot(a, x.T).T
        x = krylov(aop, b)
        assert cupy.allclose(cupy.dot(a,x)+x, b, atol=1e-5)

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

        for i in range(groups):
            Cs[i] = cupy.dot(As[i], Bs[i].T)

        res_Cs = grouped_dot(As, Bs)

        res_Cs = cupy.concatenate(res_Cs, axis=None)
        ans_Cs = cupy.concatenate(Cs, axis=None)
        assert(cupy.linalg.norm(res_Cs - ans_Cs) < 1e-8)

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

        for i in range(groups):
            Cs[i] = cupy.dot(As[i].T, Bs[i])

        res_Cs = grouped_gemm(As, Bs)

        res_Cs = cupy.concatenate(res_Cs, axis=None)
        ans_Cs = cupy.concatenate(Cs, axis=None)
        assert(cupy.linalg.norm(res_Cs - ans_Cs) < 1e-8)

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

    def test_unpack_tril(self):
        d = 10
        n = 515
        npair = n * (n+1) // 2
        atril = cupy.random.rand(d, npair) + cupy.random.rand(d, npair)*1j
        a = cupy_helper.unpack_tril(atril)
        idx, idy = cupy.tril_indices(n)
        ref = cupy.empty((d, n, n), dtype=atril.dtype)
        ref[:,idy,idx] = atril.conj()
        ref[:,idx,idy] = atril
        assert abs(a - ref).max() < 1e-12

        ref = atril
        atril = cupy_helper.pack_tril(a)
        assert abs(atril - ref).max() < 1e-12

    def test_copy_host2dev(self):
        host_array = cupy.cuda.alloc_pinned_memory(10*10*10 * 8)
        host_pointer = host_array.ptr
        host_data = numpy.ndarray(10**3, dtype=cupy.float64, buffer=host_array)
        host_data = host_data.reshape(10,10,10)
        host_data[:] = numpy.random.rand(10,10,10)
        assert host_data.ctypes.data == host_pointer

        device_data = cupy.empty_like(host_data)
        host_view = host_data[:, 8:]  # Non-contiguous view on the host
        device_view = device_data[:, 8:]  # Non-contiguous view on the device

        copy_array(host_view, device_view)
        assert numpy.linalg.norm(host_view - device_view.get()) < 1e-10

        copy_array(host_view.copy(), device_view)
        assert numpy.linalg.norm(host_view - device_view.get()) < 1e-10

        device_view = copy_array(host_view)
        assert numpy.linalg.norm(host_view - device_view.get()) < 1e-10

    def test_copy_dev2host(self):
        host_array = cupy.cuda.alloc_pinned_memory(10*10*10 * 8)
        host_data = numpy.ndarray(3*10**2, dtype=cupy.float64, buffer=host_array)
        host_data = host_data.reshape(3,10,10)

        device_data = cupy.zeros_like(host_data)
        device_data += cupy.random.rand(3,10,10)
        host_view = host_data[:, 8:]  # Non-contiguous view on the host
        device_view = device_data[:, 8:]  # Non-contiguous view on the device

        copy_array(device_view, host_view)
        assert numpy.linalg.norm(host_view - device_view.get()) < 1e-10

        copy_array(device_view.copy(), host_view)
        assert numpy.linalg.norm(host_view - device_view.get()) < 1e-10

    def test_block_diag(self):
        arrs = [cupy.random.rand(n, n) for n in range(7, 35, 3)]
        ref = cupyx.scipy.linalg.block_diag(*arrs)
        dat = cupy_helper.block_diag(arrs)
        assert cupy.array_equal(ref, dat)

    def test_block_diag_complex_mixed_blocks(self):
        blocks = [np.array([[1., 2.+3j], [2.-3j, 4.]]), np.array([[5., 6.]])]
        result = cupy_helper.block_diag([cupy.asarray(a) for a in blocks])
        self.assertEqual(result.dtype, np.result_type(*blocks))
        np.testing.assert_array_equal(result.get(), scipy.linalg.block_diag(*blocks))

        blocks = [np.array([[1., 2.], [3., 4.]], dtype=np.complex128),
                  np.array([[5., 6.]])]
        blocks[0] += 2j
        buffer = cupy.full(16, 7, dtype=np.complex128)
        result = cupy_helper.block_diag([cupy.asarray(a) for a in blocks], out=buffer)
        self.assertEqual(result.data.ptr, buffer.data.ptr)
        np.testing.assert_array_equal(result.get(), scipy.linalg.block_diag(*blocks))
        np.testing.assert_array_equal(buffer[12:].get(), 7)

    def test_condense(self):
        a = cupy.random.rand(120*6, 80*5)
        loc_x = numpy.append(numpy.arange(0, a.shape[0], 6), a.shape[0])
        loc_y = numpy.append(numpy.arange(0, a.shape[1], 5), a.shape[1])
        dat = cupy_helper.condense('sum', a, loc_x, loc_y)
        ref = a.reshape(120,6,80,5).transpose(0,2,1,3).sum(axis=(2,3))
        assert abs(dat - ref).max() < 1e-12

    def test_eigh(self):
        a = cupy.random.rand(60, 60)
        b = cupy.random.rand(60, 60)
        a = a.dot(a.T)
        b = cupy.eye(a.shape[0]) + b.dot(b.T) * .2
        eref, cref = cupy_helper.eigh(a, b)
        try:
            bakup, cusolver.MAX_EIGH_DIM = cusolver.MAX_EIGH_DIM, 2
            e, c = cupy_helper.eigh(a, b)
        finally:
            cusolver.MAX_EIGH_DIM = bakup
        assert abs(eref - e).max() < 1e-12

    def test_unpack_tril_complex(self):
        packed = cupy.asarray([1., 2.+3j, 4.], dtype=np.complex128)
        out = cupy.full((2, 2), 99, dtype=np.complex128)
        result = cupy_helper.unpack_tril(packed, out=out)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.data.ptr, out.data.ptr)
        np.testing.assert_array_equal(result.get(), [[1., 2.-3j], [2.+3j, 4.]])

    def test_copy_symmetric_complex_tensor(self):
        i, j = np.array([1, 2]), np.array([0, 1])
        source = np.arange(36, dtype=np.float64).reshape(3, 3, 4).astype(np.complex128)
        source += 2j * source
        expected = source.copy()
        expected[j,i,:] = source[i,j,:]
        array = cupy.asarray(source)
        self.assertIs(cupy_helper.copy_symmetric(array, i, j), array)
        np.testing.assert_array_equal(array.get(), expected)

    def test_scatter_add_scaled_complex_rows(self):
        indices = np.array([3, 0, 2])
        source = np.arange(20, dtype=np.float64).reshape(5, 4).astype(np.complex128)
        update = np.arange(12, dtype=np.float64).reshape(3, 4).astype(np.complex128)
        update += 2j
        expected = source.copy()
        expected[indices] -= .5 * update
        array = cupy.asarray(source)
        self.assertIs(cupy_helper.scatter_add(array, indices, cupy.asarray(update), alpha=-.5), array)
        np.testing.assert_array_equal(array.get(), expected)

    def test_add_sparse_batched_noncontiguous_update(self):
        indices = np.array([3, 0, 2])
        update = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
        array = cupy.zeros((2, 4, 4))
        result = add_sparse(array, cupy.asarray(update).transpose(0, 2, 1), indices)
        expected = np.zeros((2, 4, 4))
        for i in range(2):
            expected[i][np.ix_(indices, indices)] = update[i].T
        np.testing.assert_array_equal(result.get(), expected)

    def test_condense_signed_extrema(self):
        row, col = [0, 1, 3], [0, 2, 4]
        for opname, sign in (('max', -1), ('min', 1)):
            with self.subTest(opname=opname):
                host = sign * np.arange(1, 13, dtype=np.float64).reshape(3, 4)
                expected = np.array([[getattr(host[i0:i1,j0:j1], opname)()
                                      for j0, j1 in zip(col[:-1], col[1:])]
                                     for i0, i1 in zip(row[:-1], row[1:])])
                array = cupy.array(host, order='F')
                result = cupy_helper.condense(opname, array, row, col)
                np.testing.assert_array_equal(result.get(), expected)

    def test_batched_vec3_norm2(self):
        host = np.arange(12, dtype=np.float64).reshape(4, 3) - 4
        expected = np.einsum('ij,ij->i', host, host)
        result = cupy_helper.batched_vec3_norm2(cupy.asarray(host))
        np.testing.assert_array_equal(result.get(), expected)

    def test_batched_vec_dot_orders_and_output(self):
        first = np.arange(20, dtype=np.float64).reshape(4, 5) - 7
        second = first[::-1].copy() * .5
        expected = np.einsum('ij,ij->i', first, second)
        for order in ('C', 'F'):
            with self.subTest(order=order):
                backing = cupy.full(9, -7.)
                out = backing[2:6]
                result = cupy_helper.batched_vec_dot(cupy.array(first, order=order),
                                                     cupy.array(second, order=order), out=out)
                self.assertEqual(result.data.ptr, out.data.ptr)
                np.testing.assert_array_equal(result.get(), expected)
                expected_backing = np.full(9, -7.)
                expected_backing[2:6] = expected
                np.testing.assert_array_equal(backing.get(), expected_backing)

    def test_pinv_complex_positive_semidefinite_matrix(self):
        vectors = np.array([[1., 0.], [0., 2.], [1., 1j]])
        matrix = vectors @ vectors.conj().T
        expected = np.linalg.pinv(matrix, hermitian=True, rcond=1e-10)
        result = cupy_helper.pinv(cupy.asarray(matrix)).get()
        np.testing.assert_allclose(result, expected, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(matrix @ result @ matrix, matrix, rtol=2e-13, atol=2e-13)

    def test_krylov_residual_below_requested_tolerance(self):
        rhs = cupy.asarray([2e-6, -1e-6])
        result = krylov(lambda x: .125 * x, rhs, x0=None, tol=1e-4)
        np.testing.assert_array_equal(result.get(), np.zeros(2))
        self.assertLess(float(cupy.linalg.norm(1.125 * result - rhs)), 1e-4)


if __name__ == "__main__":
    print("Full tests for cupy helper module")
    unittest.main()
