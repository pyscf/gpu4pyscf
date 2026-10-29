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
import cupy
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib.cutensor import _contract_einsum, contraction

class KnownValues(unittest.TestCase):
    def test_contract(self):
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

    def test_complex_valued(self):
        a = cupy.random.rand(10,9,11) + cupy.random.rand(10,9,11)*1j
        b = cupy.random.rand(11,7,13) + cupy.random.rand(11,7,13)*1j
        c_einsum = cupy.einsum('ijk,ikl->jl', a[3:9,:,4:10], b[3:9,:6, 7:13])
        c_contract = contract('ijk,ikl->jl', a[3:9,:,4:10], b[3:9,:6, 7:13])
        assert cupy.linalg.norm(c_einsum - c_contract) < 1e-10

    def test_cache(self):
        a = cupy.random.rand(20,20,20,20)
        b = cupy.random.rand(20,20)

        c = contract('ijkl,ik->jl', a, b)
        c_einsum = cupy.einsum('ijkl,ik->jl', a, b)
        assert cupy.linalg.norm(c - c_einsum) < 1e-10

        c = contract('ijkl,jl->ik', a, b)
        c_einsum = cupy.einsum('ijkl,jl->ik', a, b)
        assert cupy.linalg.norm(c - c_einsum) < 1e-10

    # issue 614
    def test_zero_strides(self):
        a = cupy.ones((3, 3))
        result = contract('ij,ji->', a[0,:,None], a[0,None,:])
        self.assertEqual(result.shape, ())
        assert result == 3

    def test_einsum_fallback_contiguous(self):
        from gpu4pyscf.lib.cupy_helper import hermi_triu, transpose_sum
        ncells, nkpts, nao, comp = 8, 4, 6, 1
        expLkz = cupy.random.rand(ncells, nkpts, 2)
        mat = cupy.random.rand(ncells, comp, nao, nao)
        out = _contract_einsum('lkz,lxpq->kxpqz', expLkz, mat, 1., 0.)
        assert out.flags.c_contiguous
        ref = cupy.einsum('lkz,lxpq->kxpqz', expLkz, mat)
        assert cupy.linalg.norm(out - ref) < 1e-12
        v = out.view(numpy.complex128)[:,:,:,:,0].reshape(nkpts*comp, nao, nao)
        hermi_triu(v, 1, inplace=True)

        veff = cupy.random.rand(ncells, nao, nao)
        out = _contract_einsum('Lpq,Lkz->kpqz', veff, expLkz, 1., 0.)
        assert out.flags.c_contiguous
        ref = cupy.einsum('Lpq,Lkz->kpqz', veff, expLkz)
        assert cupy.linalg.norm(out - ref) < 1e-12
        v = out.view(numpy.complex128)[:,:,:,0]
        transpose_sum(v)

        out2 = _contract_einsum('Lpq,Lkz->kpqz', veff, expLkz, 2., 0.)
        assert out2.flags.c_contiguous
        assert cupy.linalg.norm(out2 - 2*ref) < 1e-12

class EinsumFallback(unittest.TestCase):
    def setUp(self):
        self.a = cupy.asarray([[1., 2., 3.], [4., 5., 6.]])
        self.b = cupy.asarray([[-2., 4., 1.], [1., -1., 2.]])
        self.ref = numpy.einsum('ij,ij->', self.a.get(), self.b.get())

    def test_scalar_result_can_be_used_as_energy(self):
        result = _contract_einsum('ij,ij->', self.a, self.b, 1.5, 0.)
        self.assertIsInstance(result, cupy.ndarray)
        self.assertEqual(result.shape, ())
        self.assertEqual(result.dtype, numpy.dtype(numpy.float64))
        self.assertEqual(float(result), 1.5 * self.ref)
        # EDA stores each fully contracted pair energy in a host scalar slot.
        pair_energies = numpy.zeros((2, 2))
        pair_energies[0, 1] = result
        self.assertEqual(pair_energies[0, 1], 1.5 * self.ref)

    def test_complex_scalar_result(self):
        a = self.a + 1j
        b = self.b - 2j
        result = _contract_einsum('ij,ij->', a, b, 2., 0.)
        ref = 2 * numpy.einsum('ij,ij->', a.get(), b.get())
        self.assertEqual(result.shape, ())
        self.assertEqual(result.dtype, numpy.dtype(numpy.complex128))
        numpy.testing.assert_array_equal(result.get(), ref)

    def test_scalar_output_overwrites_when_beta_is_zero(self):
        out = cupy.asarray(numpy.nan)
        result = _contract_einsum('ij,ij->', self.a, self.b, 2., 0., out=out)
        self.assertIs(result, out)
        self.assertEqual(out.shape, ())
        self.assertEqual(float(out), 2 * self.ref)

    def test_scalar_output_accumulates_alpha_and_beta(self):
        out = cupy.asarray(7.)
        result = _contract_einsum('ij,ij->', self.a, self.b, 2., -3., out=out)
        self.assertIs(result, out)
        self.assertEqual(out.shape, ())
        self.assertEqual(float(out), 2 * self.ref - 3 * 7.)

    def test_empty_contraction_preserves_scalar_shape(self):
        a = cupy.empty((0, 3))
        b = cupy.empty((0, 3))
        # Empty operands also take this fallback when cuTENSOR is available.
        result = contract('ij,ij->', a, b)
        self.assertEqual(result.shape, ())
        self.assertEqual(float(result), 0.)

    def test_singleton_vector_is_not_squeezed(self):
        result = _contract_einsum('ij,j->i', self.a[:1], self.b[0], 1., 0.)
        ref = numpy.einsum('ij,j->i', self.a[:1].get(), self.b[0].get())
        self.assertEqual(result.shape, (1,))
        self.assertTrue(result.flags.c_contiguous)
        numpy.testing.assert_array_equal(result.get(), ref)

    def test_output_overlaps_input_during_accumulation(self):
        a_host = numpy.array([[1., 2.], [3., 4.]])
        b_host = numpy.array([[2., 0.], [1., 3.]])
        a = cupy.asarray(a_host)
        out = a.T
        initial = out.get()
        expected = 2 * numpy.einsum('ik,kj->ij', a_host, b_host) + 3 * initial
        result = _contract_einsum('ik,kj->ij', a, cupy.asarray(b_host), 2., 3., out=out)
        self.assertIs(result, out)
        numpy.testing.assert_array_equal(result.get(), expected)

    def test_fortran_output_beta_zero_overwrites_nan(self):
        out = cupy.full((2, 2), numpy.nan, order='F')
        result = _contract_einsum('ij,kj->ik', self.a, self.b, 2., 0., out=out)
        self.assertIs(result, out)
        self.assertTrue(result.flags.f_contiguous)
        expected = 2 * numpy.einsum('ij,kj->ik', self.a.get(), self.b.get())
        numpy.testing.assert_array_equal(result.get(), expected)

    def test_broadcast_zero_stride_complex_operand(self):
        row = numpy.arange(3) + 1j
        a = cupy.broadcast_to(cupy.asarray(row)[None,:], (4, 3))
        b_host = numpy.arange(8, dtype=numpy.float64).reshape(4, 2)
        self.assertEqual(a.strides[0], 0)
        result = _contract_einsum('ij,ik->jk', a, cupy.asarray(b_host), 1., 0.)
        expected = numpy.einsum('ij,ik->jk', numpy.broadcast_to(row, (4, 3)), b_host)
        self.assertTrue(result.flags.c_contiguous)
        numpy.testing.assert_array_equal(result.get(), expected)

    def test_empty_contraction_accumulates_existing_matrix(self):
        # The native entry point deliberately uses its real einsum fallback
        # before requesting descriptors for unsupported empty operands.
        a = cupy.empty((2, 0), dtype=numpy.complex128)
        b = cupy.empty((0, 3), dtype=numpy.float64)
        out = cupy.arange(6, dtype=numpy.complex128).reshape(2, 3)
        expected = -.5 * out.get()
        self.assertIs(contraction('ik,kj->ij', a, b, 2., -.5, out=out), out)
        numpy.testing.assert_array_equal(out.get(), expected)
        result = contraction('ik,kj->ij', cupy.empty((0, 2)), cupy.ones((2, 3)), 1., 0.)
        self.assertEqual(result.shape, (0, 3))
        self.assertEqual(result.size, 0)


if __name__ == "__main__":
    print("Full tests for cutensor module")
    unittest.main()
