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

import pytest
import numpy as np
import scipy.linalg
import cupy as cp
from gpu4pyscf.lib.cusolver import eigh, cholesky, LinAlgError

def test_eigh_real():
    np.random.seed(6)
    n = 12
    a = np.random.rand(n, n)
    a = a + a.T
    b = np.random.rand(n, n)
    b = b.dot(b.T)
    ref = scipy.linalg.eigh(a, b)
    e, c = eigh(cp.asarray(a), cp.asarray(b))
    assert abs(e.get() - ref[0]).max() < 1e-10
    ovlp = c.get().T.dot(b).dot(ref[1])
    assert abs(abs(ovlp) - np.eye(n)).max() < 1e-10

def test_eigh_cmplx():
    np.random.seed(6)
    n = 12
    a = np.random.rand(n, n) + np.random.rand(n, n) * 1j
    a = a + a.conj().T
    b = np.random.rand(n, n) + np.random.rand(n, n) * 1j
    b = b.dot(b.conj().T)
    ref = scipy.linalg.eigh(a, b)
    e, c = eigh(cp.asarray(a), cp.asarray(b))
    assert abs(e.get() - ref[0]).max() < 1e-10
    ovlp = c.get().conj().T.dot(b).dot(ref[1])
    assert abs(abs(ovlp) - np.eye(n)).max() < 1e-10

def test_cholesky_real():
    np.random.seed(6)
    n = 12
    a = np.random.rand(n, n)
    a = a.dot(a.T)
    ref = np.linalg.cholesky(a)
    x = cholesky(cp.asarray(a))
    assert abs(x.get() - ref).max() < 1e-12

def test_cholesky_cmplx():
    np.random.seed(6)
    n = 12
    a = np.random.rand(n, n) + np.random.rand(n, n) * 1j
    a = a.dot(a.conj().T)
    ref = np.linalg.cholesky(a)
    x = cholesky(cp.asarray(a))
    assert abs(x.get() - ref).max() < 1e-12


def _hermitian_pair(dtype):
    h = np.array([[2., .25, 0., -.1], [.25, 3., .2, 0.],
                  [0., .2, 4., .3], [-.1, 0., .3, 5.]], dtype=dtype)
    s = np.array([[1.5, .1, 0., 0.], [.1, 1.7, .2, 0.],
                  [0., .2, 1.2, .1], [0., 0., .1, 1.4]], dtype=dtype)
    if dtype == np.complex128:
        h[0,1] += .2j
        h[1,0] -= .2j
        s[1,2] += .15j
        s[2,1] -= .15j
    return h, s


def _assert_eigenpairs(h, s, values, vectors):
    values, vectors = values.get(), vectors.get()
    expected = scipy.linalg.eigh(h, s)[0]
    np.testing.assert_allclose(values, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(h @ vectors, (s @ vectors) * values, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(vectors.conj().T @ s @ vectors, np.eye(len(h)),
                               rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('overwrite', [False, True])
def test_eigh_overwrite(dtype, overwrite):
    h, s = _hermitian_pair(dtype)
    h_gpu = cp.asarray(h)
    s_gpu = cp.asarray(s)
    values, vectors = eigh(h_gpu, s_gpu, overwrite=overwrite)
    _assert_eigenpairs(h, s, values, vectors)
    if not overwrite:
        np.testing.assert_array_equal(h_gpu.get(), h)
        np.testing.assert_array_equal(s_gpu.get(), s)
    elif dtype == np.float64:
        assert vectors.data.ptr == h_gpu.data.ptr
