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
import scipy.linalg
import cupy as cp
from gpu4pyscf.lib.cusolver import eigh, cholesky

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
