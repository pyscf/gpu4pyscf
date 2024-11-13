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
    ovlp = c.get().T.dot(b).dot(ref[1])
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
