# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
import cupy as cp
from gpu4pyscf.lib.diis import DIIS

def test_on_gpu():
    diis = DIIS()
    diis.incore = True
    xs = cp.random.rand(3, 5)
    diis.update(xs[0])
    diis.update(xs[1])
    diis.update(xs[2])

def test_errvec_on_gpu():
    diis = DIIS()
    diis.incore = True
    xs = cp.random.rand(3, 5)
    err = cp.random.rand(3, 5)
    diis.update(xs[0], xerr=err[0])
    diis.update(xs[1], xerr=err[1])
    diis.update(xs[2], xerr=err[2])

def test_on_cpu():
    diis = DIIS()
    diis.incore = False
    xs = cp.random.rand(3, 5)
    diis.update(xs[0])
    diis.update(xs[1])
    diis.update(xs[2])

def test_errvec_on_cpu():
    diis = DIIS()
    diis.incore = False
    xs = cp.random.rand(3, 5)
    err = cp.random.rand(3, 5)
    diis.update(xs[0], xerr=err[0])
    diis.update(xs[1], xerr=err[1])
    diis.update(xs[2], xerr=err[2])
