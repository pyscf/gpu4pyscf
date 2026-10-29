# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

from types import SimpleNamespace

import cupy
from pyscf import gto

from gpu4pyscf.mp.mp2 import MP2Base


def test_spin_paired_unequal_occupations_are_noncanonical():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    coeff = cupy.eye(2)
    mf = SimpleNamespace(
        mol=mol,
        mo_coeff=cupy.stack((coeff, coeff)),
        mo_occ=cupy.asarray([[1, 0], [0, 0]]),
        max_memory=4000,
    )
    mp = MP2Base(mf)

    assert not mp._reference_is_canonical()

    mp.mo_occ = cupy.asarray([[1, 0], [1, 0]])
    assert mp._reference_is_canonical()

    mp.mo_occ = cupy.asarray([[1, 0], [0, 0]])
    mp.mo_coeff = cupy.stack((coeff, coeff[:, ::-1]))
    assert mp._reference_is_canonical()
