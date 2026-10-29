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

import cupy
import pytest
from pyscf import gto

from gpu4pyscf.scf import hf


def _make_mol():
    return gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)


def test_scf_rejects_non_blocking_stream_before_calculation():
    mf = hf.RHF(_make_mol())
    mf.dump_flags = lambda *args, **kwargs: pytest.fail(
        'SCF proceeded past the non-blocking stream guard')
    stream = cupy.cuda.Stream(non_blocking=True)

    with stream, pytest.raises(RuntimeError, match='does not support non-blocking'):
        mf.kernel()


def test_scf_supports_blocking_stream():
    mol = _make_mol()
    mf_default = hf.RHF(mol)
    e_default = mf_default.kernel()

    mf_blocking = hf.RHF(mol)
    stream = cupy.cuda.Stream(non_blocking=False)
    with stream:
        e_blocking = mf_blocking.kernel()
    stream.synchronize()

    assert abs(e_blocking - e_default) < 1e-9
