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

from gpu4pyscf.grad.rks import _make_dR_dao_w


def test_make_dR_dao_w_matches_for_noncontiguous_weights():
    cupy.random.seed(17)
    ao = cupy.random.random((10, 5, 32)).astype(cupy.float64)
    wv_storage = cupy.random.random((4, 64)).astype(cupy.float64)
    wv = wv_storage[:, ::2]
    assert not wv.flags.c_contiguous

    result = _make_dR_dao_w(ao, wv)
    reference = _make_dR_dao_w(ao, cupy.ascontiguousarray(wv))

    cupy.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12)
