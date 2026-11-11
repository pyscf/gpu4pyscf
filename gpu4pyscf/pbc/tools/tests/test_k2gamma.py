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

import numpy as np
import pytest
from pyscf.pbc import gto
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh


@pytest.mark.parametrize('sampling_mesh, shifted_embedding', [
    ([2, 2, 2], [4, 4, 4]),
    ([2, 3, 1], [4, 3, 1]),
])
def test_kpts_to_kmesh(sampling_mesh, shifted_embedding):
    cell = gto.M(atom='He 0 0 0', basis='sto-3g',
                 a=[[5., 0., 0.], [1., 6., 0.], [0., 1., 7.]],
                 unit='Bohr', verbose=0)
    kpts = cell.make_kpts(sampling_mesh, with_gamma_point=False,
                         wrap_around=True)
    kmesh = kpts_to_kmesh(cell, kpts)
    np.testing.assert_array_equal(kmesh, shifted_embedding)

    # Each sampled Bloch function is periodic over the inferred supercell.
    supercell_vectors = kmesh[:, None] * cell.lattice_vectors()
    np.testing.assert_allclose(np.exp(1j * kpts.dot(supercell_vectors.T)),
                               1., atol=1e-12, rtol=0)
    assert not np.any(np.linalg.norm(kpts, axis=1) < 1e-12)
    assert np.prod(kmesh) > len(kpts)
