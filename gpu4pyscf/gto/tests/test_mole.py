# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

import pyscf
from gpu4pyscf.gto import mole as mole_gpu

def test_basis_seg_contraction():
    mol = pyscf.M(
        atom='C 0 0 0; O 0 1 1',
        basis=('ccpvdz', [[2, [9, .1, .1], [3, 1, .5], [1, .5, 1]]]),
    )
    pmol, c = mole_gpu.basis_seg_contraction(mol)
    ref = mol.intor('int1e_ovlp')
    c = c.get()
    dat = c.T.dot(pmol.intor('int1e_ovlp')).dot(c)
    assert abs(dat - ref).max() < 1e-15

    pmol, c = mole_gpu.basis_seg_contraction(mol, allow_replica=True)
    c = c.get()
    dat = c.T.dot(pmol.intor('int1e_ovlp')).dot(c)
    assert abs(dat - ref).max() < 1e-15

    pmol, c = mole_gpu.basis_seg_contraction(mol, allow_replica=False)
    c = c.get()
    dat = c.T.dot(pmol.intor('int1e_ovlp')).dot(c)
    assert abs(dat - ref).max() < 1e-15
