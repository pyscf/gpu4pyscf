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

from gpu4pyscf.fci.direct_spin1 import FCI
from gpu4pyscf.fci.direct_spin1 import FCISolver


def solver(mol=None, singlet=False, symm=None):
    if singlet:
        raise NotImplementedError('GPU spin-adapted FCI is not implemented')
    if symm or (symm is None and mol is not None and mol.symmetry):
        raise NotImplementedError('GPU symmetry-adapted FCI is not implemented')
    return FCISolver(mol)
