#!/usr/bin/env python
# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
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

'''
Analytical derivatives for DFT+U with kpoints sampling
'''

from gpu4pyscf.pbc.grad import kuks as kuks_grad
from gpu4pyscf.pbc.grad.krkspu import _hubbard_U_derivatives


class Gradients(kuks_grad.Gradients):
    def energy_ee(self, dm, kpts):
        # The shared routine sums the spin-resolved Hubbard responses.
        dE = _hubbard_U_derivatives(self.base, dm, kpts)
        return kuks_grad.Gradients.energy_ee(self, dm, kpts) + dE
