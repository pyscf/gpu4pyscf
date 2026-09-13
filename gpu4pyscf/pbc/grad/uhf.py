#!/usr/bin/env python
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

import gpu4pyscf.pbc.grad.rhf as rhf

__all__ = ['Gradients']


class Gradients(rhf.GradientsBase):

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        dm1e = rhf.Gradients.make_rdm1e(self, mo_energy[0], mo_coeff[0], mo_occ[0])
        dm1e += rhf.Gradients.make_rdm1e(self, mo_energy[1], mo_coeff[1], mo_occ[1])
        return dm1e

    energy_ee = rhf.Gradients.energy_ee
    grad_elec = rhf.Gradients.grad_elec
