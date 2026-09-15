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

'''
Analytical nuclear gradients for RHF with kpoints sampling
'''

import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.grad import krhf as krhf_grad
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad
from gpu4pyscf.pbc.dft import multigrid
from gpu4pyscf.pbc.gto import int1e

__all__ = ['Gradients']

class Gradients(krhf_grad.GradientsBase):
    '''Non-relativistic restricted Hartree-Fock gradients'''
    grids = None
    grid_response = False

    _keys = {'grid_response', 'grids'}

    hcore_generator = krhf_grad.hcore_generator

    energy_ee = krhf_grad.Gradients.energy_ee
    grad_elec = krhf_grad.Gradients.grad_elec

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        '''Energy weighted density matrix'''
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        dm1ea = krhf_grad.Gradients.make_rdm1e(self, mo_energy[0], mo_coeff[0], mo_occ[0])
        dm1eb = krhf_grad.Gradients.make_rdm1e(self, mo_energy[1], mo_coeff[1], mo_occ[1])
        return dm1ea + dm1eb

    as_scanner = krhf_grad.Gradients.as_scanner
    _finalize = krhf_grad.Gradients._finalize
