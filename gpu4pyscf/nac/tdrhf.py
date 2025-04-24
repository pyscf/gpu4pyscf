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

from functools import reduce
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import cphf
from pyscf import __config__
from gpu4pyscf.lib import utils
from gpu4pyscf import tdscf


def get_densities(td_nac, mf, x_yi, x_yj, singlet=True, relaxed=True):
    if singlet is None:
        singlet = True
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.init_timer(td_grad)
    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nmo = mo_coeff.shape[1]
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    x, y = x_y
    x = cp.asarray(x)
    y = cp.asarray(y)


def get_nacv(td_nac, x_y, singlet=True, atmlst=None, verbose=logger.INFO):
    pass