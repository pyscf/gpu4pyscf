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
import cupy as cp
from pyscf.pbc.dft import numint2c as numint2c_cpu
from gpu4pyscf.lib import utils
from gpu4pyscf.pbc.dft import numint
from gpu4pyscf.pbc.scf.ghf import _block_diag


def _get_vxc(ni, cell, grids, xc_code, dm, hermi, kpts, kpts_band):
    if ni.collinear[0] != 'c':
        raise NotImplementedError('Noncollinear periodic XC on GPU')
    if hermi != 1:
        raise NotImplementedError('Non-Hermitian periodic GKS density')
    dm = cp.asarray(dm)
    nao = cell.nao
    spin_dm = cp.stack((dm[..., :nao, :nao], dm[..., nao:, nao:]))
    n, exc, v = numint.nr_uks(ni, cell, grids, xc_code, spin_dm,
                             hermi=hermi, kpts=kpts, kpts_band=kpts_band)
    return n.sum(), exc, _block_diag(v[0], v[1]).astype(cp.complex128)


class KNumInt2C(numint.KNumInt):
    """Collinear GKS integrator for a k-point mesh; returns GPU arrays."""

    # Scalar response kernels inherited from KNumInt do not accept spin-AO
    # densities. Disable them until periodic two-component response is ported.
    cache_xc_kernel = cache_xc_kernel1 = NotImplemented
    nr_rks_fxc = nr_uks_fxc = nr_rks_fxc_st = NotImplemented
    get_fxc = nr_gks_fxc = nr_fxc = NotImplemented

    collinear = numint2c_cpu.KNumInt2C.collinear
    spin_samples = numint2c_cpu.KNumInt2C.spin_samples
    collinear_thrd = numint2c_cpu.KNumInt2C.collinear_thrd
    collinear_samples = numint2c_cpu.KNumInt2C.collinear_samples

    _keys = {'collinear', 'spin_samples', 'collinear_thrd', 'collinear_samples', 'kpts'}

    def __init__(self, kpts=None):
        self.kpts = np.zeros((1, 3)) if kpts is None else np.reshape(kpts, (-1, 3))

    def nr_vxc(self, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
               kpts=None, kpts_band=None, max_memory=2000, verbose=None):
        if kpts is None: kpts = self.kpts
        return _get_vxc(self, cell, grids, xc_code, dms, hermi, kpts, kpts_band)
    get_vxc = nr_gks_vxc = nr_vxc

    def get_rho(self, cell, dm, grids, kpts=None):
        if kpts is None: kpts = self.kpts
        nao = cell.nao
        charge = dm[..., :nao, :nao] + dm[..., nao:, nao:]
        return super().get_rho(cell, charge, grids, kpts)

    def to_cpu(self):
        return utils.to_cpu(self, out=numint2c_cpu.KNumInt2C(self.kpts))


class NumInt2C(KNumInt2C):
    """Collinear GKS integrator for a single k-point."""

    def nr_vxc(self, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
               kpt=None, kpts_band=None, max_memory=2000, verbose=None):
        if kpt is None: kpt = np.zeros(3)
        return _get_vxc(self, cell, grids, xc_code, dms, hermi, kpt, kpts_band)
    get_vxc = nr_gks_vxc = nr_vxc

    def to_cpu(self):
        return utils.to_cpu(self, out=numint2c_cpu.NumInt2C())
