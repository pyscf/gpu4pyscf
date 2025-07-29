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

import cupy as cp
import numpy as np

from pyscf import lib
import pyscf.pbc.grad.uhf as cpu_uhf
from pyscf.pbc.grad.rhf import _contract_vhf_dm
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.dft.multigrid.pp import vpploc_part1_nuc_grad
import gpu4pyscf.grad.uhf as mol_uhf
import gpu4pyscf.pbc.grad.rhf as rhf
from gpu4pyscf.lib.cupy_helper import return_cupy_array
from gpu4pyscf.pbc.dft import multigrid_v2
from gpu4pyscf.pbc.gto import int1e

__all__ = ['Gradients']


class Gradients(rhf.GradientsBase):

    make_rdm1e = mol_uhf.Gradients.make_rdm1e

    def get_veff(self, mol=None, dm=None, kpt=None, verbose=None):
        mf = self.base
        xc_code = getattr(mf, "xc", None)
        return mf.with_df.get_veff_ip1(dm, xc_code=xc_code, kpt=kpt)

    def grad_elec(
        self,
        mo_energy=None,
        mo_coeff=None,
        mo_occ=None,
        atmlst=None,
    ):
        mf = self.base
        cell = mf.cell
        assert hasattr(mf, '_numint')
        assert isinstance(mf._numint, multigrid_v2.MultiGridNumInt)

        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        dm0_combined_spin = dm0[0] + dm0[1]
        dm0_cpu = dm0_combined_spin.get()

        dme0 = self.make_rdm1e(mo_energy, mo_coeff, mo_occ)
        dme0_sf = (dme0[0] + dme0[1]).get()

        if atmlst is None:
            atmlst = range(cell.natm)

        ni = mf._numint
        assert hasattr(mf, 'xc'), 'HF gradients not supported'
        de = multigrid_v2.get_veff_ip1(ni, mf.xc, dm0, with_j=True).get()
        s1 = int1e.int1e_ipovlp(cell)[0].get()
        de += _contract_vhf_dm(self, s1, dme0_sf) * 2

        # the CPU code requires the attribute .rhoG
        rhoG = multigrid_v2.evaluate_density_on_g_mesh(ni, dm0).get()
        with lib.temporary_env(ni, rhoG=rhoG):
            de += vpploc_part1_nuc_grad(ni, dm0_cpu)
        de += pp_int.vpploc_part2_nuc_grad(cell, dm0_cpu)
        de += pp_int.vppnl_nuc_grad(cell, dm0_cpu)
        core_hamiltonian_gradient = int1e.int1e_ipkin(cell)[0].get()
        kinetic_contribution = _contract_vhf_dm(
            self, core_hamiltonian_gradient, dm0_cpu
        )
        de -= kinetic_contribution * 2

        return de
