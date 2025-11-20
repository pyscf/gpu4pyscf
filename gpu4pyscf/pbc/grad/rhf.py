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

import ctypes
import cupy as cp
import numpy as np

from pyscf import lib
import pyscf.pbc.grad.rhf as cpu_rhf
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.df.df_jk import _format_kpts_band
import gpu4pyscf.grad.rhf as mol_rhf
from gpu4pyscf.lib.cupy_helper import return_cupy_array
from gpu4pyscf.pbc.dft import multigrid_v2
import gpu4pyscf.pbc.dft.multigrid as multigrid_v1
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad

__all__ = ['Gradients']

class GradientsBase(mol_rhf.GradientsBase):
    get_ovlp = NotImplemented
    grad_nuc = cpu_rhf.GradientsBase.grad_nuc

    def optimizer(self):
        '''Geometry (atom positions and lattice) optimization solver
        '''
        from gpu4pyscf.geomopt.ase_solver import GeometryOptimizer
        return GeometryOptimizer(self.base)


class Gradients(GradientsBase):

    make_rdm1e = mol_rhf.Gradients.make_rdm1e

    def get_veff(self, mol=None, dm=None, kpt=None, verbose=None):
        raise NotImplementedError

    def grad_elec(
        self,
        mo_energy=None,
        mo_coeff=None,
        mo_occ=None,
        atmlst=None,
    ):
        mf = self.base
        cell = mf.cell
        kpt = mf.kpt
        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        dm0 = mf.make_rdm1(mo_coeff, mo_occ)

        dme0 = self.make_rdm1e(mo_energy, mo_coeff, mo_occ).get()

        if atmlst is None:
            atmlst = range(cell.natm)

        with_rsjk = mf.rsjk
        # TODO: handle all-electron+GGA and pseudo+GGA differently
        # pseudo+GGA does not need to evaluate the gradients with PBCJKMatrixOpt
        if with_rsjk is not None:
            from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
            assert isinstance(with_rsjk, PBCJKMatrixOpt)
            if hasattr(mf, 'xc'):
                ni = mf._numint
                assert isinstance(ni, multigrid_v2.MultiGridNumInt)
                omega, k_lr, k_sr = ni.rsh_and_hybrid_coeff(mf.xc)
                if omega != 0 and omega != with_rsjk.omega:
                    with_rsjk = PBCJKMatrixOpt(cell, omega=omega).build()
                if with_rsjk.supmol is None:
                    with_rsjk.build()
                de = multigrid_v2.get_veff_ip1(ni, mf.xc, dm0, with_j=True, with_pseudo_vloc_orbital_derivative=True).get()
                j_factor = 0
            else:
                ni = multigrid_v2.MultiGridNumInt(cell).build()
                j_factor = k_sr = k_lr = 1
                de = 0
                if cell._pseudo:
                    vpplocG = multigrid_v1.eval_vpplocG(ni.cell, ni.mesh)
                    de = multigrid_v2.convert_xc_on_g_mesh_to_fock_gradient(
                        ni, vpplocG.reshape(1,1,-1), dm0).get()
                else:
                    raise NotImplementedError
            ejk  = with_rsjk._get_ejk_sr_ip1(dm0, kpts=kpt, exxdiv=mf.exxdiv,
                                             j_factor=j_factor, k_factor=k_sr)
            ejk += with_rsjk._get_ejk_lr_ip1(dm0, kpts=kpt, exxdiv=mf.exxdiv,
                                             j_factor=j_factor, k_factor=k_lr)
            de += ejk*2
        else:
            assert hasattr(mf, 'xc'), 'HF gradients not supported'
            ni = mf._numint
            assert isinstance(ni, multigrid_v2.MultiGridNumInt)
            de = multigrid_v2.get_veff_ip1(ni, mf.xc, dm0, with_j=True, with_pseudo_vloc_orbital_derivative=True).get()

        s1 = int1e.int1e_ipovlp(cell)[0].get()
        de += cpu_rhf._contract_vhf_dm(self, s1, dme0) * 2

        # the CPU code requires the attribute .rhoG
        rhoG = multigrid_v2.evaluate_density_on_g_mesh(ni, dm0)
        rhoG = rhoG[0,0]
        dm0_cpu = dm0.get()
        if cell._pseudo:
            de += multigrid_v1.eval_vpplocG_SI_gradient(cell, ni.mesh, rhoG).get()
            de += vppnl_nuc_grad(cell, dm0_cpu)
        else:
            de += multigrid_v1.eval_nucG_SI_gradient(cell, ni.mesh, rhoG).get()
        rhoG = None
        core_hamiltonian_gradient = int1e.int1e_ipkin(cell)[0].get()
        kinetic_contribution = cpu_rhf._contract_vhf_dm(
            self, core_hamiltonian_gradient, dm0_cpu
        )
        de -= kinetic_contribution * 2

        return de

    def get_stress(self):
        from gpu4pyscf.pbc.grad import rhf_stress
        return rhf_stress.kernel(self)
