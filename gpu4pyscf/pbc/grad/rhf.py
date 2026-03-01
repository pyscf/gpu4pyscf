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

from pyscf.gto import ATOM_OF
import pyscf.pbc.grad.rhf as cpu_rhf
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.df.df_jk import _format_kpts_band
import gpu4pyscf.grad.rhf as mol_rhf
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.pbc.dft import multigrid_v2
import gpu4pyscf.pbc.dft.multigrid as multigrid_v1
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad
from gpu4pyscf.gto.mole import groupby

__all__ = ['Gradients']

class GradientsBase(mol_rhf.GradientsBase):
    get_ovlp = NotImplemented
    grad_nuc = cpu_rhf.GradientsBase.grad_nuc

    def get_veff(self, dm=None):
        '''
        Computes the first-order derivatives of the per-cell energy contribution
        from Veff per atom. This is equivalent to one half of the two-electron
        energy contribution: self.energy_ee()/2.

        NOTE: This function is provided for backward compatibility only. It is
        not consistent to the one implemented in PySCF CPU version. In the CPU
        version, get_veff returns the first order derivatives of Veff matrix
        rather than the energy contribution.
        '''
        return self.energy_ee(dm) * .5

    def energy_ee(self, dm):
        '''
        The contribution of electron-electron interactions per cell to the
        nuclear gradients.
        '''
        raise NotImplementedError

    def optimizer(self):
        '''Geometry (atom positions and lattice) optimization solver
        '''
        from gpu4pyscf.geomopt.ase_solver import GeometryOptimizer
        return GeometryOptimizer(self.base)


class Gradients(GradientsBase):

    make_rdm1e = mol_rhf.Gradients.make_rdm1e

    def energy_ee(self, dm):
        '''
        The contribution of electron-electron interactions per cell to the
        nuclear gradients.
        '''
        mf = self.base
        cell = mf.cell
        # TODO: handle all-electron+GGA and pseudo+GGA differently
        # pseudo+GGA does not need to evaluate the gradients with PBCJKMatrixOpt

        de = 0
        xc = getattr(mf, 'xc', 'HF')
        if xc.upper() == 'HF':
            ni = multigrid_v2.MultiGridNumInt(cell).build()
            j_factor = k_sr = k_lr = 1
            omega = 0
        else:
            ni = mf._numint
            omega, k_lr, k_sr = ni.rsh_and_hybrid_coeff(mf.xc)
            j_factor = 1

        if isinstance(ni, multigrid_v2.MultiGridNumInt):
            de += multigrid_v2.get_veff_ip1(
                ni, xc, dm, with_j=True, with_pseudo_vloc_orbital_derivative=True).get()
            j_factor = 0
        else:
            raise NotImplementedError

        if isinstance(ni, multigrid_v2.MultiGridNumInt):
            rhoG = multigrid_v2.evaluate_density_on_g_mesh(ni, dm)
            rhoG = rhoG[0,0]
            if cell._pseudo:
                de += multigrid_v1.eval_vpplocG_SI_gradient(cell, ni.mesh, rhoG).get()
                de += vppnl_nuc_grad(cell, dm)
            else:
                de += multigrid_v1.eval_nucG_SI_gradient(cell, ni.mesh, rhoG).get()

        de += jk_energy_per_atom(mf, dm, None, j_factor, k_sr, k_lr, omega, mf.exxdiv)
        return de

    def grad_elec(
        self,
        mo_energy=None,
        mo_coeff=None,
        mo_occ=None,
        atmlst=None,
    ):
        mf = self.base
        cell = mf.cell
        assert gamma_point(mf.kpt)
        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        de = self.energy_ee(dm0)

        dme0 = self.make_rdm1e(mo_energy, mo_coeff, mo_occ)
        s1 = int1e.int1e_ipovlp(cell)
        de += contract_h1e_dm(cell, s1, dme0, hermi=1)

        core_hamiltonian_gradient = int1e.int1e_ipkin(cell)
        de -= contract_h1e_dm(cell, core_hamiltonian_gradient, dm0, hermi=1)

        return de

    def get_stress(self):
        from gpu4pyscf.pbc.grad import rhf_stress
        return rhf_stress.kernel(self)

def contract_h1e_dm(cell, h1e, dm, hermi=0):
    '''Evaluate
    einsum('xij,ji->x', h1e[:,AO_idx_for_atom], (dm+dm.T)[:,AO_idx_for_atom])
    for all atoms. hermi=1 indicates that dm is a hermitian matrix.
    '''
    assert h1e.ndim == dm.ndim + 1
    ao_loc = cell.ao_loc
    dims = ao_loc[1:] - ao_loc[:-1]
    atm_id_for_ao = np.repeat(cell._bas[:,ATOM_OF], dims)

    if dm.ndim == 2: # RHF
        de_partial = cp.einsum('xij,ji->ix', h1e, dm).real
        if hermi != 1:
            de_partial += cp.einsum('xij,ij->ix', h1e, dm.conj()).real
    elif dm.ndim == 3: # KRHF or UHF
        de_partial = cp.einsum('kxij,kji->ix', h1e, dm).real
        if hermi != 1:
            de_partial += cp.einsum('kxij,kij->ix', h1e, dm.conj()).real
    else: # dm.ndim == 4 KUHF
        de_partial = cp.einsum('skxij,skji->ix', h1e, dm).real
        if hermi != 1:
            de_partial += cp.einsum('skxij,skji->ix', h1e, dm.conj()).real

    de_partial = de_partial.get()
    de = groupby(atm_id_for_ao, de_partial, op='sum')
    if hermi == 1:
        de *= 2

    if len(de) < cell.natm:
        # Handle the case where basis sets are not specified for certain atoms
        de, de_tmp = np.zeros((cell.natm, 3)), de
        de[np.unique(atm_id_for_ao)] = de_tmp
    return de

def jk_energy_per_atom(mf, dm, kpts=None, j_factor=1, sr_factor=1, lr_factor=1,
                       omega=0, exxdiv=None):
    '''
    Computes the first-order derivatives of the energy per atom per cell for
    j_factor * J_derivatives - sr_factor * SR_K_derivatives - lr_factor * LR_K_derivatives
    '''
    from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
    from gpu4pyscf.pbc.df.df import GDF
    if mf.rsjk is not None:
        with_rsjk = mf.rsjk
        assert isinstance(with_rsjk, PBCJKMatrixOpt)
        if with_rsjk.supmol is None:
            with_rsjk.build()
        if omega != 0:
            assert abs(omega) == with_rsjk.omega
        ejk  = with_rsjk._get_ejk_sr_ip1(dm, kpts, exxdiv, j_factor, sr_factor)
        ejk += with_rsjk._get_ejk_lr_ip1(dm, kpts, exxdiv, j_factor, lr_factor)
        ejk *= 2

    elif isinstance(mf.with_df, GDF) and omega <= 0:
        from gpu4pyscf.gto.mole import extract_pgto_params
        from gpu4pyscf.pbc.df.int3c2e import SRInt3c2eOpt
        from gpu4pyscf.pbc.df.rsdf_builder import OMEGA_MIN
        if omega != 0:
            assert abs(omega) == mf.with_df.omega
        cell = mf.with_df.cell
        auxcell = mf.with_df.auxcell
        with_long_range = omega == 0
        if with_long_range:
            cell_exps, cs = extract_pgto_params(cell, 'diffuse')
            omega = min(OMEGA_MIN, (cell_exps.min()*.5)**.5)

        hermi = 1
        if dm.ndim == 2: # RHF at gamma point
            assert kpts is None or gamma_point(kpts)
            from gpu4pyscf.pbc.df.grad.rhf import _jk_energy_per_atom
            int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega).build()
            ejk = _jk_energy_per_atom(
                int3c2e_opt, dm, hermi, j_factor, sr_factor, exxdiv,
                with_long_range)
        else: # KRHF
            assert dm.ndim == 3
            from gpu4pyscf.pbc.df.grad.krhf import _jk_energy_per_atom
            kmesh = kpts_to_kmesh(cell, kpts, rcut=cell.rcut*10, bound_by_supmol=False)
            int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
            ejk = _jk_energy_per_atom(
                int3c2e_opt, dm, kpts, hermi, j_factor, sr_factor, exxdiv,
                with_long_range)

    else: # fft or aft
        if omega <= 0:
            k_factor = sr_factor
        else:
            k_factor = lr_factor
        with mf.with_df.range_coulomb(omega) as with_df:
            ejk = 0
            if k_factor != 0:
                ejk = with_df.get_k_e1(dm, kpts, exxdiv) * k_factor
            if j_factor != 0:
                ejk += with_df.get_j_e1(dm, kpts) * j_factor
        ejk *= 2

    return ejk
