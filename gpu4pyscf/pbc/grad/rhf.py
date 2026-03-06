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
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.df.df import GDF
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad
from gpu4pyscf.gto.mole import groupby

__all__ = ['Gradients']

class GradientsBase(mol_rhf.GradientsBase):
    _keys = {'cell'}

    grad_nuc    = cpu_rhf.GradientsBase.grad_nuc
    get_hcore   = NotImplemented
    get_ovlp    = NotImplemented

    get_dispersion = NotImplemented

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.base.reset(cell)
        return self

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
        # When the MultiGridNumInt integrator is used, the J term can be
        # evaluated together with the XC term. However, if J is computed using
        # GDF approximate integrals, J from MultiGridNumInt is inconsistent with
        # the GDF-based J. In this case, j_in_xc must be disabled, and the J
        # contribution must be evaluated using the GDF jk_energy_per_atom function.
        ni = mf._numint
        j_in_xc = False
        de = 0
        xc = getattr(mf, 'xc', 'HF')
        if xc.upper() == 'HF':
            j_factor = k_sr = k_lr = 1
            # J matrix is accurately computed when rsjk or j_engine is enabled.
            # In the two cases, J from MultiGridNumInt is identical to the
            # the J computed using these real-space integral techniques.
            # FIXME: do not set j_in_xc for all-electron calculations
            j_in_xc = True
            omega = 0
        else:
            # In KS-DFT, whenever the MultiGridNumInt integrator is used,
            # the J term is evaluated along with the MultiGridNumInt integrator.
            if isinstance(ni, multigrid_v2.MultiGridNumInt):
                j_in_xc = True
            omega, k_lr, k_sr = ni.rsh_and_hybrid_coeff(mf.xc)
            j_factor = 1

        # TODO: handle all-electron+GGA and pseudo+GGA differently
        # pseudo+GGA does not need to evaluate the gradients with PBCJKMatrixOpt
        if isinstance(ni, multigrid_v2.MultiGridNumInt):
            de += multigrid_v2.get_veff_ip1(
                ni, xc, dm, with_j=j_in_xc,
                with_pseudo_vloc_orbital_derivative=True).get()
            if j_in_xc:
                j_factor = 0
        elif xc.upper() != 'HF':
            from gpu4pyscf.pbc.grad.krks import get_vxc
            de += get_vxc(ni, mf.cell, mf.grids, xc, dm[None], np.zeros((1, 3))) * 2

        if j_factor != 0 or k_sr != 0 or k_lr != 0:
            de += jk_energy_per_atom(
                mf, dm, None, j_factor, k_sr, k_lr, omega, mf.exxdiv)
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

        ni = mf._numint
        if isinstance(ni, multigrid_v2.MultiGridNumInt):
            rhoG = multigrid_v2.evaluate_density_on_g_mesh(ni, dm0)
            rhoG = rhoG[0,0]
            if cell._pseudo:
                de += multigrid_v1.eval_vpplocG_SI_gradient(cell, ni.mesh, rhoG).get()
            else:
                de += multigrid_v1.eval_nucG_SI_gradient(cell, ni.mesh, rhoG).get()

            dh1e_kin = int1e.int1e_ipkin(cell)
            de -= contract_h1e_dm(cell, dh1e_kin, dm0, hermi=1)
        else:
            from gpu4pyscf.pbc.grad.krhf import hcore_generator
            hcore_deriv = hcore_generator(self, cell, np.zeros((1, 3)))
            dh1e = cp.empty([cell.natm, 3])
            for ia in range(cell.natm):
                h1ao = hcore_deriv(ia)
                dh1e[ia] = cp.einsum('xij,ji->x', h1ao[0], dm0).real
            de += dh1e.get()

        if cell._pseudo:
            de += vppnl_nuc_grad(cell, dm0)

        dme0 = self.make_rdm1e(mo_energy, mo_coeff, mo_occ)
        s1 = int1e.int1e_ipovlp(cell)
        de += contract_h1e_dm(cell, s1, dme0, hermi=1)
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
    assert omega >= 0
    with_df = mf.with_df
    if mf.rsjk is not None:
        with_rsjk = mf.rsjk
        assert isinstance(with_rsjk, PBCJKMatrixOpt)
        if with_rsjk.supmol is None:
            with_rsjk.build()
        if omega != 0:
            assert omega == with_rsjk.omega
        ejk  = with_rsjk._get_ejk_sr_ip1(dm, kpts, exxdiv, j_factor, sr_factor)
        ejk += with_rsjk._get_ejk_lr_ip1(dm, kpts, exxdiv, j_factor, lr_factor)
        ejk *= 2

    elif isinstance(with_df, GDF):
        from pyscf.pbc.df.df import make_auxcell
        from pyscf.pbc.df.rsdf_builder import estimate_ke_cutoff_for_omega
        from gpu4pyscf.pbc.df.aft import AFTDF
        from gpu4pyscf.gto.mole import extract_pgto_params
        from gpu4pyscf.pbc.df.int3c2e import SRInt3c2eOpt
        from gpu4pyscf.pbc.df.rsdf_builder import OMEGA_MIN
        from gpu4pyscf.pbc.df.grad.krhf import _jk_energy_per_atom
        cell = with_df.cell
        auxcell = with_df.auxcell
        if auxcell is None:
            # For LDA, GGA or mGGA, J matrix is evaluated by the numint
            # integrator along with the vxc matrix. with_df might be
            # uninitialized.
            auxcell = make_auxcell(cell, with_df.auxbasis, with_df.exp_to_discard)

        def get_jk(j_factor, k_factor, omega, exxdiv):
            if omega == 0:
                with_long_range = True
                cell_exps, cs = extract_pgto_params(cell, 'diffuse')
                omega = min(OMEGA_MIN, (cell_exps.min()*.5)**.5)
            else:
                with_long_range = False
            if kpts is None:
                assert dm.ndim == 2
                kmesh = None
            else:
                assert dm.ndim == 3
                kmesh = kpts_to_kmesh(cell, kpts, rcut=cell.rcut*10, bound_by_supmol=False)
            int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
            hermi = 1
            return _jk_energy_per_atom(
                int3c2e_opt, dm, kpts, hermi, j_factor, k_factor, exxdiv,
                with_long_range)

        def get_k_lr(k_factor, omega, exxdiv):
            with AFTDF(cell).range_coulomb(omega) as mydf:
                ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
                mydf.mesh = cell.cutoff_to_mesh(ke_cutoff)
                ek_lr = mydf.get_k_e1(dm, kpts, exxdiv)
                # To scale the ek_lr, a factor of .5 is needed correspondign to
                # J-K/2; an *2 needs to be applied for the missing factor in
                # the get_k_e1 function. Here, the overall factor is 1.
                ek_lr *= -k_factor
                return ek_lr

        ejk = 0
        if omega == 0:
            ejk = get_jk(j_factor, sr_factor, 0, exxdiv)
        elif lr_factor == 0:
            if j_factor != 0:
                ejk = get_jk(j_factor, 0, 0, None)
            ejk += get_jk(0, sr_factor, omega, exxdiv)
        elif sr_factor == 0:
            if j_factor != 0:
                ejk = get_jk(j_factor, 0, 0, None)
            ejk += get_k_lr(lr_factor, omega, exxdiv)
        else:
            ejk = get_jk(j_factor, sr_factor, 0, exxdiv)
            ejk += get_k_lr(lr_factor-sr_factor, omega, exxdiv)

    else: # fft or aft
        ejk = 0
        if j_factor != 0:
            ejk = with_df.get_j_e1(dm, kpts) * j_factor
        if sr_factor != 0:
            with with_df.range_coulomb(-omega) as with_df:
                ejk -= with_df.get_k_e1(dm, kpts, exxdiv) * (.5 * sr_factor)
        if omega != 0 and lr_factor != 0:
            with with_df.range_coulomb(omega) as with_df:
                ejk -= with_df.get_k_e1(dm, kpts, exxdiv) * (.5 * lr_factor)
        ejk *= 2

    return ejk
