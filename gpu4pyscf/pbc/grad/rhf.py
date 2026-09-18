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
from pyscf import lib
from pyscf import gto
import pyscf.pbc.grad.rhf as cpu_rhf
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.df.df_jk import _format_kpts_band
from gpu4pyscf.lib import logger
import gpu4pyscf.grad.rhf as mol_rhf
from gpu4pyscf.gto.mole import SortedCell
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.pbc.dft import multigrid_v3
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.df import aft_jk, AFTDF, GDF
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.dft import KohnShamDFT, BeckeGrids
from gpu4pyscf.pbc.grad.pp import (
    vppnl_nuc_grad, _get_pp_nonloc_strain_derivatives)
from gpu4pyscf.gto.mole import groupby

__all__ = ['Gradients']

class GradientsBase(mol_rhf.GradientsBase):
    _keys = {'cell'}

    grad_nuc    = cpu_rhf.GradientsBase.grad_nuc
    get_hcore   = NotImplemented
    get_ovlp    = NotImplemented
    grad_elec   = NotImplemented

    get_dispersion = NotImplemented

    def __init__(self, method):
        mol_rhf.GradientsBase.__init__(self, method)
        self.cell = method.cell
        self.stress = None

    @property
    def mol(self):
        return self.cell
    @mol.setter
    def mol(self, x):
        self.cell = x

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.stress = None
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
        nuclear gradients and strain derivatives.
        '''
        raise NotImplementedError

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        log = logger.new_logger(self)
        t0 = log.init_timer()
        if mo_energy is None:
            if self.base.mo_energy is None:
                self.base.run()
            mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(mo_energy, mo_coeff, mo_occ)
        self.de = de[:-3] + self.grad_nuc()
        self.stress = (de[-3:] + ewald(self.cell)) / self.cell.vol
        log.timer('SCF gradients', *t0)
        self._finalize()
        return self.de

    def get_stress(self):
        if self.stress is None:
            self.kernel()
        return self.stress

    def optimizer(self):
        '''Geometry (atom positions and lattice) optimization solver
        '''
        from gpu4pyscf.geomopt.ase_solver import GeometryOptimizer
        return GeometryOptimizer(self.base)


class Gradients(GradientsBase):
    grids = None
    grid_response = False

    _keys = {'grid_response', 'grids'}

    make_rdm1e = mol_rhf.Gradients.make_rdm1e

    def energy_ee(self, dm):
        '''
        The contribution of electron-electron interactions per cell
        '''
        mf = self.base
        with_df = mf.with_df
        # When the MultiGridNumInt integrator is used, the J term can be
        # evaluated together with the XC term. However, if J is computed using
        # GDF approximate integrals, J from MultiGridNumInt is inconsistent with
        # the GDF-based J. In this case, j_in_xc must be disabled, and the J
        # contribution must be evaluated using the GDF _get_ejk_derivatives function.
        #
        # J matrix is accurately computed when rsjk or j_engine is enabled.
        # In the two cases, J from MultiGridNumInt is theoretically
        # identical to the the J computed using these real-space integral
        # techniques.
        j_in_xc = not isinstance(with_df, GDF)
        omega = 0
        j_factor = k_sr = k_lr = 1

        # TODO: handle all-electron+GGA and pseudo+GGA differently
        # pseudo+GGA does not need to evaluate the gradients with PBCJKMatrixOpt
        de = 0
        ni = mf._numint
        spin = 0 if dm.ndim == 2 else 1
        if isinstance(ni, multigrid_v3.MultiGridNumInt):
            de = ni.energy_derivatives(
                'HF', dm, spin=spin, with_j=j_in_xc, with_nuc=True)
            if j_in_xc:
                j_factor = 0

        de += _get_ejk_derivatives(mf, dm, None, j_factor, omega, k_lr, k_sr)
        return de

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        mf = self.base
        cell = mf.cell
        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ

        if getattr(mf, 'with_x2c', None):
            raise NotImplementedError('X2C gradients')

        log = logger.new_logger(cell)
        t0 = log.init_timer()
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        grad_sigma = self.energy_ee(dm0)
        t1 = log.timer_debug1('gradients of 2e part', *t0)

        is_uhf = mf.istype('UHF')
        assert gamma_point(mf.kpt)

        if is_uhf:
            dm0 = dm0[0] + dm0[1]

        ni = mf._numint
        if isinstance(ni, multigrid_v3.MultiGridNumInt):
            # Vne or pploc contribution is evaluated in energy_ee
            grad_sigma += int1e.kin_derivatives(cell, dm0)
        else:
            from gpu4pyscf.pbc.grad.krhf import hcore_generator
            hcore_deriv = hcore_generator(self, cell, np.zeros((1, 3)))
            dh1e = cp.empty([cell.natm, 3])
            for ia in range(cell.natm):
                h1ao = hcore_deriv(ia)
                dh1e[ia] = cp.einsum('xij,ji->x', h1ao[0], dm0).real
            grad_sigma[:-3] += dh1e.get()
            if isinstance(self.grids or getattr(mf, 'grids', None), BeckeGrids):
                grad_sigma[-3:] = np.nan
            else:
                # hcore_generator includes kinetic gradients, but not kinetic strain.
                grad_sigma[-3:] += int1e.kin_derivatives(cell, dm0)[-3:]
                ni = multigrid_v3.MultiGridNumInt(cell)
                grad_sigma[-3:] += ni.energy_strain_gradient(
                    'HF', dm0, spin=0, with_j=False, with_nuc=True)

        if cell._pseudo:
            grad_sigma[:-3] += vppnl_nuc_grad(cell, dm0)
            grad_sigma[-3:] += _get_pp_nonloc_strain_derivatives(cell, cell.mesh, dm0)
        t1 = log.timer_debug1('gradients of 1e part', *t1)

        dme0 = self.make_rdm1e(mo_energy, mo_coeff, mo_occ)
        grad_sigma -= int1e.ovlp_derivatives(cell, dme0)
        return grad_sigma

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

def _gdf_ejk_derivatives(mf, dm, kpts=None, j_factor=1, omega=0, lr_factor=1, sr_factor=1):
    from pyscf.pbc.df.df import make_auxcell
    from pyscf.pbc.df.rsdf_builder import estimate_ke_cutoff_for_omega
    from gpu4pyscf.pbc.df.int3c2e import SRInt3c2eOpt
    from gpu4pyscf.pbc.df.rsdf_builder import _guess_omega
    from gpu4pyscf.pbc.df.grad import krhf, kuhf
    hermi = 1

    with_df = mf.with_df
    cell = with_df.cell
    auxcell = with_df.auxcell
    if auxcell is None:
        # For LDA, GGA or mGGA, J matrix is evaluated by the numint
        # integrator along with the vxc matrix. with_df might be
        # uninitialized.
        auxcell = make_auxcell(cell, with_df.auxbasis, with_df.exp_to_discard)

    if kpts is None:
        kmesh = None
        is_rhf = dm.ndim == 2
    else:
        kmesh = kpts_to_kmesh(cell, kpts, rcut=cell.rcut+10, bound_by_supmol=False)
        is_rhf = dm.ndim == 3

    def get_jk(j_factor, k_factor, omega, exxdiv):
        if is_rhf:
            fn = krhf._get_ejk_derivatives
        else:
            fn = kuhf._get_ejk_derivatives
        rsdf_omega = max(abs(omega), _guess_omega(cell))
        # DD responses are implemented for Gamma RHF and Gamma J-only.
        opt = SRInt3c2eOpt(cell, auxcell, rsdf_omega, kmesh)
        opt.cell = SortedCell.from_cell(cell, decontract=True)
        opt.build(separate_dd=True)
        return fn(opt, dm, kpts, hermi, j_factor, k_factor, exxdiv, omega,
                  linear_dep_threshold=with_df.linear_dep_threshold)

    def get_k_lr(k_factor, omega, exxdiv):
        ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
        mydf = AFTDF(cell)
        mydf.mesh = cell.cutoff_to_mesh(ke_cutoff)
        if is_rhf:
            k_factor *= .5
        ek_sigma = aft_jk.get_ek_derivatives(
            mydf, dm, kpts, exxdiv=exxdiv, omega=omega, lr_factor=k_factor, sr_factor=0)
        return ek_sigma

    grad_sigma = 0
    if omega == 0:
        grad_sigma = get_jk(j_factor, sr_factor, 0, mf.exxdiv)
    elif lr_factor == 0:
        grad_sigma = get_jk(0, sr_factor, omega, mf.exxdiv)
        if j_factor != 0:
            grad_sigma += get_jk(j_factor, 0, 0, None)
    elif sr_factor == 0:
        grad_sigma = get_k_lr(-lr_factor, omega, mf.exxdiv)
        if j_factor != 0:
            grad_sigma += get_jk(j_factor, 0, 0, None)
    else:
        grad_sigma = get_jk(j_factor, sr_factor, 0, mf.exxdiv)
        grad_sigma -= get_k_lr(lr_factor-sr_factor, omega, mf.exxdiv)
    return grad_sigma

def _get_ejk_derivatives(mf, dm, kpts=None, j_factor=1, omega=0, lr_factor=1, sr_factor=1):
    '''
    Computes the first-order derivatives of the energy per atom per cell for
    j_factor * J_derivatives - sr_factor * SR_K_derivatives - lr_factor * LR_K_derivatives
    '''
    from gpu4pyscf.pbc.df.int3c2e import SRInt3c2eOpt
    from gpu4pyscf.pbc.df.grad.krhf import _get_ejk_derivatives
    assert omega >= 0
    with_df = mf.with_df
    cell = mf.cell
    hermi = 1
    if kpts is None:
        is_rhf = dm.ndim == 2
    else:
        is_rhf = dm.ndim == 3
    exxdiv = mf.exxdiv

    ejk_sigma = 0
    if mf.rsjk is not None:
        if j_factor != 0 and not mf.j_engine and isinstance(with_df, GDF):
            j_factor = 0
            if kpts is None:
                kmesh = None
            else:
                kmesh = kpts_to_kmesh(cell, kpts, rcut=cell.rcut)
            rsdf_omega = 0.3
            opt = SRInt3c2eOpt(cell, with_df.auxcell, rsdf_omega, kmesh)
            opt.cell = SortedCell.from_cell(cell, decontract=True)
            opt.build(separate_dd=True)
            if is_rhf:
                ejk_sigma = _get_ejk_derivatives(opt, dm, kpts, hermi, k_factor=0)
            else:
                ejk_sigma = _get_ejk_derivatives(opt, dm[0]+dm[1], kpts, hermi, k_factor=0)

        if lr_factor != 0 or sr_factor != 0:
            with_rsjk = mf.rsjk
            assert isinstance(with_rsjk, PBCJKMatrixOpt)
            ejk_sigma += with_rsjk._get_ejk_derivatives(
                dm, kpts, exxdiv=exxdiv, omega=omega, j_factor=j_factor,
                lr_factor=lr_factor, sr_factor=sr_factor)

    elif isinstance(with_df, GDF):
        return _gdf_ejk_derivatives(mf, dm, kpts, j_factor, omega, lr_factor, sr_factor)

    else: # fft or aft
        if j_factor != 0:
            if is_rhf:
                dm_sf = dm
            else:
                dm_sf = dm[0] + dm[1]
            if isinstance(with_df, AFTDF):
                ejk_sigma = with_df.get_ej_derivatives(dm_sf, kpts)
            else:
                ejk_sigma = multigrid_v3.MultiGridNumInt(cell).energy_derivatives(
                    'HF', dm_sf, kpts, spin=0, with_j=True, with_nuc=False)
            ejk_sigma *= j_factor
        if lr_factor != 0 or sr_factor != 0:
            if is_rhf:
                sr_factor *= .5
                lr_factor *= .5
            ejk_sigma -= with_df.get_ek_derivatives(
                dm, kpts, exxdiv, omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)

    return ejk_sigma

def strain_tensor_dispalcement(x, y, disp):
    E_strain = np.eye(3)
    E_strain[x,y] += disp
    return E_strain

def _finite_diff_cells(cell, x, y, disp=1e-4, precision=None):
    if precision is not None:
        cell = cell.copy()
        cell.precision = precision
    a = cell.lattice_vectors()
    r = cell.atom_coords()
    if not gto.mole.is_au(cell.unit):
        a *= lib.param.BOHR
        r *= lib.param.BOHR
    e_strain = strain_tensor_dispalcement(x, y, disp)
    cell1 = cell.set_geom_(r.dot(e_strain.T), inplace=False)
    cell1.a = a.dot(e_strain.T)
    cell1.mesh = cell.mesh

    e_strain = strain_tensor_dispalcement(x, y, -disp)
    cell2 = cell.set_geom_(r.dot(e_strain.T), inplace=False)
    cell2.a = a.dot(e_strain.T)
    cell2.mesh = cell.mesh

    if cell.space_group_symmetry:
        cell1.build(False, False)
        cell2.build(False, False)
    return cell1, cell2

def ewald(cell):
    disp = max(1e-5, (cell.precision*.1)**.5)
    out = np.empty((3, 3))
    for i in range(3):
        for j in range(i+1):
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp)
            e1 = cell1.ewald()
            e2 = cell2.ewald()
            out[j,i] = out[i,j] = (e1 - e2) / (2*disp)
    return out
