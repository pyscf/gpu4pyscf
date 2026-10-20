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

'''Nuclear gradients of the self-consistent Bengtsson slab correction.'''

import numpy as np
import cupy as cp
from gpu4pyscf.pbc.scf.addons import _dipole_grids, _sawtooth, _slab_length
from gpu4pyscf.pbc.dft import numint, gen_grid
from gpu4pyscf.pbc.grad.rhf import contract_h1e_dm
from gpu4pyscf.pbc.gto.int1e import ovlp_derivatives


def dipole_grad(grad, dm, kpts=None):
    r'''Explicit correction to dE/dR, in Hartree/Bohr.

    At fixed cell, cut and AO density matrix,
        dE_dip/dR_I = (4*pi*p/Omega) * (Z_I*n - Tr(D*dQ/dR_I)).
    Spatial AO derivatives have the opposite sign to center derivatives.
    The potential gauge adds field*offset*dS; this cancels the same shift
    in the standard energy-weighted-density Pulay term.
    '''
    mf = grad.base
    cell = mf.cell
    length = _slab_length(cell, mf.dipole_dir)
    if kpts is None:
        kpts = mf.kpts if mf.istype('KSCF') else mf.kpt
    kpts = np.asarray(kpts).reshape(-1, 3)
    nkpts = len(kpts)
    if mf.dipole_cut is None:
        mf.get_veff(dm=dm)
    dm = cp.asarray(dm)
    if mf.istype('UHF') or mf.istype('KUHF'):
        dm = dm[0] + dm[1]
    dm = dm.reshape(nkpts, cell.nao, cell.nao)
    cut = mf.cut if mf.cut is not None else mf.dipole_cut
    cut %= length
    grids = _dipole_grids(mf, cell)
    grid_response = grad.grid_response and isinstance(grids, gen_grid.BeckeGrids)

    ni = numint.KNumInt()
    vmat = None
    dp_grid = cp.zeros((cell.natm, 3))
    p_elec = cp.asarray(0.)
    z_sum = cp.asarray(0.)
    weight_sum = cp.asarray(0.)
    g1 = 0
    for ao_kpts, weight, coords in ni.block_loop(cell, grids, deriv=1, kpts=kpts):
        if vmat is None:
            vmat = cp.zeros((nkpts, 3, cell.nao, cell.nao), dtype=ao_kpts.dtype)
        z = _sawtooth(coords[:, mf.dipole_dir], cut, length)
        zw = z * weight
        rho = ni.eval_rho(cell, ao_kpts[:, 0], dm, xctype='LDA', hermi=1)
        p_elec += cp.dot(rho, zw)
        z_sum += zw.sum()
        weight_sum += weight.sum()
        for k, ao in enumerate(ao_kpts):
            aow = zw[:, None] * ao[0]
            for x in range(3):
                vmat[k, x] += ao[x + 1].conj().T @ aow

        if grid_response:
            g0, g1 = g1, g1 + weight.size
            dweight = gen_grid.get_becke_weight_derivative(grids, cell.natm, (g0, g1))
            dp_grid -= cp.einsum('axg,g->ax', dweight[:cell.natm], rho * z)
            atom = int(grids.supatm_to_atm_idx[grids.supatm_idx[g0]])
            rho_grad = ni.eval_rho(cell, ao_kpts, dm, xctype='GGA', hermi=1)[1:4]
            dp_grid[atom] -= rho_grad @ zw
            dp_grid[atom, mf.dipole_dir] -= cp.dot(rho, weight)

    p_ion = np.dot(cell.atom_charges(),
                   _sawtooth(cell.atom_coords()[:, mf.dipole_dir], cut, length))
    field = 4 * np.pi * (p_ion - float(p_elec)) / cell.vol
    dp = contract_h1e_dm(cell, vmat, dm, hermi=1) / nkpts
    dp[:, mf.dipole_dir] += cell.atom_charges()
    dp += dp_grid.get()
    de = field * dp

    offset = (float(z_sum / weight_sum) if mf.z0 is None else
              _sawtooth(mf.z0, cut, length))
    de += field * offset * ovlp_derivatives(cell, dm, kpts)[:-3]
    return de


class _SlabDipoleGradients:
    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, *args, **kwargs):
        mf = self.base
        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ
        de = super().grad_elec(mo_energy, mo_coeff, mo_occ, *args, **kwargs)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        de[:-3] += dipole_grad(self, dm)
        # Strain derivatives of the correction have not been implemented.
        de[-3:] = np.nan
        return de

    def get_stress(self):
        raise NotImplementedError('Slab dipole correction stress')
