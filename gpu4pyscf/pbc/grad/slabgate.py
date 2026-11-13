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

'''
Atomic gradients for a fixed compensating gate and barrier.
references: Phys. Rev. B 89, 245406 (2014).
'''

import numpy as np
import cupy as cp
from gpu4pyscf.pbc.scf.slabdipole import _dipole_grids, _slab_charge
from gpu4pyscf.pbc.scf.slabgate import _sheet_potential, _barrier_potential
from gpu4pyscf.pbc.dft import numint, gen_grid
from gpu4pyscf.pbc.grad.rhf import contract_h1e_dm


def gate_grad(grad, dm, kpts=None):
    '''
    Explicit d(E_gate + E_barrier)/dR at fixed AO density, Hartree/Bohr.

    AO center derivatives are minus spatial derivatives. The underlying SCF
    gradient already contains the Pulay term with corrected orbital energies.
    '''
    mf = grad.base
    cell = mf.cell
    length, zgate, start, end, height, width = mf._gate_parameters(cell)
    area = cell.vol/length
    qc = -_slab_charge(mf, cell)
    if kpts is None:
        kpts = mf.kpts if mf.istype('KSCF') else mf.kpt
    kpts = np.asarray(kpts).reshape(-1, 3)
    dm = cp.asarray(dm)
    if mf.istype('UHF') or mf.istype('KUHF'):
        dm = dm[0] + dm[1]
    dm = dm.reshape(len(kpts), cell.nao, cell.nao)
    grids = _dipole_grids(mf, cell)
    if grad.grid_response and isinstance(grids, gen_grid.BeckeGrids):
        raise NotImplementedError('Gate charge gradient grid response is not implemented')

    ni = numint.KNumInt()
    vmat = None
    for ao_kpts, weight, coords in ni.block_loop(cell, grids, deriv=1, kpts=kpts):
        if vmat is None:
            vmat = cp.zeros((len(kpts), 3, cell.nao, cell.nao), dtype=ao_kpts.dtype)
        z = coords[:, 2]
        v = -_sheet_potential(z, zgate, length, area, qc)
        if mf.barrier:
            v += _barrier_potential(z, start, end, length, height, width)
        for k, ao in enumerate(ao_kpts):
            aow = (weight*v)[:, None] * ao[0]
            # TODO: this can be optimized
            for x in range(3):
                vmat[k, x] += ao[x + 1].conj().T @ aow
    de = -contract_h1e_dm(cell, vmat, dm, hermi=1) / len(kpts)

    z = cell.atom_coords()[:, 2]
    ionic = _sheet_potential(z, zgate, length, area, qc, deriv=1)
    if mf.barrier:
        ionic -= _barrier_potential(z, start, end, length, height, width, deriv=1)
    de[:, 2] += cell.atom_charges() * ionic.get()
    return de


class _SlabGateGradients:
    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, *args, **kwargs):
        mf = self.base
        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ
        de = super().grad_elec(mo_energy, mo_coeff, mo_occ, *args, **kwargs)
        de[:-3] += gate_grad(self, mf.make_rdm1(mo_coeff, mo_occ))
        de[-3:] = np.nan
        return de

    def get_stress(self):
        raise NotImplementedError('Gate charge stress')
