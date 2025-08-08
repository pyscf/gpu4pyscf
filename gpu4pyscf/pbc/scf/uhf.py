# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
Unrestricted Hartree-Fock for periodic systems at a single k-point
'''

__all__ = [
    'UHF'
]

import numpy as np
import cupy as cp
from pyscf.pbc.scf import uhf as uhf_cpu
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.scf import uhf as mol_uhf
from gpu4pyscf.pbc.scf import hf as pbchf


class UHF(pbchf.SCF):
    '''UHF class for PBCs.
    '''

    _keys = uhf_cpu.UHF._keys

    init_guess_breaksym = uhf_cpu.UHF.init_guess_breaksym

    def __init__(self, cell, kpt=None, exxdiv='ewald'):
        pbchf.SCF.__init__(self, cell, kpt, exxdiv)
        self.nelec = None

    nelec = uhf_cpu.UHF.nelec

    dump_flags = uhf_cpu.UHF.dump_flags

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                 kpt=None, kpts_band=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt
        if isinstance(dm, cp.ndarray) and dm.ndim == 2:
            dm = cp.repeat(dm[None]*.5, 2, axis=0)
        vj, vk = self.get_jk(cell, dm, hermi, kpt, kpts_band)
        vhf = vj[0] + vj[1] - vk
        return vhf

    def get_bands(self, kpts_band, cell=None, dm=None, kpt=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt

        kpts_band = np.asarray(kpts_band)
        single_kpt_band = kpts_band.ndim == 1
        kpts_band = kpts_band.reshape(-1,3)

        fock = self.get_veff(cell, dm, kpt=kpt, kpts_band=kpts_band)
        fock += self.get_hcore(cell, kpts_band)
        s1e = self.get_ovlp(cell, kpts_band)
        nkpts = len(kpts_band)
        nao = fock.shape[-1]
        mo_energy = cp.empty((2, nkpts, nao))
        mo_coeff = cp.empty((2, nkpts, nao, nao), dtype=fock.dtype)
        for k in range(nkpts):
            e, c = self.eig(fock[:,k], s1e[k])
            mo_energy[:,k] = e
            mo_coeff[:,k] = c

        if single_kpt_band:
            mo_energy = mo_energy[:,0]
            mo_coeff = mo_coeff[:,0]
        return mo_energy, mo_coeff

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        if cell is None:
            cell = self.cell
        if s1e is None:
            s1e = self.get_ovlp(cell)
        dm = cp.asarray(mol_uhf.UHF.get_init_guess(self, cell, key))
        ne = cp.einsum('xij,ji->x', dm, s1e).real
        nelec = cp.asarray(self.nelec)
        if max(abs(ne - nelec) > 0.01):
            logger.debug(self, 'Big error detected in the electron number '
                         'of initial guess density matrix (Ne/cell = %s)!\n'
                         '  This can cause huge error in Fock matrix and '
                         'lead to instability in SCF for low-dimensional '
                         'systems.\n  DM is normalized wrt the number '
                         'of electrons %s', ne, nelec)
            dm *= (nelec / ne).reshape(2,1,1)
        return dm

    init_guess_by_1e = mol_uhf.UHF.init_guess_by_1e
    init_guess_by_chkfile = mol_uhf.UHF.init_guess_by_chkfile
    init_guess_by_minao = mol_uhf.UHF.init_guess_by_minao
    init_guess_by_atom = mol_uhf.UHF.init_guess_by_atom
    eig = mol_uhf.UHF.eig
    get_fock = mol_uhf.UHF.get_fock
    get_grad = mol_uhf.UHF.get_grad
    get_occ = mol_uhf.UHF.get_occ
    make_rdm1 = mol_uhf.UHF.make_rdm1
    make_rdm2 = mol_uhf.UHF.make_rdm2
    energy_elec = mol_uhf.UHF.energy_elec
    _finalize = mol_uhf.UHF._finalize
    get_rho = pbchf.get_rho
    analyze = NotImplemented
    mulliken_pop = NotImplemented
    mulliken_meta = NotImplemented
    mulliken_meta_spin = NotImplemented
    canonicalize = NotImplemented
    spin_square = mol_uhf.UHF.spin_square
    stability = NotImplemented

    dip_moment = NotImplemented
    to_ks = NotImplemented
    convert_from_ = NotImplemented

    density_fit = pbchf.RHF.density_fit

    def Gradients(self):
        from gpu4pyscf.pbc.grad.uhf import Gradients
        return Gradients(self)

    def to_cpu(self):
        mf = uhf_cpu.UHF(self.cell)
        utils.to_cpu(self, out=mf)
        return mf
