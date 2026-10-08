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

from functools import reduce

import cupy
from pyscf import gto
from pyscf import scf as cpu_scf
from pyscf.mcscf import casci as cpu_casci

from gpu4pyscf import scf
from gpu4pyscf.fci import direct_spin1 as gpu_direct_spin1
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import utils
from gpu4pyscf.scf import hf


def h1e_for_cas(casci, mo_coeff=None, ncas=None, ncore=None, hcore=None,
                return_corevhf=False):
    log = logger.new_logger(casci)
    t0 = log.init_timer()
    if mo_coeff is None:
        mo_coeff = casci.mo_coeff
    if ncas is None:
        ncas = casci.ncas
    if ncore is None:
        ncore = casci.ncore

    mo_coeff = cupy.asarray(mo_coeff)
    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore:ncore+ncas]

    if hcore is None:
        hcore = cupy.asarray(casci.get_hcore())
    else:
        hcore = cupy.asarray(hcore)
    energy_core = casci.energy_nuc()
    if mo_core.size == 0:
        corevhf = 0
    else:
        core_dm = cupy.dot(mo_core, mo_core.conj().T) * 2
        corevhf = cupy.asarray(casci.get_veff(casci.mol, core_dm))
        energy_core += float(cupy.einsum('ij,ji', core_dm, hcore).real.get())
        energy_core += float(cupy.einsum('ij,ji', core_dm, corevhf).real.get()) * .5

    h1eff = reduce(cupy.dot, (mo_cas.conj().T, hcore+corevhf, mo_cas))
    out = h1eff.get(), float(energy_core)
    if return_corevhf:
        out += (corevhf,)
    log.timer('CAS effective one-electron integrals', *t0)
    return out


class _CASCI(cpu_casci.CASCI):
    canonicalization = False

    get_h1eff = h1e_for_cas
    h1e_for_cas = h1e_for_cas

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    def __init__(self, mf_or_mol, ncas=0, nelecas=0, ncore=None):
        if isinstance(mf_or_mol, gto.MoleBase):
            mf_or_mol = scf.RHF(mf_or_mol)
        elif (hasattr(mf_or_mol, 'istype') and
              any(mf_or_mol.istype(x) for x in ('UHF', 'ROHF', 'GHF'))):
            raise NotImplementedError(
                'GPU CASCI supports restricted HF/KS objects only')
        elif not getattr(mf_or_mol, '__module__', '').startswith('gpu4pyscf'):
            if isinstance(mf_or_mol, cpu_scf.hf.RHF):
                logger.warn(
                    mf_or_mol,
                    'CPU restricted SCF object converted to GPU for CASCI')
                mf_or_mol = mf_or_mol.to_gpu()
            else:
                raise NotImplementedError(
                    'GPU CASCI supports restricted HF/KS objects only')

        if not isinstance(mf_or_mol, hf.RHF):
            raise NotImplementedError(
                'GPU CASCI supports restricted HF/KS objects only')

        super().__init__(mf_or_mol, ncas, nelecas, ncore)
        fcisolver = gpu_direct_spin1.FCISolver(self.mol)
        fcisolver.__dict__.update(self.fcisolver.__dict__)
        self.fcisolver = fcisolver
        self.canonicalization = False

    def energy_nuc(self):
        return self._scf.energy_nuc()

    def get_hcore(self, mol=None):
        return self._scf.get_hcore(mol)

    def get_jk(self, mol, dm, hermi=1, with_j=True, with_k=True, omega=None):
        return self._scf.get_jk(
            mol, dm, hermi, with_j=with_j, with_k=with_k, omega=omega)

    def get_veff(self, mol=None, dm=None, hermi=1):
        if mol is None:
            mol = self.mol
        if dm is None:
            mo_core = cupy.asarray(self.mo_coeff[:, :self.ncore])
            dm = mo_core @ mo_core.conj().T * 2
        vj, vk = self.get_jk(mol, dm, hermi)
        return vj - vk * .5

    def get_h2eff(self, mo_coeff=None):
        raise NotImplementedError('CASCI integral backend is not configured')

    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        if self.canonicalization:
            raise NotImplementedError('GPU CASCI canonicalization is not implemented')
        if self.natorb:
            raise NotImplementedError('GPU CASCI natural orbitals are not implemented')
        log = logger.new_logger(self, verbose)
        t0 = log.init_timer()
        out = super().kernel(mo_coeff, ci0, verbose)
        log.timer('GPU CASCI', *t0)
        return out
