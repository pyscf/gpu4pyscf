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

import numpy as np
import cupy as cp
from cupyx.scipy.linalg import block_diag
from pyscf.lib import PauliMatrices
from pyscf.scf import ghf as ghf_cpu
from gpu4pyscf.scf import hf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray, return_cupy_array
from gpu4pyscf.lib import utils

class GHF(hf.SCF):
    to_gpu = utils.to_gpu
    device = utils.device

    with_soc = None
    _keys = {'with_soc'}

    _eigh = staticmethod(hf.eigh)
    scf = kernel = hf.RHF.kernel
    make_rdm2 = NotImplemented
    newton = NotImplemented
    x2c = x2c1e = sfx2c1e = NotImplemented
    to_rhf = NotImplemented
    to_uhf = NotImplemented
    to_ghf = NotImplemented
    to_rks = NotImplemented
    to_uks = NotImplemented
    to_gks = NotImplemented
    to_ks = NotImplemented
    canonicalize = NotImplemented
    # TODO: Enable followings after testing
    analyze = NotImplemented
    stability = NotImplemented
    mulliken_pop = NotImplemented
    mulliken_meta = NotImplemented
    spin_square = NotImplemented
    #TODO: uhf._finalize depends on spin_square function
    #_finalize = ghf_cpu.GHF._finalize

    get_grad = return_cupy_array(ghf_cpu.GHF.get_grad)

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        dma = hf.RHF.get_init_guess(self, mol, key, **kwargs)
        return block_diag(dma, dma)

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        htmp = hf.get_hcore(mol)
        hcore = block_diag(htmp, htmp)

        if self.with_soc and mol.has_ecp_soc():
            # The ECP SOC contribution = <|1j * s * U_SOC|>
            s = .5 * PauliMatrices
            ecpso = np.einsum('sxy,spq->xpyq', -1j * s, mol.intor('ECPso'))
            # Convert to complex array
            hcore = hcore + asarray(ecpso.reshape(hcore.shape))
        return hcore

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        stmp = hf.SCF.get_ovlp(self, mol)
        return block_diag(stmp, stmp)

    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
               omega=None):
        vj = vk = None
        if with_j:
            vj = self.get_j(mol, dm, hermi, omega)
        if with_k:
            vk = self.get_k(mol, dm, hermi, omega)
        return vj, vk

    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        dm = asarray(dm)
        dm_shape = dm.shape
        nso = dm.shape[-1]
        nao = nso // 2
        dm = dm.reshape(-1,nso,nso)
        n_dm = dm.shape[0]
        dm = dm[:,:nao,:nao] + dm[:,nao:,nao:]
        jtmp = hf.SCF.get_j(self, mol, dm, hermi, omega)
        vj = cp.zeros((n_dm,nso,nso))
        vj[:,:nao,:nao] = vj[:,nao:,nao:] = jtmp
        return vj.reshape(dm_shape)

    def get_k(self, mol=None, dm=None, hermi=1, omega=None):
        dm = asarray(dm)
        dm_shape = dm.shape
        nso = dm.shape[-1]
        nao = nso // 2
        dm = dm.reshape(-1,nso,nso)
        n_dm = dm.shape[0]
        dmaa = dm[:,:nao,:nao]
        dmbb = dm[:,nao:,nao:]
        dmab = dm[:,:nao,nao:]
        dmba = dm[:,nao:,:nao]
        dm = cp.vstack((dmaa, dmbb, dmab, dmba))
        ktmp = hf._get_jk(self, mol, dm, hermi=0, with_j=False, omega=omega)[1]
        ktmp = ktmp.reshape(4,n_dm,nao,nao)
        vk = cp.zeros((n_dm,nso,nso), dm.dtype)
        vk[:,:nao,:nao] = ktmp[0]
        vk[:,nao:,nao:] = ktmp[1]
        vk[:,:nao,nao:] = ktmp[2]
        vk[:,nao:,:nao] = ktmp[3]
        return vk.reshape(dm_shape)

    def get_veff(mf, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        if dm is None: dm = mf.make_rdm1()
        if dm_last is not None and mf.direct_scf:
            dm = asarray(dm) - asarray(dm_last)
        vhf = mf.get_j(mol, dm, hermi)
        vk = mf.get_k(mol, dm, hermi)
        vhf -= vk
        if vhf_last is not None:
            vhf += asarray(vhf_last)
        return vhf

    def get_occ(mf, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = mf.mo_energy
        e_idx = cp.argsort(mo_energy.round(9))
        nmo = mo_energy.size
        mo_occ = cp.zeros_like(mo_energy)
        nocc = mf.mol.nelectron
        mo_occ[e_idx[:nocc]] = 1
        if mf.verbose >= logger.INFO and nocc < nmo:
            homo = float(mo_energy[e_idx[nocc-1]])
            lumo = float(mo_energy[e_idx[nocc]])
            if homo+1e-3 > lumo:
                logger.warn(mf, 'HOMO %.15g == LUMO %.15g', homo, lumo)
            else:
                logger.info(mf, '  HOMO = %.15g  LUMO = %.15g', homo, lumo)
        # TODO: depends on spin_square implmentation
        #if mo_coeff is not None and mf.verbose >= logger.DEBUG:
        #    ss, s = mf.spin_square(mo_coeff[:,mo_occ>0], mf.get_ovlp())
        #    logger.debug(mf, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
        return mo_occ

    def to_cpu(self):
        mf = ghf_cpu.GHF(self.mol)
        utils.to_cpu(self, out=mf)
        return mf
