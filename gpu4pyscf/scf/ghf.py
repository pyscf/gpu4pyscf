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
from pyscf.data.nist import HARTREE2EV
from gpu4pyscf.scf import hf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray, return_cupy_array, tag_array
from gpu4pyscf.lib import utils

def _from_rhf_init_dm(dma, breaksym=True):
    dma = dma * .5
    dm = block_diag(dma, dma)
    if breaksym:
        nao = dma.shape[0]
        idx, idy = cp.diag_indices(nao)
        dm[idx+nao,idy] = dm[idx,idy+nao] = dma.diagonal() * .05
    return dm


def _get_jk(mf, mol, dm, hermi=0, with_j=True, with_k=True, jkbuild=None,
            omega=None, lr_factor=None, sr_factor=None):
    '''
    GHF J/K adapter function.
    It takes a GHF DM, splits it into blocks, calls the 'jkbuild' strategy,
    and then reassembles the result into GHF matrices.
    
    The 'jkbuild' function is expected to compute J and K for
    a stack of density matrices.
    '''
    if jkbuild is None:
        raise ValueError("jkbuild (J/K build strategy) must be provided.")

    if mol is None: mol = mf.mol
    if dm is None: dm = mf.make_rdm1()

    dm = asarray(dm)
    dm_shape = dm.shape
    nso = dm.shape[-1]
    nao = nso // 2
    dms = dm.reshape(-1, nso, nso)
    n_dm = dms.shape[0]
    
    dmaa = dms[:, :nao, :nao]
    dmab = dms[:, :nao, nao:]
    dmba = dms[:, nao:, :nao]
    dmbb = dms[:, nao:, nao:]

    vj = vk = None # Initialize

    if dm.dtype == cp.complex128:
        if with_j:
            # --- Prepare DMs ---
            dm_j = cp.vstack((dmaa.real + dmbb.real,
                              dmaa.imag + dmbb.imag))
            jtmp = jkbuild(mf, mol, dm_j, hermi=0, with_k=False)[0]
            jtmp = jtmp[:n_dm] + 1j * jtmp[n_dm:]
            vj = cp.zeros((n_dm, nso, nso), dtype=dm.dtype)
            vj[:, :nao, :nao] = vj[:, nao:, nao:] = jtmp
            vj = vj.reshape(dm_shape)

        if with_k:
            dm_k = cp.stack((
                dmaa.real, dmbb.real, dmab.real, dmba.real,
                dmaa.imag, dmbb.imag, dmab.imag, dmba.imag))
            ktmp = jkbuild(mf, mol, dm_k, hermi=0, with_j=False, omega=omega,
                           lr_factor=lr_factor, sr_factor=sr_factor)[1]
            ktmp_r, ktmp_i = ktmp.reshape(2, 4, n_dm, nao, nao)
            vk = cp.zeros((n_dm, nso, nso), dtype=dm.dtype)
            vk[:, :nao, :nao] = ktmp_r[0] + 1j * ktmp_i[0]
            vk[:, nao:, nao:] = ktmp_r[1] + 1j * ktmp_i[1]
            vk[:, :nao, nao:] = ktmp_r[2] + 1j * ktmp_i[2]
            vk[:, nao:, :nao] = ktmp_r[3] + 1j * ktmp_i[3]
            vk = vk.reshape(dm_shape)

    else: # Real DM
        if with_j:
            dm_j = (dmaa + dmbb)
            jtmp = jkbuild(mf, mol, dm_j, hermi, with_k=False)[0]
            vj = cp.zeros((n_dm, nso, nso), dtype=dm.dtype)
            vj[:, :nao, :nao] = vj[:, nao:, nao:] = jtmp
            vj = vj.reshape(dm_shape)

        if with_k:
            dm_k = cp.vstack((dmaa, dmbb, dmab, dmba))
            ktmp = jkbuild(mf, mol, dm_k, hermi=0, with_j=False, omega=omega,
                           lr_factor=lr_factor, sr_factor=sr_factor)[1]
            vk = cp.zeros((n_dm, nso, nso), dtype=dm.dtype)
            # K from the last 4 DMs
            vk[:, :nao, :nao] = ktmp[0]
            vk[:, nao:, nao:] = ktmp[1]
            vk[:, :nao, nao:] = ktmp[2]
            vk[:, nao:, :nao] = ktmp[3]
            vk = vk.reshape(dm_shape)

    return vj, vk

def _get_jk_spin_free(mf, mol, dm, hermi, with_j=True, with_k=True,
                      omega=None, lr_factor=None, sr_factor=None):
    # The base class get_jk expects (n_dm, nao, nao)
    nao = mol.nao
    dm = dm.reshape(-1, nao, nao)
    vj = vk = None
    if with_j:
        vj = hf.SCF.get_j(mf, mol, dm, hermi)
    if with_k:
        vk = hf.SCF.get_k(mf, mol, dm, hermi, omega, lr_factor, sr_factor)
    return vj, vk


class GHF(hf.SCF):
    to_gpu = utils.to_gpu
    device = utils.device

    with_soc = None
    _keys = {'with_soc'}

    scf = kernel = hf.RHF.kernel
    make_rdm2 = NotImplemented
    newton = NotImplemented
    to_rhf = NotImplemented
    to_uhf = NotImplemented
    to_ghf = NotImplemented
    to_rks = NotImplemented
    to_uks = NotImplemented
    to_gks = NotImplemented
    to_ks = NotImplemented
    canonicalize = NotImplemented
    density_fit             = hf.RHF.density_fit
    # TODO: Enable followings after testing
    analyze = NotImplemented
    stability = NotImplemented
    mulliken_pop = NotImplemented
    mulliken_meta = NotImplemented

    get_grad = return_cupy_array(ghf_cpu.GHF.get_grad)
    energy_elec = hf.energy_elec

    def PCM(self, *args, **kwargs):
        '''
        Solvent models are not yet implemented for GHF.
        '''
        raise NotImplementedError('Solvent models are not implemented for GHF.')

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        dm = hf.RHF.get_init_guess(self, mol, key, **kwargs)
        key = key.lower()
        if key != 'hcore' and key != '1e':
            dm = _from_rhf_init_dm(dm)
        return dm

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
               omega=None, lr_factor=None, sr_factor=None):
        return _get_jk(self, mol, dm, hermi, with_j, with_k,
                       jkbuild=_get_jk_spin_free, omega=omega,
                       lr_factor=lr_factor, sr_factor=sr_factor)

    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        vj, _ = self.get_jk(mol, dm, hermi, with_j=True, with_k=False, omega=omega)
        return vj

    def get_k(self, mol=None, dm=None, hermi=1,
              omega=None, lr_factor=None, sr_factor=None):
        _, vk = self.get_jk(mol, dm, hermi, with_j=False, with_k=True,
                            omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)
        return vk

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        if dm is None: dm = self.make_rdm1()
        if dm_last is not None and self.direct_scf:
            assert vhf_last is not None
            dm_last = cp.asarray(dm_last)
            dm = cp.asarray(dm) - dm_last
        else:
            dm_last = None
            
        vj, vk = self.get_jk(mol, dm, hermi)
        vhf = vj - vk
        
        ecoul = hf._trace_ecoul(vj, dm, dm_last, vhf_last)
        if dm_last is not None:
            vhf += cp.asarray(vhf_last)
        if ecoul is not None:
            vhf = tag_array(vhf, ecoul=ecoul)
        return vhf

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        e_idx = cp.argsort(mo_energy.round(9))
        nmo = mo_energy.size
        mo_occ = cp.zeros_like(mo_energy)
        nocc = self.mol.nelectron
        
        if nocc > nmo:
            raise RuntimeError(f'Failed to assign mo_occ. Nocc ({nocc}) > Nmo ({nmo})')
            
        mo_occ[e_idx[:nocc]] = 1
        
        if nocc < nmo:
            homo, lumo = mo_energy[e_idx[nocc-1:nocc+1]].get()
            gap = (lumo - homo) * HARTREE2EV
            self.scf_summary['gap'] = gap
            if self.verbose >= logger.INFO:
                if homo+1e-3 > lumo:
                    logger.warn(self, 'HOMO %.15g == LUMO %.15g', homo, lumo)
                else:
                    logger.info(self, '  HOMO = %.15g  LUMO = %.15g  gap/eV = %.5f',
                                homo, lumo, gap)
        elif nocc > nmo:
            raise RuntimeError(f'Failed to assign mo_occ. Nocc ({nocc}) > Nmo ({nmo})')
        mo_occ[e_idx[:nocc]] = 1

        if mo_coeff is not None and self.verbose >= logger.DEBUG:
            ss, s = self.spin_square(mo_coeff[:,mo_occ>0], self.get_ovlp())
            logger.debug(self, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
        return mo_occ

    def spin_square(self, mo, s=None):
        nao = mo.shape[0] // 2
        if s is not None:
            s = s[:nao,:nao]
        mo_a = mo[:nao]
        mo_b = mo[nao:]
        saa = mo_a.conj().T.dot(s).dot(mo_a)
        sbb = mo_b.conj().T.dot(s).dot(mo_b)
        sab = mo_a.conj().T.dot(s).dot(mo_b)
        sba = sab.conj().T
        nocc_a = saa.trace().real
        nocc_b = sbb.trace().real
        ssxy = (nocc_a+nocc_b) * .5
        ssxy+= (sba.trace() * sab.trace() - cp.einsum('ij,ji->', sba, sab)).real
        ssz  = (nocc_a+nocc_b) * .25
        ssz += (nocc_a-nocc_b)**2 * .25
        tmp  = saa - sbb
        ssz -= cp.einsum('ij,ji->', tmp, tmp).real * .25
        ss = float(ssxy.get()) + ssz
        s = (ss+.25)**.5 - .5
        return ss, s*2+1

    def to_cpu(self):
        mf = ghf_cpu.GHF(self.mol)
        utils.to_cpu(self, out=mf)
        return mf

    def x2c1e(self):
        from gpu4pyscf.x2c.x2c import x2c1e_ghf
        return x2c1e_ghf(self)
    x2c = x2c1e
    sfx2c1e = NotImplemented
