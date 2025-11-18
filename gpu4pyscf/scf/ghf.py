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

def _from_rhf_init_dm(dma, breaksym=True):
    dma = dma * .5
    dm = block_diag(dma, dma)
    if breaksym:
        nao = dma.shape[0]
        idx, idy = cp.diag_indices(nao)
        dm[idx+nao,idy] = dm[idx,idy+nao] = dma.diagonal() * .05
    return dm


def get_jk(mol, dm, hermi=0, with_j=True, with_k=True, jkbuild=None, omega=None):
    '''
    GHF J/K adapter function.
    It takes a GHF DM, splits it into blocks, calls the 'jkbuild' strategy,
    and then reassembles the result into GHF matrices.
    
    The 'jkbuild' function is expected to compute J and K for
    a stack of density matrices.
    '''
    if jkbuild is None:
        raise ValueError("jkbuild (J/K build strategy) must be provided.")

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
        # --- Prepare DMs ---
        # Stack all real components: J-DM, then K-DMs
        dm_j_real = (dmaa + dmbb).real
        dm_k_real = cp.vstack((dmaa.real, dmbb.real, dmab.real, dmba.real))
        dms_real_in = cp.vstack((dm_j_real, dm_k_real))
        
        # Stack all imaginary components
        dm_j_imag = (dmaa + dmbb).imag
        dm_k_imag = cp.vstack((dmaa.imag, dmbb.imag, dmab.imag, dmba.imag))
        dms_imag_in = cp.vstack((dm_j_imag, dm_k_imag))

        # --- Call builder (real part) ---
        jtmp_real, ktmp_real = jkbuild(mol, dms_real_in, hermi=0, omega=omega)
        
        # --- Call builder (imaginary part) ---
        jtmp_imag, ktmp_imag = jkbuild(mol, dms_imag_in, hermi=0, omega=omega)
        
        # --- Reassemble ---
        if with_j:
            jtmp = jtmp_real[0] + 1j * jtmp_imag[0] # J is from the first DM in the stack
            vj = cp.zeros((n_dm, nso, nso), dtype=dm.dtype)
            vj[:, :nao, :nao] = vj[:, nao:, nao:] = jtmp
            vj = vj.reshape(dm_shape)
            
        if with_k:
            # K is from the last 4 DMs in the stack
            ktmp_r = ktmp_real[1:].reshape(4, n_dm, nao, nao) 
            ktmp_i = ktmp_imag[1:].reshape(4, n_dm, nao, nao)
            vk = cp.zeros((n_dm, nso, nso), dtype=dm.dtype)
            vk[:, :nao, :nao] = ktmp_r[0] + 1j * ktmp_i[0]
            vk[:, nao:, nao:] = ktmp_r[1] + 1j * ktmp_i[1]
            vk[:, :nao, nao:] = ktmp_r[2] + 1j * ktmp_i[2]
            vk[:, nao:, :nao] = ktmp_r[3] + 1j * ktmp_i[3]
            vk = vk.reshape(dm_shape)

    else: # Real DM
        dm_j = (dmaa + dmbb)
        dm_k = cp.vstack((dmaa, dmbb, dmab, dmba))
        
        # Stack all DMs for J and K
        dms_in = cp.vstack((dm_j, dm_k))
        
        # Call builder once
        jtmp, ktmp = jkbuild(mol, dms_in, hermi=0, omega=omega)

        # --- Reassemble ---
        if with_j:
            vj = cp.zeros((n_dm, nso, nso), dtype=dm.dtype)
            vj[:, :nao, :nao] = vj[:, nao:, nao:] = jtmp[0] # J from first DM
            vj = vj.reshape(dm_shape)
        
        if with_k:
            vk = cp.zeros((n_dm, nso, nso), dtype=dm.dtype)
            # K from the last 4 DMs
            vk[:, :nao, :nao] = ktmp[1]
            vk[:, nao:, nao:] = ktmp[2]
            vk[:, :nao, nao:] = ktmp[3]
            vk[:, nao:, :nao] = ktmp[4]
            vk = vk.reshape(dm_shape)
            
    return vj, vk


class GHF(hf.SCF):
    to_gpu = utils.to_gpu
    device = utils.device

    with_soc = None
    _keys = {'with_soc'}

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
    density_fit             = hf.RHF.density_fit
    # TODO: Enable followings after testing
    analyze = NotImplemented
    stability = NotImplemented
    mulliken_pop = NotImplemented
    mulliken_meta = NotImplemented
    spin_square = NotImplemented
    #TODO: uhf._finalize depends on spin_square function
    #_finalize = ghf_cpu.GHF._finalize

    get_grad = return_cupy_array(ghf_cpu.GHF.get_grad)
    energy_elec = hf.energy_elec

    def PCM(self, *args, **kwargs):
        '''
        Solvent models are not yet implemented for GHF.
        '''
        raise NotImplementedError('Solvent models are not implemented for GHF.')

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        dma = hf.RHF.get_init_guess(self, mol, key, **kwargs)
        return _from_rhf_init_dm(dma)

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
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        
        # Define the local "jkbuild" strategy for non-DF calculation
        # This strategy points to the base class (hf.SCF) implementation
        def jkbuild(mol_obj, dm_obj, hermi, omega=None):
            # The base class get_jk expects (n_dm, nao, nao)
            nao = mol_obj.nao
            dm_obj = dm_obj.reshape(-1, nao, nao)
            # Call super() to get the non-DF J/K
            return super(GHF, self).get_jk(mol_obj, dm_obj, hermi, 
                                           with_j=True, with_k=True, omega=omega)

        # Call the top-level adapter with the non-DF strategy
        return get_jk(mol, dm, hermi, with_j, with_k, jkbuild=jkbuild, omega=omega)

    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        vj, _ = self.get_jk(mol, dm, hermi, with_j=True, with_k=False, omega=omega)
        return vj

    def get_k(self, mol=None, dm=None, hermi=1, omega=None):
        _, vk = self.get_jk(mol, dm, hermi, with_j=False, with_k=True, omega=omega)
        return vk

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        if dm is None: dm = self.make_rdm1()
        if dm_last is not None and self.direct_scf:
            dm = asarray(dm) - asarray(dm_last)
        
        vj, vk = self.get_jk(mol, dm, hermi)
        
        vhf = vj - vk
        if vhf_last is not None:
            vhf += asarray(vhf_last)
        return vhf

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        e_idx = cp.argsort(mo_energy.round(9))
        nmo = mo_energy.size
        mo_occ = cp.zeros_like(mo_energy)
        nocc = self.mol.nelectron
        mo_occ[e_idx[:nocc]] = 1
        if self.verbose >= logger.INFO and nocc < nmo:
            homo = float(mo_energy[e_idx[nocc-1]])
            lumo = float(mo_energy[e_idx[nocc]])
            if homo+1e-3 > lumo:
                logger.warn(self, 'HOMO %.15g == LUMO %.15g', homo, lumo)
            else:
                logger.info(self, '  HOMO = %.15g  LUMO = %.15g', homo, lumo)
        # TODO: depends on spin_square implmentation
        #if mo_coeff is not None and mf.verbose >= logger.DEBUG:
        #    ss, s = mf.spin_square(mo_coeff[:,mo_occ>0], mf.get_ovlp())
        #    logger.debug(mf, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
        return mo_occ

    def to_cpu(self):
        mf = ghf_cpu.GHF(self.mol)
        utils.to_cpu(self, out=mf)
        return mf
    
    def x2c1e(self):
        '''X2C with spin-orbit coupling effects.
        '''
        from gpu4pyscf.x2c.x2c import x2c1e_ghf
        return x2c1e_ghf(self)
    
    x2c = x2c1e
