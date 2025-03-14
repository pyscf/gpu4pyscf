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
Non-relativistic Restricted Kohn-Sham for periodic systems at a single k-point
'''

__all__ = [
    'RKS', 'KohnShamDFT',
]

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.dft import rks as rks_cpu
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.dft import rks as mol_ks
from gpu4pyscf.pbc.scf import hf as pbchf, khf
from gpu4pyscf.pbc.dft import gen_grid
from gpu4pyscf.pbc.dft import numint
from gpu4pyscf.pbc.dft import multigrid
from gpu4pyscf.lib.cupy_helper import return_cupy_array, tag_array
from pyscf import __config__

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpt=None, kpts_band=None):
    '''Coulomb + XC functional

    .. note::
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Returns:
        matrix Veff = J + Vxc.  Veff can be a list matrices, if the input
        dm is a list of density matrices.
    '''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpt is None: kpt = ks.kpt
    log = logger.new_logger(ks)
    t0 = log.init_timer()

    ni = ks._numint
    hybrid = ni.libxc.is_hybrid_xc(ks.xc)

    if isinstance(ni, multigrid.MultiGridNumInt):
        if ks.do_nlc():
            raise NotImplementedError(f'MultiGrid for NLC functional {ks.xc} + {ks.nlc}')
        n, exc, vxc = ni.nr_rks(
            cell, ks.grids, ks.xc, dm, 0, hermi, kpt, kpts_band, with_j=True)
        log.debug('nelec by numeric integration = %s', n)
        if hybrid:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
            if omega == 0:
                vk = ks.get_k(cell, dm, hermi, kpt, kpts_band)
                vk *= hyb
            elif alpha == 0: # LR=0, only SR exchange
                vk = ks.get_k(cell, dm, hermi, kpt, kpts_band, omega=-omega)
                vk *= hyb
            elif hyb == 0: # SR=0, only LR exchange
                vk = ks.get_k(cell, dm, hermi, kpt, kpts_band, omega=omega)
                vk *= alpha
            else: # SR and LR exchange with different ratios
                vk = ks.get_k(cell, dm, hermi, kpt, kpts_band)
                vk *= hyb
                vklr = ks.get_k(cell, dm, hermi, kpt, kpts_band, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
            vxc -= vk * .5
            exc -= cp.einsum('ij,ji->', dm, vk).real * .5 * .5
        t0 = log.timer('veff', *t0)
        return vxc

    ground_state = (isinstance(dm, cp.ndarray) and dm.ndim == 2
                    and kpts_band is None)
    ks.initialize_grids(cell, dm, kpt, ground_state)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        n, exc, vxc = ni.nr_rks(cell, ks.grids, ks.xc, dm, 0, hermi,
                                kpt, kpts_band)
        log.debug('nelec by numeric integration = %s', n)
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(cell, ks.nlcgrids, xc, dm, 0, hermi, kpt)
            exc += enlc
            vxc += vnlc
            log.debug('nelec with nlc grids = %s', n)
        log.timer('vxc', *t0)

    if not hybrid:
        vj = ks.get_j(cell, dm, hermi, kpt, kpts_band)
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
        if omega == 0:
            vj, vk = ks.get_jk(cell, dm, hermi, kpt, kpts_band)
            vk *= hyb
        elif alpha == 0: # LR=0, only SR exchange
            vj = ks.get_j(cell, dm, hermi, kpt, kpts_band)
            vk = ks.get_k(cell, dm, hermi, kpt, kpts_band, omega=-omega)
            vk *= hyb
        elif hyb == 0: # SR=0, only LR exchange
            vj = ks.get_j(cell, dm, hermi, kpt, kpts_band)
            vk = ks.get_k(cell, dm, hermi, kpt, kpts_band, omega=omega)
            vk *= alpha
        else: # SR and LR exchange with different ratios
            vj, vk = ks.get_jk(cell, dm, hermi, kpt, kpts_band)
            vk *= hyb
            vklr = ks.get_k(cell, dm, hermi, kpt, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vxc += vj
        vxc -= vk * .5

        if ground_state:
            exc -= cp.einsum('ij,ji->', dm, vk).real * .5 * .5

    if ground_state:
        ecoul = cp.einsum('ij,ji->', dm, vj).real * .5
    else:
        ecoul = None

    log.timer('veff', *t0)
    vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

NELEC_ERROR_TOL = getattr(__config__, 'pbc_dft_rks_prune_error_tol', 0.02)
def prune_small_rho_grids_(mf, cell, dm, grids, kpts):
    rho = mf.get_rho(dm, grids, kpts)
    n = rho.dot(grids.weights)
    if abs(n-cell.nelectron) < NELEC_ERROR_TOL*n:
        rho *= grids.weights
        size0 = grids.weights.size
        idx = abs(rho) > mf.small_rho_cutoff / size0
        grids.coords  = grids.coords [idx]
        grids.weights = grids.weights[idx]
        logger.debug(mf, 'Drop grids %d', size0 - grids.weights.size)
    return grids

class KohnShamDFT(mol_ks.KohnShamDFT):
    '''PBC-KS'''

    _keys = rks_cpu.KohnShamDFT._keys

    small_rho_cutoff = getattr(
        __config__, 'dft_rks_RKS_small_rho_cutoff', 1e-7)

    def __init__(self, xc='LDA,VWN'):
        self.xc = xc
        self.grids = gen_grid.UniformGrids(self.cell)
        self.nlc = ''
        self.nlcgrids = gen_grid.UniformGrids(self.cell)
        if isinstance(self, khf.KSCF):
            self._numint = numint.KNumInt()
        else:
            self._numint = numint.NumInt()

    build = rks_cpu.KohnShamDFT.build
    reset = rks_cpu.KohnShamDFT.reset
    dump_flags = rks_cpu.KohnShamDFT.dump_flags

    get_veff = NotImplemented
    get_rho = NotImplemented

    def density_fit(self, auxbasis=None, with_df=None):
        from gpu4pyscf.pbc.df.df_jk import density_fit
        cell = self.cell
        mf = density_fit(self, auxbasis, with_df)
        mf.with_df._j_only = not self._numint.libxc.is_hybrid_xc(self.xc)
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.level = getattr(
            __config__, 'dft_rks_RKS_grids_level', mf.grids.level)
        mf.nlcgrids = gen_grid.BeckeGrids(cell)
        mf.nlcgrids.level = getattr(
            __config__, 'dft_rks_RKS_nlcgrids_level', mf.nlcgrids.level)
        return mf

    rs_density_fit = NotImplemented

    jk_method = NotImplemented

    to_rks = NotImplemented
    to_uks = NotImplemented
    to_gks = NotImplemented
    to_hf = NotImplemented

    def initialize_grids(self, cell, dm, kpts, ground_state=True):
        '''Initialize self.grids the first time call get_veff'''
        if self.grids.coords is None:
            t0 = (logger.process_clock(), logger.perf_counter())
            self.grids.build()
            if (isinstance(self.grids, gen_grid.BeckeGrids) and
                self.small_rho_cutoff > 1e-20 and ground_state):
                self.grids = prune_small_rho_grids_(
                    self, self.cell, dm, self.grids, kpts)
            t0 = logger.timer(self, 'setting up grids', *t0)
        is_nlc = self.do_nlc()
        if is_nlc and self.nlcgrids.coords is None:
            t0 = (logger.process_clock(), logger.perf_counter())
            self.nlcgrids.build()
            if (isinstance(self.grids, gen_grid.BeckeGrids) and
                self.small_rho_cutoff > 1e-20 and ground_state):
                self.nlcgrids = prune_small_rho_grids_(
                    self, self.cell, dm, self.nlcgrids, kpts)
            t0 = logger.timer(self, 'setting up nlc grids', *t0)
        return self

# Update the KohnShamDFT label in pbc.scf.hf module
pbchf.KohnShamDFT = KohnShamDFT


def get_rho(mf, dm=None, grids=None, kpt=None):
    if dm is None: dm = mf.make_rdm1()
    if grids is None: grids = mf.grids
    if kpt is None: kpt = mf.kpt
    assert dm.ndim == 2
    assert kpt.ndim == 1
    return mf._numint.get_rho(mf.cell, dm[None], grids, kpt[None])

class RKS(KohnShamDFT, pbchf.RHF):
    '''RKS class adapted for PBCs.

    This is a literal duplication of the molecular RKS class with some `mol`
    variables replaced by `cell`.
    '''

    def __init__(self, cell, kpt=np.zeros(3), xc='LDA,VWN', exxdiv='ewald'):
        pbchf.RHF.__init__(self, cell, kpt, exxdiv=exxdiv)
        KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        pbchf.RHF.dump_flags(self, verbose)
        KohnShamDFT.dump_flags(self, verbose)
        return self

    def get_hcore(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        if isinstance(self._numint, multigrid.MultiGridNumInt):
            ni = self._numint
        else:
            ni = self.with_df
        if cell.pseudo:
            nuc = ni.get_pp(kpt)
        else:
            nuc = ni.get_nuc(kpt)
        if len(cell._ecpbas) > 0:
            raise NotImplementedError('ECP in PBC SCF')
        return nuc + cp.asarray(cell.pbc_intor('int1e_kin', 1, 1, kpt))

    get_veff = get_veff
    energy_elec = mol_ks.energy_elec
    get_rho = get_rho

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        mf = rks_cpu.RKS(self.cell)
        utils.to_cpu(self, out=mf)
        return mf
