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
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.scf import hf as pbchf, khf
from gpu4pyscf.pbc.df.df import GDF
from gpu4pyscf.pbc.dft import gen_grid
from gpu4pyscf.pbc.dft import numint
from gpu4pyscf.pbc.dft import multigrid, multigrid_v2
from gpu4pyscf.lib.cupy_helper import tag_array, get_avail_mem
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
    mem_avail = get_avail_mem()
    log.debug1('available GPU memory for rks.get_veff: %.3f GB', mem_avail/1e9)

    ni = ks._numint
    hybrid = ni.libxc.is_hybrid_xc(ks.xc)

    if isinstance(ni, (multigrid_v2.MultiGridNumInt, multigrid.MultiGridNumInt)):
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
            exc -= cp.einsum('ij,ji->', dm, vk).get()[()] * .5 * .5
        log.timer_debug1('veff', *t0)
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
        log.timer_debug1('vxc', *t0)

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
            exc -= cp.einsum('ij,ji->', dm, vk).get()[()] * .5 * .5

    if ground_state:
        ecoul = cp.einsum('ij,ji->', dm, vj).get()[()] * .5
    else:
        ecoul = None

    log.timer_debug1('veff', *t0)
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

    def build(self, cell=None):
        # To handle the attribute kpt or kpts loaded from chkfile
        if 'kpts' in self.__dict__:
            self.kpts = self.__dict__.pop('kpts')
        elif 'kpt' in self.__dict__:
            self.kpt = self.__dict__.pop('kpt')

        kpts = self.kpts
        if self.rsjk:
            raise NotImplementedError('RSJK')

        # for GDF and MDF
        with_df = self.with_df
        if (isinstance(with_df, GDF) and
            self._numint.libxc.is_hybrid_xc(self.xc) and
            len(kpts) > 1 and getattr(with_df, '_j_only', False)):
            logger.warn(self, 'df.j_only cannot be used with hybrid functional')
            self.with_df._j_only = False
            self.with_df.reset()

        if isinstance(with_df, GDF):
            if isinstance(self.grids, gen_grid.UniformGrids):
                cell = self.cell
                logger.warn(cell, 'Uniform grids are used for the PBC GDF method. '
                            'Note: this differs from PySCF default settings which employ the Becke grids.')
                ngrids = np.prod(cell.mesh)
                if ngrids > 150000 * cell.natm:
                    logger.warn(cell, '''
Compact basis functions are found in the system. It is recommended to use Becke grids as that in PySCF:
    from gpu4pyscf.pbc.dft import BeckeGrids
    mf.grids = BeckeGrids(cell)
    mf.nlcgrids = BeckeGrids(cell).set(level=1)''')

        if self.verbose >= logger.WARN:
            self.check_sanity()
        return self

    def reset(self, cell=None):
        if cell is None:
            return self
        pbchf.SCF.reset(self, cell)
        self.grids.reset(cell)
        self.nlcgrids.reset(cell)
        if isinstance(self._numint, (multigrid.MultiGridNumInt, multigrid_v2.MultiGridNumInt)):
            self._numint.reset(cell)
        if hasattr(self, 'cphf_grids'):
            self.cphf_grids.reset(cell)
        return self

    dump_flags = rks_cpu.KohnShamDFT.dump_flags

    get_veff = NotImplemented
    get_rho = NotImplemented

    density_fit = NotImplemented
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

    to_gpu = utils.to_gpu
    device = utils.device
    to_cpu = NotImplemented

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

    def __init__(self, cell, kpt=None, xc='LDA,VWN', exxdiv='ewald'):
        pbchf.RHF.__init__(self, cell, kpt, exxdiv=exxdiv)
        KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        pbchf.RHF.dump_flags(self, verbose)
        KohnShamDFT.dump_flags(self, verbose)
        return self

    def get_hcore(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        if isinstance(self._numint, (multigrid.MultiGridNumInt, multigrid_v2.MultiGridNumInt)):
            ni = self._numint
        else:
            ni = self.with_df
        if cell.pseudo:
            nuc = ni.get_pp(kpt)
        else:
            nuc = ni.get_nuc(kpt)
        if len(cell._ecpbas) > 0:
            raise NotImplementedError('ECP in PBC SCF')
        t = int1e.int1e_kin(cell, kpt)
        return nuc + t

    get_veff = get_veff
    energy_elec = mol_ks.energy_elec
    get_rho = get_rho
    density_fit = pbchf.RHF.density_fit
    to_hf = NotImplemented

    def multigrid_numint(self, mesh=None):
        '''Apply the MultiGrid algorithm for XC numerical integartion'''
        mf = self.copy()
        mf._numint = multigrid.MultiGridNumInt(self.cell)
        if mesh is not None:
            mf._numint.mesh = mesh
        return mf

    def Gradients(self):
        from gpu4pyscf.pbc.grad.rks import Gradients
        return Gradients(self)

    def to_cpu(self):
        mf = rks_cpu.RKS(self.cell)
        utils.to_cpu(self, out=mf)
        return mf
