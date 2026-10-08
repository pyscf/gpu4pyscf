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

"""Generalized collinear Kohn-Sham for periodic systems at one k-point."""

__all__ = ['GKS']

import cupy as cp
from pyscf import lib, __config__
from pyscf.pbc.dft import gks as gks_cpu
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.dft import rks as mol_rks
from gpu4pyscf.pbc.scf import ghf
from gpu4pyscf.pbc.dft import rks, multigrid
from gpu4pyscf.pbc.dft.numint2c import NumInt2C


def get_veff(ks, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
             kpt=None, kpts_band=None):
    '''Coulomb + XC functional for GKS.'''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    dm = cp.asarray(dm)
    if kpt is None: kpt = ks.kpt
    t0 = (logger.process_clock(), logger.perf_counter())

    ni = ks._numint
    if ks.do_nlc():
        raise NotImplementedError(f'NLC functional {ks.xc} + {ks.nlc}')

    hybrid = ni.libxc.is_hybrid_xc(ks.xc)

    # TODO GKS with hybrid functional
    if hybrid:
        raise NotImplementedError('Hybrid functionals for periodic GKS')

    # TODO GKS with multigrid method
    if isinstance(ks._numint, multigrid.MultiGridNumIntBase):
        raise NotImplementedError('Multigrid for periodic GKS')

    # ndim = 2, dm.shape = (2*nao, 2*nao)
    ground_state = (dm.ndim == 2 and kpts_band is None)
    ks.initialize_grids(cell, dm, kpt, ground_state)

    # TODO: support non-symmetric density matrix
    assert (hermi == 1)
    dm = cp.asarray(dm)

    # ndim = 2, dm.shape = (2*nao, 2*nao)
    ground_state = (dm.ndim == 2 and kpts_band is None)

    # vxc = (vxc_aa, vxc_bb). vxc_ab is neglected in collinear DFT.
    max_memory = ks.max_memory - lib.current_memory()[0]
    ni = ks._numint
    n, exc, vxc = ni.get_vxc(cell, ks.grids, ks.xc, dm, hermi=hermi, kpt=kpt,
                             kpts_band=kpts_band, max_memory=max_memory)
    logger.info(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)

    if not hybrid:
        vj = ks.get_j(cell, dm, hermi, kpt, kpts_band)
        vxc += vj
    else:
        omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
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
        vxc += vj - vk

        if ground_state:
            exc -= cp.einsum('ij,ji', dm, vk).real * .5

    if ground_state:
        ecoul = float(cp.einsum('ij,ji', dm, vj).real) * .5
    else:
        ecoul = None

    vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc


class GKS(rks.KohnShamDFT, ghf.GHF):

    collinear = gks_cpu.GKS.collinear
    spin_samples = gks_cpu.GKS.spin_samples
    get_veff = get_veff
    energy_elec = mol_rks.energy_elec
    get_rho = ghf.GHF.get_rho
    density_fit = ghf.GHF.density_fit

    def __init__(self, cell, kpt=None, xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        ghf.GHF.__init__(self, cell, kpt, exxdiv)
        rks.KohnShamDFT.__init__(self, xc)
        self._numint = NumInt2C()

    def dump_flags(self, verbose=None):
        ghf.GHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def to_hf(self):
        return self._transfer_attrs_(ghf.GHF(self.cell, self.kpt))

    to_ghf = to_hf

    def to_gks(self, xc=None):
        '''Copy this periodic GKS object, optionally changing its functional.'''
        mf = self.copy()
        if xc is not None:
            mf.xc = xc
            mf.converged = xc == self.xc and self.converged
        return mf

    def to_cpu(self):
        return utils.to_cpu(self, out=gks_cpu.GKS(self.cell, self.kpt))
