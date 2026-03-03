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
Unrestricted Kohn-Sham for periodic systems at a single k-point
'''

__all__ = [
    'UKS',
]

import numpy as np
import cupy as cp
import pyscf.dft
from pyscf import lib
from pyscf.pbc.dft import uks as uks_cpu
from gpu4pyscf.pbc.scf import uhf as pbcuhf
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import tag_array, get_avail_mem
from gpu4pyscf.dft import uks as mol_uks
from gpu4pyscf.pbc.dft import rks
from gpu4pyscf.pbc.dft import multigrid, multigrid_v2


def get_veff(ks, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
             kpt=None, kpts_band=None):
    '''Coulomb + XC functional for UKS.  See pyscf/pbc/dft/uks.py
    :func:`get_veff` fore more details.
    '''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpt is None:
        kpt = ks.kpt
    log = logger.new_logger(ks)
    t0 = log.init_timer()
    mem_avail = get_avail_mem()
    log.debug1('available GPU memory for rks.get_veff: %.3f GB', mem_avail/1e9)

    if dm.ndim == 2:  # RHF DM
        dm = cp.repeat(dm[None]*.5, 2, axis=0)

    assert hermi != 2
    ground_state = kpts_band is None
    ni = ks._numint
    hybrid = ni.libxc.is_hybrid_xc(ks.xc)

    if isinstance(ni, (multigrid_v2.MultiGridNumInt, multigrid.MultiGridNumInt)):
        if ks.do_nlc():
            raise NotImplementedError(f'MultiGrid for NLC functional {ks.xc} + {ks.nlc}')
        n, exc, vxc = ni.nr_uks(
            cell, ks.grids, ks.xc, dm, 0, hermi, kpt, kpts_band, with_j=True)
        log.debug('nelec by numeric integration = %s', n)
        j_in_xc = True
        ecoul = vxc.ecoul
    else:
        j_in_xc = False
        ks.initialize_grids(cell, dm, kpt)
        n, exc, vxc = ni.nr_uks(cell, ks.grids, ks.xc, dm, 0, hermi, kpt, kpts_band)
        if ks.do_nlc():
            warning_message = "ATTENTION!!! VV10 is only valid for open boundary, and it is incorrect for actual periodic system! " \
                              "Lattice summation is not performed for the double integration. " \
                              "Please use only under open boundary, i.e. neighbor images are well separated, and " \
                              "all atoms belonging to one image is placed in the same image in the input."
            log.warn(warning_message)
            print(warning_message) # This is an important warning, so print even if verbose == 0.

            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(cell, ks.nlcgrids, xc, dm[0]+dm[1],
                                          0, hermi, kpt)
            exc += enlc
            vxc += vnlc
        log.debug('nelec by numeric integration = %s', n)
        log.timer('vxc', *t0)

    vj, vk = rks._get_jk(ks, cell, dm, hermi, kpt, kpts_band, not j_in_xc,
                         dm_last, vhf_last)
    if not j_in_xc:
        vxc += vj[0] + vj[1]
        ecoul = None
        if ground_state:
            ecoul = float(cp.einsum('nij,mji->', dm, vj).real.get()) * .5
    if hybrid:
        vxc -= vk
        if ground_state:
            exc -= float(cp.einsum('nij,nji->', dm, vk).real.get()) * .5
    vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    logger.timer(ks, 'veff', *t0)
    return vxc


class UKS(rks.KohnShamDFT, pbcuhf.UHF):
    '''UKS class adapted for PBCs.

    This is a literal duplication of the molecular UKS class with some `mol`
    variables replaced by `cell`.
    '''

    def __init__(self, cell, kpt=None, xc='LDA,VWN', exxdiv='ewald'):
        pbcuhf.UHF.__init__(self, cell, kpt, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        pbcuhf.UHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_hcore = rks.RKS.get_hcore
    get_veff = get_veff
    energy_elec = mol_uks.energy_elec
    density_fit = rks.RKS.density_fit
    to_hf = NotImplemented
    multigrid_numint = rks.RKS.multigrid_numint

    def get_rho(self, dm=None, grids=None, kpt=None):
        if dm is None: dm = self.make_rdm1()
        return rks.get_rho(self, dm[0]+dm[1], grids, kpt)

    def Gradients(self):
        from gpu4pyscf.pbc.grad.uks import Gradients
        return Gradients(self)

    def to_cpu(self):
        mf = uks_cpu.UKS(self.cell)
        utils.to_cpu(self, out=mf)
        return mf
