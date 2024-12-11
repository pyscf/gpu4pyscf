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
from pyscf.pbc.dft import multigrid
from gpu4pyscf.pbc.scf import uhf as pbcuhf
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import return_cupy_array, tag_array
from gpu4pyscf.dft import uks as mol_uks
from gpu4pyscf.pbc.dft import rks

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpt=None, kpts_band=None):
    '''Coulomb + XC functional for UKS.  See pyscf/pbc/dft/uks.py
    :func:`get_veff` fore more details.
    '''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpt is None: kpt = ks.kpt
    log = logger.new_logger(ks)
    t0 = log.init_timer()

    ni = ks._numint
    hybrid = ni.libxc.is_hybrid_xc(ks.xc)

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        if ks.do_nlc():
            raise NotImplementedError(f'MultiGrid for NLC functional {ks.xc} + {ks.nlc}')
        n, exc, vxc = multigrid.nr_uks(ks.with_df, ks.xc, dm, hermi,
                                       kpt.reshape(1,3), kpts_band,
                                       with_j=True, return_j=False)
        log.info('nelec by numeric integration = %s', n)
        t0 = log.timer('vxc', *t0)
        return vxc

    if dm.ndim == 2:  # RHF DM
        dm = cp.repeat(dm[None]*.5, 2, axis=0)

    # ndim = 3 : dm.shape = ([alpha,beta], nao, nao)
    ground_state = (dm.ndim == 3 and dm.shape[0] == 2 and kpts_band is None)
    ks.initialize_grids(cell, dm, kpt, ground_state)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_uks(cell, ks.grids, ks.xc, dm, 0, hermi,
                                kpt, kpts_band, max_memory=max_memory)
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(cell, ks.nlcgrids, xc, dm[0]+dm[1],
                                          0, hermi, kpt, max_memory=max_memory)
            exc += enlc
            vxc += vnlc
        log.info('nelec by numeric integration = %s', n)
        t0 = log.timer('vxc', *t0)

    if not hybrid:
        vj = ks.get_j(cell, dm[0]+dm[1], hermi, kpt, kpts_band)
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
        vj = vj[0] + vj[1]
        vxc += vj
        vxc -= vk

        if ground_state:
            exc -=(cp.einsum('ij,ji->', dm[0], vk[0]) +
                   cp.einsum('ij,ji->', dm[1], vk[1])).real * .5

    if ground_state:
        ecoul = cp.einsum('nij,ji->', dm, vj).real * .5
    else:
        ecoul = None

    vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc


class UKS(rks.KohnShamDFT, pbcuhf.UHF):
    '''UKS class adapted for PBCs.

    This is a literal duplication of the molecular UKS class with some `mol`
    variables replaced by `cell`.
    '''

    def __init__(self, cell, kpt=np.zeros(3), xc='LDA,VWN', exxdiv='ewald'):
        pbcuhf.UHF.__init__(self, cell, kpt, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    dump_flags = uks_cpu.UKS.dump_flags

    get_rho = return_cupy_array(uks_cpu.get_rho)
    get_veff = get_veff
    energy_elec = mol_uks.energy_elec

    nuc_grad_method = NotImplemented
    to_hf = NotImplemented

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        mf = uks_cpu.UKS(self.cell)
        utils.to_cpu(self, out=mf)
        return mf
