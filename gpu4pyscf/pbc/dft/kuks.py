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
Unrestricted Kohn-Sham for periodic systems with k-point sampling
'''

__all__ = [
    'KUKS',
]

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.dft import kuks as kuks_cpu
from pyscf.pbc.dft import multigrid
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import return_cupy_array, tag_array
from gpu4pyscf.pbc.scf import khf, kuhf
from gpu4pyscf.pbc.dft import rks

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    log = logger.new_logger(ks)
    t0 = log.init_timer()

    ni = ks._numint
    hybrid = ni.libxc.is_hybrid_xc(ks.xc)

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        if ks.do_nlc():
            raise NotImplementedError(f'MultiGrid for NLC functional {ks.xc} + {ks.nlc}')
        n, exc, vxc = multigrid.nr_uks(ks.with_df, ks.xc, dm, hermi,
                                       kpts, kpts_band,
                                       with_j=True, return_j=False)
        log.info('nelec by numeric integration = %s', n)
        t0 = log.timer('vxc', *t0)
        return vxc

    # ndim = 4 : dm.shape = ([alpha,beta], nkpts, nao, nao)
    ground_state = (dm.ndim == 4 and dm.shape[0] == 2 and kpts_band is None)
    ks.initialize_grids(cell, dm, kpts, ground_state)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_uks(cell, ks.grids, ks.xc, dm, 0, hermi,
                                kpts, kpts_band, max_memory=max_memory)
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(cell, ks.nlcgrids, xc, dm[0]+dm[1],
                                          0, hermi, kpts, max_memory=max_memory)
            exc += enlc
            vxc += vnlc
        log.info('nelec by numeric integration = %s', n)
        t0 = log.timer('vxc', *t0)

    nkpts = len(kpts)
    weight = 1. / nkpts

    if not hybrid:
        vj = ks.get_j(cell, dm[0]+dm[1], hermi, kpts, kpts_band)
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
        if omega == 0:
            vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
            vk *= hyb
        elif alpha == 0: # LR=0, only SR exchange
            vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
            vk = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=-omega)
            vk *= hyb
        elif hyb == 0: # SR=0, only LR exchange
            vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
            vk = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vk *= alpha
        else: # SR and LR exchange with different ratios
            vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
            vk *= hyb
            vklr = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vj = vj[0] + vj[1]
        vxc += vj
        vxc -= vk

        if ground_state:
            exc -= (cp.einsum('Kij,Kji->', dm[0], vk[0]) +
                    cp.einsum('Kij,Kji->', dm[1], vk[1])).real * .5 * weight

    if ground_state:
        ecoul = cp.einsum('nKij,Kji->', dm, vj) * .5 * weight
    else:
        ecoul = None

    vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf=None):
    if h1e_kpts is None: h1e_kpts = mf.get_hcore(mf.cell, mf.kpts)
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.cell, dm_kpts)

    weight = 1./len(h1e_kpts)
    e1 = weight *(cp.einsum('kij,kji', h1e_kpts, dm_kpts[0]) +
                  cp.einsum('kij,kji', h1e_kpts, dm_kpts[1]))
    ecoul = vhf.ecoul
    exc = vhf.exc
    tot_e = e1 + ecoul + exc
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['coul'] = ecoul.real
    mf.scf_summary['exc'] = exc.real
    logger.debug(mf, 'E1 = %s  Ecoul = %s  Exc = %s', e1, ecoul, exc)
    if abs(ecoul.imag) > mf.cell.precision*10:
        logger.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    ecoul.imag)
    return tot_e.real, ecoul.real + exc.real


class KUKS(rks.KohnShamDFT, kuhf.KUHF):
    '''UKS class adapted for PBCs with k-point sampling.
    '''

    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN', exxdiv='ewald'):
        kuhf.KUHF.__init__(self, cell, kpts, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    dump_flags = kuks_cpu.KUKS.dump_flags

    get_veff = get_veff
    energy_elec = energy_elec
    get_rho = return_cupy_array(kuks_cpu.get_rho)

    nuc_grad_method = NotImplemented
    to_hf = NotImplemented

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        mf = kuks_cpu.KUKS(self.cell)
        utils.to_cpu(self, out=mf)
        return mf
