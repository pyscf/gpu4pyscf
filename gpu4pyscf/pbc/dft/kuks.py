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
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import tag_array, get_avail_mem
from gpu4pyscf.pbc.scf import khf, kuhf
from gpu4pyscf.pbc.dft import rks, krks
from gpu4pyscf.pbc.dft import multigrid, multigrid_v2

def get_veff(ks, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
             kpts=None, kpts_band=None):
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    log = logger.new_logger(ks)
    t0 = log.init_timer()
    mem_avail = get_avail_mem()
    log.debug1('available GPU memory for kuks.get_veff: %.3f GB', mem_avail/1e9)

    assert hermi != 2
    ground_state = kpts_band is None
    ni = ks._numint
    hybrid = ni.libxc.is_hybrid_xc(ks.xc)
    nkpts = len(kpts)
    weight = 1. / nkpts

    if isinstance(ni, (multigrid_v2.MultiGridNumInt, multigrid.MultiGridNumInt)):
        if ks.do_nlc():
            raise NotImplementedError(f'MultiGrid for NLC functional {ks.xc} + {ks.nlc}')
        n, exc, vxc = ni.nr_uks(
            cell, ks.grids, ks.xc, dm, 0, hermi, kpts, kpts_band, with_j=True)
        log.debug('nelec by numeric integration = %s', n)
        j_in_xc = True
        ecoul = vxc.ecoul
    else:
        j_in_xc = False
        ks.initialize_grids(cell, dm, kpts)
        n, exc, vxc = ni.nr_uks(cell, ks.grids, ks.xc, dm, 0, hermi, kpts, kpts_band)
        if ks.do_nlc():
            raise NotImplementedError("VV10 not implemented for periodic system")
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(cell, ks.nlcgrids, xc, dm[0]+dm[1],
                                          0, hermi, kpts)
            exc += enlc
            vxc += vnlc
        log.debug('nelec by numeric integration = %s', n)
        log.timer('vxc', *t0)

    vj, vk = krks._get_jk(ks, cell, dm, hermi, kpts, kpts_band, not j_in_xc,
                          dm_last, vhf_last)
    if not j_in_xc:
        vxc = vxc + vj[0] + vj[1]
        ecoul = None
        if ground_state:
            ecoul = float(cp.einsum('nKij,mKji->', dm, vj).real.get()) * .5 * weight
    if hybrid:
        vxc = vxc - vk
        if ground_state:
            exc -= float(cp.einsum('nKij,nKji->', dm, vk).real.get()) * .5 * weight
    vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    logger.timer(ks, 'veff', *t0)
    return vxc

def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf=None):
    if h1e_kpts is None: h1e_kpts = mf.get_hcore(mf.cell, mf.kpts)
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.cell, dm_kpts)

    weight = 1./len(h1e_kpts)
    e1 = weight * cp.einsum('kij,nkji->', h1e_kpts, dm_kpts).get()
    ecoul = vhf.ecoul
    exc = vhf.exc
    if isinstance(ecoul, cp.ndarray):
        ecoul = ecoul.get()
    if isinstance(exc, cp.ndarray):
        exc = exc.get()
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

    def __init__(self, cell, kpts=None, xc='LDA,VWN', exxdiv='ewald'):
        kuhf.KUHF.__init__(self, cell, kpts, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        kuhf.KUHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_hcore = krks.KRKS.get_hcore
    get_veff = get_veff
    energy_elec = energy_elec
    density_fit = krks.KRKS.density_fit

    def get_rho(self, dm=None, grids=None, kpts=None):
        if dm is None: dm = self.make_rdm1()
        return krks.get_rho(self, dm[0]+dm[1], grids, kpts)

    def Gradients(self):
        from gpu4pyscf.pbc.grad.kuks import Gradients
        return Gradients(self)

    to_hf = NotImplemented
    multigrid_numint = krks.KRKS.multigrid_numint

    def to_cpu(self):
        mf = kuks_cpu.KUKS(self.cell)
        utils.to_cpu(self, out=mf)
        return mf
