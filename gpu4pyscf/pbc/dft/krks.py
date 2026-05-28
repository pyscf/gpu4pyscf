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
Kohn-Sham for periodic systems with k-point sampling
'''

__all__ = [
    'KRKS',
]

import cupy as cp
from pyscf import lib
from pyscf.pbc.dft import krks as krks_cpu
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import tag_array, get_avail_mem
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.scf import khf
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt
from gpu4pyscf.pbc.dft import rks
from gpu4pyscf.pbc.dft import multigrid, multigrid_v2

def get_veff(ks, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
             kpts=None, kpts_band=None):
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    log = logger.new_logger(ks)
    t0 = log.init_timer()

    assert hermi != 2
    ground_state = kpts_band is None
    ni = ks._numint
    hybrid = ni.libxc.is_hybrid_xc(ks.xc)
    nkpts = len(kpts)
    weight = 1. / nkpts

    if isinstance(ni, (multigrid_v2.MultiGridNumInt, multigrid.MultiGridNumInt)):
        if ks.do_nlc():
            raise NotImplementedError(f'MultiGrid for NLC functional {ks.xc} + {ks.nlc}')
        n, exc, vxc = ni.nr_rks(
            cell, ks.grids, ks.xc, dm, 0, hermi, kpts, kpts_band, with_j=True)
        log.debug('nelec by numeric integration = %s', n)
        j_in_xc = True
        ecoul = vxc.ecoul
    else:
        j_in_xc = False
        ks.initialize_grids(cell, dm, kpts)
        n, exc, vxc = ni.nr_rks(cell, ks.grids, ks.xc, dm, 0, hermi, kpts, kpts_band)
        log.debug('nelec by numeric integration = %s', n)
        if ks.do_nlc():
            raise NotImplementedError("VV10 not implemented for periodic system")
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(cell, ks.nlcgrids, xc, dm, 0, hermi, kpts)
            exc += enlc
            vxc += vnlc
            log.debug('nelec with nlc grids = %s', n)
        log.timer('vxc', *t0)

    vj, vk, vj_sr, vk_sr = _get_jk(
        ks, cell, dm, hermi, kpts, kpts_band, not j_in_xc, dm_last, vhf_last)
    if not j_in_xc:
        vxc = vxc + vj
        ecoul = None
        if ground_state:
            ecoul = cp.einsum('Kij,Kji->', dm, vj).get() * .5 * weight
    if hybrid:
        vxc = vxc - .5 * vk
        if ground_state:
            exc -= float(cp.einsum('Kij,Kji->', dm, vk).real.get()) * .25 * weight
    vxc = tag_array(vxc, ecoul=ecoul, exc=exc)
    if vj_sr is not None:
        vxc.vj = vj_sr
    if vk_sr is not None:
        vxc.vk = vk_sr
    logger.timer(ks, 'veff', *t0)
    return vxc

def _get_jk(mf, cell, dm, hermi, kpts, kpts_band=None, with_j=True,
            dm_last=None, vhf_last=None):
    '''J and Exx matrix. Note, Exx here is a scaled HF K term.'''
    ni = mf._numint
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    with_j = with_j and hermi != 2
    incremental_vj = dm_last is not None and getattr(vhf_last, 'vj', None) is not None
    incremental_vk = dm_last is not None and getattr(vhf_last, 'vk', None) is not None
    vj = vk = 0
    vj_sr = vk_sr = None
    if not hybrid:
        if with_j:
            if isinstance(mf.j_engine, (PBCJKMatrixOpt, PBCJMatrixOpt)):
                if mf.j_engine.supmol is None:
                    mf.j_engine.build(kpts)
                ddm = dm - dm_last if incremental_vj else dm
                vj_sr = mf.j_engine._get_j_sr(ddm, hermi, kpts, kpts_band)
                vj = mf.j_engine._get_j_lr(dm, hermi, kpts, kpts_band)
                if incremental_vj:
                    vj_sr += vhf_last.vj
                vj += vj_sr
            else:
                vj = mf.get_j(cell, dm, hermi, kpts, kpts_band)
        return vj, vk, vj_sr, vk_sr

    omega, lr_factor, sr_factor = ni.rsh_and_hybrid_coeff(mf.xc)
    if mf.rsjk:
        if with_j:
            if isinstance(mf.j_engine, (PBCJKMatrixOpt, PBCJMatrixOpt)):
                if mf.j_engine.supmol is None:
                    mf.j_engine.build(kpts)
                ddm = dm - dm_last if incremental_vj else dm
                vj_sr = mf.j_engine._get_j_sr(ddm, hermi, kpts, kpts_band)
                vj = mf.j_engine._get_j_lr(dm, hermi, kpts, kpts_band)
                if incremental_vj:
                    vj_sr += vhf_last.vj
                vj += vj_sr
            else:
                vj = mf.get_j(cell, dm, hermi, kpts, kpts_band)

        if mf.rsjk.supmol is None:
            mf.rsjk.build(kpts)
        ddm = dm - dm_last if incremental_vk else dm
        vk_sr = mf.rsjk._get_k_sr(ddm, hermi, kpts, kpts_band, mf.exxdiv,
                                  omega, lr_factor, sr_factor)
        vk = mf.rsjk._get_k_lr(dm, hermi, kpts, kpts_band, mf.exxdiv,
                               omega, lr_factor, sr_factor)
        if incremental_vk:
            vk_sr += vhf_last.vk
        vk += vk_sr
    else:
        #if getattr(mf.with_df, '_j_only', False):  # for GDF and MDF
        #    log.warn('df.j_only cannot be used with hybrid functional')
        #    mf.with_df._j_only = False
        #    # Rebuild df object due to the change of parameter _j_only
        #    if mf.with_df._cderi is not None:
        #        mf.with_df.build()
        if omega == 0:
            hyb = sr_factor
            vj, vk = mf.get_jk(cell, dm, hermi, kpts, kpts_band, with_j=with_j)
            vk *= hyb
        elif lr_factor == 0: # LR=0, only SR exchange
            if with_j:
                vj = mf.get_j(cell, dm, hermi, kpts, kpts_band)
            vk = mf.get_k(cell, dm, hermi, kpts, kpts_band, omega=-omega)
            vk *= sr_factor
        elif sr_factor == 0: # SR=0, only LR exchange
            if with_j:
                vj = mf.get_j(cell, dm, hermi, kpts, kpts_band)
            vk = mf.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vk *= lr_factor
        else: # SR and LR exchange with different ratios
            vj, vk = mf.get_jk(cell, dm, hermi, kpts, kpts_band, with_j=with_j)
            vk *= sr_factor
            vklr = mf.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vklr *= lr_factor - sr_factor
            vk += vklr
    return vj, vk, vj_sr, vk_sr

def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf=None):
    if h1e_kpts is None: h1e_kpts = mf.get_hcore(mf.cell, mf.kpts)
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.cell, dm_kpts)

    weight = 1./len(h1e_kpts)
    e1 = weight * cp.einsum('kij,kji->', h1e_kpts, dm_kpts).get()
    ecoul = vhf.ecoul
    exc = vhf.exc.real
    e2 = ecoul + exc
    tot_e = e1 + e2
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e2.real
    mf.scf_summary['coul'] = ecoul.real
    mf.scf_summary['exc'] = exc
    logger.debug(mf, 'E1 = %s  E2 = %s  Ecoul = %s  Exc = %s', e1, e2, ecoul, exc)
    if abs(ecoul.imag) > mf.cell.precision*10:
        logger.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    ecoul.imag)
    return tot_e.real, e2.real

class KRKS(rks.KohnShamDFT, khf.KRHF):
    '''RKS class adapted for PBCs with k-point sampling.
    '''

    def __init__(self, cell, kpts=None, xc='LDA,VWN', exxdiv='ewald'):
        khf.KRHF.__init__(self, cell, kpts, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        khf.KRHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def Gradients(self):
        from gpu4pyscf.pbc.grad.krks import Gradients
        return Gradients(self)

    get_veff = get_veff
    energy_elec = energy_elec
    get_rho = khf.KRHF.get_rho
    density_fit = khf.KRHF.density_fit

    to_hf = NotImplemented
    multigrid_numint = rks.RKS.multigrid_numint

    def to_cpu(self):
        mf = krks_cpu.KRKS(self.cell)
        utils.to_cpu(self, out=mf)
        return mf
