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

import cupy
from pyscf.dft import uks as uks_cpu
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.dft import rks
from gpu4pyscf.scf import hf, uhf
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.lib import utils


def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional for UKS.  See pyscf/dft/rks.py
    :func:`get_veff` fore more details.
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    assert dm.ndim == 3
    t0 = logger.init_timer(ks)
    rks.initialize_grids(ks, mol, cupy.asarray(dm[0]+dm[1]))

    if hasattr(ks, 'screen_tol') and ks.screen_tol is not None:
        ks.direct_scf_tol = ks.screen_tol
    ground_state = getattr(dm, 'ndim', 0) == 3

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_uks(mol, ks.grids, ks.xc, dm.view(cupy.ndarray), max_memory=max_memory)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(mol, ks.nlcgrids, xc, dm[0]+dm[1],
                                          max_memory=max_memory)
            exc += enlc
            vxc += vnlc
            logger.debug(ks, 'nelec with nlc grids = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    if not ni.libxc.is_hybrid_xc(ks.xc):
        vk = None
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vj', None) is not None):
            dm_last = cupy.asarray(dm_last)
            dm = cupy.asarray(dm)
            assert dm_last.ndim == 0 or dm_last.ndim == dm.ndim
            ddm = dm - dm_last
            vj = ks.get_j(mol, ddm[0]+ddm[1], hermi)
            vj += vhf_last.vj
        else:
            vj = ks.get_j(mol, dm[0]+dm[1], hermi)
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vk', None) is not None):
            dm_last = cupy.asarray(dm_last)
            dm = cupy.asarray(dm)
            assert dm_last.ndim == 0 or dm_last.ndim == dm.ndim
            ddm = dm - dm_last
            vj, vk = ks.get_jk(mol, ddm, hermi)
            vk *= hyb
            if abs(omega) > 1e-10:  # For range separated Coulomb operator
                vklr = ks.get_k(mol, ddm, hermi, omega)
                vklr *= (alpha - hyb)
                vk += vklr
            vj = vj[0] + vj[1] + vhf_last.vj
            vk += vhf_last.vk
        else:
            vj, vk = ks.get_jk(mol, dm, hermi)
            vj = vj[0] + vj[1]
            vk *= hyb
            if abs(omega) > 1e-10:
                vklr = ks.get_k(mol, dm, hermi, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
        vxc += vj - vk

        if ground_state:
            exc -=(cupy.einsum('ij,ji', dm[0], vk[0]).real +
                   cupy.einsum('ij,ji', dm[1], vk[1]).real) * .5
    if ground_state:
        ecoul = cupy.einsum('ij,ji', dm[0]+dm[1], vj).real * .5
    else:
        ecoul = None
    t0 = logger.timer_debug1(ks, 'jk total', *t0)
    vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc


def energy_elec(ks, dm=None, h1e=None, vhf=None):
    if dm is None: dm = ks.make_rdm1()
    if h1e is None: h1e = ks.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)
    if not (isinstance(dm, cupy.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]
    return rks.energy_elec(ks, dm, h1e, vhf)


class UKS(rks.KohnShamDFT, uhf.UHF):
    def __init__(self, mol, xc='LDA,VWN'):
        uhf.UHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)

    get_veff = get_veff
    get_vasp = uks_cpu.get_vsap
    energy_elec = energy_elec
    energy_tot = hf.RHF.energy_tot
    init_guess_by_vsap = uks_cpu.UKS.init_guess_by_vsap

    to_hf = NotImplemented

    def reset(self, mol=None):
        hf.SCF.reset(self, mol)
        self.grids.reset(mol)
        self.nlcgrids.reset(mol)
        self.cphf_grids.reset(mol)
        self._numint.reset()
        return self

    def nuc_grad_method(self):
        from gpu4pyscf.grad import uks as uks_grad
        return uks_grad.Gradients(self)

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        mf = uks_cpu.UKS(self.mol, xc=self.xc)
        utils.to_cpu(self, mf)
        return mf
