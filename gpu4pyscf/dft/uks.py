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
from gpu4pyscf.scf import hf, uhf, j_engine
from gpu4pyscf.lib.cupy_helper import tag_array, asarray
from gpu4pyscf.lib import utils


def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional for UKS.  See pyscf/dft/rks.py
    :func:`get_veff` fore more details.
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    if isinstance(dm, cupy.ndarray) and dm.ndim == 2:
        dm = cupy.asarray((dm*.5,dm*.5))
    else:
        dm = asarray(dm)
    assert dm.ndim == 3
    t0 = logger.init_timer(ks)
    rks.initialize_grids(ks, mol, cupy.asarray(dm[0]+dm[1]))

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

    dm_orig = dm
    vj_last = getattr(vhf_last, 'vj', None)
    if vj_last is not None:
        dm = asarray(dm) - asarray(dm_last)
    vj = ks.get_j(mol, dm[0]+dm[1], hermi)
    if vj_last is not None:
        vj += asarray(vj_last)
    vxc += vj
    if ground_state:
        ecoul = float(cupy.einsum('nij,ij->', dm_orig, vj).real) * .5
    else:
        ecoul = None

    vk = None
    if ni.libxc.is_hybrid_xc(ks.xc):
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
        if omega == 0:
            vk = ks.get_k(mol, dm, hermi)
            vk *= hyb
        elif alpha == 0: # LR=0, only SR exchange
            vk = ks.get_k(mol, dm, hermi, omega=-omega)
            vk *= hyb
        elif hyb == 0: # SR=0, only LR exchange
            vk = ks.get_k(mol, dm, hermi, omega=omega)
            vk *= alpha
        else: # SR and LR exchange with different ratios
            vk = ks.get_k(mol, dm, hermi)
            vk *= hyb
            vklr = ks.get_k(mol, dm, hermi, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        if vj_last is not None:
            vk += asarray(vhf_last.vk)
        vxc -= vk
        if ground_state:
            exc -= float(cupy.einsum('nij,nij', dm_orig, vk).real) * .5
    t0 = logger.timer_debug1(ks, 'veff', *t0)
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
