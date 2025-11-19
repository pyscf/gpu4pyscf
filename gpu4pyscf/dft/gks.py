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

import cupy as cp
from pyscf import lib
from pyscf.dft import gks
from gpu4pyscf.dft import numint2c
from gpu4pyscf.dft import rks
from gpu4pyscf.scf.ghf import GHF
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import utils
from gpu4pyscf.lib.cupy_helper import tag_array


def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional

    .. note::
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        dm_last : ndarray or a list of ndarrays or 0
            The density matrix baseline.  If not 0, this function computes the
            increment of HF potential w.r.t. the reference HF potential matrix.
        vhf_last : ndarray or a list of ndarrays or 0
            The reference Vxc potential matrix.
        hermi : int
            Whether J, K matrix is hermitian

            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian

    Returns:
        matrix Veff = J + Vxc.  Veff can be a list matrices, if the input
        dm is a list of density matrices.
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    ks.initialize_grids(mol, dm)

    t0 = (logger.process_clock(), logger.perf_counter())

    ground_state = isinstance(dm, cp.ndarray) and dm.ndim == 2

    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        ni = ks._numint
        if ni.collinear[0].lower() != 'm':
            raise NotImplementedError('Only multi-colinear GKS is implemented')
        n, exc, vxc = ni.get_vxc(mol, ks.grids, ks.xc, dm,
                                 hermi=hermi, max_memory=max_memory)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(mol, ks.nlcgrids, xc, dm,
                                          hermi=hermi, max_memory=max_memory)
            exc += enlc
            vxc += vnlc
            logger.debug(ks, 'nelec with nlc grids = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    
    dm_orig = cp.asarray(dm)
    vj_last = getattr(vhf_last, 'vj', None)
    if vj_last is not None:
        dm = cp.asarray(dm) - cp.asarray(dm_last)
    if not ni.libxc.is_hybrid_xc(ks.xc):
        vk = None
        vj = ks.get_j(mol, dm, hermi)
        if vj_last is not None:
            vj += vj_last
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
        if omega == 0:
            vj, vk = ks.get_jk(mol, dm, hermi)
            vk *= hyb
        elif alpha == 0: # LR=0, only SR exchange
            vj = ks.get_j(mol, dm, hermi)
            vk = ks.get_k(mol, dm, hermi, omega=-omega)
            vk *= hyb
        elif hyb == 0: # SR=0, only LR exchange
            vj = ks.get_j(mol, dm, hermi)
            vk = ks.get_k(mol, dm, hermi, omega=omega)
            vk *= alpha
        else: # SR and LR exchange with different ratios
            vj, vk = ks.get_jk(mol, dm, hermi)
            vk *= hyb
            vklr = ks.get_k(mol, dm, hermi, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        if vj_last is not None:
            vj += vhf_last.vj
            vk += vhf_last.vk
        vxc += vj - vk

        if ground_state:
            exc -= cp.einsum('ij,ji', dm_orig, vk).real * .5

    if ground_state:
        ecoul = cp.einsum('ij,ji', dm_orig, vj).real * .5
    else:
        ecoul = None

    vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc


class GKS(rks.KohnShamDFT, GHF):
    to_gpu = utils.to_gpu
    device = utils.device

    def __init__(self, mol, xc='LDA,VWN'):
        GHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)
        self._numint = numint2c.NumInt2C()

    def dump_flags(self, verbose=None):
        GHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        logger.info(self, 'collinear = %s', self._numint.collinear)
        if self._numint.collinear[0] == 'm':
            logger.info(self, 'mcfun spin_samples = %s', self._numint.spin_samples)
            logger.info(self, 'mcfun collinear_thrd = %s', self._numint.collinear_thrd)
            logger.info(self, 'mcfun collinear_samples = %s', self._numint.collinear_samples)
        return self

    @property
    def collinear(self):
        return self._numint.collinear
    @collinear.setter
    def collinear(self, val):
        self._numint.collinear = val

    @property
    def spin_samples(self):
        return self._numint.spin_samples
    @spin_samples.setter
    def spin_samples(self, val):
        self._numint.spin_samples = val
    
    get_veff = get_veff
    reset = rks.RKS.reset
    energy_elec = rks.RKS.energy_elec
    nuc_grad_method = NotImplemented
    to_hf = NotImplemented
    
    def to_cpu(self):
        mf = gks.GKS(self.mol)
        utils.to_cpu(self, out=mf)
        return mf
