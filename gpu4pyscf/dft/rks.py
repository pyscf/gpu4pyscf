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

# modified by Xiaojie Wu (wxj6000@gmail.com)

import cupy
from pyscf.dft import rks
from gpu4pyscf.lib import logger
from gpu4pyscf.dft import numint, gen_grid
from gpu4pyscf.scf import hf, j_engine
from gpu4pyscf.lib.cupy_helper import tag_array, asarray
from pyscf import __config__

__all__ = [
    'get_veff', 'RKS', 'KohnShamDFT',
]

def prune_small_rho_grids_(ks, mol, dm, grids):
    '''Prune grids if the electron density on the grid is small'''
    threshold = ks.small_rho_cutoff
    if threshold == 0:
        return grids
    rho = ks._numint.get_rho(mol, dm, grids, ks.max_memory, verbose=ks.verbose)
    return grids.prune_by_density_(rho, threshold)

def initialize_grids(ks, mol=None, dm=None):
    # Initialize self.grids the first time call get_veff
    if mol is None: mol = ks.mol
    if ks.grids.coords is None:
        t0 = logger.init_timer(ks)
        ks.grids.build()
        #ks.grids.build(with_non0tab=True)
        ks.grids.weights = asarray(ks.grids.weights)
        ks.grids.coords = asarray(ks.grids.coords)
        ground_state = getattr(dm, 'ndim', 0) == 2
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            # Filter grids the first time setup grids
            ks.grids = prune_small_rho_grids_(ks, ks.mol, dm, ks.grids)
        t0 = logger.timer_debug1(ks, 'setting up grids', *t0)

    if ks.do_nlc() and ks.nlcgrids.coords is None:
        t0 = logger.init_timer(ks)
        #ks.nlcgrids.build(with_non0tab=True)
        ks.nlcgrids.build()
        ks.nlcgrids.weights = asarray(ks.nlcgrids.weights)
        ks.nlcgrids.coords = asarray(ks.nlcgrids.coords)
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            # Filter grids the first time setup grids
            ks.nlcgrids = prune_small_rho_grids_(ks, ks.mol, dm, ks.nlcgrids)
        t0 = logger.timer_debug1(ks, 'setting up nlc grids', *t0)
    return ks

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functionals
    .. note::
        This function will modify the input ks object.
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
    t0 = logger.init_timer(ks)
    initialize_grids(ks, mol, dm)

    ground_state = getattr(dm, 'ndim', 0) == 2

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        n, exc, vxc = ni.nr_rks(mol, ks.grids, ks.xc, dm)
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(mol, ks.nlcgrids, xc, dm)

            exc += enlc
            vxc += vnlc
        logger.debug(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer_debug1(ks, 'vxc tot', *t0)

    dm_orig = dm
    vj_last = getattr(vhf_last, 'vj', None)
    if vj_last is not None:
        dm = asarray(dm) - asarray(dm_last)
    vj = ks.get_j(mol, dm, hermi)
    if vj_last is not None:
        vj += asarray(vj_last)
    vxc += vj
    if ground_state:
        ecoul = float(cupy.einsum('ij,ij', dm_orig, vj).real) * .5
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
        vk *= .5
        if vj_last is not None:
            vk += asarray(vhf_last.vk)
        vxc -= vk
        if ground_state:
            exc -= float(cupy.einsum('ij,ij', dm_orig, vk).real) * .5
    t0 = logger.timer_debug1(ks, 'veff', *t0)
    vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc

def energy_elec(ks, dm=None, h1e=None, vhf=None):
    r'''Electronic part of RKS energy.

    Note this function has side effects which cause mf.scf_summary updated.

    Args:
        ks : an instance of DFT class

        dm : 2D ndarray
            one-partical density matrix
        h1e : 2D ndarray
            Core hamiltonian

    Returns:
        RKS electronic energy and the 2-electron contribution
    '''
    if dm is None: dm = ks.make_rdm1()
    if h1e is None: h1e = ks.get_hcore()
    if vhf is None: vhf = ks.get_veff(ks.mol, dm)
    e1 = cupy.einsum('ij,ji->', h1e, dm).get()[()].real
    ecoul = vhf.ecoul.real
    exc = vhf.exc.real
    if isinstance(ecoul, cupy.ndarray):
        ecoul = ecoul.get()[()]
    if isinstance(exc, cupy.ndarray):
        exc = exc.get()[()]
    e2 = ecoul + exc
    ks.scf_summary['e1'] = e1
    ks.scf_summary['coul'] = ecoul
    ks.scf_summary['exc'] = exc
    logger.debug(ks, 'E1 = %s  Ecoul = %s  Exc = %s', e1, ecoul, exc)
    return e1+e2, e2

# Inherit pyscf KohnShamDFT class since this is tested in the pyscf dispersion code
class KohnShamDFT(rks.KohnShamDFT):

    _keys = {'cphf_grids', *rks.KohnShamDFT._keys}

    to_rhf = NotImplemented
    to_uhf = NotImplemented
    to_ghf = NotImplemented
    to_hf  = NotImplemented
    to_rks = NotImplemented
    to_uks = NotImplemented
    to_gks = NotImplemented

    # Use rho to filter grids
    small_rho_cutoff = getattr(__config__, 'dft_rks_RKS_small_rho_cutoff', 1e-7)

    def __init__(self, xc='LDA,VWN'):
        self.xc = xc
        self.disp = None
        self.disp_with_3body = None
        self.nlc = ''
        self.grids = gen_grid.Grids(self.mol)
        self.grids.level = getattr(
            __config__, 'dft_rks_RKS_grids_level', self.grids.level)
        self.nlcgrids = gen_grid.Grids(self.mol)
        self.nlcgrids.level = getattr(
            __config__, 'dft_rks_RKS_nlcgrids_level', self.nlcgrids.level)
        
        # Default CPHF grids is SG1 grids
        # Reference:
        # https://gaussian.com/integral/?tabid=1#Integral_keyword__Grid_option
        self.cphf_grids = gen_grid.Grids(self.mol)
        self.cphf_grids.prune = gen_grid.sg1_prune
        self.cphf_grids.atom_grid = (50,194)
##################################################
# don't modify the following attributes, they are not input options
        self._numint = numint.NumInt()

    @property
    def omega(self):
        return self._numint.omega
    @omega.setter
    def omega(self, v):
        self._numint.omega = float(v)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('XC library %s version %s\n    %s',
                 self._numint.libxc.__name__,
                 self._numint.libxc.__version__,
                 self._numint.libxc.__reference__)
        log.info('XC functionals = %s', self.xc)
        self.grids.dump_flags(verbose)

        if self.do_nlc():
            log.info('** Following is NLC and NLC Grids **')
            if self.nlc:
                log.info('NLC functional = %s', self.nlc)
            else:
                log.info('NLC functional = %s', self.xc)
            self.nlcgrids.dump_flags(verbose)

        log.info('small_rho_cutoff = %g', self.small_rho_cutoff)
        return self

    reset = rks.KohnShamDFT.reset
    do_nlc = rks.KohnShamDFT.do_nlc

hf.KohnShamDFT = KohnShamDFT
from gpu4pyscf.lib import utils

class RKS(KohnShamDFT, hf.RHF):

    to_gpu = utils.to_gpu
    device = utils.device

    def __init__(self, mol, xc='LDA,VWN'):
        hf.RHF.__init__(self, mol)
        KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        hf.RHF.dump_flags(self, verbose)
        return KohnShamDFT.dump_flags(self, verbose)

    def reset(self, mol=None):
        hf.SCF.reset(self, mol)
        self.grids.reset(mol)
        self.nlcgrids.reset(mol)
        self._numint.reset()
        # The cphf_grids attribute is not available in the PySCF CPU version.
        # In PySCF's to_gpu() function, this attribute is not properly
        # initialized. mol of the KS object must be used for initialization.
        if mol is None:
            mol = self.mol
        self.cphf_grids.reset(mol)
        return self

    def nuc_grad_method(self):
        from gpu4pyscf.grad import rks as rks_grad
        return rks_grad.Gradients(self)

    def to_cpu(self):
        mf = rks.RKS(self.mol)
        utils.to_cpu(self, out=mf)
        return mf

    energy_elec = energy_elec
    energy_tot = hf.RHF.energy_tot
    get_veff = get_veff
    to_hf = NotImplemented
    init_guess_by_vsap = rks.init_guess_by_vsap
