# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import cupy
from pyscf import lib
from pyscf.mp import mp2
from pyscf import __config__
from gpu4pyscf.lib.cupy_helper import tag_array, get_avail_mem
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import int4c2e

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)

_einsum = cupy.einsum
def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff)

    if mo_energy is None:
        mo_energy = eris.mo_energy
    mo_energy = cupy.asarray(mo_energy)

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if with_t2:
        t2 = cupy.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
    else:
        t2 = None

    emp2_ss = emp2_os = 0
    for i in range(nocc):
        if isinstance(eris.ovov, cupy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals wiht the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            gi = eris.ovov[i]
        else:
            gi = cupy.asarray(eris.ovov[i*nvir:(i+1)*nvir])

        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)

        t2i = gi.conj()/(eia[:,:,None] + eia[i])#lib.direct_sum('jb+a->jba', eia, eia[i])
        edi = _einsum('jab,jab', t2i, gi) * 2
        exi = -_einsum('jab,jba', t2i, gi)
        emp2_ss += edi*0.5 + exi
        emp2_os += edi*0.5
        if with_t2:
            t2[i] = t2i

    emp2_ss = emp2_ss.real
    emp2_os = emp2_os.real
    emp2 = tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2.real, t2

def energy(mp, t2, eris):
    '''MP2 energy'''
    nocc, nvir = t2.shape[1:3]
    eris_ovov = cupy.asarray(eris.ovov).reshape(nocc,nvir,nocc,nvir)
    ed = _einsum('ijab,iajb', t2, eris_ovov) * 2
    ex = -_einsum('ijab,ibja', t2, eris_ovov)
    emp2_ss = (ed*0.5 + ex).real
    emp2_os = ed.real*0.5
    emp2 = tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)
    return emp2

def _gamma1_intermediates(mp, t2=None, eris=None):
    if t2 is None: t2 = mp.t2
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    if t2 is None:
        if eris is None:
            eris = mp.ao2mo()
        mo_energy = eris.mo_energy
        eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
        dtype = eris.ovov.dtype
    else:
        dtype = t2.dtype

    dm1occ = cupy.zeros((nocc,nocc), dtype=dtype)
    dm1vir = cupy.zeros((nvir,nvir), dtype=dtype)
    for i in range(nocc):
        if t2 is None:
            gi = cupy.asarray(eris.ovov[i*nvir:(i+1)*nvir])
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
            t2i = gi.conj()/(eia[:,:,None] + eia[i])
            #lib.direct_sum('jb+a->jba', eia, eia[i])
        else:
            t2i = t2[i]
        l2i = t2i.conj()
        dm1vir += _einsum('jca,jcb->ba', l2i, t2i) * 2 \
                - _einsum('jca,jbc->ba', l2i, t2i)
        dm1occ += _einsum('iab,jab->ij', l2i, t2i) * 2 \
                - _einsum('iab,jba->ij', l2i, t2i)
    return -dm1occ, dm1vir

def _make_eris(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    log = logger.new_logger(mp, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    eris = mp2._ChemistsERIs()
    if isinstance(mo_coeff, np.ndarray):
        mo_coeff = cupy.asarray(mo_coeff)
    eris._common_init_(mp, mo_coeff)
    mo_coeff = eris.mo_coeff
    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc
    mem_incore, mem_outcore, mem_basic = mp2._mem_usage(nocc, nvir)
    avail_mem = get_avail_mem()
    if avail_mem/1e6 < mem_basic:
        log.warn('Not enough memory for integral transformation. '
                 'Available mem %s MB, required mem %s MB',
                 avail_mem/1e6, mem_basic)

    co = cupy.asarray(mo_coeff[:,:nocc], order='F')
    cv = cupy.asarray(mo_coeff[:,nocc:], order='F')
    if (mp.mol.incore_anyway or mp._scf._eri is not None):
        log.debug('transform (ia|jb) incore')
        eris.ovov = int4c2e.get_int4c2e_ovov(mp.mol, co, cv)

    elif getattr(mp._scf, 'with_df', None):
        # To handle the PBC or custom 2-electron with 3-index tensor.
        # Call dfmp2.MP2 for efficient DF-MP2 implementation.
        log.warn('DF-HF is found. (ia|jb) is computed based on the DF '
                 '3-tensor integrals.\n'
                 'You can switch to dfmp2.MP2 for better performance')
        log.debug('transform (ia|jb) with_df')
        eris.ovov = mp._scf.with_df.ao2mo((co,cv,co,cv))

    else:
        log.debug('transform (ia|jb) outcore')
        # TODO: use cpu memory?
        eris.ovov = int4c2e.get_int4c2e_ovov(mp.mol, co, cv)

    log.timer('Integral transformation', *time0)
    return eris

def get_nocc(mp):
    if mp._nocc is not None:
        return mp._nocc
    elif mp.frozen is None:
        nocc = np.count_nonzero(mp.mo_occ.get() > 0)
        assert (nocc > 0)
        return nocc
    elif isinstance(mp.frozen, (int, np.integer)):
        nocc = np.count_nonzero(mp.mo_occ.get() > 0) - mp.frozen
        assert (nocc > 0)
        return nocc
    elif isinstance(mp.frozen[0], (int, np.integer)):
        occ_idx = mp.mo_occ.get() > 0
        occ_idx[list(mp.frozen)] = False
        nocc = np.count_nonzero(occ_idx)
        assert (nocc > 0)
        return nocc
    else:
        raise NotImplementedError

def get_nmo(mp):
    if mp._nmo is not None:
        return mp._nmo
    elif mp.frozen is None:
        return len(mp.mo_occ)
    elif isinstance(mp.frozen, (int, np.integer)):
        return len(mp.mo_occ) - mp.frozen
    elif isinstance(mp.frozen[0], (int, np.integer)):
        return len(mp.mo_occ) - len(set(mp.frozen))
    else:
        raise NotImplementedError

def get_e_hf(mp, mo_coeff=None):
    # Get HF energy, which is needed for total MP2 energy.
    if mo_coeff is None:
        mo_coeff = mp.mo_coeff
    mo_coeff = cupy.asarray(mo_coeff)
    dm = mp._scf.make_rdm1(mo_coeff, mp.mo_occ)
    vhf = mp._scf.get_veff(mp._scf.mol, dm)
    return mp._scf.energy_tot(dm=dm, vhf=vhf)

class MP2(lib.StreamObject):
    # Use CCSD default settings for the moment
    max_cycle = getattr(__config__, 'cc_ccsd_CCSD_max_cycle', 50)
    conv_tol = getattr(__config__, 'cc_ccsd_CCSD_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_ccsd_CCSD_conv_tol_normt', 1e-5)

    _keys = {
        'max_cycle', 'conv_tol', 'conv_tol_normt', 'mol', 'max_memory',
        'frozen', 'level_shift', 'mo_coeff', 'mo_occ', 'e_hf', 'e_corr',
        'e_corr_ss', 'e_corr_os', 't2',
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen

# For iterative MP2
        self.level_shift = 0

##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.e_hf = None
        self.e_corr = None
        self.e_corr_ss = None
        self.e_corr_os = None
        self.t2 = None

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_e_hf = get_e_hf

    def set_frozen(self, method='auto', window=(-1000.0, 1000.0)):
        from pyscf import mp
        is_gmp = isinstance(self, mp.gmp2.GMP2)
        from pyscf.cc.ccsd import set_frozen
        return set_frozen(self, method=method, window=window, is_gcc=is_gmp)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen is not None:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    @property
    def emp2(self):
        return self.e_corr

    @property
    def emp2_scs(self):
        # J. Chem. Phys. 118, 9095 (2003)
        return self.e_corr_ss*1./3. + self.e_corr_os*1.2

    @property
    def e_tot(self):
        return self.e_hf + self.e_corr

    @property
    def e_tot_scs(self):
        # J. Chem. Phys. 118, 9095 (2003)
        return self.e_hf + self.emp2_scs

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        '''
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.
        '''
        if self.verbose >= logger.WARN:
            self.check_sanity()

        self.dump_flags()

        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        if eris is None:
            eris = self.ao2mo(mo_coeff)

        if self._scf.converged:
            self.e_corr, self.t2 = self.init_amps(mo_energy, mo_coeff, eris, with_t2)
        else:
            raise NotImplementedError

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr_ss = float(self.e_corr_ss)
        self.e_corr_os = float(self.e_corr_os)
        self.e_corr = float(self.e_corr)

        self._finalize()
        return self.e_corr, self.t2

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        log = logger.new_logger(self)
        log.note('E(%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot, self.e_corr)
        log.note('E(SCS-%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot_scs, self.emp2_scs)
        log.info('E_corr(same-spin) = %.15g', self.e_corr_ss)
        log.info('E_corr(oppo-spin) = %.15g', self.e_corr_os)
        return self

    def ao2mo(self, mo_coeff=None):
        return _make_eris(self, mo_coeff, verbose=self.verbose)

    def density_fit(self, auxbasis=None, with_df=None):
        raise NotImplementedError

    def nuc_grad_method(self):
        raise NotImplementedError

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

    # to_cpu can be reused only when __init__ still takes mf
    def to_cpu(self):
        mf = self._scf.to_cpu()
        from importlib import import_module
        mod = import_module(self.__module__.replace('gpu4pyscf', 'pyscf'))
        cls = getattr(mod, self.__class__.__name__)
        obj = cls(mf)
        return obj

RMP2 = MP2
from gpu4pyscf import scf
scf.hf.RHF.MP2 = lib.class_as_method(MP2)