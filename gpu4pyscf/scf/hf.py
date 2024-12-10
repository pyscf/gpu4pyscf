# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
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

import h5py
import numpy as np
import cupy
import scipy.linalg
from functools import reduce
from pyscf import gto
from pyscf import lib as pyscf_lib
from pyscf.scf import hf as hf_cpu
from pyscf.scf import chkfile
from gpu4pyscf import lib
from gpu4pyscf.lib import utils
from gpu4pyscf.lib.cupy_helper import eigh, tag_array, return_cupy_array, cond
from gpu4pyscf.scf import diis, jk
from gpu4pyscf.lib import logger

__all__ = [
    'get_jk', 'get_occ', 'get_grad', 'damping', 'level_shift', 'get_fock',
    'energy_elec', 'RHF', 'SCF'
]

def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None,
           verbose=None):
    '''Compute J, K matrices with CPU-GPU hybrid algorithm
    '''
    with mol.with_range_coulomb(omega):
        vj, vk = jk.get_jk(mol, dm, hermi, vhfopt, with_j, with_k, verbose)
    if not isinstance(dm, cupy.ndarray):
        if with_j: vj = vj.get()
        if with_k: vk = vk.get()
    return vj, vk

def _get_jk(mf, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
            omega=None):
    vhfopt = mf._opt_gpu.get(omega)
    if vhfopt is None:
        with mol.with_range_coulomb(omega):
            vhfopt = mf._opt_gpu[omega] = jk._VHFOpt(mol, mf.direct_scf_tol).build()

    vj, vk = get_jk(mol, dm, hermi, vhfopt, with_j, with_k, omega)
    return vj, vk

def make_rdm1(mf, mo_coeff=None, mo_occ=None, **kwargs):
    if mo_occ is None: mo_occ = mf.mo_occ
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    mo_coeff = cupy.asarray(mo_coeff)
    mo_occ = cupy.asarray(mo_occ)
    is_occ = mo_occ > 0
    mocc = mo_coeff[:, is_occ]
    dm = cupy.dot(mocc*mo_occ[is_occ], mocc.conj().T)
    occ_coeff = mo_coeff[:, mo_occ>1.0]
    return tag_array(dm, occ_coeff=occ_coeff, mo_occ=mo_occ, mo_coeff=mo_coeff)

def get_occ(mf, mo_energy=None, mo_coeff=None):
    if mo_energy is None: mo_energy = mf.mo_energy
    e_idx = cupy.argsort(mo_energy)
    nmo = mo_energy.size
    mo_occ = cupy.zeros(nmo)
    nocc = mf.mol.nelectron // 2
    mo_occ[e_idx[:nocc]] = 2
    return mo_occ

def get_veff(mf, mol=None, dm=None, dm_last=None, vhf_last=0, hermi=1, vhfopt=None):
    if dm is None: dm = mf.make_rdm1()
    if dm_last is None or not mf.direct_scf:
        vj, vk = mf.get_jk(mol, dm, hermi)
        return vj - vk * .5
    else:
        ddm = cupy.asarray(dm) - cupy.asarray(dm_last)
        vj, vk = mf.get_jk(mol, ddm, hermi)
        return vj - vk * .5 + vhf_last

def get_grad(mo_coeff, mo_occ, fock_ao):
    occidx = mo_occ > 0
    viridx = ~occidx
    g = reduce(cupy.dot, (mo_coeff[:,viridx].conj().T, fock_ao,
                           mo_coeff[:,occidx])) * 2
    return g.ravel()

def damping(s, d, f, factor):
    dm_vir = cupy.eye(s.shape[0]) - cupy.dot(s, d)
    f0 = reduce(cupy.dot, (dm_vir, f, d, s))
    f0 = (f0+f0.conj().T) * (factor/(factor+1.))
    return f - f0

def level_shift(s, d, f, factor):
    dm_vir = s - reduce(cupy.dot, (s, d, s))
    return f + dm_vir * factor

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    if not isinstance(s1e, cupy.ndarray): s1e = cupy.asarray(s1e)
    if not isinstance(dm, cupy.ndarray): dm = cupy.asarray(dm)
    if not isinstance(h1e, cupy.ndarray): h1e = cupy.asarray(h1e)
    if not isinstance(vhf, cupy.ndarray): vhf = cupy.asarray(vhf)
    f = h1e + vhf
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp

    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4:
        f = damping(s1e, dm*.5, f, damp_factor)
    if diis is not None and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)
    if abs(level_shift_factor) > 1e-4:
        f = level_shift(s1e, dm*.5, f, level_shift_factor)
    return f

def energy_elec(self, dm=None, h1e=None, vhf=None):
    '''
    electronic energy
    '''
    if dm is None: dm = self.make_rdm1()
    if h1e is None: h1e = self.get_hcore()
    if vhf is None: vhf = self.get_veff(self.mol, dm)
    e1 = cupy.einsum('ij,ji->', h1e, dm).real
    e_coul = cupy.einsum('ij,ji->', vhf, dm).real * .5
    e1 = e1.get()[()]
    e_coul = e_coul.get()[()]
    self.scf_summary['e1'] = e1
    self.scf_summary['e2'] = e_coul
    logger.debug(self, 'E1 = %s  E_coul = %s', e1, e_coul)
    return e1+e_coul, e_coul

def _kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    conv_tol = mf.conv_tol
    mol = mf.mol
    verbose = mf.verbose
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    if(conv_tol_grad is None):
        conv_tol_grad = conv_tol**.5
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)

    if(dm0 is None):
        dm0 = mf.get_init_guess(mol, mf.init_guess)

    dm = cupy.asarray(dm0, order='C')
    if hasattr(dm0, 'mo_coeff') and hasattr(dm0, 'mo_occ'):
        if dm0.ndim == 2:
            mo_coeff = cupy.asarray(dm0.mo_coeff)
            mo_occ = cupy.asarray(dm0.mo_occ)
            occ_coeff = cupy.asarray(mo_coeff[:,mo_occ>0])
            dm = tag_array(dm, occ_coeff=occ_coeff, mo_occ=mo_occ, mo_coeff=mo_coeff)

    h1e = cupy.asarray(mf.get_hcore(mol))
    s1e = cupy.asarray(mf.get_ovlp(mol))

    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)
    t1 = log.timer_debug1('total prep', *t0)
    scf_conv = False

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    dump_chk = dump_chk and mf.chkfile is not None
    if dump_chk:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    for cycle in range(mf.max_cycle):
        t0 = log.init_timer()
        dm_last = dm
        last_hf_e = e_tot

        f = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
        t1 = log.timer_debug1('DIIS', *t0)
        mo_energy, mo_coeff = mf.eig(f, s1e)
        t1 = log.timer_debug1('eig', *t1)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        t1 = log.timer_debug1('dm', *t1)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        t1 = log.timer_debug1('veff', *t1)
        e_tot = mf.energy_tot(dm, h1e, vhf)
        t1 = log.timer_debug1('energy', *t1)

        norm_ddm = cupy.linalg.norm(dm-dm_last)
        t1 = log.timer_debug1('total', *t0)
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_ddm)

        if dump_chk:
            mf.dump_chk(locals())

        e_diff = abs(e_tot-last_hf_e)
        norm_gorb = cupy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, f))
        if(e_diff < conv_tol and norm_gorb < conv_tol_grad):
            scf_conv = True
            break
    else:
        logger.warn(mf, "SCF failed to converge")

    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


def energy_tot(mf, dm=None, h1e=None, vhf=None):
    r'''Total Hartree-Fock energy, electronic part plus nuclear repulstion
    See :func:`scf.hf.energy_elec` for the electron part

    Note this function has side effects which cause mf.scf_summary updated.

    '''
    nuc = mf.energy_nuc()
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + nuc
    if mf.do_disp():
        if 'dispersion' in mf.scf_summary:
            e_tot += mf.scf_summary['dispersion']
        else:
            e_disp = mf.get_dispersion()
            mf.scf_summary['dispersion'] = e_disp
            e_tot += e_disp
    mf.scf_summary['nuc'] = nuc.real
    if isinstance(e_tot, cupy.ndarray):
        e_tot = e_tot.get()
    return e_tot

def scf(mf, dm0=None, **kwargs):
    cput0 = logger.init_timer(mf)

    mf.dump_flags()
    mf.build(mf.mol)

    if dm0 is None and mf.mo_coeff is not None and mf.mo_occ is not None:
        # Initial guess from existing wavefunction
        dm0 = mf.make_rdm1()

    if mf.max_cycle > 0 or mf.mo_coeff is None:
        mf.converged, mf.e_tot, \
                mf.mo_energy, mf.mo_coeff, mf.mo_occ = \
                _kernel(mf, mf.conv_tol, mf.conv_tol_grad,
                        dm0=dm0, callback=mf.callback,
                        conv_check=mf.conv_check, **kwargs)
    else:
        # Avoid to update SCF orbitals in the non-SCF initialization
        # (issue #495).  But run regular SCF for initial guess if SCF was
        # not initialized.
        mf.e_tot = _kernel(mf, mf.conv_tol, mf.conv_tol_grad,
                            dm0=dm0, callback=mf.callback,
                            conv_check=mf.conv_check, **kwargs)[1]

    logger.timer(mf, 'SCF', *cput0)
    mf._finalize()
    return mf.e_tot

def canonicalize(mf, mo_coeff, mo_occ, fock=None):
    '''Canonicalization diagonalizes the Fock matrix within occupied, open,
    virtual subspaces separatedly (without change occupancy).
    '''
    if fock is None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        fock = mf.get_fock(dm=dm)
    coreidx = mo_occ == 2
    viridx = mo_occ == 0
    openidx = ~(coreidx | viridx)
    mo = cupy.empty_like(mo_coeff)
    mo_e = cupy.empty(mo_occ.size)
    for idx in (coreidx, openidx, viridx):
        if cupy.any(idx) > 0:
            orb = mo_coeff[:,idx]
            f1 = orb.conj().T.dot(fock).dot(orb)
            e, c = cupy.linalg.eigh(f1)
            mo[:,idx] = orb.dot(c)
            mo_e[idx] = e
    return mo_e, mo

def as_scanner(mf):
    if isinstance(mf, pyscf_lib.SinglePointScanner):
        return mf

    logger.info(mf, 'Create scanner for %s', mf.__class__)
    name = mf.__class__.__name__ + SCF_Scanner.__name_mixin__
    return pyscf_lib.set_class(SCF_Scanner(mf), (SCF_Scanner, mf.__class__), name)

class SCF_Scanner(pyscf_lib.SinglePointScanner):
    def __init__(self, mf_obj):
        self.__dict__.update(mf_obj.__dict__)
        self._last_mol_fp = mf_obj.mol.ao_loc

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        # Cleanup intermediates associated to the previous mol object
        self.reset(mol)

        if 'dm0' in kwargs:
            dm0 = kwargs.pop('dm0')
        elif self.mo_coeff is None:
            dm0 = None
        else:
            dm0 = None
            if cupy.array_equal(self._last_mol_fp, mol.ao_loc):
                dm0 = self.make_rdm1()
            elif self.chkfile and h5py.is_hdf5(self.chkfile):
                dm0 = self.from_chk(self.chkfile)
        self.mo_coeff = None  # To avoid last mo_coeff being used by SOSCF
        e_tot = self.kernel(dm0=dm0, **kwargs)
        self._last_mol_fp = mol.ao_loc
        return e_tot

class SCF(pyscf_lib.StreamObject):

    # attributes
    conv_tol            = hf_cpu.SCF.conv_tol
    conv_tol_grad       = hf_cpu.SCF.conv_tol_grad
    max_cycle           = hf_cpu.SCF.max_cycle
    init_guess          = hf_cpu.SCF.init_guess
    conv_tol_cpscf      = 1e-4

    disp                = None
    DIIS                = diis.SCF_DIIS
    diis                = hf_cpu.SCF.diis
    diis_space          = hf_cpu.SCF.diis_space
    diis_damp           = hf_cpu.SCF.diis_damp
    diis_start_cycle    = hf_cpu.SCF.diis_start_cycle
    diis_file           = hf_cpu.SCF.diis_file
    diis_space_rollback = hf_cpu.SCF.diis_space_rollback
    damp                = hf_cpu.SCF.damp
    level_shift         = hf_cpu.SCF.level_shift
    direct_scf          = hf_cpu.SCF.direct_scf
    direct_scf_tol      = hf_cpu.SCF.direct_scf_tol
    conv_check          = hf_cpu.SCF.conv_check
    callback            = hf_cpu.SCF.callback
    _keys               = hf_cpu.SCF._keys

    # methods
    def __init__(self, mol):
        if not mol._built:
            mol.build()
        self.mol = mol
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.stdout = mol.stdout

        # The chkfile part is different from pyscf, we turn off chkfile by default.
        self.chkfile = None

##################################################
# don't modify the following attributes, they are not input options
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.e_tot = 0
        self.converged = False
        self.scf_summary = {}

        self._opt_gpu = {None: None}
        self._eri = None # Note: self._eri requires large amount of memory

    def check_sanity(self):
        s1e = self.get_ovlp()
        if isinstance(s1e, cupy.ndarray) and s1e.ndim == 2:
            c = cond(s1e)
        else:
            c = cupy.asarray([cond(xi) for xi in s1e])
        logger.debug(self, 'cond(S) = %s', c)
        if cupy.max(c)*1e-17 > self.conv_tol:
            logger.warn(self, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                        'SCF may be inaccurate and hard to converge.', cupy.max(c))
        return super().check_sanity()

    build                    = hf_cpu.SCF.build
    opt                      = NotImplemented
    dump_flags               = hf_cpu.SCF.dump_flags
    get_hcore                = return_cupy_array(hf_cpu.SCF.get_hcore)
    get_ovlp                 = return_cupy_array(hf_cpu.SCF.get_ovlp)
    get_fock                 = get_fock
    get_occ                  = get_occ
    get_grad                 = staticmethod(get_grad)
    init_guess_by_minao      = hf_cpu.SCF.init_guess_by_minao
    init_guess_by_atom       = hf_cpu.SCF.init_guess_by_atom
    init_guess_by_huckel     = hf_cpu.SCF.init_guess_by_huckel
    init_guess_by_mod_huckel = hf_cpu.SCF.init_guess_by_mod_huckel
    init_guess_by_1e         = hf_cpu.SCF.init_guess_by_1e
    init_guess_by_chkfile    = hf_cpu.SCF.init_guess_by_chkfile
    from_chk                 = hf_cpu.SCF.from_chk
    get_init_guess           = return_cupy_array(hf_cpu.SCF.get_init_guess)
    make_rdm1                = make_rdm1
    make_rdm2                = NotImplemented
    energy_elec              = energy_elec
    energy_tot               = energy_tot
    energy_nuc               = hf_cpu.SCF.energy_nuc
    check_convergence        = None
    _eigh                    = staticmethod(eigh)
    eig                      = hf_cpu.SCF.eig
    do_disp                  = hf_cpu.SCF.do_disp
    get_dispersion           = hf_cpu.SCF.get_dispersion
    kernel = scf             = scf
    as_scanner               = hf_cpu.SCF.as_scanner
    _finalize                = hf_cpu.SCF._finalize
    init_direct_scf          = hf_cpu.SCF.init_direct_scf
    get_jk                   = _get_jk
    get_j                    = hf_cpu.SCF.get_j
    get_k                    = hf_cpu.SCF.get_k
    get_veff                 = NotImplemented
    mulliken_meta            = hf_cpu.SCF.mulliken_meta
    pop                      = hf_cpu.SCF.pop
    _is_mem_enough           = NotImplemented
    density_fit              = NotImplemented
    newton                   = NotImplemented
    x2c = x2c1e = sfx2c1e    = NotImplemented
    stability                = NotImplemented
    nuc_grad_method          = NotImplemented
    update_                  = NotImplemented
    canonicalize             = NotImplemented
    istype                   = hf_cpu.SCF.istype
    to_rhf                   = NotImplemented
    to_uhf                   = NotImplemented
    to_ghf                   = NotImplemented
    to_rks                   = NotImplemented
    to_uks                   = NotImplemented
    to_gks                   = NotImplemented
    to_ks                    = NotImplemented
    canonicalize             = NotImplemented
    mulliken_pop             = NotImplemented
    mulliken_meta            = NotImplemented

    def dip_moment(self, mol=None, dm=None, unit='Debye', origin=None,
                   verbose=logger.NOTE):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return hf_cpu.dip_moment(mol, dm.get(), unit, origin, verbose)

    def quad_moment(self, mol=None, dm=None, unit='DebyeAngstrom', origin=None,
                    verbose=logger.NOTE):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return hf_cpu.quad_moment(mol, dm.get(), unit, origin, verbose)

    def remove_soscf(self):
        lib.logger.warn('remove_soscf has no effect in current version')
        return self

    def analyze(self, *args, **kwargs):
        return self.to_cpu().analyze()

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._opt_gpu = {None: None}
        self.scf_summary = {}
        return self

    def dump_chk(self, envs):
        assert isinstance(envs, dict)
        if self.chkfile:
            chkfile.dump_scf(
                self.mol, self.chkfile, envs['e_tot'],
                cupy.asnumpy(envs['mo_energy']), cupy.asnumpy(envs['mo_coeff']),
                cupy.asnumpy(envs['mo_occ']), overwrite_mol=False)

class KohnShamDFT:
    '''
    A mock DFT base class, to be compatible with PySCF
    '''

class RHF(SCF):

    to_gpu = utils.to_gpu
    device = utils.device

    _keys = {'e_disp', 'h1e', 's1e', 'e_mf', 'conv_tol_cpscf', 'disp_with_3body'}

    get_veff = get_veff
    canonicalize = canonicalize

    def check_sanity(self):
        mol = self.mol
        if mol.nelectron != 1 and mol.spin != 0:
            logger.warn(self, 'Invalid number of electrons %d for RHF method.',
                        mol.nelectron)
        return SCF.check_sanity(self)

    def nuc_grad_method(self):
        from gpu4pyscf.grad import rhf
        return rhf.Gradients(self)

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        import gpu4pyscf.df.df_jk
        return gpu4pyscf.df.df_jk.density_fit(self, auxbasis, with_df, only_dfj)

    def newton(self):
        from gpu4pyscf.scf.soscf import newton
        return newton(self)

    def to_cpu(self):
        mf = hf_cpu.RHF(self.mol)
        utils.to_cpu(self, out=mf)
        return mf
