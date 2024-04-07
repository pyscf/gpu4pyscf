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

from functools import reduce
import numpy as np
import cupy
from pyscf.scf import uhf
from pyscf import lib as pyscf_lib
from gpu4pyscf.scf.hf import _get_jk, eigh, damping, level_shift, _kernel
from gpu4pyscf.scf import hf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf import lib
from gpu4pyscf.scf import diis

def make_rdm1(mo_coeff, mo_occ, **kwargs):
    '''One-particle density matrix in AO representation

    Args:
        mo_coeff : tuple of 2D ndarrays
            Orbital coefficients for alpha and beta spins. Each column is one orbital.
        mo_occ : tuple of 1D ndarrays
            Occupancies for alpha and beta spins.
    Returns:
        A list of 2D ndarrays for alpha and beta spins
    '''
    mo_a = mo_coeff[0]
    mo_b = mo_coeff[1]
    dm_a = cupy.dot(mo_a*mo_occ[0], mo_a.conj().T)
    dm_b = cupy.dot(mo_b*mo_occ[1], mo_b.conj().T)
# DO NOT make tag_array for DM here because the DM arrays may be modified and
# passed to functions like get_jk, get_vxc.  These functions may take the tags
# (mo_coeff, mo_occ) to compute the potential if tags were found in the DM
# arrays and modifications to DM arrays may be ignored.
    return tag_array((dm_a, dm_b), mo_coeff=mo_coeff, mo_occ=mo_occ)


def spin_square(mo, s=1):
    r'''Spin square and multiplicity of UHF determinant

    Detailed derivataion please refers to the cpu pyscf.

    '''
    mo_a, mo_b = mo
    nocc_a = mo_a.shape[1]
    nocc_b = mo_b.shape[1]
    s = reduce(cupy.dot, (mo_a.conj().T, cupy.asarray(s), mo_b))
    ssxy = (nocc_a+nocc_b) * .5 - cupy.einsum('ij,ij->', s.conj(), s)
    ssz = (nocc_b-nocc_a)**2 * .25
    ss = (ssxy + ssz).real
    s = cupy.sqrt(ss+.25) - .5
    return ss, s*2+1


def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = cupy.asarray(mf.get_hcore())
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    if not isinstance(s1e, cupy.ndarray): s1e = cupy.asarray(s1e)
    if not isinstance(dm, cupy.ndarray): dm = cupy.asarray(dm)
    if not isinstance(h1e, cupy.ndarray): h1e = cupy.asarray(h1e)
    if not isinstance(vhf, cupy.ndarray): vhf = cupy.asarray(vhf)
    f = h1e + vhf
    if f.ndim == 2:
        f = (f, f)
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = mf.make_rdm1()

    if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor
    if isinstance(damp_factor, (tuple, list, np.ndarray)):
        dampa, dampb = damp_factor
    else:
        dampa = dampb = damp_factor

    if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4:
        f = (damping(s1e, dm[0], f[0], dampa),
             damping(s1e, dm[1], f[1], dampb))
    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)
    if abs(shifta)+abs(shiftb) > 1e-4:
        f = (level_shift(s1e, dm[0], f[0], shifta),
             level_shift(s1e, dm[1], f[1], shiftb))
    return f

def get_grad(mo_coeff, mo_occ, fock_ao):
    '''UHF Gradients'''
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb

    ga = mo_coeff[0][:,viridxa].conj().T.dot(fock_ao[0].dot(mo_coeff[0][:,occidxa]))
    gb = mo_coeff[1][:,viridxb].conj().T.dot(fock_ao[1].dot(mo_coeff[1][:,occidxb]))
    return cupy.hstack((ga.ravel(), gb.ravel()))

def energy_elec(mf, dm=None, h1e=None, vhf=None):
    '''Electronic energy of Unrestricted Hartree-Fock

    Note this function has side effects which cause mf.scf_summary updated.

    Returns:
        Hartree-Fock electronic energy and the 2-electron part contribution
    '''
    if dm is None: dm = mf.make_rdm1()
    if h1e is None:
        h1e = mf.get_hcore()
    if isinstance(dm, cupy.ndarray) and dm.ndim == 2:
        dm = cupy.array((dm*.5, dm*.5))
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)
    if h1e[0].ndim < dm[0].ndim:  # get [0] because h1e and dm may not be ndarrays
        h1e = (h1e, h1e)
    e1 = cupy.einsum('ij,ji->', h1e[0], dm[0])
    e1+= cupy.einsum('ij,ji->', h1e[1], dm[1])
    e_coul =(cupy.einsum('ij,ji->', vhf[0], dm[0]) +
             cupy.einsum('ij,ji->', vhf[1], dm[1])) * .5
    e1 = e1.get()[()]
    e_coul = e_coul.get()[()]
    e_elec = (e1 + e_coul).real
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    logger.debug(mf, 'E1 = %s  Ecoul = %s', e1, e_coul.real)
    return e_elec, e_coul

class UHF(hf.SCF):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'e_disp', 'screen_tol', 'conv_tol_cpscf', 'h1e', 's1e'}
    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        self.nelec = None

    @property
    def nelec(self):
        if self._nelec is not None:
            return self._nelec
        else:
            return self.mol.nelec
    @nelec.setter
    def nelec(self, x):
        self._nelec = x

    @property
    def nelectron_alpha(self):
        return self.nelec[0]
    @nelectron_alpha.setter
    def nelectron_alpha(self, x):
        logger.warn(self, 'WARN: Attribute .nelectron_alpha is deprecated. '
                    'Set .nelec instead')
        #raise RuntimeError('API updates')
        self.nelec = (x, self.mol.nelectron-x)

    def dump_flags(self, verbose=None):
        return

    get_jk = hf._get_jk
    _eigh = staticmethod(eigh)
    get_fock = get_fock
    get_occ = uhf.get_occ

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)

    make_rdm2                = NotImplemented
    energy_elec              = energy_elec
    get_init_guess           = hf.return_cupy_array(uhf.UHF.get_init_guess)
    init_guess_by_minao      = uhf.UHF.init_guess_by_minao
    init_guess_by_atom       = uhf.UHF.init_guess_by_atom
    init_guess_by_huckel     = uhf.UHF.init_guess_by_huckel
    init_guess_by_mod_huckel = uhf.UHF.init_guess_by_mod_huckel
    init_guess_by_1e         = uhf.UHF.init_guess_by_1e
    init_guess_by_chkfile    = NotImplemented

    analyze            = NotImplemented
    mulliken_pop       = NotImplemented
    mulliken_spin_pop  = NotImplemented
    mulliken_meta      = NotImplemented
    mulliken_meta_spin = NotImplemented
    canonicalize       = NotImplemented
    det_ovlp           = NotImplemented
    make_asym_dm       = NotImplemented
    _finalize          = uhf.UHF._finalize

    screen_tol = 1e-14
    conv_tol_cpscf = 1e-3
    DIIS = diis.SCF_DIIS
    #get_jk = _get_jk

    get_hcore = hf.RHF.get_hcore
    get_ovlp = hf.RHF.get_ovlp
    get_init_guess = hf.return_cupy_array(uhf.UHF.get_init_guess)
    density_fit = hf.RHF.density_fit
    energy_tot = hf.RHF.energy_tot
    energy_elec = energy_elec

    make_rdm2 = NotImplemented
    dump_chk = NotImplemented
    newton = NotImplemented
    x2c = x2c1e = sfx2c1e = NotImplemented
    to_rhf = NotImplemented
    to_uhf = NotImplemented
    to_ghf = NotImplemented
    to_rks = NotImplemented
    to_uks = NotImplemented
    to_gks = NotImplemented
    to_ks = NotImplemented
    canonicalize = NotImplemented
    # TODO: Enable followings after testing
    analyze = NotImplemented
    stability = NotImplemented
    mulliken_pop = NotImplemented
    mulliken_spin_pop = NotImplemented
    mulliken_meta = NotImplemented
    mulliken_meta_spin = NotImplemented
    det_ovlp = NotImplemented

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        return make_rdm1(mo_coeff, mo_occ, **kwargs)

    def eig(self, fock, s):
        e_a, c_a = self._eigh(fock[0], s)
        e_b, c_b = self._eigh(fock[1], s)
        return cupy.array((e_a,e_b)), cupy.array((c_a,c_b))

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        if isinstance(dm, cupy.ndarray) and dm.ndim == 2:
            dm = cupy.asarray((dm*.5,dm*.5))

        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, cupy.asarray(dm), hermi)
            vhf = vj[0] + vj[1] - vk
        else:
            ddm = cupy.asarray(dm) - cupy.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj[0] + vj[1] - vk
            vhf += cupy.asarray(vhf_last)
        return vhf

    scf = hf.scf
    kernel = pyscf_lib.alias(scf, alias_name='kernel')

    def spin_square(self, mo_coeff=None, s=None):
        if mo_coeff is None:
            mo_coeff = (self.mo_coeff[0][:,self.mo_occ[0]>0],
                        self.mo_coeff[1][:,self.mo_occ[1]>0])
        if s is None:
            s = self.get_ovlp()
        return spin_square(mo_coeff, s)

    def nuc_grad_method(self):
        from gpu4pyscf.grad import uhf
        return uhf.Gradients(self)

    def to_cpu(self):
        from gpu4pyscf.lib import utils
        mf = uhf.UHF(self.mol)
        utils.to_cpu(self, mf)
        return mf