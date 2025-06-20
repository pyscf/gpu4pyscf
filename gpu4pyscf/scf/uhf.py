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

from functools import reduce
import numpy as np
import cupy
from pyscf.scf import uhf as uhf_cpu
from pyscf import __config__

from gpu4pyscf.scf.hf import eigh, damping, level_shift
from gpu4pyscf.scf import hf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array, asarray

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
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    h1e = cupy.asarray(h1e)
    vhf = cupy.asarray(vhf)
    f = h1e + vhf
    if f.ndim == 2:
        f = (f, f)
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = mf.make_rdm1()
    s1e = cupy.asarray(s1e)
    dm = cupy.asarray(dm)
    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp

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

def canonicalize(mf, mo_coeff, mo_occ, fock=None):
    '''Canonicalization diagonalizes the UHF Fock matrix within occupied,
    virtual subspaces separatedly (without change occupancy).
    '''
    mo_occ = cupy.asarray(mo_occ)
    assert mo_occ.ndim == 2
    if fock is None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        fock = mf.get_fock(dm=dm)
    occidxa = mo_occ[0] == 1
    occidxb = mo_occ[1] == 1
    viridxa = mo_occ[0] == 0
    viridxb = mo_occ[1] == 0

    def eig_(fock, mo_coeff, idx, es, cs):
        if cupy.any(idx) > 0:
            orb = mo_coeff[:,idx]
            f1 = orb.conj().T.dot(fock).dot(orb)
            e, c = cupy.linalg.eigh(f1)
            es[idx] = e
            cs[:,idx] = cupy.dot(orb, c)

    mo = cupy.empty_like(mo_coeff)
    mo_e = cupy.empty(mo_occ.shape)
    eig_(fock[0], mo_coeff[0], occidxa, mo_e[0], mo[0])
    eig_(fock[0], mo_coeff[0], viridxa, mo_e[0], mo[0])
    eig_(fock[1], mo_coeff[1], occidxb, mo_e[1], mo[1])
    eig_(fock[1], mo_coeff[1], viridxb, mo_e[1], mo[1])
    return mo_e, mo

class UHF(hf.SCF):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'e_disp', 'conv_tol_cpscf', 'h1e', 's1e', 'init_guess_breaksym'}

    init_guess_breaksym = getattr(__config__, 'scf_uhf_init_guess_breaksym', 1)

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

    get_fock = get_fock
    get_occ = uhf_cpu.get_occ

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)

    make_asym_dm             = NotImplemented
    make_rdm2                = NotImplemented
    energy_elec              = energy_elec
    canonicalize             = canonicalize
    
    get_init_guess           = hf.return_cupy_array(uhf_cpu.UHF.get_init_guess)
    init_guess_by_minao      = uhf_cpu.UHF.init_guess_by_minao
    init_guess_by_atom       = uhf_cpu.UHF.init_guess_by_atom
    init_guess_by_huckel     = uhf_cpu.UHF.init_guess_by_huckel
    init_guess_by_mod_huckel = uhf_cpu.UHF.init_guess_by_mod_huckel
    init_guess_by_1e         = uhf_cpu.UHF.init_guess_by_1e
    init_guess_by_chkfile    = uhf_cpu.UHF.init_guess_by_chkfile
    _finalize                = uhf_cpu.UHF._finalize

    # TODO: Enable followings after testing
    analyze                 = NotImplemented
    stability               = NotImplemented
    mulliken_spin_pop       = NotImplemented
    mulliken_meta_spin      = NotImplemented
    det_ovlp                = NotImplemented

    density_fit             = hf.RHF.density_fit
    newton                  = hf.RHF.newton

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        return make_rdm1(mo_coeff, mo_occ, **kwargs)

    def eig(self, fock, s):
        e_a, c_a = self._eigh(fock[0], s)
        e_b, c_b = self._eigh(fock[1], s)
        return cupy.stack((e_a,e_b)), cupy.stack((c_a,c_b))

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if isinstance(dm, cupy.ndarray) and dm.ndim == 2:
            dm = cupy.asarray((dm*.5,dm*.5))
        if dm_last is not None and self.direct_scf:
            dm = asarray(dm) - asarray(dm_last)
        vj = self.get_j(mol, dm[0]+dm[1], hermi)
        vhf = self.get_k(mol, dm, hermi)
        vhf *= -1
        vhf += vj
        if vhf_last is not None:
            vhf += asarray(vhf_last)
        return vhf

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
        mf = uhf_cpu.UHF(self.mol)
        utils.to_cpu(self, mf)
        return mf
