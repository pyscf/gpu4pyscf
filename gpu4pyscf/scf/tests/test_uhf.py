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
from pyscf.scf import uhf
from gpu4pyscf.scf.hf import _get_jk, eigh, damping, level_shift
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array
import numpy as np
import cupy
from gpu4pyscf import lib
from gpu4pyscf.scf import diis
from pyscf import lib as pyscf_lib


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
    if h1e is None: h1e = cupy.asarray(mf.get_hcore())
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
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
        dm0 = mf.get_init_guess(mol)

    dm = cupy.asarray(dm0, order='C')
    if hasattr(dm0, 'mo_coeff') and hasattr(dm0, 'mo_occ'):
        mo_coeff = cupy.asarray(dm0.mo_coeff)
        mo_occ = cupy.asarray(dm0.mo_occ)
        occ_coeff = cupy.asarray(mo_coeff[:,mo_occ>0])
        dm = tag_array(dm, occ_coeff=occ_coeff, mo_occ=mo_occ, mo_coeff=mo_coeff)

    # use optimized workflow if possible
    if hasattr(mf, 'init_workflow'):
        mf.init_workflow(dm0=dm)
        h1e = mf.h1e
        s1e = mf.s1e
    else:
        h1e = cupy.asarray(mf.get_hcore(mol))
        s1e = cupy.asarray(mf.get_ovlp(mol))

    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)
    t1 = log.timer_debug1('total prep', *t0)
    scf_conv = False

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
        e_diff = abs(e_tot-last_hf_e)
        norm_gorb = cupy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, f))
        if(e_diff < conv_tol and norm_gorb < conv_tol_grad):
            scf_conv = True
            break

    if(cycle == mf.max_cycle):
        logger.warn("SCF failed to converge")

    # for dispersion correction
    e_tot = e_tot.get()
    if(hasattr(mf, 'get_dispersion')):
        e_disp = mf.get_dispersion()
        mf.e_disp = e_disp
        mf.e_mf = e_tot
        e_tot += e_disp

    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


class UHF(uhf.UHF):
    from gpu4pyscf.lib.utils import to_cpu, to_gpu, device

    DIIS = diis.SCF_DIIS
    get_jk = _get_jk
    _eigh = staticmethod(eigh)
    get_fock = get_fock
    
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
    
    def scf(self, dm0=None, **kwargs):
        cput0 = logger.init_timer(self)

        self.dump_flags()
        self.build(self.mol)

        if self.max_cycle > 0 or self.mo_coeff is None:
            self.converged, self.e_tot, \
                    self.mo_energy, self.mo_coeff, self.mo_occ = \
                    _kernel(self, self.conv_tol, self.conv_tol_grad,
                           dm0=dm0, callback=self.callback,
                           conv_check=self.conv_check, **kwargs)
        else:
            self.e_tot = _kernel(self, self.conv_tol, self.conv_tol_grad,
                                dm0=dm0, callback=self.callback,
                                conv_check=self.conv_check, **kwargs)[1]

        logger.timer(self, 'SCF', *cput0)
        self._finalize()
        return self.e_tot
    kernel = pyscf_lib.alias(scf, alias_name='kernel')
    
    def spin_square(self, mo_coeff=None, s=None):
        if mo_coeff is None:
            mo_coeff = (self.mo_coeff[0][:,self.mo_occ[0]>0],
                        self.mo_coeff[1][:,self.mo_occ[1]>0])
        if s is None:
            s = self.get_ovlp()
        return spin_square(mo_coeff, s)

