#!/usr/bin/env python
#
# Copyright 2024 The PySCF Developers. All Rights Reserved.
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
import cupy as cp
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.tdscf import rhf as tdhf_cpu
from pyscf.tdscf._lr_eig import eigh as lr_eigh, eig as lr_eig
from gpu4pyscf import scf
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.lib import utils
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import _response_functions # noqa
from pyscf import __config__

REAL_EIG_THRESHOLD = tdhf_cpu.REAL_EIG_THRESHOLD
#OUTPUT_THRESHOLD = tdhf_cpu.OUTPUT_THRESHOLD
OUTPUT_THRESHOLD = getattr(__config__, 'tdscf_rhf_get_nto_threshold', 0.3)

__all__ = [
    'TDA', 'CIS', 'TDHF', 'TDRHF', 'TDBase'
]


def gen_tda_operation(mf, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute A x
    '''
    assert fock_ao is None
    assert isinstance(mf, scf.hf.SCF)
    assert wfnsym is None
    mo_coeff = mf.mo_coeff
    assert mo_coeff.dtype == cp.float64
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidx = mo_occ == 2
    viridx = mo_occ == 0
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    orbo2 = orbo * 2. # *2 for double occupancy

    e_ia = hdiag = mo_energy[viridx] - mo_energy[occidx,None]
    hdiag = hdiag.ravel().get()
    vresp = mf.gen_response(singlet=singlet, hermi=0)
    nocc, nvir = e_ia.shape

    def vind(zs):
        zs = cp.asarray(zs).reshape(-1,nocc,nvir)
        mo1 = contract('xov,qv->xqo', zs, orbv)
        dmov = contract('po,xqo->xpq', orbo2, mo1)
        dmov = tag_array(dmov, mo1=mo1, occ_coeff=orbo)
        v1ao = vresp(dmov)
        v1ov = contract('po,xpq->xoq', orbo, v1ao)
        v1ov = contract('xoq,qv->xov', v1ov, orbv)
        #:v1ov += einsum('xia,ia->xia', zs, e_ia)
        v1ov += zs * e_ia
        return v1ov.reshape(v1ov.shape[0],-1).get()

    return vind, hdiag


class TDBase(lib.StreamObject):
    to_gpu = utils.to_gpu
    device = utils.device
    to_cpu = utils.to_cpu

    conv_tol              = tdhf_cpu.TDBase.conv_tol
    nstates               = tdhf_cpu.TDBase.nstates
    singlet               = tdhf_cpu.TDBase.singlet
    lindep                = tdhf_cpu.TDBase.lindep
    level_shift           = tdhf_cpu.TDBase.level_shift
    max_cycle             = tdhf_cpu.TDBase.max_cycle
    positive_eig_threshold = tdhf_cpu.TDBase.positive_eig_threshold
    deg_eia_thresh        = tdhf_cpu.TDBase.deg_eia_thresh

    _keys = tdhf_cpu.TDBase._keys

    __init__ = tdhf_cpu.TDBase.__init__

    nroots = tdhf_cpu.TDBase.nroots
    e_tot = tdhf_cpu.TDBase.e_tot
    dump_flags = tdhf_cpu.TDBase.dump_flags
    check_sanity = tdhf_cpu.TDBase.check_sanity
    reset = tdhf_cpu.TDBase.reset
    _finalize = tdhf_cpu.TDBase._finalize

    gen_vind = NotImplemented
    get_ab = NotImplemented
    get_precond = tdhf_cpu.TDBase.get_precond

    nuc_grad_method = NotImplemented
    as_scanner = tdhf_cpu.as_scanner

    oscillator_strength = tdhf_cpu.oscillator_strength
    transition_dipole              = tdhf_cpu.transition_dipole
    transition_quadrupole          = tdhf_cpu.transition_quadrupole
    transition_octupole            = tdhf_cpu.transition_octupole
    transition_velocity_dipole     = tdhf_cpu.transition_velocity_dipole
    transition_velocity_quadrupole = tdhf_cpu.transition_velocity_quadrupole
    transition_velocity_octupole   = tdhf_cpu.transition_velocity_octupole
    transition_magnetic_dipole     = tdhf_cpu.transition_magnetic_dipole
    transition_magnetic_quadrupole = tdhf_cpu.transition_magnetic_quadrupole

    def analyze(self, verbose=None):
        self.to_cpu().analyze(verbose)
        return self

    def get_nto(self, state=1, threshold=OUTPUT_THRESHOLD, verbose=None):
        '''
        Natural transition orbital analysis.

        Returns:
            A list (weights, NTOs).  NTOs are natural orbitals represented in AO
            basis. The first N_occ NTOs are occupied NTOs and the rest are virtual
            NTOs. weights and NTOs are all stored in nparray
        '''
        return self.to_cpu().get_nto(state, threshold, verbose)

    # needed by transition dipoles
    def _contract_multipole(tdobj, ints, hermi=True, xy=None):
        '''ints is the integral tensor of a spin-independent operator'''
        if xy is None: xy = tdobj.xy
        nstates = len(xy)
        pol_shape = ints.shape[:-2]
        nao = ints.shape[-1]

        if not tdobj.singlet:
            return np.zeros((nstates,) + pol_shape)

        mo_coeff = tdobj._scf.mo_coeff
        mo_occ = tdobj._scf.mo_occ
        orbo = mo_coeff[:,mo_occ==2]
        orbv = mo_coeff[:,mo_occ==0]
        if isinstance(orbo, cp.ndarray):
            orbo = orbo.get()
            orbv = orbv.get()

        #Incompatible to old np version
        #ints = np.einsum('...pq,pi,qj->...ij', ints, orbo.conj(), orbv)
        ints = lib.einsum('xpq,pi,qj->xij', ints.reshape(-1,nao,nao), orbo.conj(), orbv)
        pol = np.array([np.einsum('xij,ij->x', ints, x) * 2 for x,y in xy])
        if isinstance(xy[0][1], np.ndarray):
            if hermi:
                pol += [np.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
            else:  # anti-Hermitian
                pol -= [np.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
        pol = pol.reshape((nstates,)+pol_shape)
        return pol

class TDA(TDBase):
    __doc__ = tdhf_cpu.TDA.__doc__

    def gen_vind(self, mf=None):
        '''Generate function to compute Ax'''
        if mf is None:
            mf = self._scf
        return gen_tda_operation(mf, singlet=self.singlet)

    def init_guess(self, mf, nstates=None, wfnsym=None, return_symmetry=False):
        '''
        Generate initial guess for TDA

        Kwargs:
            nstates : int
                The number of initial guess vectors.
        '''
        if nstates is None: nstates = self.nstates
        assert wfnsym is None
        assert not return_symmetry

        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        if isinstance(mo_energy, cp.ndarray):
            mo_energy = mo_energy.get()
            mo_occ = mo_occ.get()
        occidx = mo_occ == 2
        viridx = mo_occ == 0
        e_ia = (mo_energy[viridx] - mo_energy[occidx,None]).ravel()
        nov = e_ia.size
        nstates = min(nstates, nov)

        # Find the nstates-th lowest energy gap
        e_threshold = float(np.partition(e_ia, nstates-1)[nstates-1])
        e_threshold += self.deg_eia_thresh

        idx = np.where(e_ia <= e_threshold)[0]
        x0 = np.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations

        return x0

    def kernel(self, x0=None, nstates=None):
        '''TDA diagonalization solver
        '''
        log = logger.Logger(self.stdout, self.verbose)
        t0 = log.init_timer()
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        mol = self.mol

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        def pickeig(w, v, nroots, envs):
            idx = np.where(w > self.positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx

        x0sym = None
        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        self.converged, self.e, x1 = lr_eigh(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        nocc = mol.nelectron // 2
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        # 1/sqrt(2) because self.x is for alpha excitation and 2(X^+*X) = 1
        self.xy = [(xi.reshape(nocc,nvir) * .5**.5, 0) for xi in x1]
        log.timer('TDA', *t0)
        self._finalize()
        return self.e, self.xy

CIS = TDA


def gen_tdhf_operation(mf, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute

    [ A   B ][X]
    [-B* -A*][Y]
    '''
    assert fock_ao is None
    assert isinstance(mf, scf.hf.SCF)
    mo_coeff = mf.mo_coeff
    assert mo_coeff.dtype == cp.float64
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidx = mo_occ == 2
    viridx = mo_occ == 0
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    e_ia = hdiag = mo_energy[viridx] - mo_energy[occidx,None]
    hdiag = cp.hstack((hdiag.ravel(), -hdiag.ravel())).get()
    vresp = mf.gen_response(singlet=singlet, hermi=0)
    nocc, nvir = e_ia.shape

    def vind(xys):
        xys = cp.asarray(xys).reshape(-1,2,nocc,nvir)
        xs, ys = xys.transpose(1,0,2,3)
        # *2 for double occupancy
        tmp = contract('xov,qv->xoq', xs*2, orbv)
        dms = contract('po,xoq->xpq', orbo, tmp)
        tmp = contract('xov,pv->xop', ys*2, orbv)
        dms += contract('xop,qo->xpq', tmp, orbo)
        v1ao = vresp(dms) # = <mb||nj> Xjb + <mj||nb> Yjb
        # A ~= <ib||aj>, B = <ij||ab>
        # AX + BY
        # = <ib||aj> Xjb + <ij||ab> Yjb
        # = (<mb||nj> Xjb + <mj||nb> Yjb) Cmi* Cna
        v1ov = contract('po,xpq->xoq', orbo, v1ao)
        v1ov = contract('xoq,qv->xov', v1ov, orbv)
        # (B*)X + (A*)Y
        # = <ab||ij> Xjb + <aj||ib> Yjb
        # = (<mb||nj> Xjb + <mj||nb> Yjb) Cma* Cni
        v1vo = contract('xpq,qo->xpo', v1ao, orbo)
        v1vo = contract('xpo,pv->xov', v1vo, orbv)
        v1ov += xs * e_ia  # AX
        v1vo += ys * e_ia  # (A*)Y
        # (AX, -AY)
        nz = xys.shape[0]
        hx = cp.hstack((v1ov.reshape(nz,-1), -v1vo.reshape(nz,-1)))
        return hx.get()

    return vind, hdiag


class TDHF(TDBase):
    __doc__ = tdhf_cpu.TDHF.__doc__

    @lib.with_doc(gen_tdhf_operation.__doc__)
    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        return gen_tdhf_operation(mf, singlet=self.singlet)

    def init_guess(self, mf, nstates=None, wfnsym=None, return_symmetry=False):
        x0 = TDA.init_guess(self, mf, nstates, wfnsym, return_symmetry)
        y0 = np.zeros_like(x0)
        return np.hstack([x0, y0])

    def kernel(self, x0=None, nstates=None):
        '''TDHF diagonalization with non-Hermitian eigenvalue solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        mol = self.mol

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        # handle single kpt PBC SCF
        if getattr(self._scf, 'kpt', None) is not None:
            from pyscf.pbc.lib.kpts_helper import gamma_point
            real_system = (gamma_point(self._scf.kpt) and
                           self._scf.mo_coeff[0].dtype == np.double)
        else:
            real_system = True

        # We only need positive eigenvalues
        def pickeig(w, v, nroots, envs):
            realidx = np.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > self.positive_eig_threshold))[0]
            # If the complex eigenvalue has small imaginary part, both the
            # real part and the imaginary part of the eigenvector can
            # approximately be used as the "real" eigen solutions.
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx, real_system)

        x0sym = None
        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        self.converged, w, x1 = lr_eig(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        nocc = mol.nelectron // 2
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        self.e = w
        def norm_xy(z):
            x, y = z.reshape(2,nocc,nvir)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            norm = np.sqrt(.5/norm)  # normalize to 0.5 for alpha spin
            return x*norm, y*norm
        self.xy = [norm_xy(z) for z in x1]

        log.timer('TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

TDRHF = TDHF

scf.hf.RHF.TDA = lib.class_as_method(TDA)
scf.hf.RHF.TDHF = lib.class_as_method(TDHF)
