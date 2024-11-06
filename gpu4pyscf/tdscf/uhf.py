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
from pyscf import lib
from pyscf.tdscf import uhf as tdhf_cpu
from pyscf.data.nist import HARTREE2EV, HARTREE2WAVENUMBER
from pyscf.tdscf._lr_eig import eigh as lr_eigh, eig as lr_eig
from gpu4pyscf import scf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.tdscf import rhf as tdhf_gpu

__all__ = [
    'TDA', 'CIS', 'TDHF', 'TDUHF', 'TDBase'
]

REAL_EIG_THRESHOLD = tdhf_cpu.REAL_EIG_THRESHOLD

def gen_tda_operation(mf, fock_ao=None, wfnsym=None):
    '''A x
    '''
    assert fock_ao is None
    assert isinstance(mf, scf.hf.SCF)
    assert wfnsym is None
    if isinstance(mf.mo_coeff, (tuple, list)):
        # The to_gpu() in pyscf is not able to convert SymAdaptedUHF.mo_coeff.
        # In this case, mf.mo_coeff has the type (NPArrayWithTag, NPArrayWithTag).
        # cp.asarray() for this object leads to an error in
        # cupy._core.core._array_from_nested_sequence
        mo_coeff = cp.asarray(mf.mo_coeff[0]), cp.asarray(mf.mo_coeff[1])
    else:
        mo_coeff = cp.asarray(mf.mo_coeff)
    assert mo_coeff[0].dtype == cp.float64
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff[0].shape
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = mo_occ[0] ==0
    viridxb = mo_occ[1] ==0
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]

    e_ia_a = mo_energy[0][viridxa] - mo_energy[0][occidxa,None]
    e_ia_b = mo_energy[1][viridxb] - mo_energy[1][occidxb,None]
    e_ia = cp.hstack((e_ia_a.reshape(-1), e_ia_b.reshape(-1)))
    hdiag = e_ia.get()
    nocca, nvira = e_ia_a.shape
    noccb, nvirb = e_ia_b.shape

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.8-mem_now)
    vresp = mf.gen_response(hermi=0, max_memory=max_memory)

    def vind(zs):
        nz = len(zs)
        zs = cp.asarray(zs)
        za = zs[:,:nocca*nvira].reshape(nz,nocca,nvira)
        zb = zs[:,nocca*nvira:].reshape(nz,noccb,nvirb)
        mo1_a = contract('xov,qv->xqo', za, orbva)
        dmova = contract('po,xqo->xpq', orboa, mo1_a)
        mo1_b = contract('xov,qv->xqo', zb, orbvb)
        dmovb = contract('po,xqo->xpq', orbob, mo1_b)
        dm1 = cp.asarray((dmova, dmovb))
        dm1 = tag_array(dm1, mo1=[mo1_a,mo1_b], occ_coeff=[orboa,orbob])
        v1ao = vresp(dm1)
        v1a = contract('po,xpq->xoq', orboa, v1ao[0])
        v1a = contract('xoq,qv->xov', v1a, orbva)
        v1b = contract('po,xpq->xoq', orbob, v1ao[1])
        v1b = contract('xoq,qv->xov', v1b, orbvb)
        v1a += za * e_ia_a
        v1b += zb * e_ia_b
        hx = cp.hstack((v1a.reshape(nz,-1), v1b.reshape(nz,-1)))
        return hx.get()

    return vind, hdiag

class TDBase(tdhf_gpu.TDBase):
    def _contract_multipole(tdobj, ints, hermi=True, xy=None):
        if xy is None: xy = tdobj.xy
        mo_coeff = tdobj._scf.mo_coeff
        mo_occ = tdobj._scf.mo_occ
        orbo_a = mo_coeff[0][:,mo_occ[0]==1]
        orbv_a = mo_coeff[0][:,mo_occ[0]==0]
        orbo_b = mo_coeff[1][:,mo_occ[1]==1]
        orbv_b = mo_coeff[1][:,mo_occ[1]==0]
        if isinstance(orbo_a, cp.ndarray):
            orbo_a = orbo_a.get()
            orbv_a = orbv_a.get()
            orbo_b = orbo_b.get()
            orbv_b = orbv_b.get()

        ints_a = np.einsum('...pq,pi,qj->...ij', ints, orbo_a.conj(), orbv_a)
        ints_b = np.einsum('...pq,pi,qj->...ij', ints, orbo_b.conj(), orbv_b)
        pol = [(np.einsum('...ij,ij->...', ints_a, x[0]) +
                np.einsum('...ij,ij->...', ints_b, x[1])) for x,y in xy]
        pol = np.array(pol)
        y = xy[0][1]
        if isinstance(y[0], np.ndarray):
            pol_y = [(np.einsum('...ij,ij->...', ints_a, y[0]) +
                      np.einsum('...ij,ij->...', ints_b, y[1])) for x,y in xy]
            if hermi:
                pol += pol_y
            else:  # anti-Hermitian
                pol -= pol_y
        return pol


class TDA(TDBase):
    __doc__ = tdhf_gpu.TDA.__doc__

    singlet = None

    def gen_vind(self, mf=None):
        '''Generate function to compute Ax'''
        if mf is None:
            mf = self._scf
        return gen_tda_operation(mf)

    def init_guess(self, mf, nstates=None, wfnsym=None, return_symmetry=False):
        if nstates is None: nstates = self.nstates
        assert wfnsym is None
        assert not return_symmetry

        mo_energy_a, mo_energy_b = mf.mo_energy
        mo_occ_a, mo_occ_b = mf.mo_occ
        if isinstance(mo_energy_a, cp.ndarray):
            mo_energy_a = mo_energy_a.get()
            mo_energy_b = mo_energy_b.get()
        if isinstance(mo_occ_a, cp.ndarray):
            mo_occ_a = mo_occ_a.get()
            mo_occ_b = mo_occ_b.get()
        occidxa = mo_occ_a >  0
        occidxb = mo_occ_b >  0
        viridxa = mo_occ_a == 0
        viridxb = mo_occ_b == 0
        e_ia_a = mo_energy_a[viridxa] - mo_energy_a[occidxa,None]
        e_ia_b = mo_energy_b[viridxb] - mo_energy_b[occidxb,None]
        nov = e_ia_a.size + e_ia_b.size
        nstates = min(nstates, nov)

        e_ia = np.append(e_ia_a.ravel(), e_ia_b.ravel())
        # Find the nstates-th lowest energy gap
        e_threshold = np.partition(e_ia, nstates-1)[nstates-1]
        e_threshold += self.deg_eia_thresh

        idx = np.where(e_ia <= e_threshold)[0]
        x0 = np.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1
        return x0

    def kernel(self, x0=None, nstates=None):
        '''TDA diagonalization solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        log = logger.Logger(self.stdout, self.verbose)

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

        nmo = self._scf.mo_occ[0].size
        nocca, noccb = self._scf.nelec
        nvira = nmo - nocca
        nvirb = nmo - noccb
        self.xy = [((xi[:nocca*nvira].reshape(nocca,nvira),  # X_alpha
                     xi[nocca*nvira:].reshape(noccb,nvirb)), # X_beta
                    (0, 0))  # (Y_alpha, Y_beta)
                   for xi in x1]

        log.timer('TDA', *cpu0)
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
    if isinstance(mf.mo_coeff, (tuple, list)):
        # The to_gpu() in pyscf is not able to convert SymAdaptedUHF.mo_coeff.
        # In this case, mf.mo_coeff has the type (NPArrayWithTag, NPArrayWithTag).
        # cp.asarray() for this object leads to an error in
        # cupy._core.core._array_from_nested_sequence
        mo_coeff = cp.asarray(mf.mo_coeff[0]), cp.asarray(mf.mo_coeff[1])
    else:
        mo_coeff = cp.asarray(mf.mo_coeff)
    assert mo_coeff[0].dtype == cp.float64
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    occidxa = mo_occ[0] >  0
    occidxb = mo_occ[1] >  0
    viridxa = mo_occ[0] == 0
    viridxb = mo_occ[1] == 0
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]

    e_ia_a = mo_energy[0][viridxa] - mo_energy[0][occidxa,None]
    e_ia_b = mo_energy[1][viridxb] - mo_energy[1][occidxb,None]
    e_ia = hdiag = cp.hstack((e_ia_a.ravel(), e_ia_b.ravel()))
    hdiag = cp.hstack((hdiag, -hdiag)).get()
    nocca, nvira = e_ia_a.shape
    noccb, nvirb = e_ia_b.shape

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.8-mem_now)
    vresp = mf.gen_response(hermi=0, max_memory=max_memory)

    def vind(xys):
        nz = len(xys)
        xys = cp.asarray(xys).reshape(nz,2,-1)
        xs, ys = xys.transpose(1,0,2)
        xa = xs[:,:nocca*nvira].reshape(nz,nocca,nvira)
        xb = xs[:,nocca*nvira:].reshape(nz,noccb,nvirb)
        ya = ys[:,:nocca*nvira].reshape(nz,nocca,nvira)
        yb = ys[:,nocca*nvira:].reshape(nz,noccb,nvirb)
        tmp = contract('xov,qv->xoq', xa, orbva)
        dmsa = contract('po,xoq->xpq', orboa, tmp)
        tmp = contract('xov,qv->xoq', xb, orbvb)
        dmsb = contract('po,xoq->xpq', orbob, tmp)
        tmp = contract('xov,pv->xop', ya, orbva)
        dmsa += contract('xop,qo->xpq', tmp, orboa)
        tmp = contract('xov,pv->xop', yb, orbvb)
        dmsb += contract('xop,qo->xpq', tmp, orbob)
        v1ao = vresp(cp.asarray((dmsa,dmsb)))
        v1aov = contract('po,xpq->xoq', orboa, v1ao[0])
        v1aov = contract('xoq,qv->xov', v1aov, orbva)
        v1bov = contract('po,xpq->xoq', orbob, v1ao[1])
        v1bov = contract('xoq,qv->xov', v1bov, orbvb)
        v1avo = contract('xpq,qo->xpo', v1ao[0], orboa)
        v1avo = contract('xpo,pv->xov', v1avo, orbva)
        v1bvo = contract('xpq,qo->xpo', v1ao[1], orbob)
        v1bvo = contract('xpo,pv->xov', v1bvo, orbvb)

        v1ov = xs * e_ia  # AX
        v1vo = ys * e_ia  # AY
        v1ov[:,:nocca*nvira] += v1aov.reshape(nz,-1)
        v1vo[:,:nocca*nvira] += v1avo.reshape(nz,-1)
        v1ov[:,nocca*nvira:] += v1bov.reshape(nz,-1)
        v1vo[:,nocca*nvira:] += v1bvo.reshape(nz,-1)
        hx = cp.hstack((v1ov, -v1vo))
        return hx.get()

    return vind, hdiag


class TDHF(TDBase):

    singlet = None

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
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx, real_system)

        x0sym = None
        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        self.converged, w, x1 = lr_eig(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        nmo = self._scf.mo_occ[0].size
        nocca, noccb = self._scf.nelec
        nvira = nmo - nocca
        nvirb = nmo - noccb
        e = []
        xy = []
        for i, z in enumerate(x1):
            x, y = z.reshape(2,-1)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            if norm > 0:
                norm = norm**-.5
                e.append(w[i])
                xy.append(((x[:nocca*nvira].reshape(nocca,nvira) * norm,  # X_alpha
                            x[nocca*nvira:].reshape(noccb,nvirb) * norm), # X_beta
                           (y[:nocca*nvira].reshape(nocca,nvira) * norm,  # Y_alpha
                            y[nocca*nvira:].reshape(noccb,nvirb) * norm)))# Y_beta
        self.e = np.array(e)
        self.xy = xy

        log.timer('TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

TDUHF = TDHF

scf.uhf.UHF.TDA = lib.class_as_method(TDA)
scf.uhf.UHF.TDHF = lib.class_as_method(TDHF)
