#!/usr/bin/env python
#
# Copyright 2024 The GPU4PySCF Developers. All Rights Reserved.
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
from gpu4pyscf.tdscf._uhf_resp_sf import gen_uhf_response_sf
from gpu4pyscf.tdscf import rhf as tdhf_gpu
from gpu4pyscf.dft import KohnShamDFT
from pyscf import __config__

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
        mo1_a = contract('xov,qv->xoq', za, orbva)
        dmova = contract('po,xoq->xpq', orboa, mo1_a)
        mo1_b = contract('xov,qv->xoq', zb, orbvb)
        dmovb = contract('po,xoq->xpq', orbob, mo1_b)
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

    def init_guess(self, mf=None, nstates=None, wfnsym=None, return_symmetry=False):
        if mf is None: mf = self._scf
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
        log = logger.new_logger(self)
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        def pickeig(w, v, nroots, envs):
            idx = np.where(w > self.positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx

        x0sym = None
        if x0 is None:
            x0 = self.init_guess()

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

class SpinFlipTDA(TDBase):
    '''
    Attributes:
        extype : int (0 or 1)
            Spin flip up: exytpe=0. Spin flip down: exytpe=1.
        collinear : str
            collinear schemes, can be
            'col': collinear, by default
            'ncol': non-collinear
            'mcol': multi-collinear
        collinear_samples : int
            Integration samples for the multi-collinear treatment
    '''

    extype = getattr(__config__, 'tdscf_uhf_SFTDA_extype', 1)
    collinear = getattr(__config__, 'tdscf_uhf_SFTDA_collinear', 'col')
    collinear_samples = getattr(__config__, 'tdscf_uhf_SFTDA_collinear_samples', 200)

    _keys = {'extype', 'collinear', 'collinear_samples'}

    def gen_vind(self):
        '''Generate function to compute A*x for spin-flip TDDFT case.
        '''
        mf = self._scf
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
        nao, nmo = mo_coeff[0].shape

        extype = self.extype
        if extype == 0:
            occidxb = mo_occ[1] > 0
            viridxa = mo_occ[0] ==0
            orbob = mo_coeff[1][:,occidxb]
            orbva = mo_coeff[0][:,viridxa]
            orbov = (orbob, orbva)
            e_ia = mo_energy[0][viridxa] - mo_energy[1][occidxb,None]
            hdiag = e_ia.ravel().get()

        elif extype == 1:
            occidxa = mo_occ[0] > 0
            viridxb = mo_occ[1] ==0
            orboa = mo_coeff[0][:,occidxa]
            orbvb = mo_coeff[1][:,viridxb]
            orbov = (orboa, orbvb)
            e_ia = mo_energy[1][viridxb] - mo_energy[0][occidxa,None]
            hdiag = e_ia.ravel().get()

        vresp = gen_uhf_response_sf(
            mf, hermi=0, collinear=self.collinear,
            collinear_samples=self.collinear_samples)

        def vind(zs):
            zs = cp.asarray(zs).reshape(-1, *e_ia.shape)
            orbo, orbv = orbov
            dmov = contract('xov,qv->xoq', zs, orbv.conj())
            dmov = contract('po,xoq->xpq', orbo, dmov)
            v1ao = vresp(dmov)
            v1ov = contract('po,xpq->xoq', orbo.conj(), v1ao)
            v1ov = contract('xoq,qv->xov', v1ov, orbv)
            v1ov += zs * e_ia
            return v1ov.reshape(len(v1ov), -1).get()

        return vind, hdiag

    def _init_guess(self, mf, nstates):
        mo_energy_a, mo_energy_b = mf.mo_energy
        mo_occ_a, mo_occ_b = mf.mo_occ
        if isinstance(mo_energy_a, cp.ndarray):
            mo_energy_a = mo_energy_a.get()
            mo_energy_b = mo_energy_b.get()
        if isinstance(mo_occ_a, cp.ndarray):
            mo_occ_a = mo_occ_a.get()
            mo_occ_b = mo_occ_b.get()

        if self.extype == 0:
            occidxb = mo_occ_b > 0
            viridxa = mo_occ_a ==0
            e_ia = mo_energy_a[viridxa] - mo_energy_b[occidxb,None]

        elif self.extype == 1:
            occidxa = mo_occ_a > 0
            viridxb = mo_occ_b ==0
            e_ia = mo_energy_b[viridxb] - mo_energy_a[occidxa,None]

        nov = e_ia.size
        nstates = min(nstates, nov)
        e_threshold = np.partition(e_ia, nstates-1)[nstates-1]
        idx = np.where(e_ia <= e_threshold)[0]
        nstates = idx.size
        e = e_ia[idx]
        idx = idx[np.argsort(e)]
        x0 = np.zeros((nstates, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1
        return np.sort(e), x0.reshape(nstates, *e_ia.shape)

    def init_guess(self, mf=None, nstates=None, wfnsym=None):
        if mf is None: mf = self._scf
        if nstates is None: nstates = self.nstates
        x0 = self._init_guess(mf, nstates)[1]
        return x0.reshape(len(x0), -1)

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, 'extype = %s', self.extype)
        logger.info(self, 'collinear = %s', self.collinear)
        if self.collinear == 'mcol':
            logger.info(self, 'collinear_samples = %s', self.collinear_samples)
        return self

    def check_sanity(self):
        super().check_sanity()
        assert self.extype in (0, 1)
        assert self.collinear in ('col', 'ncol', 'mcol')
        return self

    def kernel(self, x0=None, nstates=None):
        '''Spin-flip TDA diagonalization solver
        '''
        log = logger.new_logger(self)
        cpu0 = log.init_timer()
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        if isinstance(self._scf, KohnShamDFT):
            mf = self._scf
            ni = mf._numint
            if self.collinear == 'col' and not ni.libxc.is_hybrid_xc(mf.xc):
                self.converged = True
                self.e, xs = self._init_guess()
                self.xy = [(x, 0) for x in xs]
                return self.e, self.xy

        x0sym = None
        if x0 is None:
            x0 = self.init_guess()

        # Keep all eigenvalues as SF-TDDFT allows triplet to singlet
        # "dexcitation"
        def all_eigs(w, v, nroots, envs):
            return w, v, np.arange(w.size)

        vind, hdiag = self.gen_vind()
        precond = self.get_precond(hdiag)

        self.converged, self.e, x1 = lr_eig(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=all_eigs, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        nmo = self._scf.mo_occ[0].size
        nocca, noccb = self._scf.nelec
        nvira = nmo - nocca
        nvirb = nmo - noccb

        if self.extype == 0:
            self.xy = [(xi.reshape(noccb,nvira), 0) for xi in x1]
        elif self.extype == 1:
            self.xy = [(xi.reshape(nocca,nvirb), 0) for xi in x1]
        log.timer('SpinFlipTDA', *cpu0)
        self._finalize()
        return self.e, self.xy


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

    def init_guess(self, mf=None, nstates=None, wfnsym=None, return_symmetry=False):
        x0 = TDA.init_guess(self, mf, nstates, wfnsym, return_symmetry)
        y0 = np.zeros_like(x0)
        return np.hstack([x0, y0])

    def kernel(self, x0=None, nstates=None):
        '''TDHF diagonalization with non-Hermitian eigenvalue solver
        '''
        log = logger.new_logger(self)
        cpu0 = log.init_timer()
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

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
            x0 = self.init_guess()

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

class SpinFlipTDHF(TDBase):

    extype = SpinFlipTDA.extype
    collinear = SpinFlipTDA.collinear
    collinear_samples = SpinFlipTDA.collinear_samples

    _keys = {'extype', 'collinear', 'collinear_samples'}

    def gen_vind(self):
        '''Generate function to compute A*x for spin-flip TDDFT case.
        '''
        mf = self._scf
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
        nao, nmo = mo_coeff[0].shape

        occidxa = mo_occ[0] > 0
        occidxb = mo_occ[1] > 0
        viridxa = mo_occ[0] ==0
        viridxb = mo_occ[1] ==0
        orboa = mo_coeff[0][:,occidxa]
        orbob = mo_coeff[1][:,occidxb]
        orbva = mo_coeff[0][:,viridxa]
        orbvb = mo_coeff[1][:,viridxb]
        e_ia_b2a = mo_energy[0][viridxa] - mo_energy[1][occidxb,None]
        e_ia_a2b = mo_energy[1][viridxb] - mo_energy[0][occidxa,None]
        nocca, nvirb = e_ia_a2b.shape
        noccb, nvira = e_ia_b2a.shape

        extype = self.extype
        if extype == 0:
            hdiag = cp.hstack([e_ia_b2a.ravel(), -e_ia_a2b.ravel()]).get()
        else:
            hdiag = cp.hstack([e_ia_a2b.ravel(), -e_ia_b2a.ravel()]).get()

        vresp = gen_uhf_response_sf(
            mf, hermi=0, collinear=self.collinear,
            collinear_samples=self.collinear_samples)

        def vind(zs):
            nz = len(zs)
            zs = cp.asarray(zs).reshape(nz, -1)
            if extype == 0:
                zs_b2a = zs[:,:noccb*nvira].reshape(nz,noccb,nvira)
                zs_a2b = zs[:,noccb*nvira:].reshape(nz,nocca,nvirb)
            else:
                zs_a2b = zs[:,:nocca*nvirb].reshape(nz,nocca,nvirb)
                zs_b2a = zs[:,nocca*nvirb:].reshape(nz,noccb,nvira)

            dm_b2a = contract('xov,qv->xoq', zs_b2a, orbva.conj())
            dm_b2a = contract('po,xoq->xpq', orbob, dm_b2a)
            dm_a2b = contract('xov,qv->xoq', zs_a2b, orbvb.conj())
            dm_a2b = contract('po,xoq->xpq', orboa, dm_a2b)
            dmov = cp.vstack([dm_b2a, dm_a2b])
            v1ao = vresp(dmov)
            v1ao_b2a, v1ao_a2b = v1ao[:nz], v1ao[nz:]

            v1ov_b2a = contract('po,xpq->xoq', orbob.conj(), v1ao_b2a)
            v1ov_b2a = contract('xoq,qv->xov', v1ov_b2a, orbva)
            v1ov_a2b = contract('po,xpq->xoq', orboa.conj(), v1ao_a2b)
            v1ov_a2b = contract('xoq,qv->xov', v1ov_a2b, orbvb)
            v1vo_b2a = contract('qo,xpq->xop', orbob, v1ao_a2b)
            v1vo_b2a = contract('xop,pv->xov', v1vo_b2a, orbva.conj())
            v1vo_a2b = contract('qo,xpq->xop', orboa, v1ao_b2a)
            v1vo_a2b = contract('xop,pv->xov', v1vo_a2b, orbvb.conj())

            # add the orbital energy difference in A matrix.
            v1ov_b2a += zs_b2a * e_ia_b2a
            v1ov_a2b += zs_a2b * e_ia_a2b

            if extype == 0:
                v1_top = v1ov_b2a + v1vo_b2a
                v1_bom = (v1vo_a2b + v1ov_a2b).conj()
            else:
                v1_top = v1ov_a2b + v1vo_a2b
                v1_bom = (v1vo_b2a + v1ov_b2a).conj()
            hx = cp.hstack([v1_top.reshape(nz,-1), -v1_bom.reshape(nz,-1)])
            return hx.get()

        return vind, hdiag

    _init_guess = SpinFlipTDA._init_guess

    def init_guess(self, mf=None, nstates=None, wfnsym=None):
        if mf is None: mf = self._scf
        if nstates is None: nstates = self.nstates
        x0 = self._init_guess(mf, nstates)[1]
        nx = len(x0)
        nmo = mf.mo_occ[0].size
        nocca, noccb = mf.nelec
        nvira = nmo - nocca
        nvirb = nmo - noccb
        if self.extype == 0:
            y0 = cp.zeros((nx, nocca*nvirb))
        else:
            y0 = cp.zeros((nx, noccb*nvira))
        return cp.hstack([x0.reshape(nx,-1), y0])

    dump_flags = SpinFlipTDA.dump_flags
    check_sanity = SpinFlipTDA.check_sanity

    def kernel(self, x0=None, nstates=None):
        '''Spin-flip TDA diagonalization solver
        '''
        log = logger.new_logger(self)
        cpu0 = log.init_timer()
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        if isinstance(self._scf, KohnShamDFT):
            raise NotImplementedError

        x0sym = None
        if x0 is None:
            x0 = self.init_guess()

        # Keep all eigenvalues as SF-TDDFT allows triplet to singlet
        # "dexcitation"
        def all_eigs(w, v, nroots, envs):
            return w, v, np.arange(w.size)

        vind, hdiag = self.gen_vind()
        precond = self.get_precond(hdiag)

        self.converged, self.e, x1 = lr_eig(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=all_eigs, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        nmo = self._scf.mo_occ[0].size
        nocca, noccb = self._scf.nelec
        nvira = nmo - nocca
        nvirb = nmo - noccb

        if self.extype == 0:
            def norm_xy(z):
                x = z[:noccb*nvira].reshape(noccb,nvira)
                y = z[noccb*nvira:].reshape(nocca,nvirb)
                norm = lib.norm(x)**2 - lib.norm(y)**2
                assert norm > 0
                norm = norm ** -.5
                return x*norm, y*norm
        elif self.extype == 1:
            def norm_xy(z):
                x = z[:nocca*nvirb].reshape(nocca,nvirb)
                y = z[nocca*nvirb:].reshape(noccb,nvira)
                norm = lib.norm(x)**2 - lib.norm(y)**2
                assert norm > 0
                norm = norm ** -.5
                return x*norm, y*norm

        self.xy = [norm_xy(z) for z in x1]
        log.timer('SpinFlipTDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

scf.uhf.UHF.TDA = lib.class_as_method(TDA)
scf.uhf.UHF.TDHF = lib.class_as_method(TDHF)
scf.uhf.UHF.SFTDA = lib.class_as_method(SpinFlipTDA)
scf.uhf.UHF.SFTDHF = lib.class_as_method(SpinFlipTDHF)
