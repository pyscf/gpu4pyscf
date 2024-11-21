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
from pyscf import symm
from pyscf import lib
from pyscf.tdscf._lr_eig import eigh as lr_eigh
from gpu4pyscf.dft.rks import KohnShamDFT
from gpu4pyscf.lib.cupy_helper import contract, tag_array, transpose_sum
from gpu4pyscf.lib import logger
from gpu4pyscf.tdscf import uhf as tdhf_gpu
from gpu4pyscf import dft

__all__ = [
    'TDA', 'TDDFT', 'TDUKS', 'CasidaTDDFT', 'TDDFTNoHybrid',
]

TDA = tdhf_gpu.TDA
TDDFT = tdhf_gpu.TDHF
TDUKS = TDDFT
SpinFlipTDA = tdhf_gpu.SpinFlipTDA
SpinFlipTDDFT = tdhf_gpu.SpinFlipTDHF

class CasidaTDDFT(TDDFT):
    '''Solve the Casida TDDFT formula (A-B)(A+B)(X+Y) = (X+Y)w^2
    '''

    init_guess = TDA.init_guess
    get_precond = TDA.get_precond

    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
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
        e_ia = cp.hstack((e_ia_a.ravel(), e_ia_b.ravel()))
        d_ia = e_ia**.5
        ed_ia = e_ia * d_ia
        hdiag = e_ia ** 2
        hdiag = hdiag.get()
        vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)
        nocca, nvira = e_ia_a.shape
        noccb, nvirb = e_ia_b.shape

        def vind(zs):
            assert zs.dtype == np.float64
            nz = len(zs)
            zs = cp.asarray(zs).reshape(nz,-1)
            dmsa = (zs[:,:nocca*nvira] * d_ia[:nocca*nvira]).reshape(nz,nocca,nvira)
            dmsb = (zs[:,nocca*nvira:] * d_ia[nocca*nvira:]).reshape(nz,noccb,nvirb)
            mo1a = contract('xov,pv->xpo', dmsa, orbva)
            dmsa = contract('xpo,qo->xpq', mo1a, orboa)
            mo1b = contract('xov,pv->xpo', dmsb, orbvb)
            dmsb = contract('xpo,qo->xpq', mo1b, orbob)
            dmsa = transpose_sum(dmsa)
            dmsb = transpose_sum(dmsb)
            dms = cp.asarray((dmsa, dmsb))
            dms = tag_array(dms, mo1=[mo1a,mo1b], occ_coeff=[orboa,orbob])
            v1ao = vresp(dms)
            v1a = contract('xpq,qo->xpo', v1ao[0], orboa)
            v1a = contract('xpo,pv->xov', v1a, orbva)
            v1b = contract('xpq,qo->xpo', v1ao[1], orbob)
            v1b = contract('xpo,pv->xov', v1b, orbvb)
            hx = cp.hstack((v1a.reshape(nz,-1), v1b.reshape(nz,-1)))
            hx += ed_ia * zs
            hx *= d_ia
            return hx.get()

        return vind, hdiag

    def kernel(self, x0=None, nstates=None):
        '''TDDFT diagonalization solver
        '''
        log = logger.new_logger(self)
        cpu0 = log.init_timer()
        mf = self._scf
        if mf._numint.libxc.is_hybrid_xc(mf.xc):
            raise RuntimeError('%s cannot be used with hybrid functional'
                               % self.__class__)
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

        self.converged, w2, x1 = lr_eigh(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        mo_energy = self._scf.mo_energy
        mo_occ = self._scf.mo_occ
        occidxa = mo_occ[0] >  0
        occidxb = mo_occ[1] >  0
        viridxa = mo_occ[0] == 0
        viridxb = mo_occ[1] == 0
        e_ia_a = mo_energy[0][viridxa] - mo_energy[0][occidxa,None]
        e_ia_b = mo_energy[1][viridxb] - mo_energy[1][occidxb,None]
        nocca, nvira = e_ia_a.shape
        noccb, nvirb = e_ia_b.shape
        if isinstance(mo_energy, cp.ndarray):
            e_ia = cp.hstack((e_ia_a.reshape(-1), e_ia_b.reshape(-1)))
            e_ia = e_ia**.5
            e_ia = e_ia.get()
        else:
            e_ia = np.hstack((e_ia_a.reshape(-1), e_ia_b.reshape(-1)))
            e_ia = e_ia**.5

        e = []
        xy = []
        for i, z in enumerate(x1):
            if w2[i] < self.positive_eig_threshold:
                continue
            w = w2[i] ** .5
            zp = e_ia * z
            zm = w/e_ia * z
            x = (zp + zm) * .5
            y = (zp - zm) * .5
            norm = lib.norm(x)**2 - lib.norm(y)**2
            if norm > 0:
                norm = norm**-.5
                e.append(w)
                xy.append(((x[:nocca*nvira].reshape(nocca,nvira) * norm,  # X_alpha
                            x[nocca*nvira:].reshape(noccb,nvirb) * norm), # X_beta
                           (y[:nocca*nvira].reshape(nocca,nvira) * norm,  # Y_alpha
                            y[nocca*nvira:].reshape(noccb,nvirb) * norm)))# Y_beta
        self.e = np.array(e)
        self.xy = xy

        log.timer('TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

TDDFTNoHybrid = CasidaTDDFT

def tddft(mf):
    '''Driver to create TDDFT or CasidaTDDFT object'''
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        return TDDFT(mf)
    else:
        return CasidaTDDFT(mf)

dft.uks.UKS.TDA           = lib.class_as_method(TDA)
dft.uks.UKS.TDHF          = None
#dft.uks.UKS.TDDFT         = lib.class_as_method(TDDFT)
dft.uks.UKS.TDDFTNoHybrid = lib.class_as_method(TDDFTNoHybrid)
dft.uks.UKS.CasidaTDDFT   = lib.class_as_method(CasidaTDDFT)
dft.uks.UKS.TDDFT         = tddft
dft.uks.UKS.SFTDA         = lib.class_as_method(SpinFlipTDA)
dft.uks.UKS.SFTDDFT       = lib.class_as_method(SpinFlipTDDFT)
