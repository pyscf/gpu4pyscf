#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
# Modified by Xiaojie Wu <wxj6000@gmail.com>

import copy
import cupy
import numpy
from cupy import cublas
from pyscf import lib, __config__
from pyscf.scf import dhf
from pyscf.df import df_jk, addons
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, take_last2d, transpose_sum, load_library, get_avail_mem
from gpu4pyscf.dft import rks, uks, numint
from gpu4pyscf.scf import hf, uhf
from gpu4pyscf.df import df, int3c2e

libcupy_helper = load_library('libcupy_helper')

def _pin_memory(array):
    mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
    ret = numpy.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret

def init_workflow(mf, dm0=None):
    # build CDERI for omega = 0 and omega ! = 0
    def build_df():
        mf.with_df.build()
        if hasattr(mf, '_numint'):
            omega, _, _ = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)
            if abs(omega) <= 1e-10: return
            key = '%.6f' % omega
            if key in mf.with_df._rsh_df:
                rsh_df = mf.with_df._rsh_df[key]
            else:
                rsh_df = mf.with_df._rsh_df[key] = copy.copy(mf.with_df).reset()
            rsh_df.build(omega=omega)
        return

    # pre-compute h1e and s1e and cderi for async workflow
    with lib.call_in_background(build_df) as build:
        build()
        mf.s1e = cupy.asarray(mf.get_ovlp(mf.mol))
        mf.h1e = cupy.asarray(mf.get_hcore(mf.mol))
        # for DFT object
        if hasattr(mf, '_numint'):
            ni = mf._numint
            rks.initialize_grids(mf, mf.mol, dm0)
            ni.build(mf.mol, mf.grids.coords)
            mf._numint.xcfuns = numint._init_xcfuns(mf.xc, dm0.ndim==3)
    dm0 = cupy.asarray(dm0)
    return

def _density_fit(mf, auxbasis=None, with_df=None, only_dfj=False):
    '''For the given SCF object, update the J, K matrix constructor with
    corresponding density fitting integrals.
    Args:
        mf : an SCF object
    Kwargs:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, optimal auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.
        only_dfj : str
            Compute Coulomb integrals only and no approximation for HF
            exchange. Same to RIJONX in ORCA
    Returns:
        An SCF object with a modified J, K matrix constructor which uses density
        fitting integrals to compute J and K
    Examples:
    '''

    assert isinstance(mf, hf.SCF)

    if with_df is None:
        if isinstance(mf, dhf.UHF):
            with_df = df.DF4C(mf.mol)
        else:
            with_df = df.DF(mf.mol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    if isinstance(mf, _DFHF):
        if mf.with_df is None:
            mf.with_df = with_df
        elif getattr(mf.with_df, 'auxbasis', None) != auxbasis:
            #logger.warn(mf, 'DF might have been initialized twice.')
            mf = copy.copy(mf)
            mf.with_df = with_df
            mf.only_dfj = only_dfj
        return mf

    dfmf = _DFHF(mf, with_df, only_dfj)
    return lib.set_class(dfmf, (_DFHF, mf.__class__))

from gpu4pyscf.lib import utils
class _DFHF:
    '''
    Density fitting SCF class
    Attributes for density-fitting SCF:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.
            The default basis 'weigend+etb' means weigend-coulomb-fit basis
            for light elements and even-tempered basis for heavy elements.
        with_df : DF object
            Set mf.with_df = None to switch off density fitting mode.
    '''
    to_gpu = utils.to_gpu
    device = utils.device
    __name_mixin__ = 'DF'
    _keys = {'rhoj', 'rhok', 'disp', 'screen_tol'}

    def __init__(self, mf, dfobj, only_dfj):
        self.__dict__.update(mf.__dict__)
        self._eri = None
        self.rhoj = None
        self.rhok = None
        self.direct_scf = False
        self.with_df = dfobj
        self.only_dfj = only_dfj
        self._keys = mf._keys.union(['with_df', 'only_dfj'])

    def undo_df(self):
        '''Remove the DFHF Mixin'''
        obj = lib.view(self, lib.drop_class(self.__class__, _DFHF))
        del obj.rhoj, obj.rhok, obj.with_df, obj.only_dfj
        return obj

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return super().reset(mol)

    init_workflow = init_workflow

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if dm is None: dm = self.make_rdm1()
        if self.with_df and self.only_dfj:
            vj = vk = None
            if with_j:
                vj, vk = self.with_df.get_jk(dm, hermi, True, False,
                                             self.direct_scf_tol, omega)
            if with_k:
                vk = super().get_jk(mol, dm, hermi, False, True, omega)[1]
        elif self.with_df:
            vj, vk = self.with_df.get_jk(dm, hermi, with_j, with_k,
                                         self.direct_scf_tol, omega)
        else:
            vj, vk = super().get_jk(mol, dm, hermi, with_j, with_k, omega)
        return vj, vk

    def nuc_grad_method(self):
        if isinstance(self, rks.RKS):
            from gpu4pyscf.df.grad import rks as rks_grad
            return rks_grad.Gradients(self)
        if isinstance(self, hf.RHF):
            from gpu4pyscf.df.grad import rhf as rhf_grad
            return rhf_grad.Gradients(self)
        if isinstance(self, uks.UKS):
            from gpu4pyscf.df.grad import uks as uks_grad
            return uks_grad.Gradients(self)
        if isinstance(self, uhf.UHF):
            from gpu4pyscf.df.grad import uhf as uhf_grad
            return uhf_grad.Gradients(self)
        raise NotImplementedError()

    def Hessian(self):
        from gpu4pyscf.dft.rks import KohnShamDFT
        if isinstance(self, hf.RHF):
            if isinstance(self, KohnShamDFT):
                from gpu4pyscf.df.hessian import rks as rks_hess
                return rks_hess.Hessian(self)
            else:
                from gpu4pyscf.df.hessian import rhf as rhf_hess
                return rhf_hess.Hessian(self)
        elif isinstance(self, uhf.UHF):
            if isinstance(self, KohnShamDFT):
                from gpu4pyscf.df.hessian import uks as uks_hess
                return uks_hess.Hessian(self)
            else:
                from gpu4pyscf.df.hessian import uhf as uhf_hess
                return uhf_hess.Hessian(self)
        else:
            raise NotImplementedError
    @property
    def auxbasis(self):
        return getattr(self.with_df, 'auxbasis', None)

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=0, hermi=1):
        '''
        effective potential
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        # for DFT
        if isinstance(self, rks.KohnShamDFT):
            if dm.ndim == 2:
                return rks.get_veff(self, dm=dm)
            elif dm.ndim == 3:
                return uks.get_veff(self, dm=dm)

        if dm.ndim == 2:
            if self.direct_scf:
                ddm = cupy.asarray(dm) - dm_last
                vj, vk = self.get_jk(mol, ddm, hermi=hermi)
                return vhf_last + vj - vk * .5
            else:
                vj, vk = self.get_jk(mol, dm, hermi=hermi)
                return vj - vk * .5
        elif dm.ndim == 3:
            if self.direct_scf:
                ddm = cupy.asarray(dm) - dm_last
                vj, vk = self.get_jk(mol, ddm, hermi=hermi)
                vhf = vj[0] + vj[1] - vk
                vhf += cupy.asarray(vhf_last)
                return vhf
            else:
                vj, vk = self.get_jk(mol, dm, hermi=hermi)
                return vj[0] + vj[1] - vk
        else:
            raise NotImplementedError("Please check the dimension of the density matrix, it should not reach here.")

    def to_cpu(self):
        obj = self.undo_df().to_cpu().density_fit()
        return utils.to_cpu(self, obj)

def get_jk(dfobj, dms_tag, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-14, omega=None):
    '''
    get jk with density fitting
    outputs and input are on the same device
    TODO: separate into three cases: j only, k only, j and k
    '''

    log = logger.new_logger(dfobj.mol, dfobj.verbose)
    out_shape = dms_tag.shape
    out_cupy = isinstance(dms_tag, cupy.ndarray)
    if not isinstance(dms_tag, cupy.ndarray):
        dms_tag = cupy.asarray(dms_tag)

    assert(with_j or with_k)
    if dms_tag is None: logger.error("dm is not given")
    nao = dms_tag.shape[-1]
    dms = dms_tag.reshape([-1,nao,nao])
    nset = dms.shape[0]
    t1 = t0 = log.init_timer()
    if dfobj._cderi is None:
        log.debug('Build CDERI ...')
        dfobj.build(direct_scf_tol=direct_scf_tol, omega=omega)
        t1 = log.timer_debug1('init jk', *t0)

    assert nao == dfobj.nao
    vj = vk = None
    ao_idx = dfobj.intopt.ao_idx
    dms = take_last2d(dms, ao_idx)
    dms_shape = dms.shape
    rows = dfobj.intopt.cderi_row
    cols = dfobj.intopt.cderi_col
    
    if with_j:
        dm_sparse = dms[:,rows,cols]
        dm_sparse[:, dfobj.intopt.cderi_diag] *= .5

    if with_k:
        vk = cupy.zeros_like(dms)
    
    # SCF K matrix with occ
    if getattr(dms_tag, 'mo_coeff', None) is not None:
        mo_occ = dms_tag.mo_occ
        mo_coeff = dms_tag.mo_coeff
        nmo = mo_occ.shape[-1]
        mo_coeff = mo_coeff.reshape(-1,nao,nmo)
        mo_occ   = mo_occ.reshape(-1,nmo)
        nocc = 0
        occ_coeff = [0]*nset
        for i in range(nset):
            occ_idx = mo_occ[i] > 0
            occ_coeff[i] = mo_coeff[i][:,occ_idx][ao_idx] * mo_occ[i][occ_idx]**0.5
            nocc += mo_occ[i].sum()
        blksize = dfobj.get_blksize(extra=nao*nocc)
        if with_j:
            vj_packed = cupy.zeros_like(dm_sparse)
        for cderi, cderi_sparse in dfobj.loop(blksize=blksize, unpack=with_k):
            # leading dimension is 1
            if with_j:
                rhoj = 2.0*dm_sparse.dot(cderi_sparse)
                vj_packed += cupy.dot(rhoj, cderi_sparse.T)
            cderi_sparse = rhoj = None
            for i in range(nset):
                if with_k:
                    rhok = contract('Lji,jk->Lki', cderi, occ_coeff[i])
                    # In most cases, syrk does not outperform cupy.dot
                    #cublas.syrk('T', rhok.reshape([-1,nao]), out=vk[i], alpha=1.0, beta=1.0, lower=True)
                    rhok = rhok.reshape([-1,nao])
                    vk[i] += cupy.dot(rhok.T, rhok)
                rhok = None
        if with_j:
            vj = cupy.zeros(dms_shape)
            vj[:,rows,cols] = vj_packed
            vj[:,cols,rows] = vj_packed

    # CP-HF K matrix
    elif hasattr(dms_tag, 'mo1'):
        occ_coeffs = dms_tag.occ_coeff
        mo1s = dms_tag.mo1
        mo_occ = dms_tag.mo_occ
        if not isinstance(occ_coeffs, list):
            occ_coeffs = [occ_coeffs * 2.0] # For restricted
        if not isinstance(mo1s, list):
            mo1s = [mo1s]

        occ_coeffs = [occ_coeff[ao_idx] for occ_coeff in occ_coeffs]
        mo1s = [mo1[:,ao_idx] for mo1 in mo1s]

        if with_j:
            vj_sparse = cupy.zeros_like(dm_sparse)

        nocc = max([mo1.shape[2] for mo1 in mo1s])
        blksize = dfobj.get_blksize(extra=2*nao*nocc)
        for cderi, cderi_sparse in dfobj.loop(blksize=blksize, unpack=with_k):
            if with_j:
                rhoj = 2.0*dm_sparse.dot(cderi_sparse)
                vj_sparse += cupy.dot(rhoj, cderi_sparse.T)
                rhoj = None
            cderi_sparse = None
            if with_k:
                iset = 0
                for occ_coeff, mo1 in zip(occ_coeffs, mo1s):
                    rhok = contract('Lij,jk->Lki', cderi, occ_coeff).reshape([-1,nao])
                    for i in range(mo1.shape[0]):
                        rhok1 = contract('Lij,jk->Lki', cderi, mo1[i]).reshape([-1,nao])
                        #contract('Lki,Lkj->ij', rhok, rhok1, alpha=1.0, beta=1.0, out=vk[iset])
                        vk[iset] += cupy.dot(rhok.T, rhok1)
                        iset += 1
                mo1 = rhok1 = rhok = None
            cderi = None
        mo1s = None
        if with_j:
            vj = cupy.zeros(dms_shape)
            vj[:,rows,cols] = vj_sparse
            vj[:,cols,rows] = vj_sparse
        if with_k:
            transpose_sum(vk)
        vj_sparse = None
    # general K matrix with density matrix
    else:
        if with_j:
            vj_sparse = cupy.zeros_like(dm_sparse)
        blksize = dfobj.get_blksize()
        for cderi, cderi_sparse in dfobj.loop(blksize=blksize, unpack=with_k):
            if with_j:
                rhoj = 2.0*dm_sparse.dot(cderi_sparse)
                vj_sparse += cupy.dot(rhoj, cderi_sparse.T)
            if with_k:
                for k in range(nset):
                    rhok = contract('Lij,jk->Lki', cderi, dms[k]).reshape([-1,nao])
                    #vk[k] += contract('Lki,Lkj->ij', cderi, rhok)
                    vk[k] += cupy.dot(cderi.reshape([-1,nao]).T, rhok)
        if with_j:
            vj = cupy.zeros(dms_shape)
            vj[:,rows,cols] = vj_sparse
            vj[:,cols,rows] = vj_sparse
        rhok = None

    rev_ao_idx = dfobj.intopt.rev_ao_idx
    if with_j:
        vj = take_last2d(vj, rev_ao_idx)
        vj = vj.reshape(out_shape)
    if with_k:
        vk = take_last2d(vk, rev_ao_idx)
        vk = vk.reshape(out_shape)
    t1 = log.timer_debug1('vj and vk', *t1)
    if out_cupy:
        return vj, vk
    else:
        if vj is not None:
            vj = vj.get()
        if vk is not None:
            vk = vk.get()
        return vj, vk

def _get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True,
            direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
            omega=None):
    if omega is None:
        return get_jk(dfobj, dm, hermi, with_j, with_k, direct_scf_tol)

    # A temporary treatment for RSH-DF integrals
    key = '%.6f' % omega
    if key in dfobj._rsh_df:
        rsh_df = dfobj._rsh_df[key]
    else:
        rsh_df = dfobj._rsh_df[key] = copy.copy(dfobj).reset()
        logger.info(dfobj, 'Create RSH-DF object %s for omega=%s', rsh_df, omega)

    with rsh_df.mol.with_range_coulomb(omega):
        return get_jk(rsh_df, dm, hermi, with_j, with_k, direct_scf_tol)

def get_j(dfobj, dm, hermi=1, direct_scf_tol=1e-13):
    intopt = getattr(dfobj, 'intopt', None)
    if intopt is None:
        dfobj.build(direct_scf_tol=direct_scf_tol)
        intopt = dfobj.intopt
    j2c = dfobj.j2c
    rhoj = int3c2e.get_j_int3c2e_pass1(intopt, dm)
    if dfobj.cd_low.tag == 'eig':
        rhoj, _, _, _ = cupy.linalg.lstsq(j2c, rhoj)
    else:
        rhoj = cupy.linalg.solve(j2c, rhoj)

    rhoj *= 2.0
    vj = int3c2e.get_j_int3c2e_pass2(intopt, rhoj)
    return vj

density_fit = _density_fit
