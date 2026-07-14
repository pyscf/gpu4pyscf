# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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
from concurrent.futures import ThreadPoolExecutor
import cupy
import numpy
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
from pyscf import lib
from pyscf.scf import dhf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, transpose_sum, tag_array, CPArrayWithTag, cholesky)
from gpu4pyscf.dft import rks, uks, numint, gks
from gpu4pyscf.scf import hf, uhf, rohf
from gpu4pyscf.scf.jk import _check_rsh_factors
from gpu4pyscf.scf import ghf
from gpu4pyscf.lib.cupy_helper import asarray, ndarray, get_avail_mem
from gpu4pyscf.lib import multi_gpu
num_devices = multi_gpu.num_devices

def density_fit(mf, auxbasis=None, with_df=None, only_dfj=False):
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
    from gpu4pyscf.df import df
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
            mf = mf.copy()
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
    _keys = {'disp', 'screen_tol', 'with_df', 'only_dfj'}

    def __init__(self, mf, dfobj, only_dfj):
        self.__dict__.update(mf.__dict__)
        self._eri = None
        self.direct_scf = False
        self.with_df = dfobj
        self.only_dfj = only_dfj

    def undo_df(self):
        '''Remove the DFHF Mixin'''
        obj = lib.view(self, lib.drop_class(self.__class__, _DFHF))
        del obj.with_df, obj.only_dfj
        return obj

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return super().reset(mol)

    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        return self.get_jk(mol, dm, hermi, with_j=True, with_k=False, omega=omega)[0]

    def get_k(self, mol=None, dm=None, hermi=1, omega=None,
              lr_factor=None, sr_factor=None):
        omega, lr_factor, sr_factor = _check_rsh_factors(mol, omega, lr_factor, sr_factor)
        vk = self.get_jk(mol, dm, hermi, False, True, omega=omega)[1]
        vk *= sr_factor
        if omega == 0:
            return vk

        vklr = self.get_jk(mol, dm, hermi, False, True, omega=omega)[1]
        vklr *= lr_factor - sr_factor
        vk += vklr
        return vk

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        if isinstance(self, ghf.GHF):
            nao = mol.nao
            if with_k and not self.only_dfj:
                factor_l, factor_r = factorize_dm(dm, hermi)
                if factor_r is None:
                    factor_r = factor_l
                n2c, nocc = factor_l.shape[-2:]
                nao = n2c // 2
                factor_l = factor_l.reshape(-1, n2c, nocc)
                factor_r = factor_r.reshape(-1, n2c, nocc)
                la = factor_l[:,:nao]
                lb = factor_l[:,nao:]
                ra = factor_r[:,:nao]
                rb = factor_r[:,nao:]
                is_real = dm.dtype == cp.float64
                if dm.ndim == 2:
                    n_dm = 1
                else:
                    n_dm = len(dm)

            def get_jk_spin_free(mf, mol, dm, hermi, with_j=True, with_k=True,
                                 omega=None, lr_factor=None, sr_factor=None):
                vj = vk = None
                if with_j:
                    vj = mf.with_df.get_jk(dm, hermi, with_k=False)[0]

                if not with_k:
                    return vj, vk

                if self.only_dfj:
                    vk = hf.SCF.get_k(self, mol, dm, hermi, omega=omega)
                    return vj, vk

                if is_real:
                    # leading dimension of dm corresponds to aa,bb,ab,ba
                    factor_l = cp.vstack([la, lb, la, lb])
                    factor_r = cp.vstack([ra, rb, rb, ra])
                else:
                    # leading dimension of dm corresponds to
                    # aaR,bbR,abR,baR,aaI,bbI,abI,baI
                    factor_l = cp.stack([
                        la.real, la.imag, # laR*raR - laI*raI
                        lb.real, lb.imag, # lbR*rbR - lbI*rbI
                        la.real, la.imag, # laR*rbR - laI*rbI
                        lb.real, lb.imag, # lbR*raR - lbI*raI
                        la.real, la.imag, # laR*raI + laI*raR
                        lb.real, lb.imag, # lbR*rbI + lbI*rbR
                        la.real, la.imag, # laR*rbI + laI*rbR
                        lb.real, lb.imag, # lbR*raI + lbI*raR
                    ]).reshape(8,2, n_dm,nao,nocc).transpose(0,2,3,1,4).reshape(8*n_dm, nao, 2*nocc)
                    factor_r = cp.stack([
                        ra.real, -ra.imag,
                        rb.real, -rb.imag,
                        rb.real, -rb.imag,
                        ra.real, -ra.imag,
                        ra.imag, ra.real,
                        rb.imag, rb.real,
                        rb.imag, rb.real,
                        ra.imag, ra.real,
                    ]).reshape(8,2, n_dm,nao,nocc).transpose(0,2,3,1,4).reshape(8*n_dm, nao, 2*nocc)
                dm = tag_array(dm, factor_l=factor_l, factor_r=factor_r)
                vk = self.with_df.get_jk(dm, hermi=0, with_j=False, omega=omega)[1]
                return vj, vk

            return ghf._get_jk(self, mol, dm, hermi, with_j, with_k,
                               jkbuild=get_jk_spin_free, omega=omega)

        vj = vk = None
        if self.only_dfj:
            if with_j:
                vj = self.with_df.get_j(mol, dm, hermi, omega)
            if with_k:
                vk = hf.SCF.get_k(self, mol, dm, hermi, omega=omega)
        else:
            # Full DF mode (DF J + DF K)
            vj, vk = self.with_df.get_jk(dm, hermi, with_j, with_k, omega=omega)
        return vj, vk

    def Gradients(self):
        if self.istype('_Solvation'):
            raise NotImplementedError(
                'Gradients of solvent are not computed. '
                'Solvent must be applied after density fitting method, e.g.\n'
                'mf = mol.RKS().to_gpu().density_fit().PCM()')
        from gpu4pyscf.dft import ucdft
        if isinstance(self, ucdft.CDFT_UKS):
            from gpu4pyscf.df.grad import ucdft as ucdft_grad
            return ucdft_grad.Gradients(self)
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
        if self.istype('_Solvation'):
            raise NotImplementedError(
                'Hessian of solvent are not computed. '
                'Solvent must be applied after density fitting method, e.g.\n'
                'mf = mol.RKS().to_gpu().density_fit().PCM()')
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

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        '''
        effective potential
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        assert not self.direct_scf
        log = logger.new_logger(self)

        if isinstance(self, rohf.ROHF):
            if getattr(dm, 'mo_coeff', None) is not None:
                mo_coeff = cupy.repeat(dm.mo_coeff[None], 2, axis=0)
                mo_occ = cupy.stack([dm.mo_occ>0, dm.mo_occ==2]).astype(numpy.double)
                if dm.ndim == 2:  # RHF DM
                    dm = cupy.repeat(dm[None]*.5, 2, axis=0)
                dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
            elif dm.ndim == 2:  # RHF DM
                dm = cupy.repeat(dm[None]*.5, 2, axis=0)

        # for DFT
        if isinstance(self, rks.KohnShamDFT):
            t0 = log.init_timer()
            ni = self._numint
            if isinstance(self, (uhf.UHF, rohf.ROHF)): # UKS
                if self.grids.coords is None:
                    rks.initialize_grids(self, mol, dm[0]+dm[1])
                n, exc, vxc = ni.nr_uks(mol, self.grids, self.xc, dm)
                log.debug('nelec by numeric integration = %s', n)
                if self.do_nlc():
                    if ni.libxc.is_nlc(self.xc):
                        xc = self.xc
                    else:
                        assert ni.libxc.is_nlc(self.nlc)
                        xc = self.nlc
                    n, enlc, vnlc = ni.nr_nlc_vxc(mol, self.nlcgrids, xc, dm)
                    exc += enlc
                    vxc += vnlc
                    log.debug('nelec with nlc grids = %s', n)
                t0 = log.timer('vxc', *t0)

                if not ni.libxc.is_hybrid_xc(self.xc):
                    vj = self.get_j(mol, dm[0]+dm[1], hermi)
                    vxc += vj
                else:
                    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)
                    vj, vk = self.get_jk(mol, dm, hermi)
                    vj = vj[0] + vj[1]
                    vxc += vj
                    vk *= hyb
                    if abs(omega) > 1e-10:
                        vklr = self.get_k(mol, dm, hermi, omega=omega)
                        vklr *= (alpha - hyb)
                        vk += vklr
                    vxc -= vk
                    exc -= float(cupy.einsum('sij,sji->', dm, vk).real.get()) * .5
                ecoul = float(cupy.einsum('sij,ji->', dm, vj).real.get()) * .5

            elif isinstance(self, hf.RHF):
                rks.initialize_grids(self, mol, dm)
                n, exc, vxc = ni.nr_rks(mol, self.grids, self.xc, dm)
                log.debug('nelec by numeric integration = %s', n)
                if self.do_nlc():
                    if ni.libxc.is_nlc(self.xc):
                        xc = self.xc
                    else:
                        assert ni.libxc.is_nlc(self.nlc)
                        xc = self.nlc
                    n, enlc, vnlc = ni.nr_nlc_vxc(mol, self.nlcgrids, xc, dm)
                    exc += enlc
                    vxc += vnlc
                    log.debug('nelec with nlc grids = %s', n)
                t0 = log.timer('vxc', *t0)

                if not ni.libxc.is_hybrid_xc(self.xc):
                    vj = self.get_j(mol, dm, hermi)
                    vxc += vj
                else:
                    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)
                    vj, vk = self.get_jk(mol, dm, hermi)
                    vxc += vj
                    vk *= hyb
                    if omega != 0:
                        vklr = self.get_k(mol, dm, hermi, omega=abs(omega))
                        vklr *= (alpha - hyb)
                        vk += vklr
                    vxc -= vk * .5
                    exc -= float(cupy.einsum('ij,ji->', dm, vk).real.get()) * .25
                ecoul = float(cupy.einsum('ij,ji->', dm, vj).real.get()) * .5
            elif isinstance(self, ghf.GHF):
                if hermi == 2:  # because rho = 0
                    n, exc, vxc = 0, 0, 0
                else:
                    max_memory = self.max_memory - lib.current_memory()[0]
                    if ni.collinear[0].lower() != 'm':
                        raise NotImplementedError('Only multi-colinear GKS is implemented for DF')
                    if self.grids.coords is None:
                        self.initialize_grids(mol, dm)
                    
                    if self.grids.coords is None:
                        self.initialize_grids(mol, dm)
                    
                    n, exc, vxc = ni.get_vxc(mol, self.grids, self.xc, dm,
                                             hermi=hermi, max_memory=max_memory)
                    log.debug('nelec by numeric integration = %s', n)
                    t0 = log.timer('vxc', *t0)
                    
                    if self.do_nlc():
                        if ni.libxc.is_nlc(self.xc):
                            xc = self.xc
                        else:
                            assert ni.libxc.is_nlc(self.nlc)
                            xc = self.nlc
                        n_nlc, enlc, vnlc = ni.nr_nlc_vxc(mol, self.nlcgrids, xc, dm,
                                                      hermi=hermi, max_memory=max_memory)
                        exc += enlc
                        vxc += vnlc
                        log.debug('nelec with nlc grids = %s', n_nlc)

                if not ni.libxc.is_hybrid_xc(self.xc):
                    vk = None
                    vj = self.get_j(mol, dm, hermi)
                    vxc += vj
                else:
                    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)
                    if omega == 0:
                        vj, vk = self.get_jk(mol, dm, hermi)
                        vk *= hyb
                    elif alpha == 0:
                        vj = self.get_j(mol, dm, hermi)
                        vk = self.get_k(mol, dm, hermi, omega=-omega)
                        vk *= hyb
                    elif hyb == 0:
                        vj = self.get_j(mol, dm, hermi)
                        vk = self.get_k(mol, dm, hermi, omega=omega)
                        vk *= alpha
                    else:
                        vj, vk = self.get_jk(mol, dm, hermi)
                        vk *= hyb
                        vklr = self.get_k(mol, dm, hermi, omega=omega)
                        vklr *= (alpha - hyb)
                        vk += vklr
                    vxc += vj - vk
                    exc -= cupy.einsum('ij,ji->', dm, vk).real * .5
                    ecoul = cupy.einsum('ij,ji->', dm, vj).real * .5
            else:
                raise NotImplementedError("DF only supports R/U/RO KS.")
            t0 = log.timer('veff', *t0)
            return tag_array(vxc, ecoul=ecoul, exc=exc)

        if isinstance(self, (uhf.UHF, rohf.ROHF)):
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            vj = vj[0] + vj[1]
            vhf = vj - vk
            ecoul = float(cp.einsum('sij,ji->', dm, vj).real.get()) * .5
            return tag_array(vhf, ecoul=ecoul)
        elif isinstance(self, hf.RHF):
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            vhf = vj - vk * .5
            ecoul = float(cp.einsum('ij,ji->', dm, vj).real.get()) * .5
            return tag_array(vhf, ecoul=ecoul)
        elif isinstance(self, ghf.GHF): # GHF branch
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            vhf = vj - vk
            ecoul = float(cp.einsum('ij,ji->', dm, vj).real.get()) * .5
            return tag_array(vhf, ecoul=ecoul)
        else:
            raise NotImplementedError("DF only supports R/U/RO/G HF.")

    def to_cpu(self):
        obj = self.undo_df().to_cpu().density_fit()
        obj = utils.to_cpu(self, obj)
        if hasattr(self, 'collinear'):
            obj.collinear = self.collinear
        if hasattr(self, 'spin_samples'):
            obj.spin_samples = self.spin_samples
        return obj

def get_jk(dfobj, dms, hermi=0, with_j=True, with_k=True, omega=None):
    '''
    get jk with density fitting
    '''
    log = logger.new_logger(dfobj.mol, dfobj.verbose)
    t1 = t0 = log.init_timer()
    if dfobj._cderi is None:
        log.debug('Build CDERI ...')
        dfobj.build(omega=omega)
        t1 = log.timer_debug1('init jk', *t0)

    out_cupy = isinstance(dms, cp.ndarray)
    dm_factor_l, dm_factor_r = factorize_dm(dms, hermi)
    symmetrize = getattr(dms, 'symmetrize', 0)

    if dm_factor_r is None:
        dm_factor_mode = 0
    elif dm_factor_l.ndim == dm_factor_r.ndim:
        dm_factor_mode = 1
    elif dm_factor_l.ndim < dm_factor_r.ndim:
        dm_factor_mode = 2
    else: # dm_factor_l.ndim > dm_factor_r.ndim:
        dm_factor_mode = 3

    nao, nocc = dm_factor_l.shape[-2:]
    dms_3d = cp.asarray(dms).reshape(-1,nao,nao)
    n_dm = dms_3d.shape[0]

    if nocc == 0:
        # dms equals to 0. vj and vk must be all zeros.
        return dms, dms

    if with_j:
        pair_addresses, diags = dfobj._cderi_idx
        rows, cols = divmod(cp.asarray(pair_addresses), nao)
        dm_sparse = dms_3d[:,rows,cols]
        if hermi == 0:
            dm_sparse += dms_3d[:,cols,rows]
        else:
            dm_sparse *= 2
        dm_sparse[:,diags] *= .5

    def proc():
        factor_l = cp.asarray(dm_factor_l).reshape(-1,nao,nocc)
        factor_r = dm_factor_r
        if factor_r is not None:
            factor_r = cp.asarray(factor_r).reshape(-1,nao,nocc)

        vj = vk = None
        if with_j:
            _dm_sparse = cp.asarray(dm_sparse)
            vj = cp.zeros_like(dm_sparse)

        blksize = dfobj.get_blksize(mem_fraction=0.4)
        if with_k:
            vk = cupy.zeros_like(dms_3d)
            mem_avail = get_avail_mem(exclude_memory_pool=True)
            dm_batch_size = int(mem_avail * 0.6 / (blksize*nao*nocc * 8))
            if dm_factor_mode == 1:
                dm_batch_size = dm_batch_size // 2
            dm_batch_size = min(dm_batch_size, n_dm)
            assert dm_batch_size > 0
            log.debug1('blksize=%d, dm_batch_size=%d', blksize, dm_batch_size)

            if dm_factor_mode == 0:
                buf = cp.empty((dm_batch_size, blksize * nao*nocc))
            elif dm_factor_mode == 1:
                buf = cp.empty((2 * dm_batch_size, blksize * nao*nocc))
            else:
                buf = cp.empty((dm_batch_size+1, blksize * nao*nocc))
                buf1 = buf[-1]

        for cderi, cderi_tril in dfobj.loop(blksize=blksize, unpack=with_k):
            if with_j:
                auxvec = contract('Lp,np->nL', cderi_tril, _dm_sparse)
                contract('Lp,nL->np', cderi_tril, auxvec, beta=1, out=vj)

            if with_k:
                nL = len(cderi)
                if dm_factor_mode == 0:
                    for i0, i1 in lib.prange(0, n_dm, dm_batch_size):
                        rhok = ndarray((i1-i0,nao,nocc,nL), buffer=buf)
                        contract('Lij,njk->nikL', cderi, factor_l[i0:i1], out=rhok)
                        contract('nikL,njkL->nij', rhok, rhok, beta=1, out=vk[i0:i1])
                elif dm_factor_mode == 1:
                    for i0, i1 in lib.prange(0, n_dm, dm_batch_size):
                        rhok, rhok1 = ndarray((2,i1-i0,nao,nocc,nL), buffer=buf)
                        contract('Lij,njk->nikL', cderi, factor_l[i0:i1], out=rhok)
                        contract('Lij,njk->nikL', cderi, factor_r[i0:i1], out=rhok1)
                        contract('nikL,njkL->nij', rhok, rhok1, beta=1, out=vk[i0:i1])
                elif dm_factor_mode == 2:
                    rhok = ndarray((nao,nocc,nL), buffer=buf1)
                    contract('Lij,jk->ikL', cderi, factor_l[0], out=rhok)
                    for i0, i1 in lib.prange(0, n_dm, dm_batch_size):
                        rhok1 = ndarray((i1-i0,nao,nocc,nL), buffer=buf)
                        contract('Lij,njk->nikL', cderi, factor_r[i0:i1], out=rhok1)
                        contract('nikL,jkL->nij', rhok, rhok1, beta=1, out=vk[i0:i1])
                else:
                    rhok1 = ndarray((nao,nocc,nL), buffer=buf1)
                    contract('Lij,jk->ikL', cderi, factor_r[0], out=rhok1)
                    for i0, i1 in lib.prange(0, n_dm, dm_batch_size):
                        rhok = ndarray((i1-i0,nao,nocc,nL), buffer=buf)
                        contract('Lij,njk->nikL', cderi, factor_l[i0:i1], out=rhok)
                        contract('nikL,jkL->nij', rhok, rhok1, beta=1, out=vk[i0:i1])
                rhok1 = rhok = None
        return vj, vk

    results = multi_gpu.run(proc, non_blocking=True)

    vj = vk = None
    if with_j:
        vj_sparse = multi_gpu.array_reduce([x[0] for x in results], inplace=True)
        vj = cp.zeros_like(dms_3d)
        vj[:,cols,rows] = vj[:,rows,cols] = vj_sparse
        vj = vj.reshape(dms.shape)
        if not out_cupy: vj = vj.get()

    if with_k:
        vk = multi_gpu.array_reduce([x[1] for x in results], inplace=True)
        if symmetrize != 0:
            vk = transpose_sum(vk, hermi=symmetrize)
        vk = vk.reshape(dms.shape)
        if not out_cupy: vk = vk.get()
    t1 = log.timer_debug1('vj and vk', *t1)
    return vj, vk

def get_j(dfobj, dm, hermi=1):
    if dfobj.intopt is None:
        dfobj.build(build_cderi=False)

    if dfobj._cd_j2c is None:
        from gpu4pyscf.df.int3c2e_bdiv import int2c2e
        j2c = int2c2e(dfobj.auxmol)
        try:
            dfobj._cd_j2c = cholesky(j2c), 'cd'
        except RuntimeError:
            dfobj._cd_j2c = j2c, None

    dm_shape = dm.shape
    intopt = dfobj.intopt
    mol = intopt.mol
    auxmol = intopt.auxmol
    dm = mol.apply_C_mat_CT(dm)
    rhoj = intopt.contract_dm(dm, hermi=hermi)
    rhoj = auxmol.apply_CT_dot(rhoj, axis=-1)

    j2c, tag = dfobj._cd_j2c
    if tag == 'cd':
        rhoj = solve_triangular(j2c, rhoj.T, lower=True)
        rhoj = solve_triangular(j2c.T, rhoj, lower=False)
    else:
        rhoj = cp.linalg.solve(j2c, rhoj.T)

    auxvec = auxmol.apply_C_dot(rhoj, axis=0).T
    vj = intopt.contract_auxvec(auxvec)
    vj = intopt.mol.apply_CT_mat_C(vj)
    return vj.reshape(dm_shape)

def factorize_dm(dm, hermi=0):
    '''
    Factorize density matrices to the product of two low-rank tensors.

    Returns:
        orbol : list of ndarrays of shape (nao,*)
            Contains non-null eigenvectors of density matrix.
            When the input dm contains the mo_coeff attribute, orbol stores
            eigenvectors * sqrt(occupancies).
        orbor : list of ndarrays of shape (nao,*)
            Contains orbol * eigenvalues (occupancies).
            When the input dm contains the mo_coeff attribute, orbor is None
    '''
    if isinstance(dm, CPArrayWithTag):
        if hasattr(dm, 'mo_coeff'):
            mo_coeff = cp.asarray(dm.mo_coeff)
            mo_occ = cp.asarray(dm.mo_occ)
            assert mo_coeff.ndim == mo_occ.ndim + 1
            if mo_coeff.ndim == 2:
                mask = mo_occ > 0
                dm_factor = mo_coeff[:,mask]
                dm_factor *= cp.sqrt(mo_occ[mask])
            elif mo_coeff.ndim == 3:
                mask = (mo_occ > 0).any(axis=0)
                dm_factor = mo_coeff[:,:,mask]
                dm_factor *= cp.sqrt(mo_occ[:,None,mask])
            else:
                mask = (mo_occ > 0).any(axis=(0, 1))
                dm_factor = mo_coeff[:,:,:,mask]
                dm_factor *= cp.sqrt(mo_occ[:,:,None,mask])
            return dm_factor, None
        if hasattr(dm, 'factor_l'):
            return dm.factor_l, dm.factor_r

    dm = cp.asarray(dm)
    shape = dm.shape
    if len(shape) > 3:
        dm = dm.reshape(-1, *shape[-2:])
    l, r = decompose_rdm1_svd(dm, hermi)
    if len(shape) > 3:
        shape = shape[:-2] + l.shape[-2:]
        l = l.reshape(shape)
        r = r.reshape(shape)
    return l, r

def decompose_rdm1_svd(dm, hermi=0):
    '''Decompose density matrix as U.Vh using SVD

    Args:
        dm : ndarray or sequence of ndarrays of shape (*,nao,nao)
            Density matrices

    Returns:
        orbol : list of ndarrays of shape (nao,*)
            Contains non-null eigenvectors of density matrix
        orbor : list of ndarrays of shape (nao,*)
            Contains orbol * eigenvalues (occupancies)
    '''
    if hermi == 1:
        s, u = cp.linalg.eigh(dm)
        mask = abs(s) > 1e-8
        if dm.ndim == 2:
            c = u[:,mask]
            return c, contract('i,pi->pi', s[mask], c).conj()
        else:
            mask = mask.any(axis=0)
            c = u[:,:,mask]
            return c, contract('si,spi->spi', s[:,mask], c).conj()

    u, s, vh = cp.linalg.svd(dm)
    mask = s > 1e-8
    if dm.ndim == 2:
        mask = cp.where(mask)[0]
        return u[:,mask], contract('i,ip->pi', s[mask], vh[mask])
    else:
        mask = cp.where(mask.any(axis=0))[0]
        return u[:,:,mask], contract('si,sip->spi', s[:,mask], vh[:,mask])

def _make_factorized_dm(factor_l, factor_r, symmetrize=1):
    dm = cp.matmul(factor_l, factor_r.swapaxes(-1, -2))
    if symmetrize == 1 or symmetrize == 2:
        nao = dm.shape[-1] # dm1 may have dimensions > 3
        transpose_sum(dm.reshape(-1,nao,nao), inplace=True, hermi=symmetrize)
    return tag_array(dm, factor_l=factor_l, factor_r=factor_r, symmetrize=symmetrize)

def _tag_factorize_dm(dm, hermi=0):
    if hasattr(dm, 'symmetrize'):
        # This dm should be created by the _make_factorized_dm
        return dm
    l, r = factorize_dm(dm, hermi)
    return tag_array(dm, factor_l=l, factor_r=r, symmetrize=0)

def _transpose_dm(dm):
    dm_T = dm.T
    if hasattr(dm, 'symmetrize'):
        dm_T.factor_l = dm.factor_r
        dm_T.factor_r = dm.factor_l
        dm_T.symmetrize = dm.symmetrize
    else:
        dm_T = dm_T.view(cp.ndarray)
    return dm_T

def _aggregate_dm_factor_l(dms):
    factor_l = cp.stack([x.factor_l for x in dms])
    factor_r = dms[0].factor_r
    assert all(x.symmetrize == 0 for x in dms)
    return tag_array(cp.stack(dms), factor_l=factor_l, factor_r=factor_r,
                     symmetrize=0)
