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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
# Modified by Xiaojie Wu <wxj6000@gmail.com>

import copy
from concurrent.futures import ThreadPoolExecutor
import cupy
import numpy
from pyscf import lib, __config__
from pyscf.scf import dhf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, transpose_sum, reduce_to_device, tag_array)
from gpu4pyscf.dft import rks, uks, numint
from gpu4pyscf.scf import hf, uhf, rohf
from gpu4pyscf.df import df, int3c2e
from gpu4pyscf.__config__ import _streams, num_devices

def _pin_memory(array):
    mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
    ret = numpy.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret

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
    _keys = {'rhoj', 'rhok', 'disp', 'screen_tol', 'with_df', 'only_dfj'}

    def __init__(self, mf, dfobj, only_dfj):
        self.__dict__.update(mf.__dict__)
        self._eri = None
        self.rhoj = None
        self.rhok = None
        self.direct_scf = False
        self.with_df = dfobj
        self.only_dfj = only_dfj

    def undo_df(self):
        '''Remove the DFHF Mixin'''
        obj = lib.view(self, lib.drop_class(self.__class__, _DFHF))
        del obj.rhoj, obj.rhok, obj.with_df, obj.only_dfj
        return obj

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return super().reset(mol)

    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        return self.with_df.get_jk(dm, hermi, True, False, self.direct_scf_tol, omega)[0]

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if dm is None: dm = self.make_rdm1()
        if self.with_df and self.only_dfj:
            vj = vk = None
            if with_j:
                vj = self.get_j(mol, dm, hermi, omega)
            if with_k:
                vk = super().get_jk(mol, dm, hermi, False, True, omega)[1]
        elif self.with_df:
            vj, vk = self.with_df.get_jk(dm, hermi, with_j, with_k,
                                         self.direct_scf_tol, omega)
        else:
            vj, vk = super().get_jk(mol, dm, hermi, with_j, with_k, omega)
        return vj, vk

    def nuc_grad_method(self):
        if self.istype('_Solvation'):
            raise NotImplementedError(
                'Gradients of solvent are not computed. '
                'Solvent must be applied after density fitting method, e.g.\n'
                'mf = mol.RKS().to_gpu().density_fit().PCM()')
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

    Gradients = nuc_grad_method

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

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=0, hermi=1):
        '''
        effective potential
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        assert not self.direct_scf

        if isinstance(self, rohf.ROHF):
            if getattr(dm, 'mo_coeff', None) is not None:
                mo_coeff = cupy.repeat(dm.mo_coeff[None], 2, axis=0)
                mo_occ = cupy.asarray([dm.mo_occ>0, dm.mo_occ==2],
                                      dtype=numpy.double)
                if dm.ndim == 2:  # RHF DM
                    dm = cupy.repeat(dm[None]*.5, 2, axis=0)
                dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
            elif dm.ndim == 2:  # RHF DM
                dm = cupy.repeat(dm[None]*.5, 2, axis=0)

        # for DFT
        if isinstance(self, rks.KohnShamDFT):
            t0 = logger.init_timer(self)
            rks.initialize_grids(self, mol, dm)
            ni = self._numint
            if isinstance(self, (uhf.UHF, rohf.ROHF)): # UKS
                n, exc, vxc = ni.nr_uks(mol, self.grids, self.xc, dm)
                logger.debug(self, 'nelec by numeric integration = %s', n)
                if self.do_nlc():
                    if ni.libxc.is_nlc(self.xc):
                        xc = self.xc
                    else:
                        assert ni.libxc.is_nlc(self.nlc)
                        xc = self.nlc
                    n, enlc, vnlc = ni.nr_nlc_vxc(mol, self.nlcgrids, xc, dm[0]+dm[1])
                    exc += enlc
                    vxc += vnlc
                    logger.debug(self, 'nelec with nlc grids = %s', n)
                t0 = logger.timer(self, 'vxc', *t0)

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
                    exc -= cupy.einsum('sij,sji->', dm, vk).real * .5
                ecoul = cupy.einsum('sij,ji->', dm, vj).real * .5

            elif isinstance(self, hf.RHF):
                n, exc, vxc = ni.nr_rks(mol, self.grids, self.xc, dm)
                logger.debug(self, 'nelec by numeric integration = %s', n)
                if self.do_nlc():
                    if ni.libxc.is_nlc(self.xc):
                        xc = self.xc
                    else:
                        assert ni.libxc.is_nlc(self.nlc)
                        xc = self.nlc
                    n, enlc, vnlc = ni.nr_nlc_vxc(mol, self.nlcgrids, xc, dm)
                    exc += enlc
                    vxc += vnlc
                    logger.debug(self, 'nelec with nlc grids = %s', n)
                t0 = logger.timer_debug1(self, 'vxc tot', *t0)

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
                    exc -= cupy.einsum('ij,ji', dm, vk).real * .25
                ecoul = cupy.einsum('ij,ji', dm, vj).real * .5

            else:
                raise NotImplementedError("DF only supports R/U/RO KS.")
            t0 = logger.timer_debug1(self, 'jk total', *t0)
            return tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)

        if isinstance(self, (uhf.UHF, rohf.ROHF)):
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj[0] + vj[1] - vk
        elif isinstance(self, hf.RHF):
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj - vk * .5
        else:
            raise NotImplementedError("DF only supports R/U/RO HF.")

    def to_cpu(self):
        obj = self.undo_df().to_cpu().density_fit()
        return utils.to_cpu(self, obj)

def _jk_task_with_mo(dfobj, dms, mo_coeff, mo_occ,
                     with_j=True, with_k=True, hermi=0, device_id=0):
    ''' Calculate J and K matrices on single GPU
    '''
    with cupy.cuda.Device(device_id), _streams[device_id]:
        assert isinstance(dfobj.verbose, int)
        log = logger.new_logger(dfobj.mol, dfobj.verbose)
        t0 = log.init_timer()
        dms = cupy.asarray(dms)
        mo_coeff = cupy.asarray(mo_coeff)
        mo_occ = cupy.asarray(mo_occ)
        nao = dms.shape[-1]
        intopt = dfobj.intopt
        rows = intopt.cderi_row
        cols = intopt.cderi_col
        nset = dms.shape[0]
        dms_shape = dms.shape
        vj = vk = None
        if with_j:
            dm_sparse = dms[:,rows,cols]
            if hermi == 0:
                dm_sparse += dms[:,cols,rows]
            else:
                dm_sparse *= 2
            dm_sparse[:, intopt.cderi_diag] *= .5

        if with_k:
            vk = cupy.zeros_like(dms)

        # SCF K matrix with occ
        if mo_coeff is not None:
            assert hermi == 1
            nocc = 0
            occ_coeff = [0]*nset
            for i in range(nset):
                occ_idx = mo_occ[i] > 0
                occ_coeff[i] = mo_coeff[i][:,occ_idx] * mo_occ[i][occ_idx]**0.5
                nocc += int(mo_occ[i].sum())
            blksize = dfobj.get_blksize(extra=nao*nocc)
            if with_j:
                vj_packed = cupy.zeros_like(dm_sparse)
            for cderi, cderi_sparse in dfobj.loop(blksize=blksize, unpack=with_k):
                # leading dimension is 1
                if with_j:
                    rhoj = dm_sparse.dot(cderi_sparse)
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
        t0 = log.timer_debug1(f'vj and vk on Device {device_id}', *t0)
    return vj, vk

def _jk_task_with_mo1(dfobj, dms, mo1s, occ_coeffs,
                      with_j=True, with_k=True, hermi=0, device_id=0):
    ''' Calculate J and K matrices with mo response
        For CP-HF or TDDFT
    '''
    vj = vk = None
    with cupy.cuda.Device(device_id), _streams[device_id]:
        assert isinstance(dfobj.verbose, int)
        log = logger.new_logger(dfobj.mol, dfobj.verbose)
        t0 = log.init_timer()
        dms = cupy.asarray(dms)
        mo1s = [cupy.asarray(mo1) for mo1 in mo1s]
        occ_coeffs = [cupy.asarray(occ_coeff) for occ_coeff in occ_coeffs]

        nao = dms.shape[-1]
        intopt = dfobj.intopt
        rows = intopt.cderi_row
        cols = intopt.cderi_col
        dms_shape = dms.shape
        if with_j:
            dm_sparse = dms[:,rows,cols]
            if hermi == 0:
                dm_sparse += dms[:,cols,rows]
            else:
                dm_sparse *= 2
            dm_sparse[:, intopt.cderi_diag] *= .5

        if with_k:
            vk = cupy.zeros_like(dms)

        if with_j:
            vj_sparse = cupy.zeros_like(dm_sparse)

        nocc = max([mo1.shape[2] for mo1 in mo1s])
        blksize = dfobj.get_blksize(extra=2*nao*nocc)
        for cderi, cderi_sparse in dfobj.loop(blksize=blksize, unpack=with_k):
            if with_j:
                rhoj = dm_sparse.dot(cderi_sparse)
                vj_sparse += cupy.dot(rhoj, cderi_sparse.T)
                rhoj = None
            cderi_sparse = None
            if with_k:
                iset = 0
                for occ_coeff, mo1 in zip(occ_coeffs, mo1s):
                    rhok = contract('Lij,jk->Lki', cderi, occ_coeff).reshape([-1,nao])
                    for i in range(mo1.shape[0]):
                        rhok1 = contract('Lij,jk->Lki', cderi, mo1[i]).reshape([-1,nao])
                        #contract('Lki,Lkj->ij', rhok1, rhok, alpha=1.0, beta=1.0, out=vk[iset])
                        vk[iset] += cupy.dot(rhok1.T, rhok)
                        iset += 1
                mo1 = rhok1 = rhok = None
            cderi = None
        mo1s = None
        if with_j:
            vj = cupy.zeros(dms_shape)
            vj[:,rows,cols] = vj_sparse
            vj[:,cols,rows] = vj_sparse
        if with_k and hermi:
            transpose_sum(vk)
        vj_sparse = None

        t0 = log.timer_debug1(f'vj and vk on Device {device_id}', *t0)
    return vj, vk

def _jk_task_with_dm(dfobj, dms, with_j=True, with_k=True, hermi=0, device_id=0):
    ''' Calculate J and K matrices with density matrix
    '''
    with cupy.cuda.Device(device_id), _streams[device_id]:
        assert isinstance(dfobj.verbose, int)
        log = logger.new_logger(dfobj.mol, dfobj.verbose)
        t0 = log.init_timer()
        dms = cupy.asarray(dms)
        intopt = dfobj.intopt
        rows = intopt.cderi_row
        cols = intopt.cderi_col
        nao = dms.shape[-1]
        dms_shape = dms.shape
        vj = vk = None
        if with_j:
            dm_sparse = dms[:,rows,cols]
            if hermi == 0:
                dm_sparse += dms[:,cols,rows]
            else:
                dm_sparse *= 2
            dm_sparse[:, intopt.cderi_diag] *= .5
            vj_sparse = cupy.zeros_like(dm_sparse)

        if with_k:
            vk = cupy.zeros_like(dms)

        nset = dms.shape[0]
        blksize = dfobj.get_blksize()
        for cderi, cderi_sparse in dfobj.loop(blksize=blksize, unpack=with_k):
            if with_j:
                rhoj = dm_sparse.dot(cderi_sparse)
                vj_sparse += cupy.dot(rhoj, cderi_sparse.T)
            if with_k:
                for k in range(nset):
                    rhok = contract('Lij,jk->Lki', cderi, dms[k]).reshape([-1,nao])
                    #vk[k] += contract('Lki,Lkj->ij', rhok, cderi)
                    vk[k] += cupy.dot(rhok.T, cderi.reshape([-1,nao]))
        if with_j:
            vj = cupy.zeros(dms_shape)
            vj[:,rows,cols] = vj_sparse
            vj[:,cols,rows] = vj_sparse

        t0 = log.timer_debug1(f'vj and vk on Device {device_id}', *t0)
    return vj, vk

def get_jk(dfobj, dms_tag, hermi=0, with_j=True, with_k=True, direct_scf_tol=1e-14, omega=None):
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
    t1 = t0 = log.init_timer()
    if dfobj._cderi is None:
        log.debug('Build CDERI ...')
        dfobj.build(direct_scf_tol=direct_scf_tol, omega=omega)
        t1 = log.timer_debug1('init jk', *t0)

    assert nao == dfobj.nao
    intopt = dfobj.intopt

    nao = dms_tag.shape[-1]
    dms = dms_tag.reshape([-1,nao,nao])
    intopt = dfobj.intopt
    dms = intopt.sort_orbitals(dms, axis=[1,2])

    if getattr(dms_tag, 'mo_coeff', None) is not None:
        mo_occ = dms_tag.mo_occ
        mo_coeff = dms_tag.mo_coeff
        nmo = mo_occ.shape[-1]
        mo_coeff = mo_coeff.reshape(-1,nao,nmo)
        mo_occ   = mo_occ.reshape(-1,nmo)
        mo_coeff = intopt.sort_orbitals(mo_coeff, axis=[1])
        cupy.cuda.get_current_stream().synchronize()

        futures = []
        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            for device_id in range(num_devices):
                future = executor.submit(
                    _jk_task_with_mo,
                    dfobj, dms, mo_coeff, mo_occ,
                    hermi=hermi, device_id=device_id,
                    with_j=with_j, with_k=with_k)
                futures.append(future)

    elif hasattr(dms_tag, 'mo1'):
        occ_coeffs = dms_tag.occ_coeff
        mo1s = dms_tag.mo1
        if not isinstance(occ_coeffs, (tuple, list)):
            # *2 for double occupancy in RHF/RKS
            occ_coeffs = [occ_coeffs * 2.0]
        if not isinstance(mo1s, (tuple, list)):
            mo1s = [mo1s]
        occ_coeffs = [intopt.sort_orbitals(occ_coeff, axis=[0]) for occ_coeff in occ_coeffs]
        mo1s = [intopt.sort_orbitals(mo1, axis=[1]) for mo1 in mo1s]
        cupy.cuda.get_current_stream().synchronize()

        futures = []
        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            for device_id in range(num_devices):
                future = executor.submit(
                    _jk_task_with_mo1,
                    dfobj, dms, mo1s, occ_coeffs,
                    hermi=hermi, device_id=device_id,
                    with_j=with_j, with_k=with_k)
                futures.append(future)

    # general K matrix with density matrix
    else:
        cupy.cuda.Stream.null.synchronize()
        futures = []
        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            for device_id in range(num_devices):
                future = executor.submit(
                    _jk_task_with_dm, dfobj, dms,
                    hermi=hermi, device_id=device_id,
                    with_j=with_j, with_k=with_k)
                futures.append(future)

    vj = vk = None
    if with_j:
        vj = [future.result()[0] for future in futures]
        vj = reduce_to_device(vj, inplace=True)
        vj = intopt.unsort_orbitals(vj, axis=[1,2])
        vj = vj.reshape(out_shape)

    if with_k:
        vk = [future.result()[1] for future in futures]
        vk = reduce_to_device(vk, inplace=True)
        vk = intopt.unsort_orbitals(vk, axis=[1,2])
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
