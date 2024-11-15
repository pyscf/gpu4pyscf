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
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.dft import numint as numint_cpu
from pyscf.dft.gen_grid import CUTOFF
from pyscf.pbc.lib.kpts import KPoints
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import transpose_sum, contract, get_avail_mem
from gpu4pyscf.lib import utils

MIN_BLK_SIZE = numint.MIN_BLK_SIZE
ALIGNED = numint.ALIGNED

def eval_ao(cell, coords, kpt=np.zeros(3), deriv=0, relativity=0, shls_slice=None,
            non0tab=None, cutoff=None, out=None, verbose=None):
    '''Collocate AO crystal orbitals (opt. gradients) on the real-space grid.

    Args:
        cell : instance of :class:`Cell`

        coords : (nx*ny*nz, 3) ndarray
            The real-space grid point coordinates.

    Kwargs:
        kpt : (3,) ndarray
            The k-point corresponding to the crystal AO.
        deriv : int
            AO derivative order.  It affects the shape of the return array.
            If deriv=0, the returned AO values are stored in a (N,nao) array.
            Otherwise the AO values are stored in an array of shape (M,N,nao).
            Here N is the number of grids, nao is the number of AO functions,
            M is the size associated to the derivative deriv.

    Returns:
        aoR : ([4,] nx*ny*nz, nao=cell.nao_nr()) ndarray
            The value of the AO crystal orbitals on the real-space grid by default.
            If deriv=1, also contains the value of the orbitals gradient in the
            x, y, and z directions.  It can be either complex or float array,
            depending on the kpt argument.  If kpt is not given (gamma point),
            aoR is a float array.
    '''
    ao_kpts = eval_ao_kpts(cell, coords, np.reshape(kpt, (-1,3)), deriv)
    return ao_kpts[0]

def eval_ao_kpts(cell, coords, kpts=None, deriv=0, relativity=0,
                 shls_slice=None, non0tab=None, cutoff=None, out=None, verbose=None):
    '''
    Returns:
        ao_kpts: (nkpts, [comp], ngrids, nao) ndarray
            AO values at each k-point
    '''
    return [cp.asarray(ao) for ao in numint_cpu.eval_ao_kpts(cell, coords.get(), kpts, deriv)]


def eval_rho(cell, ao, dm, non0tab=None, xctype='LDA', hermi=0, with_lapl=False,
             verbose=None):
    '''Collocate the density (opt. gradients) on the real-space grid.

    Args:
        cell : instance of :class:`Mole` or :class:`Cell`

        ao : ([4,] nx*ny*nz, nao=cell.nao_nr()) ndarray
            The value of the AO crystal orbitals on the real-space grid by default.
            If xctype='GGA', also contains the value of the gradient in the x, y,
            and z directions.

    Returns:
        rho : ([4,] nx*ny*nz) ndarray
            The value of the density on the real-space grid. If xctype='GGA',
            also contains the value of the gradient in the x, y, and z
            directions.

    See Also:
        pyscf.dft.numint.eval_rho

    '''
    if np.iscomplexobj(ao) or np.iscomplexobj(dm):
        ngrids, nao = ao.shape[-2:]
        ao_loc = cell.ao_loc_nr()
        assert nao == ao_loc[-1]
        dm = cp.asarray(dm, dtype=np.complex128)

        if hermi == 1:
            def dot_bra(bra, aodm):
                rho = contract('pi,pi->p', bra.real, aodm.real)
                rho += contract('pi,pi->p', bra.imag, aodm.imag)
                return rho
            dtype = np.float64
        else:
            def dot_bra(bra, aodm):
                return contract('pi,pi->p', bra.conj(), aodm)
            dtype = np.complex128

        if xctype == 'LDA' or xctype == 'HF':
            c0 = ao.dot(dm)
            rho = dot_bra(ao, c0)

        elif xctype == 'GGA':
            rho = cp.empty((4,ngrids), dtype=dtype)
            c0 = ao[0].dot(dm)
            rho[0] = dot_bra(ao[0], c0)
            for i in range(1, 4):
                rho[i] = dot_bra(ao[i], c0)
            if hermi == 1:
                rho[1:4] *= 2
            else:
                c1 = ao[0].dot(dm.conj().T)
                for i in range(1, 4):
                    rho[i] += dot_bra(c1, ao[i])

        else: # MGGA
            assert not with_lapl
            rho = cp.empty((5,ngrids), dtype=dtype)
            tau_idx = 4
            c0 = ao[0].dot(dm)
            rho[0] = dot_bra(ao[0], c0)
            rho[tau_idx] = 0
            for i in range(1, 4):
                c1 = ao[i].dot(dm)
                rho[tau_idx] += dot_bra(ao[i], c1)
                rho[i] = dot_bra(ao[i], c0)
                if hermi == 1:
                    rho[i] *= 2
                else:
                    rho[i] += dot_bra(ao[0], c1)
            rho[tau_idx] *= .5
    else:
        # real orbitals and real DM
        # TODO: call numint.eval_rho. However, the structure of ao is not compatible
        # rho = numint.eval_rho(cell, ao, dm, non0tab, xctype, hermi, with_lapl, verbose)
        ngrids, nao = ao.shape[-2:]
        ao_loc = cell.ao_loc_nr()
        assert nao == ao_loc[-1]

        def dot_bra(bra, aodm):
            return contract('pi,pi->p', bra, aodm)

        if xctype == 'LDA' or xctype == 'HF':
            c0 = ao.dot(dm)
            rho = dot_bra(ao, c0)

        elif xctype == 'GGA':
            rho = cp.empty((4,ngrids))
            c0 = ao[0].dot(dm)
            rho[0] = dot_bra(ao[0], c0)
            for i in range(1, 4):
                rho[i] = dot_bra(ao[i], c0)
            if hermi == 1:
                rho[1:4] *= 2
            else:
                c1 = ao[0].dot(dm.T)
                for i in range(1, 4):
                    rho[i] += dot_bra(c1, ao[i])

        else: # MGGA
            assert not with_lapl
            rho = cp.empty((5,ngrids))
            tau_idx = 4
            c0 = ao[0].dot(dm)
            rho[0] = dot_bra(ao[0], c0)
            rho[tau_idx] = 0
            for i in range(1, 4):
                c1 = ao[i].dot(dm)
                rho[tau_idx] += dot_bra(ao[i], c1)
                rho[i] = dot_bra(ao[i], c0)
                if hermi == 1:
                    rho[i] *= 2
                else:
                    rho[i] += dot_bra(ao[0], c1)
            rho[tau_idx] *= .5
    return rho

nr_uks_vxc = nr_uks = NotImplemented
nr_nlc_vxc = NotImplemented
nr_rks_fxc = NotImplemented
nr_rks_fxc_st = NotImplemented
nr_uks_fxc = NotImplemented
cache_xc_kernel = NotImplemented
cache_xc_kernel1 = NotImplemented


def get_rho(ni, cell, dm, grids, kpts=np.zeros((1,3)), max_memory=2000):
    '''Density in real space
    '''
    assert dm.ndim == 2 or dm.shape[0] == 1
    rho = cp.empty(grids.size)
    nao = cell.nao
    p1 = 0
    for ao_k1, ao_k2, mask, weight, coords \
            in ni.block_loop(cell, grids, nao, 0, kpts, None, max_memory):
        p0, p1 = p1, p1 + weight.size
        rho[p0:p1] = ni.eval_rho(cell, ao_k1, dm, xctype='LDA', hermi=1)
    return rho

def _scale_ao(ao, wv, out=None):
    # TODO: reuse gpu4pyscf.dft.numint._scale_ao
    if wv.ndim == 1:
        return ao * wv[:,None]
    else:
        return contract('ngi,ng->gi', ao, wv)

def _tau_dot(bra, ket, wv):
    '''1/2 <nabla i| v | nabla j>'''
    # TODO: reuse gpu4pyscf.dft.numint._tau_dot
    wv = .5 * wv
    mat  = bra[1].conj().T.dot(_scale_ao(ket[1], wv))
    mat += bra[2].conj().T.dot(_scale_ao(ket[2], wv))
    mat += bra[3].conj().T.dot(_scale_ao(ket[3], wv))
    return mat

class NumInt(lib.StreamObject, numint.LibXCMixin):
    '''Generalization of pyscf's NumInt class for a single k-point shift and
    periodic images.
    '''

    get_vxc = nr_vxc = numint_cpu.NumInt.nr_vxc

    def nr_rks(self, cell, grids, xc_code, dms, relativity=0, hermi=1,
               kpt=None, kpts_band=None, max_memory=2000, verbose=None):
        if kpt is None:
            kpt = np.zeros(3)
        xctype = self._xc_type(xc_code)
        if xctype == 'LDA':
            ao_deriv = 0
            nvar = 1
        elif xctype == 'GGA':
            ao_deriv = 1
            nvar = 4
        elif xctype == 'MGGA':
            ao_deriv = 1
            nvar = 5
        elif xctype == 'HF':
            return 0, 0, cp.zeros_like(dms)
        else:
            raise NotImplementedError(f'nr_rks for functional {xc_code}')

        dms = cp.asarray(dms)
        dm_shape = dms.shape
        nao = dm_shape[-1]
        dms = dms.reshape(nao,nao)
        ngrids = grids.size

        rho = cp.empty([nvar,ngrids])
        p0 = p1 = 0
        for ao_ks, weight, coords \
                in self.block_loop(cell, grids, ao_deriv, kpt=kpt):
            p0, p1 = p1, p1 + weight.size
            rho[:,p0:p1] = eval_rho(cell, ao_ks[0], dms, xctype=xctype, hermi=hermi)

        if xctype == 'LDA':
            exc, vxc = self.eval_xc_eff(xc_code, rho[0], deriv=1, xctype=xctype)[:2]
        else:
            exc, vxc = self.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[:2]
        den = rho[0] * grids.weights
        nelec = den.sum()
        excsum = cp.sum(den * exc[:,0])

        wv = vxc * grids.weights
        # *.5 for v+v.conj().T at the end
        if xctype == 'GGA':
            wv[0] *= .5
        elif xctype == 'MGGA':
            wv[[0,4]] *= .5

        kpts_band, input_band = _format_kpts_band(kpts_band, kpt), kpts_band
        nband = len(kpts_band)
        if is_zero(kpts_band):
            vmat = cp.zeros((nband, nao, nao))
        else:
            vmat = cp.zeros((nband, nao, nao), dtype=np.complex128)
        v_hermi = 1  # the output matrix must be hermitian
        p0 = p1 = 0
        for ao_ks, weight, coords \
                in self.block_loop(cell, grids, ao_deriv, kpts_band=kpts_band):
            p0, p1 = p1, p1 + weight.size
            for k, ao in enumerate(ao_ks):
                if xctype == 'LDA':
                    aow = _scale_ao(ao, wv[0,p0:p1])
                    vmat[k] += ao.conj().T.dot(aow)
                elif xctype == 'GGA':
                    aow = _scale_ao(ao[:4], wv[:4,p0:p1])
                    vmat[k] += ao[0].conj().T.dot(aow)
                elif xctype == 'MGGA':
                    aow = _scale_ao(ao[:4], wv[:4,p0:p1])
                    vmat[k] += ao[0].conj().T.dot(aow)
                    vmat[k] += _tau_dot(ao, ao, wv[4,p0:p1])

        if v_hermi and xctype != 'LDA':
            vmat = vmat + vmat.transpose(0, 2, 1).conj()
        if input_band is None:
            vmat = vmat[0]
        return nelec, excsum, vmat

    def nr_uks(self, cell, grids, xc_code, dms, relativity=0, hermi=1,
               kpt=None, kpts_band=None, max_memory=2000, verbose=None):
        raise NotImplementedError

    def block_loop(self, cell, grids, deriv=0, kpt=None, kpts_band=None):
        '''Define this macro to loop over grids by blocks.
        '''
        nao = cell.nao
        grids_coords = grids.coords
        grids_weights = grids.weights
        ngrids = grids_coords.shape[0]
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6

        #cupy.get_default_memory_pool().free_all_blocks()
        mem_avail = get_avail_mem()
        blksize = int((mem_avail*.2/8/((comp+1)*nao))/ ALIGNED) * ALIGNED
        blksize = min(blksize, MIN_BLK_SIZE)
        if blksize < ALIGNED:
            raise RuntimeError('Not enough GPU memory')

        if kpts_band is None:
            if kpt is None:
                kpts = np.zeros((1, 3))
            else:
                kpts = np.reshape(kpt, (1, 3))
        elif kpt is None:
            kpts = np.reshape(kpts_band, (-1, 3))
        else:
            raise RuntimeError('Cannot produce AOs for kpt and kpts_band in the same run')

        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            coords = grids_coords[ip0:ip1]
            weight = grids_weights[ip0:ip1]
            ao_ks = eval_ao_kpts(cell, coords, kpts, deriv=deriv)
            yield ao_ks, weight, coords
            ao_ks = None

    eval_xc_eff = numint.eval_xc_eff
    _init_xcfuns = numint.NumInt._init_xcfuns

    get_fxc = nr_fxc = numint_cpu.NumInt.nr_fxc
    nr_rks_fxc = nr_rks_fxc
    nr_uks_fxc = nr_uks_fxc
    nr_rks_fxc_st = nr_rks_fxc_st
    nr_nlc_vxc = nr_nlc_vxc
    cache_xc_kernel = cache_xc_kernel
    cache_xc_kernel1 = cache_xc_kernel1
    get_rho = get_rho

    eval_ao = staticmethod(eval_ao)
    eval_rho = staticmethod(eval_rho)
    eval_rho2 = NotImplemented
    eval_rho1 = NotImplemented

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        return numint_cpu.NumInt()

_NumInt = NumInt


class KNumInt(lib.StreamObject, numint.LibXCMixin):
    '''Generalization of pyscf's NumInt class for k-point sampling and
    periodic images.
    '''
    def __init__(self, kpts=np.zeros((1,3))):
        self.kpts = np.reshape(kpts, (-1,3))

    eval_ao = staticmethod(eval_ao_kpts)

    make_mask = NotImplemented

    def eval_rho(self, cell, ao_kpts, dm_kpts, non0tab=None, xctype='LDA',
                 hermi=0, with_lapl=True, verbose=None):
        '''Collocate the density (opt. gradients) on the real-space grid.

        Args:
            cell : Mole or Cell object
            ao_kpts : (nkpts, ngrids, nao) ndarray
                AO values at each k-point
            dm_kpts: (nkpts, nao, nao) ndarray
                Density matrix at each k-point

        Returns:
           rhoR : (ngrids,) ndarray
        '''
        nkpts = len(ao_kpts)
        rho_ks = [eval_rho(cell, ao_kpts[k], dm_kpts[k], non0tab, xctype,
                           hermi, with_lapl, verbose)
                  for k in range(nkpts)]
        dtype = np.result_type(*rho_ks)
        rho = cp.zeros(rho_ks[0].shape, dtype=dtype)
        for k in range(nkpts):
            rho += rho_ks[k]
        rho *= 1./nkpts
        return rho

    get_vxc = nr_vxc = numint_cpu.KNumInt.nr_vxc
    eval_rho1 = NotImplemented
    nr_rks = NotImplemented
    nr_uks = NotImplemented

    block_loop = NotImplemented
    eval_rho2 = NotImplemented
    get_vxc = nr_vxc = numint_cpu.KNumInt.nr_vxc
    nr_rks_fxc = nr_rks_fxc
    nr_uks_fxc = nr_uks_fxc
    nr_rks_fxc_st = nr_rks_fxc_st
    cache_xc_kernel  = cache_xc_kernel
    cache_xc_kernel1 = cache_xc_kernel1
    get_rho = get_rho

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        return numint_cpu.KNumInt()

_KNumInt = KNumInt
