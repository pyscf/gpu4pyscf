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
from pyscf.pbc.dft import numint as numint_cpu
from pyscf.dft.gen_grid import CUTOFF
from pyscf.pbc.lib.kpts import KPoints
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import return_cupy_array, contract
from gpu4pyscf.lib import utils

eval_ao = return_cupy_array(numint_cpu.eval_ao)
eval_ao_kpts = return_cupy_array(numint_cpu.eval_ao_kpts)


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
            rho = np.empty((4,ngrids))
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
            rho = np.empty((5,ngrids))
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

nr_rks_vxc = nr_rks = NotImplemented
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
    rho = np.empty(grids.size)
    nao = cell.nao
    p1 = 0
    for ao_k1, ao_k2, mask, weight, coords \
            in ni.block_loop(cell, grids, nao, 0, kpts, None, max_memory):
        p0, p1 = p1, p1 + weight.size
        rho[p0:p1] = ni.eval_rho(cell, ao_k1, dm, xctype='LDA', hermi=1)
    return rho


class NumInt(lib.StreamObject, numint.LibXCMixin):
    '''Generalization of pyscf's NumInt class for a single k-point shift and
    periodic images.
    '''

    get_vxc = nr_vxc = numint_cpu.NumInt.nr_vxc
    nr_rks = NotImplemented
    nr_uks = NotImplemented
    block_loop = NotImplemented

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
        obj = utils.to_cpu(self)
        return obj.reset()

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
        rho = np.zeros(rho_ks[0].shape, dtype=dtype)
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
        obj = utils.to_cpu(self)
        return obj.reset()

_KNumInt = KNumInt
