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

'''
Transform XC functional derivatives between different representations
'''
import numpy as np
import cupy
from pyscf.dft.xc_deriv import _stack_fg, _stack_frr, _stack_fgg
from gpu4pyscf.lib.cupy_helper import contract

def transform_vxc(rho, vxc, xctype, spin=0):
    r'''
    The output tensor has the shape:
        * spin polarized
            LDA : [2,1,N]
            GGA : [2,4,N]
            MGGA: [2,5,N]
        * spin unpolarized
            LDA : [1,N]
            GGA : [4,N]
            MGGA: [5,N]
    '''
    rho = cupy.asarray(rho, order='C')
    if xctype == 'GGA':
        order = 1
        nvar = 4
        fr = vxc[0].T
        fg = vxc[1].T
    elif xctype == 'MGGA':
        order = 2
        nvar = 5
        fr = vxc[0].T
        fg = vxc[1].T
        ft = vxc[3].T
    else:  # LDA
        order = 0
        nvar = 1
        fr = vxc[0].T

    ngrids = rho.shape[-1]
    if spin == 1:
        if order == 0:
            vp = fr.reshape(2, nvar, ngrids)
        else:
            vp = cupy.empty((2, nvar, ngrids))
            vp[:,0] = fr
            #vp[:,1:4] = _stack_fg(fg, rho=rho)
            vp[:,1:4] = contract('abg,bxg->axg', _stack_fg(fg), rho[:,1:4])
        if order > 1:
            vp[:,4] = ft
    else:
        if order == 0:
            vp = fr.reshape(nvar, ngrids)
        else:
            vp = cupy.empty((nvar, ngrids))
            vp[0] = fr
            vp[1:4] = 2 * fg * rho[1:4]
        if order > 1:
            vp[4] = ft
    return vp

def transform_fxc(rho, vxc, fxc, xctype, spin=0):
    r'''
    The output tensor has the shape:
        * spin polarized
            LDA : [2,1,2,1,N]
            GGA : [2,4,2,4,N]
            MGGA: [2,5,2,5,N]
        * spin unpolarized is not implemented
    '''
    rho = cupy.asarray(rho, order='C')
    if xctype == 'GGA':
        order = 1
        nvar = 4
        fg = vxc[1].T
        frr = fxc[0].T
        frg = fxc[1].T
        fgg = fxc[2].T
    elif xctype == 'MGGA':
        order = 2
        nvar = 5
        fg = vxc[1].T
        frr, frg, fgg, ftt, frt, fgt = [fxc[i].T for i in [0, 1, 2, 4, 6, 9]]
    else:  # LDA
        order = 0
        nvar = 1
        frr = fxc[0].T

    ngrids = rho.shape[-1]
    if spin == 1:
        if order == 0:
            vp = _stack_frr(frr).reshape(2,nvar, 2,nvar, ngrids).transpose(1,3,0,2,4)
        else:
            vp = cupy.empty((2,nvar, 2,nvar, ngrids)).transpose(1,3,0,2,4)
            vp[0,0] = _stack_frr(frr)
            i3 = np.arange(3)
            qgg = _stack_fgg(fgg)
            qgg = cupy.einsum('abcdg,axg->xbcdg', qgg, rho[:,1:4])
            qgg = cupy.einsum('xbcdg,cyg->xybdg', qgg, rho[:,1:4])
            #qgg = _stack_fgg(fgg, rho=rho).transpose(1,3,0,2,4)
            qgg[i3,i3] += _stack_fg(fg)
            vp[1:4,1:4] = qgg

            frg = frg.reshape(2,3,ngrids)
            qrg = _stack_fg(frg, axis=1)
            qrg = cupy.einsum('rabg,axg->xrbg', qrg, rho[:,1:4])
            #qrg = _stack_fg(frg, axis=1, rho=rho).transpose(2,0,1,3)
            vp[0,1:4] = qrg
            vp[1:4,0] = qrg.transpose(0,2,1,3)

        if order > 1:
            fgt = fgt.reshape(3,2,ngrids)
            qgt = _stack_fg(fgt, axis=0)
            qgt = cupy.einsum('abrg,axg->xbrg', qgt, rho[:,1:4])
            # qgt = _stack_fg(fgt, axis=0, rho=rho).transpose(1,0,2,3)
            vp[1:4,4] = qgt
            vp[4,1:4] = qgt.transpose(0,2,1,3)

            qrt = frt.reshape(2,2,ngrids)
            vp[0,4] = qrt
            vp[4,0] = qrt.transpose(1,0,2)

            vp[4,4] = _stack_frr(ftt)

        vp = vp.transpose(2,0,3,1,4)

    else:
        if order == 0:
            vp = frr.reshape(nvar, nvar, ngrids)
        else:
            vp = cupy.empty((nvar, nvar, ngrids))
            vp[0,0] = frr
            i3 = np.arange(3)
            qgg = 4 * fgg * rho[1:4] * rho[1:4,None]
            qgg[i3,i3] += fg * 2
            vp[1:4,1:4] = qgg
            vp[0,1:4] = vp[1:4,0] = 2 * frg * rho[1:4]
        if order > 1:
            vp[4,1:4] = vp[1:4,4] = 2 * fgt * rho[1:4]
            vp[0,4] = frt
            vp[4,0] = frt
            vp[4,4] = ftt
    return vp

def transform_kxc(rho, fxc, kxc, xctype, spin=0):
    r'''
    The output tensor has the shape:
        * spin polarized
            LDA : [2,1,2,1,2,1,N]
            GGA : [2,4,2,4,2,4,N]
            MGGA: [2,5,2,5,2,5,N]
        * spin unpolarized
            LDA : [1,1,1,N]
            GGA : [4,4,4,N]
            MGGA: [5,5,5,N]
    '''
    rho = cupy.asarray(rho, order='C')
    if xctype == 'GGA':
        order = 1
        nvar = 4
        frg = fxc[1].T
        fgg = fxc[2].T
        frrr, frrg, frgg, fggg = [x.T for x in kxc[:4]]
    elif xctype == 'MGGA':
        order = 2
        nvar = 5
        frg = fxc[1].T
        fgg = fxc[2].T
        fgt = fxc[9].T
        frrr, frrg, frgg, fggg, frrt, frgt, frtt, fggt, fgtt, fttt = \
                [kxc[i].T for i in [0, 1, 2, 3, 5, 7, 10, 12, 15, 19]]
    else:  # LDA
        order = 0
        nvar = 1
        frrr = kxc[0].T

    ngrids = rho.shape[-1]
    if spin == 1:
        if order == 0:
            vp = _stack_frrr(frrr).reshape(2,nvar, 2,nvar, 2,nvar, ngrids).transpose(1,3,5,0,2,4,6)
        else:
            vp = cupy.empty((2,nvar, 2,nvar, 2,nvar, ngrids)).transpose(1,3,5,0,2,4,6)
            vp[0,0,0] = _stack_frrr(frrr)
            i3 = np.arange(3)
            qggg = _stack_fggg(fggg)
            qggg = contract('abcdefg,axg->xbcdefg', qggg, rho[:,1:4])
            qggg = contract('xbcdefg,cyg->xybdefg', qggg, rho[:,1:4])
            qggg = contract('xybdefg,ezg->xyzbdfg', qggg, rho[:,1:4])
            # qggg = _stack_fggg(fggg, rho=rho).transpose(1,3,5,0,2,4,6)
            # qggg = cupy.asarray(qggg)
            qgg = _stack_fgg(fgg)
            qgg = contract('abcdg,axg->xbcdg', qgg, rho[:,1:4])
            for i in range(3):
                qggg[:,i,i] += qgg
                qggg[i,:,i] += qgg.transpose(0,2,1,3,4)
                qggg[i,i,:] += qgg.transpose(0,2,3,1,4)
            vp[1:4,1:4,1:4] = qggg

            frgg = frgg.reshape(2,6,ngrids)
            qrgg = _stack_fgg(frgg, axis=1)
            qrgg = contract('rabcdg,axg->xrbcdg', qrgg, rho[:,1:4])
            qrgg = contract('xrbcdg,cyg->xyrbdg', qrgg, rho[:,1:4])
            # qrgg = _stack_fgg(frgg.get(), axis=1, rho=rho.get()).transpose(2,4,0,1,3,5)
            qrg = _stack_fg(frg.reshape(2,3,ngrids), axis=1)
            # qrgg = cupy.asarray(qrgg)
            qrgg[i3,i3] += qrg
            vp[0,1:4,1:4] = qrgg
            vp[1:4,0,1:4] = qrgg.transpose(0,1,3,2,4,5)
            vp[1:4,1:4,0] = qrgg.transpose(0,1,3,4,2,5)

            frrg = frrg.reshape(3,3,ngrids)
            qrrg = _stack_frr(frrg, axis=0)
            qrrg = _stack_fg(qrrg, axis=2)
            qrrg = contract('rsabg,axg->rsxbg', qrrg, rho[:,1:4]).transpose(2,0,1,3,4)
            # qrrg = _stack_fg(qrrg.get(), axis=2, rho=rho.get()).transpose(3,0,1,2,4)
            # qrrg = cupy.asarray(qrrg)
            vp[0,0,1:4] = qrrg
            vp[0,1:4,0] = qrrg.transpose(0,1,3,2,4)
            vp[1:4,0,0] = qrrg.transpose(0,3,1,2,4)

        if order > 1:
            fggt = fggt.reshape(6,2,ngrids)
            qggt = _stack_fgg(fggt, axis=0)
            qggt = contract('abcdrg,axg->xbcdrg', qggt, rho[:,1:4])
            qggt = contract('xbcdrg,cyg->xybdrg', qggt, rho[:,1:4])
            # qggt = _stack_fgg(fggt, axis=0, rho=rho).transpose(1,3,0,2,4,5)
            qgt = _stack_fg(fgt.reshape(3,2,ngrids), axis=0)
            i3 = np.arange(3)
            qggt[i3,i3] += qgt
            vp[1:4,1:4,4] = qggt
            vp[1:4,4,1:4] = qggt.transpose(0,1,2,4,3,5)
            vp[4,1:4,1:4] = qggt.transpose(0,1,4,2,3,5)

            qgtt = _stack_frr(fgtt.reshape(3,3,ngrids), axis=1)
            qgtt = _stack_fg(qgtt, axis=0)
            qgtt = contract('abrsg,axg->xbrsg', qgtt, rho[:,1:4])
            # qgtt = _stack_fg(qgtt, axis=0, rho=rho).transpose(1,0,2,3,4)
            vp[1:4,4,4] = qgtt
            vp[4,1:4,4] = qgtt.transpose(0,2,1,3,4)
            vp[4,4,1:4] = qgtt.transpose(0,2,3,1,4)

            frgt = frgt.reshape(2,3,2,ngrids)
            qrgt = _stack_fg(frgt, axis=1)
            qrgt = contract('rabsg,axg->xrbsg', qrgt, rho[:,1:4])
            # qrgt = _stack_fg(frgt, axis=1, rho=rho).transpose(2,0,1,3,4)
            vp[0,1:4,4] = qrgt
            vp[0,4,1:4] = qrgt.transpose(0,1,3,2,4)
            vp[1:4,0,4] = qrgt.transpose(0,2,1,3,4)
            vp[4,0,1:4] = qrgt.transpose(0,3,1,2,4)
            vp[1:4,4,0] = qrgt.transpose(0,2,3,1,4)
            vp[4,1:4,0] = qrgt.transpose(0,3,2,1,4)

            qrrt = _stack_frr(frrt.reshape(3,2,ngrids), axis=0)
            vp[0,0,4] = qrrt
            vp[0,4,0] = qrrt.transpose(0,2,1,3)
            vp[4,0,0] = qrrt.transpose(2,0,1,3)

            qrtt = _stack_frr(frtt.reshape(2,3,ngrids), axis=1)
            vp[0,4,4] = qrtt
            vp[4,0,4] = qrtt.transpose(1,0,2,3)
            vp[4,4,0] = qrtt.transpose(1,2,0,3)

            vp[4,4,4] = _stack_frrr(fttt, axis=0)

        vp = vp.transpose(3,0,4,1,5,2,6)

    else:
        if order == 0:
            vp = frrr.reshape(nvar, nvar, nvar, ngrids)
        else:
            vp = cupy.empty((nvar, nvar, nvar, ngrids))
            vp[0,0,0] = frrr
            i3 = np.arange(3)
            qggg = 8 * fggg * rho[1:4] * rho[1:4,None] * rho[1:4,None,None]
            qgg = 4 * fgg * rho[1:4]
            for i in range(3):
                qggg[i,i,:] += qgg
                qggg[i,:,i] += qgg
                qggg[:,i,i] += qgg
            vp[1:4,1:4,1:4] = qggg

            qrgg = 4 * frgg * rho[1:4] * rho[1:4,None]
            qrgg[i3,i3] += frg * 2
            vp[0,1:4,1:4] = qrgg
            vp[1:4,0,1:4] = qrgg
            vp[1:4,1:4,0] = qrgg

            qrrg = 2 * frrg * rho[1:4]
            vp[0,0,1:4] = qrrg
            vp[0,1:4,0] = qrrg
            vp[1:4,0,0] = qrrg

        if order > 1:
            qggt = 4 * fggt * rho[1:4] * rho[1:4,None]
            qggt[i3,i3] += fgt * 2
            vp[1:4,1:4,4] = qggt
            vp[1:4,4,1:4] = qggt
            vp[4,1:4,1:4] = qggt

            qgtt = 2 * fgtt * rho[1:4]
            vp[1:4,4,4] = qgtt
            vp[4,1:4,4] = qgtt
            vp[4,4,1:4] = qgtt

            qrgt = 2 * frgt * rho[1:4]
            vp[0,1:4,4] = qrgt
            vp[0,4,1:4] = qrgt
            vp[1:4,0,4] = qrgt
            vp[4,0,1:4] = qrgt
            vp[1:4,4,0] = qrgt
            vp[4,1:4,0] = qrgt

            vp[0,0,4] = frrt
            vp[0,4,0] = frrt
            vp[4,0,0] = frrt
            vp[0,4,4] = frtt
            vp[4,0,4] = frtt
            vp[4,4,0] = frtt
            vp[4,4,4] = fttt
    return vp


def _stack_frrr(frrr, axis=0):
    '''
    frrr [u_u_u, u_u_d, u_d_d, d_d_d]
    -> tensor with shape [2, 2, 2, ...]
    '''
    if frrr.shape[axis] != 4:
        frrr = frrr.reshape(frrr.shape[:axis] + (4, -1) + frrr.shape[axis+1:])
    slices = [slice(None)] * frrr.ndim
    slices[axis] = [[[0, 1], [1, 2]],
                    [[1, 2], [2, 3]]]
    return frrr[tuple(slices)]

def _stack_fggg(fggg, axis=0, rho=None):
    '''
    fggg [uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, ud_ud_dd, ud_dd_dd, dd_dd_dd]
    -> tensor with shape [2,2, 2,2, 2,2, ...]
    '''
    if fggg.shape[axis] != 10:
        fggg = fggg.reshape(fggg.shape[:axis] + (10, 2) + fggg.shape[axis+1:])
    slices = [slice(None)] * fggg.ndim
    slices[axis] = [[[0, 1, 2], [1, 3, 4], [2, 4, 5]],
                    [[1, 3, 4], [3, 6, 7], [4, 7, 8]],
                    [[2, 4, 5], [4, 7, 8], [5, 8, 9]]]
    fggg = fggg[tuple(slices)]
    fggg = _stack_fg(fggg, axis=axis+2, rho=rho)
    fggg = _stack_fg(fggg, axis=axis+1, rho=rho)
    return _stack_fg(fggg, axis=axis, rho=rho)