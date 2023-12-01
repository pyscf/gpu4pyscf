# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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
        * spin unpolarized is not implemented
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
        raise NotImplementedError()

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