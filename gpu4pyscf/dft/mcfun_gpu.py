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


import math
import cupy as cp
import numpy as np
from pyscf.dft.LebedevGrid import MakeAngularGrid

MAX_GRIDS_PER_TASK = 65536

def eval_xc_eff(func, rho_tm, deriv=1, spin_samples=770):
    '''Multi-collinear effective potential and effective kernel.

    Parameters
    ----------
    func : Function to evaluate collinear functionals.
        The collinear energy must be invariant under exchange of the two spin channels.
        The function signature is
            exc, vxc, fxc, ... = func((rho, s), deriv)
        The input rho and s have shape
        * (1, Ngrids) for LDA functionals
        * (4, Ngrids) for GGA functionals where the four variables are
        rho, nabla_x(rho), nabla_y(rho), nabla_z(rho)
        * (5, Ngrids) for meta-GGA functionals where the five variables are
        rho, nabla_x(rho), nabla_y(rho), nabla_z(rho), tau
        The returns of func should have the shape
        * (2, Nvar, Ngrids) for vxc
        * (2, Nvar, 2, Nvar, Ngrids) for fxc
        * (2, Nvar, 2, Nvar, 2, Nvar, Ngrids) for kxc
        The dimension Nvar can be
        * 1 for LDA
        * 4 for GGA: rho, nabla_x(rho), nabla_y(rho), nabla_z(rho)
        * 5 for meta-GGA: rho, nabla_x(rho), nabla_y(rho), nabla_z(rho), tau
        Note: for GGA functionals the required returns have different conventions
        to the returns of libxc or xcfun in functional derivatives. Here func
        needs to return the derivatives to rho, nabla_x(rho), nabla_y(rho),
        nabla_z(rho) while libxc (or xcfun) returns derivatives to rho, sigma_uu,
        sigma_ud, sigma_dd (sigma_ud = nabla(rho_u) dot nabla(rho_d)). Please
        see example 04-xc_wrapper.py for the transformation between the two
        conventions.
    rho_tm : cp array with shape (4, Nvar, Ngrids)
        rho_tm[0] is density. rho_tm[1:4] is the magnetization spin vector. Nvar can be
        * 1 for LDA functionals
        * 4 for GGA functionals: rho, nabla_x(rho), nabla_y(rho), nabla_z(rho)
        * 5 for meta-GGA functionals: rho, nabla_x(rho), nabla_y(rho), nabla_z(rho), tau
    deriv : int, optional
        Functional derivatives. The current version (the kernel) supports maximum value 2.
    spin_samples : int, optional
        Number of points in the full Lebedev grid. Only one direction from
        each antipodal pair is evaluated.

    Returns
    -------
    XC functional and derivatives for each grid.
    * [exc] if deriv = 0
    * [exc, vxc] if deriv = 1
    * [exc, vxc, fxc] if deriv = 2
    '''

    assert deriv < 5
    rho_tm = cp.asarray(rho_tm)
    if rho_tm.dtype != cp.double:
        raise RuntimeError('rho and m must be real')
    ngrids = rho_tm.shape[-1]

    results = []
    for p0, p1 in _prange(0, ngrids, MAX_GRIDS_PER_TASK):
        r = _eval_xc_lebedev(func, rho_tm[...,p0:p1], deriv, spin_samples)
        results.append(r)

    return [None if x[0] is None else cp.concatenate(x, axis=-1) for x in zip(*results)]


def _eval_xc_lebedev(func, rho_tm, deriv, spin_samples):
    '''Multi-collinear effective potential and effective kernel with projection
    samples on spherical surface (the Lebedev grid samples)
    '''
    ngrids = rho_tm.shape[-1]
    sgrids, weights = _make_sph_samples(spin_samples)
    sgrids = cp.asarray(sgrids)
    weights = cp.asarray(weights)

    if rho_tm.ndim == 2:
        nvar = 1
    else:
        nvar = rho_tm.shape[1]
    if nvar >=5:
        ndim = 2
    else:
        ndim = 4
    blksize = math.ceil(ndim*1e5 / ngrids) * 8
    exc_eff = vxc_eff = fxc_eff = kxc_eff = 0
    for p0, p1 in _prange(0, weights.size, blksize):
        nsg = p1 - p0
        p_sgrids = sgrids[p0:p1]
        p_weights = weights[p0:p1]
        rho = _project_spin_sph(rho_tm, p_sgrids)

        xc_orig = func(rho, deriv+1)

        exc = xc_orig[0].reshape(ngrids, nsg)
        vxc = xc_orig[1].reshape(2, nvar, ngrids, nsg)

        rho = rho.reshape(2, nvar, ngrids, nsg)
        s = rho[1]
        rho_pure = rho[0,0]
        exc_rho = exc * rho_pure + cp.einsum('xgo,xgo->go', vxc[1], s)
        exc_eff += cp.einsum('go,o->g', exc_rho, p_weights)

        if deriv > 0:
            fxc = xc_orig[2].reshape(2, nvar, 2, nvar, ngrids, nsg)
            # vs * 2 + s*f_s_st
            vxc[1] *= 2
            vxc += cp.einsum('xbygo,xgo->bygo', fxc[1], s)
            c_tm = _ts2tm_transformation(p_sgrids)
            cw_tm = c_tm * p_weights
            vxc_eff += cp.einsum('rao,axgo->rxg', cw_tm, vxc)

        if deriv > 1:
            kxc = xc_orig[3].reshape(2, nvar, 2, nvar, 2, nvar, ngrids, nsg)
            fxc[1,:,1] *= 3
            fxc[0,:,1] *= 2
            fxc[1,:,0] *= 2
            fxc += cp.einsum('xbyczgo,xgo->byczgo', kxc[1], s)
            fxc = cp.einsum('rao,axbygo->rxbygo', c_tm, fxc)
            fxc_eff += cp.einsum('sbo,rxbygo->rxsyg', cw_tm, fxc)

        if deriv > 2:
            lxc = xc_orig[4].reshape(2, nvar, 2, nvar, 2, nvar, 2, nvar, ngrids, nsg)
            kxc[1,:,1,:,1] *= 4
            kxc[1,:,1,:,0] *= 3
            kxc[1,:,0,:,1] *= 3
            kxc[0,:,1,:,1] *= 3
            kxc[1,:,0,:,0] *= 2
            kxc[0,:,1,:,0] *= 2
            kxc[0,:,0,:,1] *= 2

            kxc += cp.einsum('wbxcydzgo,wgo->bxcydzgo', lxc[1], s)
            kxc = cp.einsum('rao,axbyczgo->rxbyczgo', c_tm, kxc)
            kxc = cp.einsum('sbo,rxbyczgo->rxsyczgo', c_tm, kxc)
            kxc_eff += cp.einsum('tco,rxsyczgo->rxsytzg', cw_tm, kxc)

    # exc in libxc is defined as Exc per particle. exc_eff calculated above is exc*rho.
    # Divide exc_eff by rho so as to follow the convention of libxc
    if rho_tm.ndim == 2:
        rho_pure = rho_tm[0]
    else:
        rho_pure = rho_tm[0,0]

    exc_eff[rho_pure == 0] = 0
    exc_eff[rho_pure != 0] /= rho_pure[rho_pure != 0]

    ret = [exc_eff]
    if deriv > 0:
        ret.append(vxc_eff)
    if deriv > 1:
        ret.append(fxc_eff)
    if deriv > 2:
        ret.append(kxc_eff)
    if deriv > 3:
        raise NotImplementedError
    return ret

def _make_sph_samples(spin_samples):
    '''One direction per antipodal Lebedev pair, with the combined weight.'''
    ang_grids = MakeAngularGrid(spin_samples)
    x, y, z = ang_grids[:,:3].T
    keep = (z > 0) | ((z == 0) & ((y > 0) | ((y == 0) & (x > 0))))
    directions = np.asfortranarray(ang_grids[keep,:3])
    weights = 2 * ang_grids[keep,3]
    return directions, weights

def _prange(start, end, step):
    '''Partitions range into segments: i0:i1, i1:i2, i2:i3, ...'''
    if start < end:
        for i in range(start, end, step):
            yield i, min(i+step, end)

def _project_spin_sph(rho_tm, sgrids):
    '''Projects spin onto spherical surface'''
    rho = rho_tm[0]
    m = rho_tm[1:]
    nsg = sgrids.shape[0]
    ngrids = rho.shape[-1]
    if rho_tm.ndim == 2:
        rho_ts = cp.empty((2, ngrids, nsg))
        rho_ts[0] = rho[:, cp.newaxis]
        rho_ts[1] = cp.einsum('mg,om->go', m, sgrids)
        rho_ts = rho_ts.reshape(2, ngrids*nsg)
    else:
        nvar = rho_tm.shape[1]
        rho_ts = cp.empty((2, nvar, ngrids, nsg))
        rho_ts[0] = rho[:, :, cp.newaxis]
        rho_ts[1] = cp.einsum('mxg,om->xgo', m, sgrids)
        rho_ts = rho_ts.reshape(2, nvar, ngrids*nsg)
    return rho_ts

def _ts2tm_transformation(sgrids):
    '''
    Transformation that projects v_ts(rho,s) to rho/m representation (rho,mx,my,mz)
    '''
    nsg = sgrids.shape[0]
    c_tm = cp.zeros((4, 2, nsg))
    c_tm[0,0] = 1
    c_tm[1:,1] = sgrids.T
    return c_tm
