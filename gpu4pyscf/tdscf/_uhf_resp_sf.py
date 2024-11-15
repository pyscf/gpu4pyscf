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


# TODO: merge this function into scf._response_functions.py

import functools
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import numint2c, xc_deriv
from gpu4pyscf.scf import hf, uhf
from gpu4pyscf.dft.numint import _scale_ao, _tau_dot, eval_rho, eval_rho2
from gpu4pyscf.lib.cupy_helper import transpose_sum, add_sparse, contract

def gen_uhf_response_sf(mf, mo_coeff=None, mo_occ=None, hermi=0,
                        collinear='mcol', collinear_samples=200):
    '''Generate a function to compute the product of Spin Flip UKS response function
    and UKS density matrices.
    '''
    assert isinstance(mf, (uhf.UHF))
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    assert hermi == 0

    if isinstance(mf, hf.KohnShamDFT):
        if mf.do_nlc():
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')

        ni = mf._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        if collinear in ('ncol', 'mcol'):
            fxc = cache_xc_kernel_sf(ni, mol, mf.grids, mf.xc, mo_coeff, mo_occ,
                                     collinear_samples)[2]
        dm0 = None

        def vind(dm1):
            if collinear in ('ncol', 'mcol'):
                v1 = nr_uks_fxc_sf(ni, mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                   None, None, fxc)
            else:
                v1 = cp.zeros_like(dm1)
            if hybrid:
                # j = 0 in spin flip part.
                if omega == 0:
                    vk = mf.get_k(mol, dm1, hermi) * hyb
                elif alpha == 0: # LR=0, only SR exchange
                    vk = mf.get_k(mol, dm1, hermi, omega=-omega) * hyb
                elif hyb == 0: # SR=0, only LR exchange
                    vk = mf.get_k(mol, dm1, hermi, omega=omega) * alpha
                else: # SR and LR exchange with different ratios
                    vk = mf.get_k(mol, dm1, hermi) * hyb
                    vk += mf.get_k(mol, dm1, hermi, omega=omega) * (alpha-hyb)
                v1 -= vk
            return v1
        return vind

    else: #HF
        def vind(dm1):
            vk = mf.get_k(mol, dm1, hermi)
            return -vk
        return vind

# This function is copied from pyscf.dft.numint2c.py
def __mcfun_fn_eval_xc(ni, xc_code, xctype, rho, deriv):
    evfk = ni.eval_xc_eff(xc_code, rho, deriv=deriv, xctype=xctype)
    evfk = list(evfk)
    for order in range(1, deriv+1):
        if evfk[order] is not None:
            evfk[order] = xc_deriv.ud2ts(evfk[order])
    return evfk

# Edited based on pyscf.dft.numint2c.mcfun_eval_xc_adapter
def mcfun_eval_xc_adapter_sf(ni, xc_code, collinear_samples):
    '''Wrapper to generate the eval_xc function required by mcfun
    '''

    try:
        import mcfun
    except ImportError:
        raise ImportError('This feature requires mcfun library.\n'
                          'Try install mcfun with `pip install mcfun`')

    ni = numint2c.NumInt2C()
    ni.collinear = 'mcol'
    ni.collinear_samples = collinear_samples
    xctype = ni._xc_type(xc_code)
    fn_eval_xc = functools.partial(__mcfun_fn_eval_xc, ni, xc_code, xctype)
    nproc = lib.num_threads()

    def eval_xc_eff(xc_code, rho, deriv=1, omega=None, xctype=None, verbose=None):
        res = mcfun.eval_xc_eff_sf(
            fn_eval_xc, rho.get(), deriv,
            collinear_samples=collinear_samples, workers=nproc)
        return [x if x is None else cp.asarray(x) for x in res]
    return eval_xc_eff

def cache_xc_kernel_sf(ni, mol, grids, xc_code, mo_coeff, mo_occ,
                       collinear_samples):
    '''Compute the fxc_sf, which can be used in SF-TDDFT/TDA
    '''
    xctype = ni._xc_type(xc_code)
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 1
    else:
        ao_deriv = 0
    assert isinstance(mo_coeff, cp.ndarray)
    assert mo_coeff.ndim == 3

    nao = mo_coeff[0].shape[0]
    rhoa = []
    rhob = []

    with_lapl = False
    opt = getattr(ni, 'gdftopt', None)
    if opt is None or mol not in [opt.mol, opt._sorted_mol]:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])

    for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
        rhoa_slice = eval_rho2(_sorted_mol, ao_mask, mo_coeff[0,idx,:],
                               mo_occ[0], None, xctype, with_lapl)
        rhob_slice = eval_rho2(_sorted_mol, ao_mask, mo_coeff[1,idx,:],
                               mo_occ[1], None, xctype, with_lapl)
        rhoa.append(rhoa_slice)
        rhob.append(rhob_slice)
    rho_ab = (cp.hstack(rhoa), cp.hstack(rhob))
    rho_z = cp.array([rho_ab[0]+rho_ab[1],
                      rho_ab[0]-rho_ab[1]])
    eval_xc_eff = mcfun_eval_xc_adapter_sf(ni, xc_code, collinear_samples)
    vxc, fxc = eval_xc_eff(xc_code, rho_z, deriv=2, xctype=xctype)[1:3]
    return rho_ab, vxc, fxc

def nr_uks_fxc_sf(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                  rho0=None, vxc=None, fxc=None):
    if fxc is None:
        raise RuntimeError('fxc was not initialized')
    assert hermi == 0
    assert dms.dtype == np.double

    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None or mol not in [opt.mol, opt._sorted_mol]:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    mol = None
    _sorted_mol = opt._sorted_mol
    nao, nao0 = opt.coeff.shape
    dm_shape = dms.shape

    dms = cp.asarray(dms).reshape(-1,nao0,nao0)
    dms = opt.sort_orbitals(dms, axis=[1,2])

    nset = len(dms)
    vmat = cp.zeros((nset, nao, nao))

    if xctype == 'LDA':
        ao_deriv = 0
    elif xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 1
    else:
        raise RuntimeError(f'Unknown xctype {xctype}')
    p0 = p1 = 0
    for ao, mask, weights, coords in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
        p0, p1 = p1, p1+len(weights)
        # precompute fxc_w. *2.0 becausue xx + yy
        fxc_w = fxc[:,:,p0:p1] * weights * 2.

        for i in range(nset):
            rho1 = eval_rho(_sorted_mol, ao, dms[i,mask[:,None],mask],
                            xctype=xctype, hermi=hermi)
            if xctype == 'LDA':
                wv = rho1 * fxc_w[0,0]
                vtmp = ao.dot(_scale_ao(ao, wv).T)
            elif xctype == 'GGA':
                wv = contract('bg,abg->ag', rho1, fxc_w)
                wv[0] *= .5 # for transpose_sum at the end
                vtmp = ao[0].dot(_scale_ao(ao, wv).T)
            elif xctype == 'MGGA':
                wv = contract('bg,abg->ag', rho1, fxc_w)
                wv[[0,4]] *= .5 # for transpose_sum at the end
                vtmp = ao[0].dot(_scale_ao(ao[:4], wv[:4]).T)
                vtmp += _tau_dot(ao, ao, wv[4])
            add_sparse(vmat[i], vtmp, mask)

    vmat = opt.unsort_orbitals(vmat, axis=[1,2])
    if xctype != 'LDA':
        transpose_sum(vmat)
    if len(dm_shape) == 2:
        vmat = vmat[0]
    return vmat
