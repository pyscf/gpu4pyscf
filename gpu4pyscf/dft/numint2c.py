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
Numerical integration functions for (2-component) GKS with real AO basis
'''

import functools
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.dft import numint2c
from gpu4pyscf.dft import numint, mcfun_gpu
from gpu4pyscf.dft.numint import _dot_ao_dm, _dot_ao_ao, _scale_ao
from gpu4pyscf.dft import xc_deriv
from gpu4pyscf.lib import utils
from gpu4pyscf.lib.cupy_helper import add_sparse
from pyscf import __config__


def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0,
             with_lapl=True, verbose=None):
    nao = ao.shape[-2]
    assert dm.ndim == 2 and nao * 2 == dm.shape[0]
    if not isinstance(dm, cp.ndarray):
        dm = cp.asarray(dm)

    nao, ngrids = ao.shape[-2:]
    xctype = xctype.upper()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc

    if xctype == 'LDA':
        c0a = _dot_ao_dm(mol, ao, dm[:nao], non0tab, shls_slice, ao_loc)
        c0b = _dot_ao_dm(mol, ao, dm[nao:], non0tab, shls_slice, ao_loc)
        rho_m = _contract_rho_m((ao, ao), (c0a, c0b), hermi, True)
    elif xctype == 'GGA':
        # first 4 ~ (rho, m), second 4 ~ (0th order, dx, dy, dz)
        if hermi:
            rho_m = cp.empty((4, 4, ngrids))
        else:
            rho_m = cp.empty((4, 4, ngrids), dtype=cp.complex128)
        c0a = _dot_ao_dm(mol, ao[0], dm[:nao], non0tab, shls_slice, ao_loc)
        c0b = _dot_ao_dm(mol, ao[0], dm[nao:], non0tab, shls_slice, ao_loc)
        c0 = (c0a, c0b)
        rho_m[:,0] = _contract_rho_m((ao[0], ao[0]), c0, hermi, True)
        for i in range(1, 4):
            rho_m[:,i] = _contract_rho_m((ao[i], ao[i]), c0, hermi, False)
        if hermi:
            rho_m[:,1:4] *= 2  # *2 for |ao> dm < dx ao| + |dx ao> dm < ao|
        else:
            for i in range(1, 4):
                c1a = _dot_ao_dm(mol, ao[i], dm[:nao], non0tab, shls_slice, ao_loc)
                c1b = _dot_ao_dm(mol, ao[i], dm[nao:], non0tab, shls_slice, ao_loc)
                rho_m[:,i] += _contract_rho_m((ao[0], ao[0]), (c1a, c1b), hermi, False)
    else: # meta-GGA
        if hermi:
            dtype = cp.double
        else:
            dtype = cp.complex128
        if with_lapl:
            rho_m = cp.empty((4, 6, ngrids), dtype=dtype)
            tau_idx = 5
        else:
            rho_m = cp.empty((4, 5, ngrids), dtype=dtype)
            tau_idx = 4
        c0a = _dot_ao_dm(mol, ao[0], dm[:nao], non0tab, shls_slice, ao_loc)
        c0b = _dot_ao_dm(mol, ao[0], dm[nao:], non0tab, shls_slice, ao_loc)
        c0 = (c0a, c0b)
        rho_m[:,0] = _contract_rho_m((ao[0], ao[0]), c0, hermi, True)
        rho_m[:,tau_idx] = 0
        for i in range(1, 4):
            c1a = _dot_ao_dm(mol, ao[i], dm[:nao], non0tab, shls_slice, ao_loc)
            c1b = _dot_ao_dm(mol, ao[i], dm[nao:], non0tab, shls_slice, ao_loc)
            rho_m[:,tau_idx] += _contract_rho_m((ao[i], ao[i]), (c1a, c1b), hermi, True)

            rho_m[:,i] = _contract_rho_m((ao[i], ao[i]), c0, hermi, False)
            if hermi:
                rho_m[:,i] *= 2
            else:
                rho_m[:,i] += _contract_rho_m((ao[0], ao[0]), (c1a, c1b), hermi, False)
        if with_lapl:
            # TODO: rho_m[:,4] = \nabla^2 rho
            raise NotImplementedError
        # tau = 1/2 (\nabla f)^2
        rho_m[:,tau_idx] *= .5
    return rho_m

def _gks_mcol_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
                  max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    n2c = dms[0].shape[-1]
    nao = n2c // 2
    opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol

    nelec = 0
    excsum = 0
    # vmat = cp.zeros((n2c,n2c), dtype=cp.complex128)
    vmat_aa_real = cp.zeros((nao,nao), dtype=cp.float64)
    vmat_ab_real = cp.zeros((nao,nao), dtype=cp.float64)
    vmat_ba_real = cp.zeros((nao,nao), dtype=cp.float64)
    vmat_bb_real = cp.zeros((nao,nao), dtype=cp.float64)
    vmat_aa_imag = cp.zeros((nao,nao), dtype=cp.float64)
    vmat_ab_imag = cp.zeros((nao,nao), dtype=cp.float64)
    vmat_ba_imag = cp.zeros((nao,nao), dtype=cp.float64)
    vmat_bb_imag = cp.zeros((nao,nao), dtype=cp.float64)

    if xctype in ('LDA', 'GGA', 'MGGA'):
        f_eval_mat = {
            ('LDA' , 'n'): (_ncol_lda_vxc_mat , 0),
            ('LDA' , 'm'): (_mcol_lda_vxc_mat , 0),
            ('GGA' , 'm'): (_mcol_gga_vxc_mat , 1),
            ('MGGA', 'm'): (_mcol_mgga_vxc_mat, 1),
        }
        fmat, ao_deriv = f_eval_mat[(xctype, ni.collinear[0])]

        if ni.collinear[0] == 'm':  # mcol
            eval_xc = ni.mcfun_eval_xc_adapter(xc_code)
        else:
            raise NotImplementedError('locally-collinear vxc is not implemented')

        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            mask_2c = np.concatenate([mask, mask + nao])
            dm_mask = dms[mask_2c[:,None],mask_2c]
            rho = eval_rho(_sorted_mol, ao, dm_mask, non0tab=None, xctype=xctype, hermi=hermi,
                    with_lapl=False, verbose=None)
            exc, vxc = eval_xc(xc_code, rho, deriv=1, xctype=xctype)[:2]
            if xctype == 'LDA':
                den = rho[0] * weight
            else:
                den = rho[0,0] * weight
            nelec += den.sum()
            excsum += cp.dot(den, exc)
            vtmpaa, vtmpab, vtmpba, vtmpbb = fmat(mol, ao, weight, rho, vxc, mask, shls_slice,
                            ao_loc, hermi)
            add_sparse(vmat_aa_real, cp.ascontiguousarray(vtmpaa.real), mask)
            add_sparse(vmat_ab_real, cp.ascontiguousarray(vtmpab.real), mask)
            add_sparse(vmat_ba_real, cp.ascontiguousarray(vtmpba.real), mask)
            add_sparse(vmat_bb_real, cp.ascontiguousarray(vtmpbb.real), mask)
            add_sparse(vmat_aa_imag, cp.ascontiguousarray(vtmpaa.imag), mask)
            add_sparse(vmat_ab_imag, cp.ascontiguousarray(vtmpab.imag), mask)
            add_sparse(vmat_ba_imag, cp.ascontiguousarray(vtmpba.imag), mask)
            add_sparse(vmat_bb_imag, cp.ascontiguousarray(vtmpbb.imag), mask)

        row1 = cp.concatenate([vmat_aa_real, vmat_ab_real], axis=1)
        row2 = cp.concatenate([vmat_ba_real, vmat_bb_real], axis=1)
        vmat_real = cp.concatenate([row1, row2], axis=0)

        row1 = cp.concatenate([vmat_aa_imag, vmat_ab_imag], axis=1)
        row2 = cp.concatenate([vmat_ba_imag, vmat_bb_imag], axis=1)
        vmat_imag = cp.concatenate([row1, row2], axis=0)

        vmat = vmat_real + 1j * vmat_imag

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint2c.get_vxc for functional {xc_code}')

    if hermi:
        vmat = vmat + vmat.conj().transpose(1,0)

    return nelec, excsum, vmat

def _gks_mcol_fxc(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                  rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    raise NotImplementedError('non-collinear lda fxc')

def _ncol_lda_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, hermi):
    '''Vxc matrix of non-collinear LDA'''
    # NOTE vxc in u/d representation
    raise NotImplementedError('non-collinear lda vxc mat')


# * Mcfun requires functional derivatives to total-density and spin-density.
# * Make it a global function than a closure so as to be callable by multiprocessing
def __mcfun_fn_eval_xc(ni, xc_code, xctype, rho, deriv):
    t, s = rho
    if not isinstance(t, cp.ndarray):
        t = cp.asarray(t)
    if not isinstance(s, cp.ndarray):
        s = cp.asarray(s)
    rho = cp.stack([(t + s) * .5, (t - s) * .5])
    spin = 1
    evfk = ni.eval_xc_eff(xc_code, rho, deriv=deriv, xctype=xctype, spin=spin)
    evfk = list(evfk)
    for order in range(1, deriv+1):
        if evfk[order] is not None:
            evfk[order] = xc_deriv.ud2ts(evfk[order])
    return evfk

def mcfun_eval_xc_adapter(ni, xc_code):
    '''Wrapper to generate the eval_xc function required by mcfun'''

    xctype = ni._xc_type(xc_code)
    fn_eval_xc = functools.partial(__mcfun_fn_eval_xc, ni, xc_code, xctype)
    def eval_xc_eff(xc_code, rho, deriv=1, omega=None, xctype=None,
                    verbose=None, spin=None):
        return mcfun_gpu.eval_xc_eff(
            fn_eval_xc, rho, deriv, spin_samples=ni.spin_samples,
            collinear_threshold=ni.collinear_thrd,
            collinear_samples=ni.collinear_samples)
    return eval_xc_eff

def _mcol_lda_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, hermi, assemble_spin_components=False):
    '''Vxc matrix of multi-collinear LDA'''
    wv = weight * vxc
    if hermi:
        wv *= .5  # * .5 because of v+v.conj().T in r_vxc
    wr, wmx, wmy, wmz = wv

    # einsum('g,g,xgi,xgj->ij', vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sx, vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sy, vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sz, vxc, weight, ao, ao)
    aow = None
    aow = _scale_ao(ao, wmx[0], out=aow)  # Mx
    tmpx = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, wmy[0], out=aow)  # My
    tmpy = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    if hermi:
        # conj(mx+my*1j) == mx-my*1j, tmpx and tmpy should be real
        matba = (tmpx + tmpx.T) + (tmpy + tmpy.T) * 1j
        matab = cp.zeros_like(matba)
    else:
        # conj(mx+my*1j) != mx-my*1j, tmpx and tmpy should be complex
        matba = tmpx + tmpy * 1j
        matab = tmpx - tmpy * 1j
    tmpx = tmpy = None
    aow = _scale_ao(ao, wr[0]+wmz[0], out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, wr[0]-wmz[0], out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    if assemble_spin_components:
        row1 = cp.concatenate([mataa, matab], axis=1)
        row2 = cp.concatenate([matba, matbb], axis=1)

        mat = cp.concatenate([row1, row2], axis=0)
        return mat
    else:
        return mataa, matab, matba, matbb

def _mcol_gga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, hermi, assemble_spin_components=False):
    '''Vxc matrix of multi-collinear LDA'''
    wv = weight * vxc
    if hermi:
        wv[:,0] *= .5  # * .5 because of v+v.conj().T in r_vxc
    wr, wmx, wmy, wmz = wv

    aow = None
    aow = _scale_ao(ao[:4], wr[:4]+wmz[:4], out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wr[:4]-wmz[:4], out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wmx[:4], out=aow)  # Mx
    tmpx = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wmy[:4], out=aow)  # My
    tmpy = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    if hermi:
        assert vxc.dtype == cp.double
        # conj(mx+my*1j) == mx-my*1j, tmpx and tmpy should be real
        matba = (tmpx + tmpx.T) + (tmpy + tmpy.T) * 1j
        matab = cp.zeros_like(matba)
    else:
        # conj(mx+my*1j) != mx-my*1j, tmpx and tmpy should be complex
        aow = _scale_ao(ao[1:4], wmx[1:4].conj(), out=aow)  # Mx
        tmpx += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(ao[1:4], wmy[1:4].conj(), out=aow)  # My
        tmpy += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        matba = tmpx + tmpy * 1j
        matab = tmpx - tmpy * 1j
        aow = _scale_ao(ao[1:4], (wr[1:4]+wmz[1:4]).conj(), out=aow)  # Mz
        mataa += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(ao[1:4], (wr[1:4]-wmz[1:4]).conj(), out=aow)  # Mz
        matbb += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

    if assemble_spin_components:
        row1 = cp.concatenate([mataa, matab], axis=1)
        row2 = cp.concatenate([matba, matbb], axis=1)

        mat = cp.concatenate([row1, row2], axis=0)
        return mat
    else:
        return mataa, matab, matba, matbb

def _tau_dot(mol, bra, ket, wv, mask, shls_slice, ao_loc):
    '''nabla_ao dot nabla_ao
    numpy.einsum('p,xpi,xpj->ij', wv, bra[1:4].conj(), ket[1:4])
    '''
    aow = _scale_ao(ket[1], wv)
    mat = _dot_ao_ao(mol, bra[1], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ket[2], wv, aow)
    mat += _dot_ao_ao(mol, bra[2], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ket[3], wv, aow)
    mat += _dot_ao_ao(mol, bra[3], aow, mask, shls_slice, ao_loc)
    return mat

def _mcol_mgga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, hermi, assemble_spin_components=False):
    '''Vxc matrix of multi-collinear MGGA'''
    wv = weight * vxc
    tau_idx = 4
    wv[:,tau_idx] *= .5  # *.5 for 1/2 in tau
    if hermi:
        wv[:,0] *= .5  # * .5 because of v+v.conj().T in r_vxc
        wv[:,tau_idx] *= .5
    wr, wmx, wmy, wmz = wv

    aow = None
    aow = _scale_ao(ao[:4], wr[:4]+wmz[:4], out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    mataa += _tau_dot(mol, ao, ao, wr[tau_idx]+wmz[tau_idx], mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wr[:4]-wmz[:4], out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    matbb += _tau_dot(mol, ao, ao, wr[tau_idx]-wmz[tau_idx], mask, shls_slice, ao_loc)

    aow = _scale_ao(ao[:4], wmx[:4], out=aow)  # Mx
    tmpx = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    tmpx += _tau_dot(mol, ao, ao, wmx[tau_idx], mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wmy[:4], out=aow)  # My
    tmpy = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    tmpy += _tau_dot(mol, ao, ao, wmy[tau_idx], mask, shls_slice, ao_loc)
    if hermi:
        assert vxc.dtype == cp.double
        # conj(mx+my*1j) == mx-my*1j, tmpx and tmpy should be real
        matba = (tmpx + tmpx.T) + (tmpy + tmpy.T) * 1j
        matab = cp.zeros_like(matba)
    else:
        # conj(mx+my*1j) != mx-my*1j, tmpx and tmpy should be complex
        aow = _scale_ao(ao[1:4], wmx[1:4].conj(), out=aow)  # Mx
        tmpx += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(ao[1:4], wmy[1:4].conj(), out=aow)  # My
        tmpy += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        matba = tmpx + tmpy * 1j
        matab = tmpx - tmpy * 1j
        aow = _scale_ao(ao[1:4], (wr[1:4]+wmz[1:4]).conj(), out=aow)  # Mz
        mataa += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(ao[1:4], (wr[1:4]-wmz[1:4]).conj(), out=aow)  # Mz
        matbb += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

    if assemble_spin_components:
        row1 = cp.concatenate([mataa, matab], axis=1)
        row2 = cp.concatenate([matba, matbb], axis=1)

        mat = cp.concatenate([row1, row2], axis=0)
        return mat
    else:
        return mataa, matab, matba, matbb

def _mcol_lda_fxc_mat(mol, ao, weight, rho0, rho1, fxc,
                      mask, shls_slice, ao_loc, hermi):
    raise NotImplementedError('non-collinear lda fxc')

def _mcol_gga_fxc_mat(mol, ao, weight, rho0, rho1, fxc,
                      mask, shls_slice, ao_loc, hermi):
    raise NotImplementedError('non-collinear gga fxc')

def _mcol_mgga_fxc_mat(mol, ao, weight, rho0, rho1, fxc,
                       mask, shls_slice, ao_loc, hermi):
    raise NotImplementedError('non-collinear mgga fxc')

def _contract_rho_m(bra, ket, hermi=0, bra_eq_ket=False):
    '''
    hermi indicates whether the density matrix is hermitian.
    bra_eq_ket indicates whether bra and ket basis are the same AOs.
    '''
    # rho = einsum('xgi,ij,xgj->g', ket, dm, bra.conj())
    # mx = einsum('xy,ygi,ij,xgj->g', sx, ket, dm, bra.conj())
    # my = einsum('xy,ygi,ij,xgj->g', sy, ket, dm, bra.conj())
    # mz = einsum('xy,ygi,ij,xgj->g', sz, ket, dm, bra.conj())
    ket_a, ket_b = ket
    bra_a, bra_b = bra
    nao = min(ket_a.shape[-2], bra_a.shape[-2])
    ngrids = ket_a.shape[-1]
    if hermi:
        raa = cp.einsum('ip,ip->p', bra_a.real, ket_a[:nao].real)
        raa+= cp.einsum('ip,ip->p', bra_a.imag, ket_a[:nao].imag)
        rab = cp.einsum('ip,ip->p', bra_a.conj(), ket_b[:nao])
        rbb = cp.einsum('ip,ip->p', bra_b.real, ket_b[nao:].real)
        rbb+= cp.einsum('ip,ip->p', bra_b.imag, ket_b[nao:].imag)
        rho_m = cp.empty((4, ngrids))
        rho_m[0,:] = raa + rbb     # rho
        rho_m[1,:] = rab.real      # mx
        rho_m[2,:] = rab.imag      # my
        rho_m[3,:] = raa - rbb     # mz
        if bra_eq_ket:
            rho_m[1,:] *= 2
            rho_m[2,:] *= 2
        else:
            rba = cp.einsum('ip,ip->p', bra_b.conj(), ket_a[nao:])
            rho_m[1,:] += rba.real
            rho_m[2,:] -= rba.imag
    else:
        raa = cp.einsum('ip,ip->p', bra_a.conj(), ket_a[:nao])
        rba = cp.einsum('ip,ip->p', bra_b.conj(), ket_a[nao:])
        rab = cp.einsum('ip,ip->p', bra_a.conj(), ket_b[:nao])
        rbb = cp.einsum('ip,ip->p', bra_b.conj(), ket_b[nao:])
        rho_m = cp.empty((4, ngrids), dtype=cp.complex128)
        rho_m[0,:] = raa + rbb         # rho
        rho_m[1,:] = rab + rba         # mx
        rho_m[2,:] = (rba - rab) * 1j  # my
        rho_m[3,:] = raa - rbb         # mz
    return rho_m


class NumInt2C(lib.StreamObject, numint.LibXCMixin):
    '''Numerical integration methods for 2-component basis (used by GKS)'''
    _keys = {'gdftopt'}
    to_gpu = utils.to_gpu
    device = utils.device

    gdftopt      = None

    # collinear schemes:
    #   'col' (collinear, by default)
    #   'ncol' (non-collinear, also known as locally collinear)
    #   'mcol' (multi-collinear)
    collinear = getattr(__config__, 'dft_numint_RnumInt_collinear', 'col')
    spin_samples = getattr(__config__, 'dft_numint_RnumInt_spin_samples', 770)
    collinear_thrd = getattr(__config__, 'dft_numint_RnumInt_collinear_thrd', 0.99)
    collinear_samples = getattr(__config__, 'dft_numint_RnumInt_collinear_samples', 200)

    eval_ao = staticmethod(numint.eval_ao)
    eval_rho = staticmethod(eval_rho)

    def build(self, mol, coords):
        self.gdftopt = _GDFTOpt2C.from_mol(mol)
        self.grid_blksize = None
        self.non0ao_idx = {}
        return self

    def eval_rho1(self, mol, ao, dm, screen_index=None, xctype='LDA', hermi=0,
                  with_lapl=True, cutoff=None, ao_cutoff=None, pair_mask=None,
                  verbose=None):
        return self.eval_rho(mol, ao, dm, screen_index, xctype, hermi,
                             with_lapl, verbose=verbose)

    def eval_rho2(self, mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
                  with_lapl=True, verbose=None):
        '''Calculate the electron density for LDA functional and the density
        derivatives for GGA functional in the framework of 2-component basis.
        '''
        if not isinstance(mo_occ, cp.ndarray):
            mo_occ = cp.asarray(mo_occ)
        if not isinstance(mo_coeff, cp.ndarray):
            mo_coeff = cp.asarray(mo_coeff)
        if self.collinear[0] in ('n', 'm'):
            # TODO:
            dm = cp.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
            hermi = 1
            rho = self.eval_rho(mol, ao, dm, non0tab, xctype, hermi, with_lapl, verbose)
            return rho

        raise NotImplementedError(self.collinear)

    def cache_xc_kernel(self, mol, grids, xc_code, mo_coeff, mo_occ, spin=0,
                        max_memory=2000):
        '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
        DFT hessian module etc.
        '''
        raise NotImplementedError("Kxc calculation is not supported.")

    def get_rho(self, mol, dm, grids, max_memory=2000):
        '''Density in real space
        '''
        nao = dm.shape[-1] // 2
        dm_a = dm[:nao,:nao].real
        dm_b = dm[nao:,nao:].real
        ni = self._to_numint1c()
        return ni.get_rho(mol, dm_a+dm_b, grids, max_memory)

    _gks_mcol_vxc = _gks_mcol_vxc
    _gks_mcol_fxc = _gks_mcol_fxc

    @lib.with_doc(numint.nr_rks.__doc__)
    def nr_vxc(self, mol, grids, xc_code, dms, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        if not isinstance(dms, cp.ndarray):
            dms = cp.asarray(dms)
        if self.collinear[0] in ('m',):  # mcol or ncol
            opt = getattr(self, 'gdftopt', None)
            if opt is None:
                self.build(mol, grids.coords)
                opt = self.gdftopt
            assert dms.ndim == 2
            dms = cp.asarray(dms)
            dms = opt.sort_orbitals(dms, axis=[0,1])
            n, exc, vmat = self._gks_mcol_vxc(mol, grids, xc_code, dms, relativity,
                                              hermi, max_memory, verbose)
            vmat = opt.unsort_orbitals(vmat, axis=[0,1])
        else:
            raise NotImplementedError("Locally collinear and collinear is not implemented")
        return n.sum(), exc, vmat
    get_vxc = nr_gks_vxc = nr_vxc

    @lib.with_doc(numint.nr_nlc_vxc.__doc__)
    def nr_nlc_vxc(self, mol, grids, xc_code, dm, spin=0, relativity=0, hermi=1,
                   max_memory=2000, verbose=None):
        raise NotImplementedError('non-collinear nlc vxc')

    @lib.with_doc(numint.nr_rks_fxc.__doc__)
    def nr_fxc(self, mol, grids, xc_code, dm0, dms, spin=0, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
        raise NotImplementedError('non-collinear fxc')
    get_fxc = nr_gks_fxc = nr_fxc
    
    def _init_xcfuns(self, xc_code, spin=0):
        return numint._init_xcfuns(xc_code, spin)
    eval_xc_eff = numint.eval_xc_eff
    mcfun_eval_xc_adapter = mcfun_eval_xc_adapter

    block_loop = numint.NumInt.block_loop
    reset = numint.NumInt.reset

    def _to_numint1c(self):
        '''Converts to the associated class to handle collinear systems'''
        return self.view(numint.NumInt)
    
    def to_cpu(self):
        ni = numint2c.NumInt2C()
        return ni
    

class _GDFTOpt2C(numint._GDFTOpt):

    def sort_orbitals(self, mat, axis=[]):
        ''' Transform given axis of a 2-component matrix (GKS) into sorted AO
        
        This assumes the axes specified in 'axis' have a dimension of 2*nao,
        representing alpha (:nao) and beta (nao:) components. Both components
        are sorted using the same AO sorting index.
        '''
        idx = self._ao_idx
        nao = len(idx)
        
        # Create the 2-component sorting index:
        # [sorted_alpha_indices, sorted_beta_indices]
        # E.g., if idx = [0, 2, 1] (nao=3),
        # idx_2c = [0, 2, 1, 3+0, 3+2, 3+1] = [0, 2, 1, 3, 5, 4]
        # We use numpy to build the index, consistent with self._ao_idx
        idx_2c = np.concatenate([idx, idx + nao])

        shape_ones = (1,) * mat.ndim
        fancy_index = []
        for dim, n in enumerate(mat.shape):
            if dim in axis:
                # Check if the dimension matches the 2-component size
                if n != 2 * nao:
                    raise ValueError(f"Axis {dim} has dimension {n}, expected {2*nao} for 2-component sorting")
                indices = idx_2c
            else:
                # Use cp.arange for non-sorted axes, as in the original sort_orbitals
                indices = cp.arange(n)
            
            idx_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(indices.reshape(idx_shape))
            
        # Perform the sorting using advanced indexing
        return mat[tuple(fancy_index)]

    def unsort_orbitals(self, sorted_mat, axis=[], out=None):
        ''' Transform given axis of a 2-component matrix from sorted AO to original AO
        
        This assumes the axes specified in 'axis' have a dimension of 2*nao.
        This is the inverse operation of sort_orbitals_2c.
        '''
        idx = self._ao_idx
        nao = len(idx)
        
        # The 2-component index is created identically to sort_orbitals_2c
        idx_2c = np.concatenate([idx, idx + nao])

        shape_ones = (1,) * sorted_mat.ndim
        fancy_index = []
        for dim, n in enumerate(sorted_mat.shape):
            if dim in axis:
                # Check if the dimension matches the 2-component size
                if n != 2 * nao:
                    raise ValueError(f"Axis {dim} has dimension {n}, expected {2*nao} for 2-component unsorting")
                indices = idx_2c
            else:
                indices = cp.arange(n)
            
            idx_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(indices.reshape(idx_shape))
            
        if out is None:
            out = cp.empty_like(sorted_mat)
            
        # Perform the unsorting assignment
        out[tuple(fancy_index)] = sorted_mat
        return out
    
    