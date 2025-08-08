# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import ctypes
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.gto import ANG_OF, ATOM_OF, PTR_COORD
from pyscf.pbc.dft import numint as numint_cpu
from pyscf.pbc.df.fft_jk import _format_kpts_band
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.gto.eval_gto import get_lattice_Ls
from gpu4pyscf.lib import logger
from gpu4pyscf.gto.mole import group_basis, PTR_BAS_COORD, extract_pgto_params
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks
from gpu4pyscf.pbc.df.ft_ao import libpbc, PBCIntEnvVars
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.pbc.dft.gen_grid import UniformGrids
from gpu4pyscf.scf.jk import _nearest_power2, _scale_sp_ctr_coeff
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import (
    transpose_sum, contract, get_avail_mem, asarray)
from gpu4pyscf.lib import utils

__all__ = ['NumInt', 'KNumInt']

MIN_BLK_SIZE = 8192
ALIGNED = 256
LMAX = 4

def eval_ao(cell, coords, kpt=np.zeros(3), deriv=0, relativity=0, shls_slice=None,
            non0tab=None, cutoff=None, out=None, verbose=None, opt=None):
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
    ao_kpts = eval_ao_kpts(cell, coords, np.reshape(kpt, (-1,3)), deriv,
                           out=out, verbose=verbose, opt=opt)
    return ao_kpts[0]

def eval_ao_kpts(cell, coords, kpts=None, deriv=0, relativity=0,
                 shls_slice=None, non0tab=None, cutoff=None, out=None,
                 verbose=None, opt=None):
    '''
    Returns:
        ao_kpts: (nkpts, [comp], ngrids, nao) ndarray
            AO values at each k-point
    '''
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    assert deriv <= 2
    if opt is None:
        opt = _GTOvalOpt(cell, kpts, deriv=deriv)
    else:
        assert kpts is opt.kpts
    bvkcell = opt.bvkcell
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    ngrids = len(coords)
    coords = cp.asarray(coords.T, order='C')
    bvk_ncells = opt.bvk_ncells
    nao = cell.nao
    out = cp.empty((comp, bvk_ncells, nao, ngrids))

    drv = libpbc.PBCeval_gto_deriv
    err = drv(ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.byref(opt.gto_envs),
        ctypes.cast(coords.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(bvk_ncells*nao), ctypes.c_int(bvkcell.nbas),
        ctypes.c_int(deriv), ctypes.c_int(cell.cart),
        ctypes.cast(opt.bas_rcut.data.ptr, ctypes.c_void_p))
    if err != 0:
        raise RuntimeError('PBCeval_gto_deriv failed')

    if bvk_ncells == 1: # gamma point
        out = out.transpose(1,0,3,2)
    else:
        bvk_ncells, nkpts = opt.expLk.shape
        expLk = opt.expLk.view(np.float64).reshape(bvk_ncells, nkpts, 2)
        out = contract('Lks,cLig->kcigs', expLk, out)
        out = out.view(np.complex128)[:,:,:,:,0].transpose(0,1,3,2)

    if deriv == 0:
        out = out[:,0]
    log.timer_debug2('eval_ao_kpts', *t0)
    return out

class _GTOvalOpt:
    def __init__(self, cell, kpts=None, bvk_kmesh=None, deriv=0):
        self.cell = cell
        assert kpts is None or kpts.ndim == 2
        self.kpts = kpts
        self.bvk_kmesh = bvk_kmesh
        self.deriv = deriv

        sorted_cell, ao_idx, _, uniq_l_ctr, _, bas_mapping = group_basis(
            cell, tile=1, return_bas_mapping=True, sparse_coeff=True)
        uniq_l = uniq_l_ctr[:,0]
        lmax = uniq_l.max()
        assert lmax <= LMAX
        sorted_cell.cart = cell.cart
        self.sorted_cell = sorted_cell
        self.ao_idx = ao_idx
        self.bas_mapping = bas_mapping
        rcut = _estimate_rcut(sorted_cell, deriv)
        self.bas_rcut = cp.asarray(rcut)

        if bvk_kmesh is None:
            if kpts is None:
                bvk_kmesh = np.ones(3, dtype=np.int32)
            else:
                bvk_kmesh = kpts_to_kmesh(cell, kpts)
            self.bvk_kmesh = bvk_kmesh
        bvk_ncells = np.prod(bvk_kmesh)
        if bvk_ncells == 1:
            bvkcell = sorted_cell
            expLk = None
        else:
            bvkcell = super_cell(sorted_cell, bvk_kmesh, wrap_around=True)
            # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
            bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
            bvkmesh_Ls = translation_vectors_for_kmesh(sorted_cell, bvk_kmesh, True)
            expLk = cp.exp(1j * asarray(bvkmesh_Ls).dot(asarray(kpts).T))
        self.bvk_ncells = bvk_ncells
        self.bvkcell = bvkcell
        self.expLk = expLk
        ao_loc = bvkcell.ao_loc
        cell0_nao = ao_loc[sorted_cell.nbas]
        nao = ao_loc[-1]
        assert nao == cell0_nao * bvk_ncells

        rcut = rcut.max()
        Ls = _get_bvkcell_lattice_Ls(cell, bvkcell, rcut)
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
        nimgs = len(Ls)
        logger.debug1(cell, 'eval_ao_kpts rcut=%g nimgs=%d', rcut, nimgs)
        _atm = cp.array(bvkcell._atm, dtype=np.int32)
        _bas = cp.array(bvkcell._bas, dtype=np.int32)
        _env = cp.array(_scale_sp_ctr_coeff(bvkcell), dtype=np.float64)

        original_cell_dims = (ao_loc[1:] - ao_loc[:-1])[bas_mapping]
        original_cell_ao_loc = np.append(np.int32(0), np.cumsum(original_cell_dims))
        rev_bas_mapping = np.empty_like(bas_mapping)
        rev_bas_mapping[bas_mapping] = np.arange(len(bas_mapping), dtype=np.int32)
        sorted_ao_loc = (original_cell_ao_loc[rev_bas_mapping] +
                         np.arange(bvk_ncells, dtype=np.int32)[:,None] * cell0_nao)
        ao_loc_gpu = cp.array(sorted_ao_loc.ravel(), dtype=np.int32)
        gto_envs = PBCIntEnvVars(
            sorted_cell.natm, sorted_cell.nbas, bvk_ncells, nimgs,
            _atm.data.ptr, _bas.data.ptr, _env.data.ptr,
            ao_loc_gpu.data.ptr, Ls.data.ptr,
        )
        gto_envs._env_ref_holder = (_atm, _bas, _env, ao_loc_gpu, Ls)
        self.gto_envs = gto_envs

def _estimate_rcut(cell, deriv=0):
    '''Analogous to pyscf.pbc.gto.eval_gto._estimate_rcut, improved value
    estimation.
    '''
    es, cs = extract_pgto_params(cell, 'diffused')
    ls = cell._bas[:,ANG_OF]

    vol = cell.vol
    weight_penalty = vol # ~ V[r] * (vol/ngrids) * ngrids
    rad = vol**(-1./3) * cell.rcut + 1
    surface = 4*np.pi * rad**2
    lattice_sum_factor = surface
    precision = cell.precision / max(weight_penalty*lattice_sum_factor, 1)

    norm_ang = ((2*ls+1)/(4*np.pi))**.5
    fac = 2*np.pi/vol * cs*norm_ang/es / precision

    r = cell.rcut
    r = (np.log(fac * r**(ls+1)*(2*es*r)**deriv + 1.) / es)**.5
    r = (np.log(fac * r**(ls+1)*(2*es*r)**deriv + 1.) / es)**.5
    return r

def _get_bvkcell_lattice_Ls(cell, bvkcell, rcut=None):
    '''
    Analogous to pyscf.pbc.gto.eval_gto.get_lattice_ls, but tailored for the BvK
    supercell.
    It generates lattice summation vectors for the BvK-cell, with the cutoff
    based on distance from the original unit cell positioned at the center of
    the supercell. This produces fewer images than using
    bvkcell.get_lattice_Ls() directly.
    '''
    pcell = cell.copy()
    pcell.a = bvkcell.lattice_vectors()
    pcell.unit = 'Bohr'
    return asarray(get_lattice_Ls(pcell, rcut=rcut))

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
    if cp.iscomplexobj(ao) or cp.iscomplexobj(dm):
        ngrids, nao = ao.shape[-2:]
        ao_loc = cell.ao_loc_nr()
        assert nao == ao_loc[-1]
        dm = cp.asarray(dm, dtype=np.complex128)
        ao = cp.asarray(ao, dtype=np.complex128)

        if hermi == 1:
            def dot_bra(bra, aodm):
                rho = contract('pi,pi->p', bra.conj(), aodm).real
                return cp.asarray(rho, order='C')
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
        assert ao.dtype == dm.dtype

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

def nr_rks(ni, cell, grids, xc_code, dm_kpts, relativity=0, hermi=1,
           kpts=None, kpts_band=None, max_memory=2000, verbose=None):
    if kpts is None:
        kpts = np.zeros((1,3))
    elif isinstance(kpts, KPoints):
        kpts = kpts.kpts

    is_single_kpt = kpts.ndim == 1
    if is_single_kpt:
        assert dm_kpts.ndim == 2
        kpts = kpts.reshape(1, 3)
        dm_kpts = dm_kpts[None]
    assert dm_kpts.ndim == 3
    nao = dm_kpts.shape[-1]
    ngrids = grids.size

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    if is_zero(kpts_band):
        vmat = cp.zeros((nband,nao,nao))
    else:
        vmat = cp.zeros((nband,nao,nao), dtype=np.complex128)

    xctype = ni._xc_type(xc_code)
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
        return 0, 0, vmat
    else:
        raise NotImplementedError(f'r_vxc for functional {xc_code}')

    rho = cp.empty([nvar,ngrids])
    p0 = p1 = 0
    for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts,
                                               sort_grids=True):
        p0, p1 = p1, p1 + weight.size
        rho[:,p0:p1] = ni.eval_rho(cell, ao_ks, dm_kpts, xctype=xctype, hermi=hermi)

    exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[:2]
    den = rho[0] * grids.weights
    nelec = den.sum()
    excsum = den.dot(exc[:,0]).get()[()]

    wv = vxc * grids.weights
    # *.5 for v+v.conj().T at the end
    if xctype == 'GGA':
        wv[0] *= .5
    elif xctype == 'MGGA':
        wv[[0,4]] *= .5

    v_hermi = 1  # the output matrix must be hermitian
    p0 = p1 = 0
    for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts_band,
                                               sort_grids=True):
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
    if input_band is None and is_single_kpt:
        vmat = vmat[0]
    if is_zero(kpts_band):
        vmat = vmat.real
    return nelec, excsum, vmat

def nr_uks(ni, cell, grids, xc_code, dm_kpts, relativity=0, hermi=1,
           kpts=None, kpts_band=None, max_memory=2000, verbose=None):
    if kpts is None:
        kpts = np.zeros((1,3))
    elif isinstance(kpts, KPoints):
        kpts = kpts.kpts

    is_single_kpt = kpts.ndim == 1
    if is_single_kpt:
        assert dm_kpts.ndim == 3
        kpts = kpts.reshape(1, 3)
        dm_kpts = dm_kpts[:,None]
    assert dm_kpts.ndim == 4 and len(dm_kpts) == 2
    nao = dm_kpts.shape[-1]
    ngrids = grids.size

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    if is_zero(kpts_band):
        vmat = cp.zeros((2,nband,nao,nao))
    else:
        vmat = cp.zeros((2,nband,nao,nao), dtype=np.complex128)

    xctype = ni._xc_type(xc_code)
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
        return 0, 0, vmat
    else:
        raise NotImplementedError(f'r_vxc for functional {xc_code}')

    rho = cp.empty([2,nvar,ngrids])
    p0 = p1 = 0
    for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts,
                                               sort_grids=True):
        p0, p1 = p1, p1 + weight.size
        rho[0,:,p0:p1] = ni.eval_rho(cell, ao_ks, dm_kpts[0], xctype=xctype, hermi=hermi)
        rho[1,:,p0:p1] = ni.eval_rho(cell, ao_ks, dm_kpts[1], xctype=xctype, hermi=hermi)

    exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[:2]
    den = rho[:,0] * grids.weights
    nelec = den.sum(axis=1)
    excsum = den.dot(exc[:,0]).sum().get()[()]

    wv = vxc * grids.weights
    # *.5 for v+v.conj().T at the end
    if xctype == 'GGA':
        wv[:,0] *= .5
    elif xctype == 'MGGA':
        wv[:,[0,4]] *= .5

    v_hermi = 1  # the output matrix must be hermitian
    p0 = p1 = 0
    for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpts_band,
                                               sort_grids=True):
        p0, p1 = p1, p1 + weight.size
        for k, ao in enumerate(ao_ks):
            if xctype == 'LDA':
                aow = _scale_ao(ao, wv[0,0,p0:p1])
                vmat[0,k] += ao.conj().T.dot(aow)
                aow = _scale_ao(ao, wv[1,0,p0:p1])
                vmat[1,k] += ao.conj().T.dot(aow)
            elif xctype == 'GGA':
                aow = _scale_ao(ao[:4], wv[0,:4,p0:p1])
                vmat[0,k] += ao[0].conj().T.dot(aow)
                aow = _scale_ao(ao[:4], wv[1,:4,p0:p1])
                vmat[1,k] += ao[0].conj().T.dot(aow)
            elif xctype == 'MGGA':
                aow = _scale_ao(ao[:4], wv[0,:4,p0:p1])
                vmat[0,k] += ao[0].conj().T.dot(aow)
                aow = _scale_ao(ao[:4], wv[1,:4,p0:p1])
                vmat[1,k] += ao[0].conj().T.dot(aow)
                vmat[0,k] += _tau_dot(ao, ao, wv[0,4,p0:p1])
                vmat[1,k] += _tau_dot(ao, ao, wv[1,4,p0:p1])

    if v_hermi and xctype != 'LDA':
        vmat = vmat + vmat.conj().transpose(0, 1, 3, 2)
    if input_band is None and is_single_kpt:
        vmat = vmat[:,0]
    if is_zero(kpts_band):
        vmat = vmat.real
    return nelec, excsum, vmat

nr_nlc_vxc = NotImplemented
nr_rks_fxc = NotImplemented
nr_rks_fxc_st = NotImplemented
nr_uks_fxc = NotImplemented
cache_xc_kernel = NotImplemented
cache_xc_kernel1 = NotImplemented

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


class KNumInt(lib.StreamObject, numint.LibXCMixin):
    eval_ao = staticmethod(eval_ao_kpts)

    make_mask = NotImplemented

    def reset(self, cell=None):
        return self

    def get_rho(self, cell, dm, grids, kpts=np.zeros((1,3))):
        '''Density in real space
        '''
        kpts = kpts.reshape(-1, 3)
        assert dm.ndim == 2 or len(dm) == len(kpts)
        rho = cp.empty(grids.size)
        p1 = 0
        for ao_ks, weight, coords in self.block_loop(cell, grids, 0, kpts,
                                                     sort_grids=True):
            p0, p1 = p1, p1 + weight.size
            rho[p0:p1] = self.eval_rho(cell, ao_ks, dm, xctype='LDA', hermi=1)
        return rho

    def eval_rho(self, cell, ao_kpts, dm_kpts, non0tab=None, xctype='LDA',
                 hermi=0, with_lapl=False, verbose=None):
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
        if nkpts == 1:
            return rho_ks[0]
        dtype = np.result_type(*rho_ks)
        rho = cp.zeros(rho_ks[0].shape, dtype=dtype)
        for k in range(nkpts):
            rho += rho_ks[k]
        rho *= 1./nkpts
        return rho

    def block_loop(self, cell, grids, deriv=0, kpts=None, sort_grids=False):
        '''Define this macro to loop over grids by blocks.
        '''
        nao = cell.nao
        grids_coords = grids.coords
        grids_weights = grids.weights
        ngrids = grids_coords.shape[0]
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6

        if sort_grids and isinstance(grids, UniformGrids):
            idx = grids.argsort(tile=8)
            grids_coords = grids_coords[idx]
            grids_weights = grids_weights[idx]

        #cupy.get_default_memory_pool().free_all_blocks()
        mem_avail = get_avail_mem()
        blksize = int((mem_avail*.2/8/((comp+1)*nao))/ ALIGNED) * ALIGNED
        blksize = min(blksize, MIN_BLK_SIZE)
        if blksize < ALIGNED:
            raise RuntimeError('Not enough GPU memory')

        if kpts is not None:
            kpts = kpts.reshape(-1, 3)

        eval_gto_opt = _GTOvalOpt(cell, kpts, deriv=deriv)
        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            coords = grids_coords[ip0:ip1]
            weight = grids_weights[ip0:ip1]
            ao_ks = self.eval_ao(cell, coords, kpts, deriv=deriv, opt=eval_gto_opt)
            yield ao_ks, weight, coords
            ao_ks = None

    eval_xc_eff = numint.eval_xc_eff
    _init_xcfuns = numint.NumInt._init_xcfuns

    nr_rks = nr_rks
    nr_uks = nr_uks
    get_vxc = nr_vxc = numint_cpu.KNumInt.nr_vxc
    eval_rho1 = NotImplemented
    eval_rho2 = NotImplemented

    nr_rks_fxc = NotImplemented
    nr_uks_fxc = NotImplemented
    nr_rks_fxc_st = NotImplemented
    cache_xc_kernel  = NotImplemented
    cache_xc_kernel1 = NotImplemented

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        return numint_cpu.KNumInt()

class NumInt(KNumInt):
    def to_cpu(self):
        return numint_cpu.NumInt()
