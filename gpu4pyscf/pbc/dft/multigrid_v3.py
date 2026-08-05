#!/usr/bin/env python
# Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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

from dataclasses import dataclass
import math
import ctypes
import numpy as np
import cupy as cp
import cupyx.scipy.fft as fft
from pyscf import lib
from pyscf.gto import ANG_OF, PTR_EXP, PTR_COEFF
from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.tools.pbc import mesh_to_cutoff, cutoff_to_mesh
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, transpose_sum, ndarray, tag_array, load_library)
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.dft import numint
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.tools import k2gamma
from gpu4pyscf.pbc.lib.kpts_helper import fft_matrix
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks
from gpu4pyscf.gto.mole import (
    PTR_BAS_COORD, SortedGTO, PBCIntEnvVars, _scale_sp_ctr_coeff)
from gpu4pyscf.pbc.df.ft_ao import libpbc

libmgrid = load_library('libmgrid')
NBAS_MAX = 16777216
LMAX = 4

def _aft_eval_density(ni, dm_sc):
    cell = ni.cell
    bvkcell = ni.bvkcell
    envs = bvkcell.rys_envs # FIXME
    envs = ni.mg_envs # FIXME

    a = cell.lattice_vectors()
    assert abs(a - np.diag(a.diagonal())).max() < 1e-5, 'Must be orthogonal lattice'
    b = cell.reciprocal_vectors()

    n_dm = dm_sc.shape[0]
    assert n_dm == 1
    rhoG = cp.zeros((n_dm, *ni.mesh), dtype=np.complex128)

    for bucket in ni.aft_buckets:
        mesh = bucket.mesh
        mesh_cum = cp.array(np.append(0, np.cumsum(mesh)), dtype=np.int32)
        nimgs = bucket.nimgs
        nimgs_cum = cp.array(np.append(0, np.cumsum(nimgs)), dtype=np.int32)
        G_bases = _get_G_bases(mesh, b)
        L_bases = _get_L_bases(nimgs, a)

        ngrids = np.prod(mesh)
        bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
            bucket.bas_ij_cache, min(16, ngrids//4096))
        nbatches_shl_pair = len(shl_pair_offsets) - 1

        rhoR = cp.zeros(mesh)
        rhoI = cp.zeros(mesh)
        libpbc.contract_orth_aopair_dm(
            ctypes.cast(rhoR.data.ptr, ctypes.c_void_p),
            ctypes.cast(rhoI.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.byref(envs),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(G_bases.data.ptr, ctypes.c_void_p),
            ctypes.cast(L_bases.data.ptr, ctypes.c_void_p),
            ctypes.cast(mesh_cum.data.ptr, ctypes.c_void_p),
            ctypes.cast(nimgs_cum.data.ptr, ctypes.c_void_p),
            mesh.ctypes, ctypes.c_int(nbatches_shl_pair))
        tmp_rhoG = cp.empty(mesh, dtype=np.complex128)
        tmp_rhoG.real = rhoR
        tmp_rhoG.imag = rhoI
        _takebak_4d(rhoG, tmp_rhoG, mesh)
    return rhoG.ravel()

def _aft_eval_coul_matrix(ni, coulG):
    cell = ni.cell
    bvkcell = ni.bvkcell
    envs = bvkcell.rys_envs # FIXME
    envs = cell.rys_envs # FIXME

    a = cell.lattice_vectors()
    b = cell.reciprocal_vectors()

    assert coulG.ndim == 1
    nao = cell.nao
    vj = cp.zeros((nao, nao))

    for bucket in ni.aft_buckets:
        mesh = bucket.mesh
        mesh_cum = cp.array(np.append(0, np.cumsum(mesh)), dtype=np.int32)
        nimgs = bucket.nimgs
        nimgs_cum = cp.array(np.append(0, np.cumsum(nimgs)), dtype=np.int32)
        G_bases = _get_G_bases(mesh, b)
        L_bases = _get_L_bases(nimgs, a)

        ngrids = np.prod(mesh)
        bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
            bucket.bas_ij_cache, min(16, ngrids//4096))

        sub_vG = _take_4d(coulG, mesh)
        libpbc.contract_orth_aopair_coulG(
            ctypes.cast(vj.data.ptr, ctypes.c_void_p),
            ctypes.cast(sub_vG.data.ptr, ctypes.c_void_p),
            ctypes.byref(envs),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(G_bases.data.ptr, ctypes.c_void_p),
            ctypes.cast(L_bases.data.ptr, ctypes.c_void_p),
            ctypes.cast(mesh_cum.data.ptr, ctypes.c_void_p),
            ctypes.cast(nimgs_cum.data.ptr, ctypes.c_void_p),
            mesh.ctypes, ctypes.c_int(len(bas_ij_idx)))

    return transpose_sum(vj)

def _get_G_bases(mesh, b):
    Gx = np.fft.fftfreq(mesh[0], 1./mesh[0]) * b[0,0]
    Gy = np.fft.fftfreq(mesh[1], 1./mesh[1]) * b[1,1]
    Gz = np.fft.fftfreq(mesh[2], 1./mesh[2]) * b[2,2]
    G_bases = cp.array(np.hstack([Gx, Gy, Gz]))
    return G_bases

def _get_L_bases(nimgs, a):
    Tx = np.arange(-nimgs[0], nimgs[0]+1) * a[0,0]
    Ty = np.arange(-nimgs[1], nimgs[1]+1) * a[1,1]
    Tz = np.arange(-nimgs[2], nimgs[2]+1) * a[2,2]
    L_bases = cp.array(np.hstack([Tx, Ty, Tz]))
    return L_bases

def _esitmate_grid_ranges(bas_ij_cache, mg_envs, xctype, log_threshold):
    bas_ij_idx = cp.hstack([bas_ij for bas_ij, _ in bas_ij_cache.values()], dtype=np.int64)
    npairs = len(bas_ij_idx)
    grid_frac_ranges = cp.empty((3,npairs,2), dtype=np.float32)
    li_inc = lj_inc = 0
    if xctype == 'MGGA':
        li_inc = lj_inc = 1
    err = libmgrid.gaussian_prod_grid_ranges(
        ctypes.cast(grid_frac_ranges.data.ptr, ctypes.c_void_p),
        ctypes.byref(mg_envs),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.c_int(npairs),
        ctypes.c_int(li_inc), ctypes.c_int(lj_inc),
        ctypes.c_float(log_threshold))

    grid_frac_ranges_cache = {}
    p0 = p1 = 0
    for key, (bas_ij, _) in bas_ij_cache.items():
        p0, p1 = p1, p1 + len(bas_ij)
        grid_frac_ranges_cache[key] = grid_frac_ranges[:,p0:p1]

    if err != 0:
        raise RuntimeError('grid range kernel failed')
    return grid_frac_ranges_cache

@dataclass
class _Bucket:
    ke_lower: float
    ke_upper: float
    mesh: np.ndarray
    nimgs: np.ndarray = None

    # bas_ij_cache[key] are shell-pairs (one shell in the unit cell, the other
    # in supmol)
    bas_ij_cache = {}

    # grid_ranges_cache[key] = grid_frac_ranges[3,N,2]
    #   For each shell pair in bas_ij_idx, stores the fractional-coordinate
    #   bounds of the real-space grids that are not negligible.
    grid_ranges_cache = {}

    # grid_tile_cache[key] = (grid_tile_idx, supmol_pair_idx, shl_pair_offsets)
    # - grid_tile_idx:
    #     Unique grid tile indices that contributes to the density.
    # - supmol_pair_idx:
    #     Shell-pair indices contributing to the tiles in grid_tile_idx.
    # - shl_pair_offsets:
    #     Partition the shell pairs in supmol_pair_idx by grid tile.
    grid_tile_cache = None

def _balance_init_mesh(a, mesh):
    ke = mesh_to_cutoff(a, mesh)
    mesh = cutoff_to_mesh(np.mean(ke))
    return mesh // 2 * 2

def _prepare_buckets(ni, bas_ij_cache, init_mesh, ke_max, log,
                     with_grid_ranges=None):
    cell = ni.cell
    a = cell.lattice_vectors()
    mesh = np.asarray(init_mesh, dtype=np.int32)
    ke_lower = 0
    ke_upper = min(mesh_to_cutoff(a, init_mesh).min(), ke_max)

    nimgs = np.asarray(cell.nimgs, dtype=np.int32)

    buckets = []
    while ke_lower < ke_max:
        ke_upper = min(ke_upper, ke_max)

        # TODO: nimgs is related to ke_lower. It can be reduced for large Ecut

        filtered_pairs = {}
        filtered_grid_ranges = {}
        for key, (bas_ij_idx, pair_ke) in bas_ij_cache.items():
            idx = cp.where((ke_lower < pair_ke) & (pair_ke <= ke_upper))[0]
            filtered_pairs[key] = bas_ij_idx[idx]
            if with_grid_ranges:
                filtered_grid_ranges[key] = with_grid_ranges[key][:,idx]

        buckets.append(_Bucket(
            ke_lower=ke_lower, ke_upper=ke_upper, mesh=mesh, nimgs=nimgs,
            bas_ij_cache=filtered_pairs,
            grid_ranges_cache=filtered_grid_ranges,
        ))
        log.debug('Add bucket: mesh=%s, shl_pairs=%d', tuple(mesh),
                  sum(len(x) for x in filtered_pairs.values()))

        mesh = (mesh * 0.75).astype(np.int32) * 2
        ke_lower, ke_upper = ke_upper, mesh_to_cutoff(a, mesh).min()
    return buckets

def _Ecut_estimation(cell, ke_max, log_cutoff, xctype='LDA'):
    nbas = cell.nbas
    ls = cp.asarray(cell._bas[:,ANG_OF], dtype=np.int32)
    es = cp.asarray(cell._env[cell._bas[:,PTR_EXP]], dtype=np.float32)
    cs = cp.asarray(abs(cell._env[cell._bas[:,PTR_COEFF]]), dtype=np.float32)
    is_mgga = 0
    if xctype == 'MGGA':
        cs *= es*2
        is_mgga = 1
    log_cs = cp.asarray(cp.log(cs), dtype=np.float32)
    ptr_coords = cell._bas[:,PTR_BAS_COORD]
    bas_coords = cp.asarray(cell._env[ptr_coords[:,None] + np.arange(3)])
    bas_coords = cp.asarray(bas_coords.T, dtype=np.float32, order='C')

    nimgs = np.asarray(cell.nimgs)
    nimgs = np.prod(nimgs*2+1)

    Ecut = cp.empty((nbas, nimgs, nbas), dtype=np.float32)
    err = libmgrid.aft_Ecut_kernel(
        ctypes.cast(Ecut.data.ptr, ctypes.c_void_p),
        ctypes.cast(es.data.ptr, ctypes.c_void_p),
        ctypes.cast(log_cs.data.ptr, ctypes.c_void_p),
        ctypes.cast(ls.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(nimgs.data.ptr, ctypes.c_void_p),
        ctypes.c_int(nbas),
        ctypes.c_float(log_cutoff),
        # Set the upper limit of Ecut. This ensures all high-Ecut orbital pairs
        # are handled by the last bucket in fft_buckets
        ctypes.c_float(ke_max),
        ctypes.c_int(is_mgga))
    if err != 0:
        raise RuntimeError('Ecut kernel failed')
    return Ecut

def _cache_grid_range_to_tiles(fft_buckets, cell):
    buf_size = 0
    for bucket in fft_buckets:
        tiles_per_cell = cp.asarray(bucket.mesh / 4, dtype=np.float32)
        for (bas_ij, grid_range) in zip(
                bucket.bas_ij_cache.values(), bucket.grid_ranges_cache.values()):
            raw_tiles = grid_range[:,:,1] - grid_range[:,:,0]
            raw_tiles *= tiles_per_cell[:,None]
            raw_tiles = cp.ceil(raw_tiles)
            raw_tiles += 3 # penalty for roundings near boundary
            n = (raw_tiles[0] * raw_tiles[1] * raw_tiles[2]).sum().get()
            buf_size = max(buf_size, int(n))

    work = cp.empty(buf_size+10, dtype=np.int32)
    tile_counts = work[-1:]
    work1 = cp.empty(buf_size+10, dtype=np.int64)

    # sentinels are used to mark the first and last elements in a sorted array
    sentinel_marks = cp.array([-0x7fffffff, 0x7fffffff], dtype=np.int32)

    nbas = cell.nbas
    kern = libmgrid.grid_range_to_tiles
    for bucket in fft_buckets:
        bucket.grid_tile_cache = grid_tile_cache = {}
        for key, bas_ij_idx in bucket.bas_ij_cache.items():
            if len(bas_ij_idx) == 0: continue
            grid_range = bucket.grid_ranges_cache[key]
            err = kern(
                ctypes.cast(work.data.ptr, ctypes.c_void_p),
                ctypes.cast(work1.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(grid_range.data.ptr, ctypes.c_void_p),
                bucket.nimgs.ctypes, bucket.mesh.ctypes,
                ctypes.c_int(len(bas_ij_idx)),
                ctypes.c_int(nbas),
                ctypes.cast(tile_counts.data.ptr, ctypes.c_void_p))
            if err != 0:
                raise RuntimeError('grid_range_to_tiles failed')
            n = int(tile_counts.get())

            # Add two sentinels. They will be placed at the first and last
            # position after sorting the target.
            work[n:n+2] = sentinel_marks
            # Sort along with the sentinels
            sorted_idx = cp.argsort(work[:n+2])
            grid_tile_ids = work[sorted_idx]

            # Exclude the first and last entries. They are the sentinels.
            supmol_bas_ij_idx = work1[sorted_idx[1:-1]]

            # The start and last position of each grid tile. These offsets also
            # partition supmol_bas_ij_idx by the corresponding grid tile index.
            tile_boundary_offsets = cp.where(grid_tile_ids[:-1] != grid_tile_ids[1:])[0]
            shl_pair_offsets = tile_boundary_offsets

            # TODO: Further divide large entry in shell_pair_offsets for better
            # load balance.

            # Store only the unique grid tile ids. The leading and trailing
            # sentinel boundaries are excluded.
            grid_tile_idx = grid_tile_ids[1:][tile_boundary_offsets[:-1]]
            grid_tile_cache[key] = (
                grid_tile_idx, supmol_bas_ij_idx, shl_pair_offsets)

def fft_in_place(x):
    return fft.fftn(x, axes=(-3, -2, -1), overwrite_x=True)

def ifft_in_place(x):
    return fft.ifftn(x, axes=(-3, -2, -1), overwrite_x=True)

def _take_4d(a, mesh, out=None):
    assert a.dtype == np.complex128
    out_shape = tuple(mesh)
    inp_shape = a.shape
    counts = 1
    if a.ndim == 4:
        counts, inp_shape = inp_shape[0], inp_shape[1:]
        out_shape = (counts,) + out_shape
    out = ndarray(out_shape, dtype=np.complex128, buffer=out)
    err = libmgrid.fft_take(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(a.data.ptr, ctypes.c_void_p),
        mesh.ctypes,
        (ctypes.c_int*3)(*inp_shape),
        ctypes.c_int(counts))
    if err != 0:
        raise RuntimeError('fft_take kernel failed')
    return out

def _takebak_4d(out, a, mesh):
    assert out.dtype == a.dtype == np.complex128
    assert out.ndim == a.ndim
    out_shape = out.shape
    counts = 1
    if out.ndim == 4:
        counts, out_shape = out_shape[0], out_shape[1:]
    err = libmgrid.fft_takebak(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(a.data.ptr, ctypes.c_void_p),
        mesh.ctypes,
        (ctypes.c_int*3)(*out_shape),
        ctypes.c_int(counts))
    if err != 0:
        raise RuntimeError('fft_take kernel failed')
    return out

def nr_rks(ni, cell, grids, xc_code, dm_kpts, relativity=0, hermi=1,
           kpts=None, kpts_band=None, with_j=False, verbose=None):
    '''Compute the XC energy and RKS XC matrix at sampled k-points.
    multigrid version of function pbc.dft.numint.nr_rks.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.
        with_j : bool
            Whether to add the Coulomb matrix into the XC matrix.

    Returns:
        exc : XC energy
        nelec : number of electrons obtained from the numerical integration
        veff : (nkpts, nao, nao) ndarray
            or list of veff if the input dm_kpts is a list of DMs
    '''
    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        nvar = 1
    elif xctype == 'GGA':
        nvar = 4
    elif xctype == 'MGGA':
        nvar = 5
    else:
        raise NotImplementedError(f'XC functional {xc_code}')

    assert kpts is None or is_zero(kpts)
    kpts = np.zeros((1, 3))

    cell = ni.cell
    log = logger.new_logger(cell, verbose)
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    n_dm, nkpts, nao = dms.shape[:3]
    assert n_dm == 1

    kmesh = k2gamma.kpts_to_kmesh(cell, kpts)






    expLk = fft_matrix(kmesh)
    dm_sc, dms = contract('skpq,Lk->sLpq', dms, expLk), None
    #FIXME: ignore imaginary part?
    dm_sc = dm_sc.real
    dm_sc = cell.apply_C_mat_CT(dm_sc)

    #FIXME: We only need the tril part. Is it correct to symmetrize dm_sc?
    dm_sc = transpose_sum(dm_sc)




    _eval_rhoG(ni, dm_sc)

    vol = cell.vol
    mesh = ni.mesh
    ngrids = np.prod(mesh)
    rhoG = _eval_rhoG(ni, dms, hermi, kpts, xctype)
    rhoG = rhoG.reshape(1,ngrids)
    if xctype != 'LDA':
        Gv = cp.asarray(cell.get_Gv(mesh))
        rhoG = cp.repeat(rhoG, nvar, axis=0)
        rhoG[1:4] *= 1j
        rhoG[1:4] *= Gv.T
        if xctype == 'MGGA':
            rhoG[4] = _eval_tauG(ni, dms, kmesh)

    coulG = tools.get_coulG(cell, mesh=mesh)
    vG = rhoG[0] * coulG
    ecoul = .5 * float(rhoG[0].conj().dot(vG).real.get()) / vol
    log.debug('Multigrid Coulomb energy %s', ecoul)

    weight = vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rhoR = cp.asarray(rhoR.reshape(nvar,ngrids), order='C')
    nelec = float(rhoR[0].sum().real.get()) * weight

    exc, vxc = ni.eval_xc_eff(xc_code, rhoR, deriv=1, xctype=xctype, spin=0)[:2]
    excsum = float(rhoR[0].dot(exc).real.get()) * weight
    wv = weight * vxc
    wv_freq = tools.fft(wv, mesh).reshape(nvar,ngrids)
    rhoR = rhoG = exc = vxc = wv = None
    log.debug('Multigrid exc %s  nelec %s', excsum, nelec)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if with_j:
        wv_freq[0] += vG
    if xctype == 'LDA':
        veff = _eval_lda_mat(ni, wv_freq[None,0], kmesh, verbose=log)
    else:
        #veff = _get_gga_pass2(ni, wv_freq[None,:4], hermi, kpts_band, verbose=log)
        wv_freq[0] -= contract('xg,gx->g', wv_freq[1:4], Gv) * 1j
        veff = _eval_lda_mat(ni, wv_freq[None,0], kmesh, verbose=log)
        if xctype == 'MGGA':
            veff += _eval_mgga_mat(ni, wv_freq[None,4], kmesh, verbose=log)
    veff = _format_jks(veff, dm_kpts, input_band, kpts)

    shape = list(dm_kpts.shape)
    if len(shape) == 3 and shape[0] != kpts_band.shape[0]:
        shape[0] = kpts_band.shape[0]
    veff = veff.reshape(shape)
    veff = tag_array(veff, ecoul=ecoul, exc=excsum)
    return nelec, excsum, veff

@multi_gpu.lru_cache(10)
def _images_supmol_to_bvkcell(nimgs, kmesh):
    if kmesh is None:
        kmesh = (1, 1, 1)
    mapping_a = np.arange(-nimgs[0], nimgs[0]+1, dtype=np.int32) % kmesh[0]
    mapping_b = np.arange(-nimgs[1], nimgs[1]+1, dtype=np.int32) % kmesh[1]
    mapping_c = np.arange(-nimgs[2], nimgs[2]+1, dtype=np.int32) % kmesh[2]
    mapping = (mapping_a[:,None,None] * kmesh[1] + mapping_b[:,None]) * kmesh[2] + mapping_c
    return cp.asarray(mapping, dtype=np.int32)

def _map_images_supmol_to_bvkcell(nimgs, kmesh):
    '''
    For translation vectors Ls that defines supmol, constructs mapping_index,
    which makes
        exp(Ls.dot(kpts.T)) == exp(bvk_Ls.dot(kpts.T))[mapping_index]
    where, bvk_Ls is the translation vectors for BvK cell, constructed via
    the function pyscf.pbc.tools.k2gamma.translation_vectors_for_kmesh
    '''
    return _images_supmol_to_bvkcell(tuple(nimgs), tuple(kmesh))

def _eval_rhoG(ni, dm_sc, kmesh=None):
    cell = ni.cell
    assert isinstance(cell, SortedGTO)
    log = logger.new_logger(cell)
    t0 = log.init_timer()

    n_dm = dm_sc.shape[0]
    if ni.aft_buckets is not None:
        rhoG = _aft_eval_density(ni, dm_sc)
    else:
        rhoG = cp.zeros((n_dm, *ni.mesh), dtype=np.complex128)

    a = cell.lattice_vectors()
    supmol_to_bvk_mapping = _map_images_supmol_to_bvkcell(cell.nimgs, kmesh)

    vol = cell.vol
    nk = 1
    if kmesh is not None:
        nk = np.prod(kmesh)

    # TODO: Adjust these parameters for each li, lj values
    tiles_per_block = 4
    nsp_per_block = 4
    shm_size = 1024*47

    mg_envs = ni.mg_envs
    kern = libmgrid.evaluate_density
    uniq_l = cell.uniq_l_ctr[:,0]
    work = None
    for bucket in ni.fft_buckets:
        assert bucket.grid_tile_cache is not None
        mesh = bucket.mesh

        dxyz_dabc = a / mesh[:,None]
        libmgrid.update_dxyz_dabc(dxyz_dabc.ctypes)

        rhoR = ndarray(mesh, buffer=work)
        rhoR[:] = 0.
        for (i, j), (grid_tile_idx, bas_ij_idx, shl_pair_offsets) \
                in bucket.grid_tile_cache.items():
            if len(bas_ij_idx) == 0: continue
            li = uniq_l[i]
            lj = uniq_l[j]
            err = kern(
                ctypes.cast(rhoR.data.ptr, ctypes.c_void_p),
                ctypes.cast(dm_sc[0].data.ptr, ctypes.c_void_p),
                ctypes.byref(mg_envs),
                dxyz_dabc.ctypes,
                ctypes.c_int(tiles_per_block),
                ctypes.c_int(nsp_per_block),
                ctypes.c_int(shm_size),
                ctypes.c_int(li), ctypes.c_int(lj),
                ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(grid_tile_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(supmol_to_bvk_mapping.data.ptr, ctypes.c_void_p),
                ctypes.c_int(len(grid_tile_idx)),
                mesh.ctypes)
            if err != 0:
                raise RuntimeError('evaluate_density kernel failed')

        density = fft_in_place(rhoR)
        ngrids = np.prod(mesh)
        weight = vol/ngrids / nk
        density *= weight
        _takebak_4d(rhoG[i:i+1], density.reshape(-1, *mesh), mesh)

    log.timer_debug1('eval_rhoG', *t0)
    return rhoG.reshape(n_dm,-1)

def _eval_tauG(ni, dm_sc, kmesh=None, verbose=None):
    pass

def _eval_lda_mat(ni, vxcG, kmesh=None, verbose=None):
    cell = ni.cell
    assert isinstance(cell, SortedGTO)
    log = logger.new_logger(cell)
    t0 = log.init_timer()

    nk = 1
    if kmesh is not None:
        nk = np.prod(kmesh)

    if ni.aft_buckets is not None:
        vxc_mat = _aft_eval_coul_matrix(ni, vxcG)
    else:
        n_dm = 1
        vxc_mat = cp.zeros((n_dm, nao, nk, nao), dtype=np.complex128)

    supmol_to_bvk_mapping = _map_images_supmol_to_bvkcell(cell.nimgs, kmesh)

    a = cell.lattice_vectors()
    b = cell.reciprocal_vectors()
    libmgrid.update_lattice_vectors(a.ctypes, b.ctypes)

    # TODO: Adjust these parameters for each li, lj values
    tiles_per_block = 4
    nsp_per_block = 4
    shm_size = 1024*47

    mg_envs = ni.mg_envs
    kern = libmgrid.evaluate_lda_mat
    kern1 = libmgrid.evaluate_lda_mat_v2
    uniq_l = cell.uniq_l_ctr[:,0]
    for bucket in ni.fft_buckets:
        mesh = bucket.mesh

        dxyz_dabc = a / mesh[:,None]
        libmgrid.update_dxyz_dabc(dxyz_dabc.ctypes)

        sub_vxcG = _take_4d(vxcG, mesh)
        vxc = ifft_in_place(sub_vxcG)
        vxcR = cp.asarray(vxc.real, order='C')

        if 1:
            for (i, j), (grid_tile_idx, bas_ij_idx, shl_pair_offsets) \
                    in bucket.grid_tile_cache.items():
                if len(bas_ij_idx) == 0: continue
                li = uniq_l[i]
                lj = uniq_l[j]
                err = kern(
                    ctypes.cast(vxc_mat.data.ptr, ctypes.c_void_p),
                    ctypes.cast(vxcR[0].data.ptr, ctypes.c_void_p),
                    ctypes.byref(mg_envs),
                    dxyz_dabc.ctypes,
                    ctypes.c_int(tiles_per_block),
                    ctypes.c_int(nsp_per_block),
                    ctypes.c_int(shm_size),
                    ctypes.c_int(li), ctypes.c_int(lj),
                    ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(grid_tile_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(supmol_to_bvk_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(len(grid_tile_idx)),
                    mesh.ctypes)
                if err != 0:
                    raise RuntimeError('evaluate_lda_mat kernel failed')
        else:
            for (i, j), bas_ij_idx in bucket.bas_ij_cache.items():
                if len(bas_ij_idx) == 0: continue
                grid_frac_ranges = bucket.grid_ranges_cache[i,j]
                li = uniq_l[i]
                lj = uniq_l[j]
                err = kern1(
                    ctypes.cast(vxc_mat.data.ptr, ctypes.c_void_p),
                    ctypes.cast(vxcR[0].data.ptr, ctypes.c_void_p),
                    ctypes.byref(mg_envs),
                    dxyz_dabc.ctypes,
                    ctypes.c_int(li),
                    ctypes.c_int(lj),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(grid_frac_ranges.data.ptr, ctypes.c_void_p),
                    ctypes.cast(supmol_to_bvk_mapping.data.ptr, ctypes.c_void_p),
                    mesh.ctypes, ctypes.c_int(len(bas_ij_idx)))
                if err != 0:
                    raise RuntimeError('evaluate_lda_mat kernel failed')

    log.timer_debug1('eval_rhoG', *t0)
    return vxc_mat

def _eval_mgga_mat(ni, vxc, kmesh=None, verbose=None):
    pass

class MultiGridNumInt(lib.StreamObject, numint.LibXCMixin):

    def build(self, xctype='LDA'):
        log = logger.new_logger(self.cell)
        t0 = log.init_timer()
        cell = self.cell = SortedGTO.from_cell(
            self.cell, decontract=True, diffuse_cutoff=1e200)
        assert cell.uniq_l_ctr[:,0].max() <= LMAX
        nbas = cell.nbas

        a = cell.lattice_vectors()
        b = cell.reciprocal_vectors()
        libmgrid.update_lattice_vectors(a.ctypes, b.ctypes)

        vol = cell.vol
        weight_penalty = vol
        precision = cell.precision / max(weight_penalty, 1)
        log_cutoff = math.log(precision)

        # Initialize bas_ij_cache
        # FIXME: ni.mesh and ni.ke_cutoff are coupled, might need only one of them
        mesh = self.mesh
        ke_max = mesh_to_cutoff(a, mesh).mean()
        Ecut = _Ecut_estimation(cell, ke_max, xctype)

        bas_ij_cache = {}
        groups = len(cell.uniq_l_ctr)
        l_ctr_offsets = np.append(0, np.cumsum(cell.l_ctr_counts))
        for i in range(groups):
            for j in range(i+1):
                ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
                jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
                mask = Ecut[ish0:ish1,:,jsh0:jsh1] > 0
                ish, jL, jsh = cp.where(mask)
                bas_ij = cp.asarray(ish * NBAS_MAX + jL * nbas + jsh, dtype=np.int64)
                bas_ij_cache[i, j] = (bas_ij, Ecut[ish, jL, jsh])
        Ecut = None
        t0 = log.timer_debug1('Initialize Ecut', *t0)

        nimgs = cell.nimgs
        Tx = cp.array(np.arange(-nimgs[0], nimgs[0]+1)[:,None] * a[0])
        Ty = cp.array(np.arange(-nimgs[1], nimgs[1]+1)[:,None] * a[1])
        Tz = cp.array(np.arange(-nimgs[2], nimgs[2]+1)[:,None] * a[2])
        Ls = (Tx[:,None,None] + Ty[:,None] + Tz).reshape(-1, 3)

        _env = _scale_sp_ctr_coeff(cell)
        ao_loc = cell.ao_loc
        self.mg_envs = PBCIntEnvVars.new(
            cell.natm, cell.nbas, 1, len(Ls),
            cell._atm, cell._bas, _env, ao_loc, Ls)

        # Initialize buckets
        is_orth_lattice = abs(a - np.diag(a.diagonal())).max() < 1e-5
        self.aft_buckets = None
        self.fft_buckets = None

        if is_orth_lattice:
            aft_init_mesh = _balance_init_mesh(a, [16]*3)
            aft_final_mesh = aft_init_mesh * 5 # or 7.5
            aft_ke_max = mesh_to_cutoff(a, aft_final_mesh).max()
            self.aft_buckets = _prepare_buckets(
                cell, bas_ij_cache, aft_init_mesh, aft_ke_max, log)

            # Filter shell pairs that are not handled by AFT. The remaining pairs
            # are handled by FFT.
            aft_ke_cutoff = self.aft_buckets[-1].ke_upper
            for key, (bas_ij_idx, pair_ke) in bas_ij_cache.items():
                idx = cp.where(pair_ke > aft_ke_cutoff)[0]
                bas_ij_cache[key] = (bas_ij_idx[idx], pair_ke[idx])

            fft_init_mesh = _balance_init_mesh(a, aft_final_mesh * 3//2)
            grid_ranges_cache = _esitmate_grid_ranges(
                bas_ij_cache, self.mg_envs, xctype, log_cutoff)
            self.fft_buckets = _prepare_buckets(
                cell, bas_ij_cache, fft_init_mesh, ke_max, log,
                with_grid_ranges=grid_ranges_cache)
        else:
            fft_init_mesh = _balance_init_mesh(a, [32]*3)
            grid_ranges_cache = _esitmate_grid_ranges(
                bas_ij_cache, self.mg_envs, xctype, log_cutoff)
            self.fft_buckets = _prepare_buckets(
                cell, bas_ij_cache, fft_init_mesh, ke_max, log,
                with_grid_ranges=grid_ranges_cache)

        # TODO: skip grid_tile_cache when memory is insufficient
        cache_tile_idx = True
        if cache_tile_idx:
            _cache_grid_range_to_tiles(self.fft_buckets, cell)
        t0 = log.timer_debug1('Initialize buckets', *t0)
        return self

if __name__ == '__main__':
    import pyscf
    from gpu4pyscf.pbc.df import ft_ao
    cell = pyscf.M(
        atom='C1 0 0 0; C2 .2 .3 .7',
        basis=[[0, [3, 1]], [1, [1, 1]], [1, [.8, 1]]],
        a=np.eye(3) * 3.2,
        mesh=[1,3,3]
    )

    cp.random.seed(4)
    nao = cell.nao
    dm = cp.random.rand(nao, nao)
    dm = dm + dm.T

    Gv = cell.get_Gv()
    Gpq = ft_ao.ft_aopair(cell, Gv)
    rho_ref = cp.einsum('pq,Gpq->G', dm, Gpq)
    vj_ref = cp.einsum('G,Gpq->pq', rho_ref, Gpq)
    print('----------')

    cell = SortedGTO.from_cell(cell)
    rhoG = _aft_eval_density(cell, dm)
    print(abs(rho_ref.real - rhoG.real).max())
    print(abs(rho_ref.imag - rhoG.imag).max())

    vj = _aft_eval_coul_matrix(cell, rhoG)
    print(abs(vj_ref.real - vj).max())
