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

import math
import ctypes
import numpy as np
import cupy as cp
import cupyx.scipy.fft as fft
from pyscf import lib
from pyscf.gto import ANG_OF, PTR_EXP, PTR_COEFF, gto_norm
from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.tools.pbc import mesh_to_cutoff, cutoff_to_mesh, super_cell
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, transpose_sum, ndarray, asarray, tag_array, load_library, absmax)
from gpu4pyscf.lib.utils import nearest_power2
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.dft import numint
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.tools import k2gamma, get_coulG
from gpu4pyscf.pbc.lib.kpts_helper import fft_matrix
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks
from gpu4pyscf.pbc.gto.cell import get_Gv
from gpu4pyscf.gto.mole import (
    PTR_BAS_COORD, SortedGTO, SortedCell, PBCIntEnvVars, _scale_sp_ctr_coeff)

libmgrid = load_library('libmgrid_v3')
NBAS_MAX = 16777216
LMAX = 4

def _aft_eval_density(ni, dm_sc):
    bvkcell = ni.bvkcell
    envs = ni.mg_envs

    a = bvkcell.lattice_vectors()
    assert abs(a - np.diag(a.diagonal())).max() < 1e-5, 'Must be orthogonal lattice'
    b = bvkcell.reciprocal_vectors()

    rhoG = cp.zeros(ni.mesh, dtype=np.complex128)

    for bucket in ni.aft_buckets:
        mesh = bucket['mesh']
        mesh_cum = cp.array(np.append(0, np.cumsum(mesh)), dtype=np.int32)
        nimgs = bucket['nimgs']
        nimgs_cum = cp.array(np.append(0, np.cumsum(nimgs*2+1)), dtype=np.int32)
        G_bases = _get_G_bases(mesh, b)
        L_bases = _get_L_bases(nimgs, a)

        # To reduce the overhead of atomicAdd, process multiple pairs for each
        # cuda block.
        pairs_per_block = 60
        shl_pair_offsets = bucket['shl_pair_offsets']
        offsets = []
        for p0, p1 in zip(shl_pair_offsets[:-1], shl_pair_offsets[1:]):
            offsets.append(cp.arange(p0, p1, pairs_per_block, dtype=np.int32))
        offsets.append(np.int32(shl_pair_offsets[-1]))
        shl_pair_offsets = cp.hstack(offsets, dtype=np.int32)
        nbatches_shl_pair = len(shl_pair_offsets) - 1

        rhoR = cp.zeros(mesh)
        rhoI = cp.zeros(mesh)
        err = libmgrid.contract_orth_aopair_dm(
            ctypes.cast(rhoR.data.ptr, ctypes.c_void_p),
            ctypes.cast(rhoI.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm_sc.data.ptr, ctypes.c_void_p),
            ctypes.byref(envs),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bucket['bas_ij_idx'].data.ptr, ctypes.c_void_p),
            ctypes.cast(G_bases.data.ptr, ctypes.c_void_p),
            ctypes.cast(L_bases.data.ptr, ctypes.c_void_p),
            ctypes.cast(mesh_cum.data.ptr, ctypes.c_void_p),
            ctypes.cast(nimgs_cum.data.ptr, ctypes.c_void_p),
            mesh.ctypes, ctypes.c_int(nbatches_shl_pair))
        if err != 0:
            raise RuntimeError('contract_orth_aopair_dm kernel failed')
        tmp_rhoG = cp.empty(mesh, dtype=np.complex128)
        tmp_rhoG.real = rhoR
        tmp_rhoG.imag = rhoI
        _takebak_4d(rhoG, tmp_rhoG, mesh)
    return rhoG.ravel()

def _aft_eval_lda_matrix(ni, vxcG):
    cell = ni.cell
    bvkcell = ni.bvkcell
    envs = ni.mg_envs

    a = bvkcell.lattice_vectors()
    b = bvkcell.reciprocal_vectors()

    vxcG = vxcG.reshape(ni.mesh)

    nao = cell.nao
    nkpts = len(ni.bvkmesh_Ls)
    vxc_mat = cp.zeros((nao, nkpts, nao))

    for bucket in ni.aft_buckets:
        mesh = bucket['mesh']
        mesh_cum = cp.array(np.append(0, np.cumsum(mesh)), dtype=np.int32)
        nimgs = bucket['nimgs']
        nimgs_cum = cp.array(np.append(0, np.cumsum(nimgs*2+1)), dtype=np.int32)
        # In real space formula, VxcG in reciprocal space is first IFFT to real
        # space. Here, AFT integrals for -G are identical to the inverse FT.
        G_bases = -_get_G_bases(mesh, b)
        L_bases = _get_L_bases(nimgs, a)

        bas_ij_idx = bucket['bas_ij_idx']

        sub_vG = _take_4d(vxcG, mesh)
        err = libmgrid.contract_orth_aopair_coulG(
            ctypes.cast(vxc_mat.data.ptr, ctypes.c_void_p),
            ctypes.cast(sub_vG.data.ptr, ctypes.c_void_p),
            ctypes.byref(envs),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(G_bases.data.ptr, ctypes.c_void_p),
            ctypes.cast(L_bases.data.ptr, ctypes.c_void_p),
            ctypes.cast(mesh_cum.data.ptr, ctypes.c_void_p),
            ctypes.cast(nimgs_cum.data.ptr, ctypes.c_void_p),
            mesh.ctypes, ctypes.c_int(len(bas_ij_idx)))
        if err != 0:
            raise RuntimeError('contract_orth_aopair_coulG kernel failed')

    # See get_Gv_weights
    weight = abs(np.linalg.det(b)) / (2*np.pi)**3
    vxc_mat *= weight
    return vxc_mat

def _eval_rhoG(ni, dm_sc):
    cell = ni.cell
    n_dm = dm_sc.shape[0]
    if ni.aft_buckets is not None:
        rhoG = _aft_eval_density(ni, dm_sc)
    else:
        rhoG = cp.zeros(ni.mesh, dtype=np.complex128)

    a = cell.lattice_vectors()

    vol = cell.vol
    nkpts = np.prod(ni.kmesh)

    mg_envs = ni.mg_envs
    kern = libmgrid.evaluate_density
    uniq_l = cell.uniq_l_ctr[:,0]
    work = None

    fft_buckets = ni.fft_buckets or []
    for bucket in fft_buckets:
        assert bucket['grid_tile_cache'] is not None
        mesh = bucket['mesh']
        ngrids = np.prod(mesh)

        weight = vol / ngrids / nkpts

        dxyz_dabc = a / mesh[:,None]
        libmgrid.update_dxyz_dabc(dxyz_dabc.ctypes)

        rhoR = ndarray(mesh, buffer=work)
        rhoR.fill(0)
        for (li, lj), (grid_tile_idx, dressed_bas_ij_idx, shl_pair_offsets) \
                in bucket['grid_tile_cache'].items():
            if len(dressed_bas_ij_idx) == 0: continue
            ntiles = len(grid_tile_idx)
            tiles_per_block = min(60, max(1, ntiles // 10000))
            err = kern(
                ctypes.cast(rhoR.data.ptr, ctypes.c_void_p),
                ctypes.cast(dm_sc[0].data.ptr, ctypes.c_void_p),
                ctypes.byref(mg_envs),
                dxyz_dabc.ctypes,
                ctypes.cast(ni.supmol_img_coords.data.ptr, ctypes.c_void_p),
                ctypes.c_int(li), ctypes.c_int(lj),
                ctypes.c_int(tiles_per_block),
                ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(dressed_bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(grid_tile_idx.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ntiles),
                (ctypes.c_int*3)(*mesh),
                ctypes.c_double(weight))
            if err != 0:
                raise RuntimeError('evaluate_density kernel failed')

        density = fft_in_place(rhoR)
        _takebak_4d(rhoG, density.reshape(mesh), mesh)

    return rhoG.reshape(n_dm,-1)

def _eval_tauG(ni, dm_sc, kmesh=None, verbose=None):
    cell = ni.cell
    n_dm = dm_sc.shape[0]
    if ni.aft_buckets is not None:
        tauG = _aft_eval_tau(ni, dm_sc)
    else:
        tauG = cp.zeros(ni.mesh, dtype=np.complex128)

    a = cell.lattice_vectors()

    vol = cell.vol
    nkpts = np.prod(ni.kmesh)

    mg_envs = ni.mg_envs
    kern = libmgrid.evaluate_tau
    uniq_l = cell.uniq_l_ctr[:,0]
    work = None

    fft_buckets = ni.fft_buckets or []
    for bucket in fft_buckets:
        assert bucket['grid_tile_cache'] is not None
        mesh = bucket['mesh']
        ngrids = np.prod(mesh)

        weight = vol / ngrids / nkpts

        dxyz_dabc = a / mesh[:,None]
        libmgrid.update_dxyz_dabc(dxyz_dabc.ctypes)

        rhoR = ndarray(mesh, buffer=work)
        tauR = None#ndarray(mesh, buffer=work)
        rhoR.fill(0)
        tauR.fill(0)
        for (li, lj), (grid_tile_idx, dressed_bas_ij_idx, shl_pair_offsets) \
                in bucket['grid_tile_cache'].items():
            if len(dressed_bas_ij_idx) == 0: continue
            ntiles = len(grid_tile_idx)
            tiles_per_block = min(60, max(1, ntiles // 10000))
            err = kern(
                ctypes.cast(rhoR.data.ptr, ctypes.c_void_p),
                ctypes.cast(tauR.data.ptr, ctypes.c_void_p),
                ctypes.cast(dm_sc[0].data.ptr, ctypes.c_void_p),
                ctypes.byref(mg_envs),
                dxyz_dabc.ctypes,
                ctypes.cast(ni.supmol_img_coords.data.ptr, ctypes.c_void_p),
                ctypes.c_int(li), ctypes.c_int(lj),
                ctypes.c_int(tiles_per_block),
                ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(dressed_bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(grid_tile_idx.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ntiles),
                (ctypes.c_int*3)(*mesh),
                ctypes.c_double(weight))
            if err != 0:
                raise RuntimeError('evaluate_density kernel failed')

        density_tmp = fft_in_place(rhoR)
        tau_tmp = fft_in_place(tauR)
        _takebak_4d(rhoG, density_tmp.reshape(mesh), mesh)
        _takebak_4d(tauG, tau_tmp.reshape(mesh), mesh)
    return tauG.reshape(n_dm,-1)

def _eval_lda_mat(ni, vxcG):
    cell = ni.cell
    if ni.aft_buckets is not None:
        vxc_mat = _aft_eval_lda_matrix(ni, vxcG)
    else:
        n_dm = 1
        nkpts = len(ni.bvkmesh_Ls)
        nao = cell.nao
        vxc_mat = cp.zeros((nao, nkpts, nao))

    a = cell.lattice_vectors()

    vxcG = vxcG.reshape(ni.mesh)

    mg_envs = ni.mg_envs
    kern = libmgrid.evaluate_lda_mat
    kern1 = libmgrid.evaluate_lda_mat_v2

    fft_buckets = ni.fft_buckets or []
    for bucket in fft_buckets:
        mesh = bucket['mesh']

        dxyz_dabc = a / mesh[:,None]
        libmgrid.update_dxyz_dabc(dxyz_dabc.ctypes)

        sub_vxcG = _take_4d(vxcG, mesh)
        vxc = ifft_in_place(sub_vxcG)
        vxcR = cp.asarray(vxc.real, order='C')

        if 1:
            for (li, lj), (grid_tile_idx, dressed_bas_ij_idx, shl_pair_offsets) \
                    in bucket['grid_tile_cache'].items():
                if len(dressed_bas_ij_idx) == 0: continue
                ntiles = len(grid_tile_idx)
                tiles_per_block = min(60, max(1, ntiles // 10000))
                err = kern(
                    ctypes.cast(vxc_mat.data.ptr, ctypes.c_void_p),
                    ctypes.cast(vxcR[0].data.ptr, ctypes.c_void_p),
                    ctypes.byref(mg_envs),
                    dxyz_dabc.ctypes,
                    ctypes.cast(ni.supmol_img_coords.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(li), ctypes.c_int(lj),
                    ctypes.c_int(tiles_per_block),
                    ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dressed_bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(grid_tile_idx.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(len(grid_tile_idx)),
                    (ctypes.c_int*3)(*mesh))
                if err != 0:
                    raise RuntimeError('evaluate_lda_mat kernel failed')
        else:
            grid_ranges_cache = bucket['grid_ranges_cache']
            for (li, lj), bas_ij_idx in bucket['bas_ij_cache'].items():
                if len(bas_ij_idx) == 0: continue
                grid_frac_ranges = grid_ranges_cache[li, lj]
                err = kern1(
                    ctypes.cast(vxc_mat.data.ptr, ctypes.c_void_p),
                    ctypes.cast(vxcR[0].data.ptr, ctypes.c_void_p),
                    ctypes.byref(mg_envs),
                    dxyz_dabc.ctypes,
                    ctypes.c_int(li),
                    ctypes.c_int(lj),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(grid_frac_ranges.data.ptr, ctypes.c_void_p),
                    (ctypes.c_int*3)(*mesh),
                    ctypes.c_int(len(bas_ij_idx)))
                if err != 0:
                    raise RuntimeError('evaluate_lda_mat kernel failed')
    return vxc_mat

def _eval_mgga_mat(ni, vxc, kmesh=None, verbose=None):
    pass

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

def _estimate_Ecut_and_grid_ranges(cell, mg_envs, bas_ij_idx, ke_max, precision, xctype):
    '''Estimate the FFT energy cutoff and the spread of each orbital pair
    in real space'''

    # Some orbitals may require high Ecut, sometimes higher than ni.ke_cutoff.
    # Use ke_max to limit the highest Ecut. This ensures that these orbital
    # pairs are included in the last bucket in _partition_ke_for_fft.
    Ecut_by_shell = _estimate_fft_Ecut_per_shell(cell, precision)
    Ecut_by_shell[Ecut_by_shell > ke_max] = ke_max
    Ecut_by_shell = cp.asarray(Ecut_by_shell, dtype=np.float32)

    npairs = len(bas_ij_idx)
    pair_ke = cp.empty(npairs, dtype=np.float32)
    grid_frac_ranges = cp.empty((3,npairs,2), dtype=np.float32)

    li_inc = lj_inc = 0
    if xctype == 'MGGA':
        li_inc = lj_inc = 1

    err = libmgrid.gaussian_prod_grid_ranges(
        ctypes.cast(grid_frac_ranges.data.ptr, ctypes.c_void_p),
        ctypes.cast(pair_ke.data.ptr, ctypes.c_void_p),
        ctypes.cast(Ecut_by_shell.data.ptr, ctypes.c_void_p),
        ctypes.byref(mg_envs),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.c_int(npairs),
        ctypes.c_int(li_inc), ctypes.c_int(lj_inc),
        ctypes.c_float(math.log(precision)))
    if err != 0:
        raise RuntimeError('grid range kernel failed')
    return pair_ke, grid_frac_ranges

def _balance_init_mesh(a, mesh):
    ke = mesh_to_cutoff(a, mesh)
    mesh = cutoff_to_mesh(a, np.mean(ke))
    return mesh // 2 * 2

def _partition_ke_for_aft(ni, pair_idx, pair_ke, init_mesh, ke_max, precision,
                          xctype, log):
    cell = ni.cell
    bvkcell = ni.bvkcell
    a = cell.lattice_vectors()
    mesh = np.asarray(init_mesh, dtype=np.int32)
    ke_lower = 0
    ke_cutoff = ni.ke_cutoff
    ke_max = min(ke_max, ke_cutoff)
    ke_upper = min(mesh_to_cutoff(a, init_mesh).min(), ke_max)

    ang_per_shell = cp.array(bvkcell._bas[:,ANG_OF])

    nimgs = np.asarray(bvkcell.nimgs, dtype=np.int32)

    buckets = []
    while ke_lower < ke_max:
        ke_upper = mesh_to_cutoff(a, mesh).min()
        if ke_upper >= ke_max:
            if ke_upper >= ke_cutoff:
                mesh = np.asarray(ni.mesh, dtype=np.int32)
            else:
                mesh_upper = cutoff_to_mesh(a, ke_max)
                mesh = np.where(mesh < mesh_upper, mesh, mesh_upper)

        filtered_pairs = pair_idx[(ke_lower < pair_ke) & (pair_ke <= ke_upper)]
        if len(filtered_pairs) > 0:
            ish, jsh = divmod(filtered_pairs, NBAS_MAX)
            lij = ang_per_shell[ish] * 5 + ang_per_shell[jsh]
            idx = cp.argsort(lij)
            filtered_pairs = filtered_pairs[idx]
            lij = lij[idx]
            shl_pair_offsets = _segment_offsets(lij)

            # TODO: nimgs can be reduced for large Ecut
            # filtered_pairs -> rcut_for_each_pair -> max_rcut -> nimgs

            buckets.append({
                'ke_cutoff': ke_upper,
                'mesh': np.asarray(mesh, dtype=np.int32),
                'nimgs': nimgs,
                'bas_ij_idx': filtered_pairs,
                'shl_pair_offsets': shl_pair_offsets,
            })
            log.debug('Add bucket: mesh=%s, shl_pairs=%d', tuple(mesh),
                      len(filtered_pairs))

        mesh = (mesh * 0.75).astype(np.int32) * 2
        ke_lower = ke_upper
    return buckets

def _estimate_fft_Ecut_per_shell(cell, precision):
    # To accurately describe the orbital in real space, the resolution for
    # real-space grid cannot be reduced, even a small normalized function is
    # associated with the orbital. The resolution is estimated in terms of the
    # energy cutoff for regular orbital with standard normalization.
    ai = cell._env[cell._bas[:,PTR_EXP]]
    li = cell._bas[:,ANG_OF]
    ci = gto_norm(li, ai)

    # Ecut ~ ci * (Ecut/2/ai**2)**(li/2) * exp(-Ecut/(2*ai))
    #      = ci * ai**li * E2**(li/2) * exp(-E2/ai), where E2 = Ecut/2
    log_fac = np.log(ci) - li*np.log(ai) - np.log(precision)
    log_fac[log_fac <= 0] = 1e-9
    E2 = log_fac * ai
    E2 = (log_fac + .5 * li * np.log(E2)) * ai
    Ecut = E2 * 2
    return Ecut

def _partition_ke_for_fft(ni, pair_idx, init_mesh, precision, xctype, log):
    cell = ni.cell
    a = cell.lattice_vectors()
    mesh = np.asarray(init_mesh, dtype=np.int32)
    ke_lower = 0
    ke_max = ni.ke_cutoff
    ke_upper = min(mesh_to_cutoff(a, init_mesh).min(), ke_max)

    bvkcell = ni.bvkcell
    ang_per_shell = cp.array(bvkcell._bas[:,ANG_OF])
    nimgs = np.asarray(cell.nimgs, dtype=np.int32)

    supmol_bas_ij_idx = _bvk_pairs_to_supmol_pairs(
        ni.mg_envs, pair_idx, precision, xctype)

    pair_ke, grid_frac_ranges = _estimate_Ecut_and_grid_ranges(
        cell, ni.mg_envs, supmol_bas_ij_idx, ke_max, precision, xctype)

    buckets = []
    while ke_lower < ke_max:
        ke_upper = mesh_to_cutoff(a, mesh).min()
        if ke_upper >= ke_max:
            mesh = np.asarray(ni.mesh, dtype=np.int32)

        idx = cp.where((ke_lower < pair_ke) & (pair_ke <= ke_upper))[0]
        if len(idx) > 0:
            filtered_pairs = supmol_bas_ij_idx[idx]
            filtered_grid_ranges = grid_frac_ranges[:,idx]

            ish, jsh = divmod(filtered_pairs, NBAS_MAX)
            lij = ang_per_shell[ish] * 5 + ang_per_shell[jsh % bvkcell.nbas]
            idx = cp.argsort(lij)
            lij = lij[idx]
            split_points = cp.where(lij[1:] != lij[:-1])[0] + 1

            # Group bas_ij_idx and grid_frac_ranges by (li, lj) patterns
            idx_by_pattern = cp.split(idx, split_points.get())
            lij_patterns = np.append(lij[0].get(), lij[split_points].get())
            lij_patterns = [divmod(x, 5) for x in lij_patterns.tolist()]

            bas_ij_cache = {key: filtered_pairs[idx]
                            for key, idx in zip(lij_patterns, idx_by_pattern)}
            grid_ranges_cache = {key: filtered_grid_ranges[:,idx]
                                 for key, idx in zip(lij_patterns, idx_by_pattern)}

            # * bas_ij_cache[key] are shell-pairs (one shell in the unit cell,
            #   the other in supmol)
            # * grid_ranges_cache[key] = grid_frac_ranges[3,N,2]
            #   For each shell pair in bas_ij_idx, stores the fractional-coordinate
            #   bounds of the real-space grids that are not negligible.
            # * grid_tile_cache[key] = (grid_tile_idx, supmol_pair_idx, shl_pair_offsets)
            #   - grid_tile_idx:
            #     Unique grid tile indices that contributes to the density.
            #   - dressed_pair_idx:
            #     Shell-pair indices contributing to the tiles in grid_tile_idx.
            #   - shl_pair_offsets:
            #     Partition the shell pairs in supmol_pair_idx by grid tile.
            buckets.append({
                'ke_cutoff': ke_upper,
                'mesh': np.asarray(mesh, dtype=np.int32),
                'bas_ij_cache': bas_ij_cache,
                'grid_ranges_cache': grid_ranges_cache,
                'grid_tile_cache': None
            })
            log.debug('Add bucket: mesh=%s, shl_pairs=%d', tuple(mesh),
                      len(filtered_pairs))

        mesh = (mesh * 0.75).astype(np.int32) * 2
        ke_lower = ke_upper
    return buckets

def _non_trivial_bvk_pairs(ni, precision):
    '''Search non-negligible pairs for <cell0|bvk-cell> overlaps'''
    cell = ni.cell
    bvkcell = ni.bvkcell
    if isinstance(cell, SortedCell):
        a = bvkcell.lattice_vectors()
        Ls = cp.asarray(lib.cartesian_prod([np.array([0., -1., 1.])]*3).dot(a))
    else:
        Ls = cp.zeros((1, 3))
    nimgs = len(Ls)

    nbas = cell.nbas
    bvk_nbas = bvkcell.nbas
    ovlp_mask = cp.zeros((nbas, bvk_nbas), dtype=bool)
    err = libmgrid.bvk_ovlp_mask_estimation(
        ctypes.cast(ovlp_mask.data.ptr, ctypes.c_void_p),
        ctypes.byref(ni.mg_envs),
        ctypes.cast(Ls.data.ptr, ctypes.c_void_p),
        ctypes.c_int(nimgs),
        ctypes.c_float(math.log(precision)))
    if err != 0:
        raise RuntimeError('bvk_ovlp_mask_estimation kernel failed')

    ish, jsh = cp.where(ovlp_mask)
    bas_ij = ish * NBAS_MAX + jsh
    return bas_ij

def _bvk_pairs_to_supmol_pairs(mg_envs, bas_ij_idx, precision, xctype):
    # The bas_ij_idx stores the effective shells in bvkcell. Each of these
    # shells involve multiple primitive shells in supmol. Unpack the bvk-shells
    # and provide the primitive pair indices in supmol.
    nimgs = mg_envs.nimgs
    npairs = len(bas_ij_idx)
    supmol_bas_ij_idx = cp.empty(npairs * nimgs, dtype=np.int64)
    is_mgga = 1 if xctype == 'MGGA' else 0
    counts = cp.empty(1, dtype=np.int32)
    err = libmgrid.supmol_non_trivial_pairs(
        ctypes.cast(supmol_bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.byref(mg_envs),
        ctypes.c_int(npairs),
        ctypes.c_float(math.log(precision)),
        ctypes.c_int(is_mgga),
        ctypes.cast(counts.data.ptr, ctypes.c_void_p))
    if err != 0:
        raise RuntimeError('bvk_ovlp_mask_estimation kernel failed')
    supmol_bas_ij_idx = supmol_bas_ij_idx[:int(counts[0].get())]
    return supmol_bas_ij_idx

def _aft_Ecut_estimation(ni, bas_ij_idx, ke_max, precision, xctype='LDA'):
    bvkcell = ni.bvkcell
    if isinstance(bvkcell, SortedCell):
        a = bvkcell.lattice_vectors()
        Ls = cp.asarray(lib.cartesian_prod([np.array([0., -1., 1.])]*3).dot(a))
    else:
        Ls = cp.zeros((1, 3))
    nimgs = len(Ls)
    npairs = len(bas_ij_idx)
    is_mgga = 1 if xctype == 'MGGA' else 0

    Ecut = cp.empty(npairs, dtype=np.float32)
    err = libmgrid.estimate_aft_Ecut(
        ctypes.cast(Ecut.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.byref(ni.mg_envs),
        ctypes.cast(Ls.data.ptr, ctypes.c_void_p),
        ctypes.c_int(nimgs),
        ctypes.c_int(npairs),
        ctypes.c_float(math.log(precision)),
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
        tiles_per_cell = cp.asarray(bucket['mesh'] / 4, dtype=np.float32)
        for (bas_ij, grid_range) in zip(
                bucket['bas_ij_cache'].values(), bucket['grid_ranges_cache'].values()):
            raw_tiles = grid_range[:,:,1] - grid_range[:,:,0]
            raw_tiles *= tiles_per_cell[:,None]
            raw_tiles = cp.ceil(raw_tiles)
            raw_tiles += 3 # penalty for roundings near boundary
            n = (raw_tiles[0] * raw_tiles[1] * raw_tiles[2]).sum().get()
            buf_size = max(buf_size, int(n))

    # temporary space to store grid_tile_idx
    work = cp.empty(buf_size+10, dtype=np.int32)
    tile_counts = work[-1:]
    # temporary space to store dressed_bas_ij
    work1 = cp.empty(buf_size+10, dtype=np.int64)

    nimgs = cell.nimgs
    nbas = cell.nbas
    kern = libmgrid.grid_range_to_tiles
    for bucket in fft_buckets:
        bucket['grid_tile_cache'] = grid_tile_cache = {}
        grid_ranges_cache = bucket['grid_ranges_cache']
        for key, bas_ij_idx in bucket['bas_ij_cache'].items():
            if len(bas_ij_idx) == 0: continue
            grid_range = grid_ranges_cache[key]
            npairs = len(bas_ij_idx)
            err = kern(
                ctypes.cast(work.data.ptr, ctypes.c_void_p),
                ctypes.cast(work1.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(grid_range.data.ptr, ctypes.c_void_p),
                (ctypes.c_int*3)(*nimgs),
                (ctypes.c_int*3)(*bucket['mesh']),
                ctypes.c_int(npairs),
                ctypes.c_int(nbas),
                ctypes.cast(tile_counts.data.ptr, ctypes.c_void_p))
            if err != 0:
                raise RuntimeError('grid_range_to_tiles failed')
            n = int(tile_counts[0].get())
            assert n < 2**31, 'int32 indexing in shl_pair_offsets'

            sorted_idx = cp.argsort(work[:n])
            grid_tile_ids = work[sorted_idx]
            dressed_bas_ij = work1[sorted_idx]

            shl_pair_offsets = _segment_offsets(grid_tile_ids)

            # TODO: Further divide large entry in shell_pair_offsets for better
            # load balance.

            # Store only the unique grid tile ids.
            grid_tile_idx = grid_tile_ids[shl_pair_offsets[:-1]]
            grid_tile_cache[key] = (
                grid_tile_idx, dressed_bas_ij, shl_pair_offsets)

def fft_in_place(x):
    return fft.fftn(x, axes=(-3, -2, -1), overwrite_x=True)

def ifft_in_place(x):
    return fft.ifftn(x, axes=(-3, -2, -1), overwrite_x=True)

def _take_4d(a, mesh, out=None):
    assert a.dtype == np.complex128
    assert a.ndim >= 3
    out_shape = mesh = tuple(mesh)
    inp_shape = a.shape
    if inp_shape[-3:] == out_shape:
        return a

    counts = 1
    if a.ndim == 4:
        counts, inp_shape = inp_shape[0], inp_shape[1:]
        out_shape = (counts,) + mesh
    out = ndarray(out_shape, dtype=np.complex128, buffer=out)
    err = libmgrid.fft_take(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(a.data.ptr, ctypes.c_void_p),
        (ctypes.c_int*3)(*mesh),
        (ctypes.c_int*3)(*inp_shape),
        ctypes.c_int(counts))
    if err != 0:
        raise RuntimeError('fft_take kernel failed')
    return out

def _takebak_4d(out, a, mesh):
    assert out.dtype == a.dtype == np.complex128
    assert out.ndim == a.ndim
    mesh = tuple(mesh)
    assert a.shape[-3:] == mesh
    out_shape = out.shape
    counts = 1
    if out.ndim == 4:
        counts, out_shape = out_shape[0], out_shape[1:]
    err = libmgrid.fft_takebak(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(a.data.ptr, ctypes.c_void_p),
        (ctypes.c_int*3)(*out_shape),
        (ctypes.c_int*3)(*mesh),
        ctypes.c_int(counts))
    if err != 0:
        raise RuntimeError('fft_take kernel failed')
    return out

def _segment_offsets(label, dtype=np.int32):
    split_points = cp.nonzero(label[:-1] != label[1:])[0] + 1
    offsets = cp.empty(len(split_points)+2, dtype=dtype)
    offsets[0] = 0
    offsets[1:-1] = split_points
    offsets[-1] = len(label)
    return offsets

def _conj_dot(a, b):
    '''a.conj().dot(b).real'''
    return a.view(np.float64).dot(b.view(np.float64))

def get_j_kpts(ni, dm_kpts, hermi=1, kpts=None, kpts_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (*, nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (*, nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    return nr_rks(ni, ni.cell, None, 'HF', dm_kpts,
                  kpts=kpts, kpts_band=kpts_band, with_j=True)[2]

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
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()

    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA' or xctype == 'HF':
        nvar = 1
    elif xctype == 'GGA':
        nvar = 4
    elif xctype == 'MGGA':
        nvar = 5
    else:
        raise NotImplementedError(f'XC functional {xc_code}')

    assert kpts_band is None
    kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
    if ni.bvkcell is None or any(ni.kmesh != kmesh):
        ni.build(kmesh, xctype)

    cell = ni.cell
    log = logger.new_logger(cell, verbose)
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    n_dm, nkpts, nao = dms.shape[:3]
    assert n_dm == 1
    dms = dms[0]

    dms = cell.apply_C_mat_CT(dms)
    #Only the tril part is processed by the integral code
    dms = transpose_sum(dms)

    #expLk = fft_matrix(kmesh)
    if nkpts == 1:
        dm_sc = dms
    else:
        expLk = cp.exp(1j*cp.asarray(ni.bvkmesh_Ls).dot(cp.asarray(kpts).T))
        dm_sc, dms = contract('kpq,Lk->qLp', dm_sc, expLk), None
        assert absmax(dms.imag) < cell.precision*5e2
        dm_sc = cp.asarray(dm_sc.real, order='C')

    rhoG = _eval_rhoG(ni, dm_sc)
    if xctype == 'MGGA':
        raise

    vol = cell.vol
    mesh = ni.mesh
    ngrids = np.prod(mesh)
    rhoG = rhoG.reshape(-1, ngrids)
    Gv = get_Gv(cell, mesh)
    if xctype == "LDA" or xctype == 'HF':
        pass
    else:
        rhoG = cp.repeat(rhoG, nvar, axis=0)
        rhoG[1:4] *= 1j
        rhoG[1:4] *= Gv.T
        if xctype == 'MGGA':
            rhoG[4] = _eval_tauG(ni, dms, kmesh)

    coulG = get_coulG(cell, Gv=Gv)
    coulomb_on_g_mesh = rhoG[0] * coulG
    coulG = None
    ecoul = .5 * float(_conj_dot(rhoG[0], coulomb_on_g_mesh).get())
    log.debug('Multigrid Coulomb energy %s', ecoul)
    t0 = log.timer("coulomb", *t0)

    n_electrons = float(rhoG[0].sum().real.get()) / vol

    if xctype == 'HF':
        assert with_j
        xc_for_fock = coulomb_on_g_mesh
        rhoG = coulomb_on_g_mesh = None
        xc_energy_sum = 0
    else:
        weight = vol / ngrids
        density = ifft_in_place(rhoG.reshape(-1, *mesh)).real.reshape(-1, ngrids)
        rhoG = None
        # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
        # computing rhoR with IFFT, the weight factor is not needed.
        density /= weight

        # eval_xc_eff supports float64 only
        density = cp.asarray(density, dtype=np.float64, order='C')
        xc_for_energy, xc_for_fock = ni.eval_xc_eff(
            xc_code, density, deriv=1, xctype=xctype, spin=0
        )[:2]

        rho_sf = density[0].real
        xc_energy_sum = float(rho_sf.dot(xc_for_energy.ravel()).get()) * weight

        # To reduce the memory usage, we reuse the xc_for_fock name.
        # Now xc_for_fock represents xc on G space
        xc_for_fock *= weight
        xc_for_fock = fft_in_place(xc_for_fock.reshape(-1, *mesh)).reshape(-1, ngrids)

        log.debug("Multigrid exc %s  nelec %s", xc_energy_sum, n_electrons)

        if xctype == "LDA":
            pass

        else:
            xc_for_fock[0] -= cp.einsum("gp, pg -> p", xc_for_fock[1:4], Gv) * 1j
            xc_for_fock = xc_for_fock[0].reshape((-1, ngrids))

            if xctype == "MGGA":
                raise

        if with_j:
            xc_for_fock[0] += coulomb_on_g_mesh
        coulomb_on_g_mesh = None

    #if kpts_band is not None:
    #    ni = ni.copy().reset().build()
    veff = _eval_lda_mat(ni, xc_for_fock)
    if nkpts == 1:
        veff = veff[:,0]
    else:
        veff = contract('pLq,Lk->kpq', veff, expLk)
    veff = cell.apply_CT_mat_C(veff)
    veff = transpose_sum(veff)

    #veff = _format_jks(veff, dm_kpts, input_band, kpts)
    veff = tag_array(veff, ecoul=ecoul, exc=xc_energy_sum)
    t0 = log.timer("xc", *t0)
    return n_electrons, xc_energy_sum, veff

class MultiGridNumInt(lib.StreamObject, numint.LibXCMixin):
    enable_aft = True

    def __init__(self, cell):
        self.reset(cell)

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
            self.mesh = cell.mesh
        self.bvkcell = None
        self.mg_envs = None
        self.supmol_img_coords = None
        self.aft_buckets = None
        self.fft_buckets = None

    def build(self, kmesh=None, xctype='MGGA'):
        log = logger.new_logger(self.cell)
        t0 = log.init_timer()
        cell = self.cell = SortedGTO.from_cell(
            self.cell, decontract=True, diffuse_cutoff=1e200)
        assert cell.uniq_l_ctr[:,0].max() <= LMAX

        self.xctype = xctype
        self.kmesh = kmesh
        if kmesh is None:
            bvkcell = cell
            bvkmesh_Ls = np.zeros((1, 3))
        else:
            bvkcell = super_cell(cell, kmesh, wrap_around=True)
            bvkmesh_Ls = translation_vectors_for_kmesh(cell, kmesh, wrap_around=True)
        self.bvkcell = bvkcell
        self.bvkmesh_Ls = bvkmesh_Ls
        bvk_ncells = len(bvkmesh_Ls)

        Ls = cp.asarray(bvkcell.get_lattice_Ls())
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
        nimgs = len(Ls)
        log.debug('ft_ao bvk_ncells=%d, nimgs=%d', bvk_ncells, nimgs)
        _env = _scale_sp_ctr_coeff(cell)
        ao_loc = cell.ao_loc
        self.mg_envs = PBCIntEnvVars.new(
            cell.natm, cell.nbas, bvk_ncells, nimgs,
            bvkcell._atm, bvkcell._bas, _env, ao_loc, Ls)

        a = cell.lattice_vectors()
        b = cell.reciprocal_vectors(norm_to=1)
        libmgrid.update_lattice_vectors(a.ctypes, b.ctypes)

        # FIXME: weight_penalty in terms of overlap integrals?
        vol = cell.vol
        weight_penalty = vol
        precision = cell.precision / max(weight_penalty, 1)
        bas_ij_idx = _non_trivial_bvk_pairs(self, precision)

        # Initialize buckets
        is_orth_lattice = abs(a - np.diag(a.diagonal())).max() < 1e-5
        self.aft_buckets = None
        self.fft_buckets = None

        # FIXME: ni.mesh and ni.ke_cutoff are coupled, might need only one of them
        mesh = self.mesh
        self.ke_cutoff = max(0.1, mesh_to_cutoff(a, mesh).min())

        if self.enable_aft and is_orth_lattice:
            # Estimate Ecut for AFT integrals. These can be potentially handled by
            # aft_eval_* functions.
            # Use self.ke_cutoff to limit the highest Ecut. This ensures to handle
            # shell-pairs even if their Ecuts are higher than ke_cutoff.
            aft_Ecut = _aft_Ecut_estimation(
                self, bas_ij_idx, self.ke_cutoff, precision, xctype)

            aft_init_mesh = _balance_init_mesh(a, [16]*3)
            aft_final_mesh = aft_init_mesh * 5 # or 7.5
            aft_ke_max = mesh_to_cutoff(a, aft_final_mesh).max()
            self.aft_buckets = _partition_ke_for_aft(
                self, bas_ij_idx, aft_Ecut, aft_init_mesh, aft_ke_max,
                precision, xctype, log)

            # Filter shell pairs that are not handled by AFT. The remaining pairs
            # are handled by FFT.
            if aft_ke_max < self.ke_cutoff:
                bas_ij_idx = bas_ij_idx[aft_Ecut > aft_ke_max]
            else:
                bas_ij_idx = None

            fft_init_mesh = aft_final_mesh * 3//2
        else:
            fft_init_mesh = [32]*3

        if bas_ij_idx is not None and len(bas_ij_idx) > 0:
            # bas_ij_idx are the effective paris between cell0 and bvkcell.
            # The FFT-MultiGrid code operates on cell0-supmol paris.
            # Every bvkcell shell in bas_ij_idx needs to be unpacked to several
            # primitive shells in supmol.
            fft_init_mesh = _balance_init_mesh(a, fft_init_mesh)
            self.fft_buckets = _partition_ke_for_fft(
                self, bas_ij_idx, fft_init_mesh, precision, xctype, log)

            nimgs = cell.nimgs
            Tx = np.arange(-nimgs[0], nimgs[0]+1, dtype=np.float64)
            Ty = np.arange(-nimgs[1], nimgs[1]+1, dtype=np.float64)
            Tz = np.arange(-nimgs[2], nimgs[2]+1, dtype=np.float64)
            self.supmol_img_coords = cp.asarray(lib.cartesian_prod([Tx, Ty, Tz]).dot(a))

            # TODO: skip grid_tile_cache when memory is insufficient
            cache_tile_idx = True
            if cache_tile_idx:
                _cache_grid_range_to_tiles(self.fft_buckets, cell)
        t0 = log.timer_debug1('Initialize buckets', *t0)
        return self

    get_j = get_j_kpts

    nr_rks = nr_rks

#    get_nuc = get_nuc
#    get_pp = get_pp
#
#    get_rho = get_rho
#    nr_rks = nr_rks
#    nr_uks = nr_uks
#    nr_vxc = get_vxc = multigrid_v1.MultiGridNumInt.get_vxc
#
#    eval_xc_eff = numint.NumInt.eval_xc_eff
#    _init_xcfuns = numint.NumInt._init_xcfuns
#
#    def nr_rks_fxc(self, cell, grids, xc_code, dm0, dms, hermi=0, fxc=None,
#                   kpts=None, with_j=False):
#        if kpts is None:
#            kpts = np.zeros((1,3))
#        elif isinstance(kpts, KPoints):
#            kpts = kpts.kpts_ibz
#
#        assert kpts.ndim == 2
#        assert dms.ndim == 4
#        nset, nkpts, nao = dms.shape[:3]
#        assert len(kpts) == nkpts
#
#        # The transition density matrices dm1 must be hermitian. The
#        # evaluate_density_on_g_mesh function only supports real density.
#        assert hermi == 1
#        v_hermi = hermi
#
#        xctype = self._xc_type(xc_code)
#        if xctype == 'HF':
#            return cp.zeros_like(dms)
#
#        assert xctype in ('LDA', 'GGA', 'MGGA')
#
#        if fxc is None:
#            spin = 0
#            fxc = self.cache_xc_kernel1(cell, grids, xc_code, dm0, spin, kpts, is_rhf=True)[2]
#
#        mesh = self.mesh
#        Gv = get_Gv(cell, mesh)
#        ngrids = len(Gv)
#        rho1 = evaluate_density_on_g_mesh(self, dms, kpts, xctype)
#        if with_j:
#            coulG = pbc_tools.get_coulG(cell, Gv=Gv)
#            coulomb_on_g_mesh = rho1[:,0] * coulG
#        rho1 = ifft_in_place(rho1.reshape(-1, *mesh)).real.reshape(nset, -1, ngrids)
#        wv = cp.einsum('nxg,xyg->nyg', rho1, fxc)
#        wv = fft_in_place(wv.reshape(-1, *mesh)).reshape(wv.shape)
#
#        if with_j:
#            wv[:,0] += coulomb_on_g_mesh
#
#        if 'GGA' in xctype:
#            wv[:,0] -= contract('nxp,xp->np', wv[:,1:4], Gv.T) * 1j
#            if xctype == 'GGA':
#                wv = cp.asarray(wv[:,0], order='C')
#            elif xctype == 'MGGA':
#                wv = cp.asarray(wv[:,[0, 4]], order='C')
#
#        with_tau = (xctype == 'MGGA')
#        vmat = convert_xc_on_g_mesh_to_fock(self, wv, v_hermi, kpts, with_tau=with_tau)
#        return vmat.reshape(dms.shape)
#
#    def nr_rks_fxc_st(self, cell, grids, xc_code, dm0, dms, hermi=0, singlet=True,
#                      fxc=None, kpts=None, with_j=False):
#        if fxc is None:
#            spin = 1
#            fxc = self.cache_xc_kernel1(cell, grids, xc_code, dm0, spin, kpts,
#                                      is_rhf=True)[2]
#        if singlet:
#            fxc = fxc[0,:,0] + fxc[0,:,1]
#        else:
#            fxc = fxc[0,:,0] - fxc[0,:,1]
#        return self.nr_rks_fxc(cell, grids, xc_code, dm0, dms, hermi, fxc, kpts, with_j)
#
#    def nr_uks_fxc(self, cell, grids, xc_code, dm0, dms, hermi=0, fxc=None,
#                   kpts=None, with_j=False):
#        if kpts is None:
#            kpts = np.zeros((1,3))
#        elif isinstance(kpts, KPoints):
#            kpts = kpts.kpts_ibz
#
#        assert kpts.ndim == 2
#        assert dms.ndim == 5
#        nset, nkpts, nao = dms.shape[1:4]
#        assert len(kpts) == nkpts
#
#        # The transition density matrices dm1 must be hermitian. The
#        # evaluate_density_on_g_mesh function only supports real density.
#        assert hermi == 1
#        v_hermi = hermi
#
#        xctype = self._xc_type(xc_code)
#        if xctype == 'HF':
#            return cp.zeros_like(dms)
#
#        assert xctype in ('LDA', 'GGA', 'MGGA')
#
#        if fxc is None:
#            spin = 1
#            fxc = self.cache_xc_kernel1(cell, grids, xc_code, dm0, spin, kpts, is_rhf=False)[2]
#
#        mesh = self.mesh
#        Gv = get_Gv(cell, mesh)
#        ngrids = len(Gv)
#        rho1 = evaluate_density_on_g_mesh(self, dms.reshape(-1,nkpts,nao,nao), kpts, xctype)
#        if with_j:
#            coulG = pbc_tools.get_coulG(cell, Gv=Gv)
#            coulomb_on_g_mesh = rho1[:,0].reshape(2, nset, ngrids).sum(axis=0) * coulG
#        rho1 = ifft_in_place(rho1.reshape(-1, *mesh)).real.reshape(2, nset, -1, ngrids)
#        wv = cp.einsum('anxg,axbyg->bnyg', rho1, fxc)
#        wv = fft_in_place(wv.reshape(-1, *mesh)).reshape(wv.shape)
#
#        if with_j:
#            wv[:,:,0] += coulomb_on_g_mesh
#
#        if 'GGA' in xctype:
#            wv[:,:,0] -= contract('anxp,xp->anp', wv[:,:,1:4], Gv.T) * 1j
#            if xctype == 'GGA':
#                wv = cp.asarray(wv[:,:,0], order='C')
#            elif xctype == 'MGGA':
#                wv = cp.asarray(wv[:,:,[0, 4]], order='C')
#
#        wv = wv.reshape(2*nset, -1, ngrids)
#
#        with_tau = (xctype == 'MGGA')
#        vmat = convert_xc_on_g_mesh_to_fock(self, wv, v_hermi, kpts, with_tau=with_tau)
#        return vmat.reshape(dms.shape)
#
#    def cache_xc_kernel1(self, cell, grids, xc_code, dm, spin=0, kpts=None, is_rhf=None):
#        if isinstance(kpts, KPoints):
#            raise NotImplementedError
#
#        dms = _format_dms(dm, kpts)
#        if is_rhf is None:
#            is_rhf = len(dms) == 1
#        elif is_rhf:
#            assert len(dms) == 1
#        else:
#            assert spin == 1
#            assert len(dms) == 2
#
#        xctype = self._xc_type(xc_code)
#        mesh = self.mesh
#        ngrids = np.prod(mesh)
#        rho = evaluate_density_on_g_mesh(self, dms, kpts, xctype)
#        # Remove the grid weights. rho is scaled by the grid weights
#        # (vol/ngrids) in evaluate_density_on_g_mesh
#        rho *= ngrids / cell.vol
#        rho = ifft_in_place(rho.reshape(-1, *mesh)).real.reshape(rho.shape)
#
#        if is_rhf:
#            if spin == 1:
#                rho *= .5
#                rho = cp.repeat(rho, 2, axis=0)
#            else:
#                rho = rho[0]
#
#        vxc, fxc = self.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype, spin=spin)[1:3]
#        return rho, vxc, fxc
#
#    cache_xc_kernel = NotImplemented

    to_cpu = NotImplemented
    to_gpu = NotImplemented
