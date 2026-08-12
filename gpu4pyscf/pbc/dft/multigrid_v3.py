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

'''
MultiGrid numerical integration for computing XC matrices

In the current implementation, the memory required by Vxc integrals are
       RKS     UKS
LDA    5*N^3   8*N^3
GGA    8*N^3   14*N^3
MGGA   9*N^3   16*N^3

For 80 GB memory, the upper limits of mesh are approximately
       RKS     UKS
LDA    1200    1000
GGA    1000    850
MGGA   1000    800
'''

import math
import ctypes
import numpy as np
import cupy as cp
import cupyx.scipy.fft as fft
from pyscf import lib
from pyscf.gto import ANG_OF, ATOM_OF, PTR_COORD, PTR_EXP, PTR_COEFF, gto_norm
from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, transpose_sum, ndarray, asarray, tag_array, load_library, absmax,
    get_avail_mem)
from gpu4pyscf.lib.utils import nearest_power2
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.dft import numint
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.tools import k2gamma, get_coulG
from gpu4pyscf.pbc.lib.kpts_helper import fft_matrix
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks
from gpu4pyscf.pbc.gto.cell import get_Gv, get_Gv_base
from gpu4pyscf.gto.mole import (
    PTR_BAS_COORD, SortedGTO, SortedCell, PBCIntEnvVars, _scale_sp_ctr_coeff)

libmgrid = load_library('libmgrid_v3')
NBAS_MAX = 16777216
LMAX = 4

_kernel_registery = {}

def _aft_eval_density(ni, dm_sc, with_tau=False):
    cell = ni.cell
    bvkcell = ni.bvkcell

    a = bvkcell.lattice_vectors()
    assert abs(a - np.diag(a.diagonal())).max() < 1e-5, 'Must be orthogonal lattice'
    b = cell.reciprocal_vectors()

    nkpts = len(ni.bvkmesh_Ls)
    weight = 1./nkpts

    rhoG = cp.zeros(ni.mesh, dtype=np.complex128)
    kern = libmgrid.orth_contract_aopair_dm
    tauG = None
    if with_tau:
        tauG = cp.zeros(ni.mesh, dtype=np.complex128)
        kern = libmgrid.orth_contract_ft_tau_dm

    for bucket in ni.aft_buckets:
        mesh = bucket['mesh']
        mesh_cum = cp.array(np.append(0, np.cumsum(mesh)), dtype=np.int32)
        nimgs = bucket['nimgs']
        nimgs_cum = cp.array(np.append(0, np.cumsum(nimgs*2+1)), dtype=np.int32)
        Gx, Gy, Gz = _get_G_bases(mesh, b)
        G_bases = cp.hstack([Gx[0], Gy[1], Gz[2]])
        L_bases = _get_L_bases(nimgs, a)

        # To reduce the overhead of atomicAdd, process multiple pairs for each
        # cuda block.
        pairs_per_block = 100
        shl_pair_offsets = bucket['shl_pair_offsets']
        offsets = []
        for p0, p1 in zip(shl_pair_offsets[:-1], shl_pair_offsets[1:]):
            offsets.append(cp.arange(p0, p1, pairs_per_block, dtype=np.int32))
        offsets.append(np.int32(shl_pair_offsets[-1]))
        shl_pair_offsets = cp.hstack(offsets, dtype=np.int32)
        nbatches_shl_pair = len(shl_pair_offsets) - 1

        rhoR = cp.zeros(mesh)
        rhoI = cp.zeros(mesh)
        tauR = tauI = rhoR
        if with_tau:
            tauR = cp.zeros(mesh)
            tauI = cp.zeros(mesh)
        err = kern(
            ctypes.cast(rhoR.data.ptr, ctypes.c_void_p),
            ctypes.cast(rhoI.data.ptr, ctypes.c_void_p),
            ctypes.cast(tauR.data.ptr, ctypes.c_void_p),
            ctypes.cast(tauI.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm_sc.data.ptr, ctypes.c_void_p),
            ctypes.byref(ni.mg_envs),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bucket['bas_ij_idx'].data.ptr, ctypes.c_void_p),
            ctypes.cast(G_bases.data.ptr, ctypes.c_void_p),
            ctypes.cast(L_bases.data.ptr, ctypes.c_void_p),
            ctypes.cast(mesh_cum.data.ptr, ctypes.c_void_p),
            ctypes.cast(nimgs_cum.data.ptr, ctypes.c_void_p),
            (ctypes.c_int*3)(*mesh),
            ctypes.c_int(nbatches_shl_pair),
            ctypes.c_double(weight))
        if err != 0:
            raise RuntimeError('contract_orth_aopair_dm kernel failed')
        _takebak_4d(rhoG, (rhoR, rhoI), mesh)
        if with_tau:
            _takebak_4d(tauG, (tauR, tauI), mesh)
    return rhoG, tauG

def _aft_eval_xc_matrix(ni, vxcG, out=None):
    cell = ni.cell
    bvkcell = ni.bvkcell

    a = bvkcell.lattice_vectors()
    b = cell.reciprocal_vectors()

    if isinstance(vxcG, cp.ndarray):
        vrhoG = vxcG.reshape(ni.mesh)
        vtauG = None
        kern = libmgrid.orth_aft_lda_mat
    else:
        vrhoG, vtauG = vxcG
        vrhoG = vrhoG.reshape(ni.mesh)
        vtauG = vtauG.reshape(ni.mesh)
        kern = libmgrid.orth_aft_mgga_mat

    nao = cell.nao
    nkpts = len(ni.bvkmesh_Ls)
    vxc_mat = ndarray((nao, nkpts, nao), dtype=np.float64, buffer=out)
    vxc_mat.fill(0.)

    for bucket in ni.aft_buckets:
        mesh = bucket['mesh']
        mesh_cum = cp.array(np.append(0, np.cumsum(mesh)), dtype=np.int32)
        nimgs = bucket['nimgs']
        nimgs_cum = cp.array(np.append(0, np.cumsum(nimgs*2+1)), dtype=np.int32)
        # In real space formula, VxcG in reciprocal space is first IFFT to real
        # space. Here, AFT integrals for -G are identical to the inverse FT.
        Gx, Gy, Gz = _get_G_bases(mesh, b)
        G_bases = -cp.hstack([Gx[0], Gy[1], Gz[2]])
        L_bases = _get_L_bases(nimgs, a)

        bas_ij_idx = bucket['bas_ij_idx']

        sub_vrhoG = _take_4d(vrhoG, mesh)
        sub_vtauG = sub_vrhoG
        if vtauG is not None:
            sub_vtauG = _take_4d(vtauG, mesh)
        err = kern(
            ctypes.cast(vxc_mat.data.ptr, ctypes.c_void_p),
            ctypes.cast(sub_vrhoG.data.ptr, ctypes.c_void_p),
            ctypes.cast(sub_vtauG.data.ptr, ctypes.c_void_p),
            ctypes.byref(ni.mg_envs),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(G_bases.data.ptr, ctypes.c_void_p),
            ctypes.cast(L_bases.data.ptr, ctypes.c_void_p),
            ctypes.cast(mesh_cum.data.ptr, ctypes.c_void_p),
            ctypes.cast(nimgs_cum.data.ptr, ctypes.c_void_p),
            (ctypes.c_int*3)(*mesh),
            ctypes.c_int(len(bas_ij_idx)))
        if err != 0:
            raise RuntimeError('contract_orth_aopair_coulG kernel failed')

    # See get_Gv_weights
    weight = abs(np.linalg.det(b)) / (2*np.pi)**3
    vxc_mat *= weight
    return vxc_mat

def _eval_density(ni, dm_sc, with_tau=False):
    cell = ni.cell
    if ni.aft_buckets is not None:
        rhoG, tauG = _aft_eval_density(ni, dm_sc, with_tau)
    else:
        rhoG = cp.zeros(ni.mesh, dtype=np.complex128)
        tauG = None
        if with_tau:
            tauG = cp.zeros(ni.mesh, dtype=np.complex128)

    a = cell.lattice_vectors()
    vol = np.linalg.det(a)
    nkpts = len(ni.bvkmesh_Ls)

    work = cp.empty_like(rhoG)
    if not with_tau:
        kern = libmgrid.evaluate_density
    else:
        kern = libmgrid.evaluate_tau
        work1 = cp.empty_like(rhoG)
    mg_envs = ni.mg_envs

    fft_buckets = ni.fft_buckets or []
    for bucket in fft_buckets:
        assert bucket['grid_tile_cache'] is not None
        mesh = bucket['mesh']
        ngrids = np.prod(mesh)

        weight = vol / ngrids / nkpts

        dxyz_dabc = a / mesh[:,None]
        libmgrid.update_dxyz_dabc(dxyz_dabc.ctypes)

        rhoR = ndarray(mesh, dtype=np.complex128, buffer=work)
        rhoR.fill(0)
        tauR = rhoR # placeholder
        if with_tau:
            tauR = ndarray(mesh, dtype=np.complex128, buffer=work1)
            tauR.fill(0)
        for ((li, lj), (grid_tile_idx, dressed_bas_ij_idx, shl_pair_offsets)) \
                in zip(bucket['lij_patterns'], bucket['grid_tile_cache']):
            if len(dressed_bas_ij_idx) == 0: continue
            ntiles = len(grid_tile_idx)
            tiles_per_block = min(100, max(1, ntiles // 10000))
            err = kern(
                ctypes.cast(rhoR.data.ptr, ctypes.c_void_p),
                ctypes.cast(tauR.data.ptr, ctypes.c_void_p),
                ctypes.cast(dm_sc.data.ptr, ctypes.c_void_p),
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
                ctypes.c_double(weight),
                ctypes.c_double(bucket['negligible']))
            if err != 0:
                raise RuntimeError('evaluate_density kernel failed')

        _takebak_4d(rhoG, fft_in_place(rhoR).reshape(mesh), mesh)
        if with_tau:
            _takebak_4d(tauG, fft_in_place(tauR).reshape(mesh), mesh)

    return rhoG, tauG

def _eval_xc_mat(ni, vxcG, out=None, work=None):
    '''Note, contents of vxcG will be destroyed in this function
    '''
    cell = ni.cell
    if ni.aft_buckets is not None:
        vxc_mat = _aft_eval_xc_matrix(ni, vxcG, out)
    else:
        nkpts = len(ni.bvkmesh_Ls)
        nao = cell.nao
        vxc_mat = ndarray((nao, nkpts, nao), dtype=np.float64, buffer=out)
        vxc_mat.fill(0.)

    a = cell.lattice_vectors()

    if isinstance(vxcG, cp.ndarray):
        vrhoG = vxcG.reshape(ni.mesh)
        vtauG = None
        work = ndarray((3,vrhoG.size), dtype=np.float64, buffer=work)
        kern = libmgrid.evaluate_lda_mat
        kern1 = libmgrid.evaluate_lda_mat_v2
    else:
        vrhoG, vtauG = vxcG
        vrhoG = vrhoG.reshape(ni.mesh)
        vtauG = vtauG.reshape(ni.mesh)
        work = ndarray((4,vrhoG.size), dtype=np.float64, buffer=work)
        kern = libmgrid.evaluate_mgga_mat
        kern1 = libmgrid.evaluate_mgga_mat_v2

    mg_envs = ni.mg_envs

    fft_buckets = ni.fft_buckets or []
    for bucket in fft_buckets:
        mesh = bucket['mesh']

        dxyz_dabc = a / mesh[:,None]
        libmgrid.update_dxyz_dabc(dxyz_dabc.ctypes)

        # _take_4d does not always make a copy. In the last bucket, the contents
        # of vrhoG will be overwritten by ifft_in_place
        sub_vrhoG = _take_4d(vrhoG, mesh, work[:2])
        sub_vrhoR = ndarray(mesh, dtype=np.float64, buffer=work[2])
        sub_vrhoR[:] = ifft_in_place(sub_vrhoG).real
        sub_vtauR = sub_vrhoR # placeholder

        if vtauG is not None:
            sub_vtauG = _take_4d(vtauG, mesh, work[:2])
            sub_vtauR = ndarray(mesh, dtype=np.float64, buffer=work[3])
            sub_vtauR[:] = ifft_in_place(sub_vtauG).real

        if 0:
            for ((li, lj), (grid_tile_idx, dressed_bas_ij_idx, shl_pair_offsets)) \
                    in zip(bucket['lij_patterns'], bucket['grid_tile_cache']):
                if len(dressed_bas_ij_idx) == 0: continue
                ntiles = len(grid_tile_idx)
                tiles_per_block = min(100, max(1, ntiles // 10000))
                err = kern(
                    ctypes.cast(vxc_mat.data.ptr, ctypes.c_void_p),
                    ctypes.cast(sub_vrhoR.data.ptr, ctypes.c_void_p),
                    ctypes.cast(sub_vtauR.data.ptr, ctypes.c_void_p),
                    ctypes.byref(mg_envs),
                    dxyz_dabc.ctypes,
                    ctypes.cast(ni.supmol_img_coords.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(li), ctypes.c_int(lj),
                    ctypes.c_int(tiles_per_block),
                    ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dressed_bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(grid_tile_idx.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(len(grid_tile_idx)),
                    (ctypes.c_int*3)(*mesh),
                    ctypes.c_double(bucket['negligible']))
                if err != 0:
                    raise RuntimeError('evaluate_xc_mat kernel failed')
        else:
            for (li, lj), bas_ij_idx, grid_frac_ranges in zip(
                    bucket['lij_patterns'], bucket['bas_ij_cache'],
                    bucket['grid_ranges_cache']):
                if len(bas_ij_idx) == 0: continue
                err = kern1(
                    ctypes.cast(vxc_mat.data.ptr, ctypes.c_void_p),
                    ctypes.cast(sub_vrhoR.data.ptr, ctypes.c_void_p),
                    ctypes.cast(sub_vtauR.data.ptr, ctypes.c_void_p),
                    ctypes.byref(mg_envs),
                    dxyz_dabc.ctypes,
                    ctypes.c_int(li),
                    ctypes.c_int(lj),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(grid_frac_ranges.data.ptr, ctypes.c_void_p),
                    (ctypes.c_int*3)(*mesh),
                    ctypes.c_int(len(bas_ij_idx)),
                    ctypes.c_double(bucket['negligible']))
                if err != 0:
                    raise RuntimeError('evaluate_xc_mat kernel failed')
    return vxc_mat

def _get_G_bases(mesh, b):
    Gx = cp.array(np.fft.fftfreq(mesh[0], 1./mesh[0]) * b[0,:,None])
    Gy = cp.array(np.fft.fftfreq(mesh[1], 1./mesh[1]) * b[1,:,None])
    Gz = cp.array(np.fft.fftfreq(mesh[2], 1./mesh[2]) * b[2,:,None])
    return (Gx, Gy, Gz)

def _get_L_bases(nimgs, a):
    Tx = np.arange(-nimgs[0], nimgs[0]+1) * a[0,0]
    Ty = np.arange(-nimgs[1], nimgs[1]+1) * a[1,1]
    Tz = np.arange(-nimgs[2], nimgs[2]+1) * a[2,2]
    L_bases = cp.array(np.hstack([Tx, Ty, Tz]))
    return L_bases

def _estimate_Ecut_and_grid_ranges(ni, bas_ij_idx, ke_max, precision, xctype):
    '''Estimate the FFT energy cutoff and the spread of each orbital pair
    in real space'''
    cell = ni.cell
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
        ctypes.byref(ni.mg_envs),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.c_int(npairs),
        ctypes.c_int(li_inc), ctypes.c_int(lj_inc),
        ctypes.c_float(math.log(precision)))
    if err != 0:
        raise RuntimeError('grid range kernel failed')
    return pair_ke, grid_frac_ranges

def _estimate_fft_Ecut_per_shell(cell, precision):
    # To accurately describe the orbital in real space, the resolution for
    # real-space grid cannot be reduced, even a small normalized function is
    # associated with the orbital. The resolution is estimated in terms of the
    # energy cutoff for regular orbital with standard normalization.
    ai = cell._env[cell._bas[:,PTR_EXP]]
    li = cell._bas[:,ANG_OF]
    ci = gto_norm(li, ai)

    # Ecut ~ ci * (Ecut/2/ai**2)**(li/2) * exp(-Ecut/(2*ai))
    #      = ci / ai**li * E2**(li/2) * exp(-E2/ai), where E2 = Ecut/2
    log_fac = np.log(ci) + 1.717 - (li+1.5)*np.log(ai) - np.log(precision)
    log_fac[log_fac <= 0] = 1e-9
    E2 = log_fac * ai
    E2 = (log_fac + .5 * li * np.log(E2)) * ai
    Ecut = E2 * 2
    return Ecut

def ke_to_mesh(a, cutoff):
    '''
    Based on pyscf.pbc.tools.pbc.cutoff_to_mesh
    '''
    b = 2 * np.pi * np.linalg.inv(a.T)
    rx = np.linalg.qr(b[[1,2,0]].T)[1][2,2]
    ry = np.linalg.qr(b[[2,0,1]].T)[1][2,2]
    rz = np.linalg.qr(b.T)[1][2,2]

    Gmax = (2*cutoff)**.5 / np.abs([rx, ry, rz])
    mesh = np.ceil(Gmax * 2).astype(np.int32)
    return mesh

def mesh_to_ke(a, mesh):
    '''
    Based on pyscf.pbc.tools.pbc.mesh_to_cutoff
    '''
    b = 2 * np.pi * np.linalg.inv(a.T)
    rx = np.linalg.qr(b[[1,2,0]].T)[1][2,2]
    ry = np.linalg.qr(b[[2,0,1]].T)[1][2,2]
    rz = np.linalg.qr(b.T)[1][2,2]

    gs = np.asarray(mesh) / 2
    Gmax = gs * np.array([rx, ry, rz])
    ke_cutoff = Gmax**2 / 2
    return ke_cutoff.min()

def _partition_ke_for_aft(ni, pair_idx, pair_ke, init_ke, ke_max, precision,
                          xctype, log):
    cell = ni.cell
    bvkcell = ni.bvkcell
    a = cell.lattice_vectors()
    mesh = ke_to_mesh(a, init_ke)

    ke_cutoff = ni.ke_cutoff
    mesh_max = np.asarray(ni.mesh, dtype=np.int32)
    if ke_max < ke_cutoff:
        mesh_final = ke_to_mesh(a, ke_max)
        mesh_final = np.where(mesh_final < mesh_max, mesh_final, mesh_max)
    else:
        ke_max = ke_cutoff
        mesh_final = mesh_max

    ang_per_shell = cp.array(bvkcell._bas[:,ANG_OF])
    nimgs = np.asarray(bvkcell.nimgs, dtype=np.int32)

    buckets = []

    ke_lower, ke_upper = 0, init_ke
    while ke_lower < ke_max:
        if ke_upper >= ke_max:
            mesh = mesh_final

        filtered_pairs = pair_idx[(ke_lower < pair_ke) & (pair_ke <= ke_upper)]
        if len(filtered_pairs) > 0:
            ish, jsh = divmod(filtered_pairs, NBAS_MAX)
            lij = ang_per_shell[ish] * 5 + ang_per_shell[jsh]
            idx = cp.argsort(lij)
            filtered_pairs = filtered_pairs[idx]
            lij = lij[idx]
            shl_pair_offsets = _segment_offsets(lij).get()

            # TODO: nimgs can be reduced for large Ecut
            # filtered_pairs -> rcut_for_each_pair -> max_rcut -> nimgs

            buckets.append({
                'ke_cutoff': ke_upper,
                'mesh': np.asarray(mesh, dtype=np.int32),
                'nimgs': nimgs,
                'bas_ij_idx': filtered_pairs,
                'shl_pair_offsets': shl_pair_offsets,
            })
            log.debug('Add aft bucket: ke=%g mesh=%s, shl_pairs=%d', ke_upper,
                      tuple(mesh), len(filtered_pairs))

        mesh = (mesh * 0.75).astype(np.int32) * 2
        ke_lower, ke_upper = ke_upper, mesh_to_ke(a, mesh)
    return buckets

def _partition_ke_for_fft(ni, pair_idx, init_ke, precision, xctype, log):
    cell = ni.cell
    bvkcell = ni.bvkcell

    a = cell.lattice_vectors()
    mesh = ke_to_mesh(a, init_ke)
    ke_max = ni.ke_cutoff
    mesh_final = ni.mesh

    ang_per_shell = cp.array(bvkcell._bas[:,ANG_OF])

    # a penalty to encounter for lattice sum
    rad = cell.rcut / ni.bvkcell.vol**(1./3) + 1
    surface = 4*np.pi * rad**2
    lattice_sum_factor = surface
    log.debug1('lattice_sum_factor for fft_buckets = %g', lattice_sum_factor)
    precision /= lattice_sum_factor

    vol = cell.vol

    supmol_bas_ij_idx = _bvk_pairs_to_supmol_pairs(
        ni, pair_idx, precision, xctype)

    pair_ke, grid_frac_ranges = _estimate_Ecut_and_grid_ranges(
        ni, supmol_bas_ij_idx, ke_max, precision, xctype)

    buckets = []

    ke_lower, ke_upper = 0, init_ke
    while ke_lower < ke_max:
        ke_upper = mesh_to_ke(a, mesh).min()
        if ke_upper >= ke_max:
            mesh = ke_to_mesh(a, ke_max)
            mesh = np.where(mesh_final < mesh, mesh_final, mesh)

        idx = cp.where((ke_lower < pair_ke) & (pair_ke <= ke_upper))[0]
        if len(idx) > 0:
            filtered_pairs = supmol_bas_ij_idx[idx]
            filtered_grid_ranges = grid_frac_ranges[:,idx]

            ish, jsh = divmod(filtered_pairs, NBAS_MAX)
            lij = ang_per_shell[ish] * 5 + ang_per_shell[jsh % bvkcell.nbas]
            idx = cp.argsort(lij)
            lij = lij[idx]
            split_points = (cp.where(lij[1:] != lij[:-1])[0] + 1).get()

            #TODO: to avoid too many bas_ij in each sub-bucket, Add more
            # split_points and divide idx into more segments.

            # Group bas_ij_idx and grid_frac_ranges by (li, lj) patterns
            idx_by_pattern = cp.split(idx, split_points)
            lilj_patterns = np.append(lij[0].get(), lij[split_points].get())
            lilj_patterns = [divmod(x, 5) for x in lilj_patterns.tolist()]

            bas_ij_cache = [filtered_pairs[idx] for idx in idx_by_pattern]
            grid_ranges_cache = [filtered_grid_ranges[:,idx] for idx in idx_by_pattern]

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
            weight = vol / np.prod(mesh)
            ao_val_threshold = precision*1e-2 / (12.56*40**2 * weight)
            buckets.append({
                'ke_cutoff': ke_upper,
                'mesh': np.asarray(mesh, dtype=np.int32),
                'lij_patterns': lilj_patterns,
                'bas_ij_cache': bas_ij_cache,
                'grid_ranges_cache': grid_ranges_cache,
                'grid_tile_cache': None,
                'negligible': ao_val_threshold
            })
            log.debug('Add fft bucket: ke=%g mesh=%s, shl_pairs=%d, ao_val_threshold=%g',
                      ke_upper, tuple(mesh), len(filtered_pairs), ao_val_threshold)

        mesh = (mesh * 1.2).astype(np.int32)
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

def _bvk_pairs_to_supmol_pairs(ni, bas_ij_idx, precision, xctype):
    # The bas_ij_idx stores the effective shells in bvkcell. Each of these
    # shells involve multiple primitive shells in supmol. Unpack the bvk-shells
    # and provide the primitive pair indices in supmol.
    nimgs = ni.mg_envs.nimgs
    npairs = len(bas_ij_idx)
    supmol_bas_ij_idx = cp.empty(npairs * nimgs, dtype=np.int64)
    is_mgga = 1 if xctype == 'MGGA' else 0
    counts = cp.empty(1, dtype=np.int32)
    err = libmgrid.supmol_non_trivial_pairs(
        ctypes.cast(supmol_bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.byref(ni.mg_envs),
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
        tiles_per_cell = cp.asarray((bucket['mesh']+3) / 4, dtype=np.float32)
        for (bas_ij, grid_range) in zip(
                bucket['bas_ij_cache'], bucket['grid_ranges_cache']):
            raw_tiles = grid_range[:,:,1] - grid_range[:,:,0]
            raw_tiles *= tiles_per_cell[:,None]
            raw_tiles = cp.ceil(raw_tiles)
            raw_tiles += 1 # penalty for rounding on the boundary
            n = (raw_tiles[0] * raw_tiles[1] * raw_tiles[2]).sum().get()
            buf_size = max(buf_size, int(n))

    # temporary space to store grid_tile_idx
    work = cp.empty(buf_size+1, dtype=np.int32)
    # temporary space to store dressed_bas_ij
    work1 = cp.empty(buf_size, dtype=np.int64)

    nimgs = cell.nimgs
    nbas = cell.nbas
    for bucket in fft_buckets:
        bucket['grid_tile_cache'] = grid_tile_cache = []
        mesh = bucket['mesh']
        for bas_ij_idx, grid_range in zip(
                bucket['bas_ij_cache'], bucket['grid_ranges_cache']):
            grid_tile_idx, dressed_bas_ij, shl_pair_offsets = _group_pairs_in_tile(
                bas_ij_idx, grid_range, nimgs, mesh, nbas, work, work1)
            grid_tile_cache.append((
                grid_tile_idx, dressed_bas_ij, shl_pair_offsets))

def _group_pairs_in_tile(bas_ij_idx, grid_range, nimgs, mesh, nbas, work, work1):
    npairs = len(bas_ij_idx)
    assert npairs > 0
    tile_counts = work[-1:]
    err = libmgrid.grid_range_to_tiles(
        ctypes.cast(work.data.ptr, ctypes.c_void_p),
        ctypes.cast(work1.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(grid_range.data.ptr, ctypes.c_void_p),
        (ctypes.c_int*3)(*nimgs),
        (ctypes.c_int*3)(*mesh),
        ctypes.c_int(npairs),
        ctypes.c_int(nbas),
        ctypes.cast(tile_counts.data.ptr, ctypes.c_void_p))
    if err != 0:
        raise RuntimeError('grid_range_to_tiles failed')
    n = int(tile_counts[0].get())
    assert n < 2**31, 'int32 indexing in shl_pair_offsets'

    sorted_idx = cp.argsort(work[:n])
    grid_tile_idx = work[sorted_idx]
    shl_pair_offsets = _segment_offsets(grid_tile_idx)

    # TODO: Further divide large entry in shell_pair_offsets for better
    # load balance.

    # Store only the unique grid tile ids.
    grid_tile_idx = grid_tile_idx[shl_pair_offsets[:-1]]

    dressed_bas_ij = work1[sorted_idx]
    return grid_tile_idx, dressed_bas_ij, shl_pair_offsets

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
    if isinstance(a, cp.ndarray):
        assert a.dtype == np.complex128
    else:
        aR, aI = a
        a = cp.empty(aR.shape, dtype=np.complex128)
        a.real = aR
        a.imag = aI
    assert out.dtype == np.complex128
    assert out.ndim == a.ndim
    mesh = tuple(mesh)
    assert a.shape[-3:] == mesh
    out_shape = out.shape
    counts = 1
    if out.ndim == 4:
        counts, out_shape = out_shape[0], out_shape[1:]
    assert all(x <= y for x, y in zip(mesh, out_shape)), \
            'folding frequency down unsupported'
    err = libmgrid.fft_takebak(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(a.data.ptr, ctypes.c_void_p),
        (ctypes.c_int*3)(*out_shape),
        (ctypes.c_int*3)(*mesh),
        ctypes.c_int(counts))
    if err != 0:
        raise RuntimeError('fft_takebak kernel failed')
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

def _apply_Gv_1j(rhoG, Gx, Gy, Gz, out=None):
    '''einsum('g,g->g', rhoG, Gv[:,n]*1j), n is 0, 1 or 2'''
    fn_name = 'apply_Gv_1j'
    if fn_name not in _kernel_registery:
        kernel_code = ('''\
#include <cuComplex.h>
extern "C" __global__
void ''' + fn_name + r'''(cuDoubleComplex* __restrict__ out, cuDoubleComplex *rhoG,
    double *Gx, double *Gy, double *Gz, long long nx, long long ny, long long nz) {
    size_t nyz = ny * nz;
    size_t ng = nx * nyz;
    size_t g = blockDim.x * (size_t)blockIdx.x + threadIdx.x;
    if (g >= ng) return;
    int ix = g / nyz;
    int iyz = g - nyz * ix;
    int iy = iyz / nz;
    int iz = iyz - nz * iy;
    cuDoubleComplex rho = rhoG[g];
    double Gv = Gx[ix] + Gy[iy] + Gz[iz];
    out[g] = make_cuDoubleComplex(-Gv * cuCimag(rho), Gv * cuCreal(rho));
}''')
        _kernel_registery[fn_name] = cp.RawKernel(kernel_code, fn_name)

    kernel = _kernel_registery[fn_name]
    out = ndarray(rhoG.shape, buffer=out, dtype=np.complex128)
    kernel(((rhoG.size + 1023) // 1024,), (1024,),
           (out, rhoG, Gx, Gy, Gz, len(Gx), len(Gy), len(Gz)))
    return out

def _contract_Gv_1j(out, xc, Gx, Gy, Gz):
    '''out += einsum('g,g->g', xc[n], Gv[:,n]*-1j), n is 0, 1 or 2'''
    fn_name = 'contract_Gv_1j'
    if fn_name not in _kernel_registery:
        kernel_code = ('''\
#include <cuComplex.h>
extern "C" __global__
void ''' + fn_name + r'''(cuDoubleComplex* __restrict__ out, cuDoubleComplex *vxcG,
    double *Gx, double *Gy, double *Gz, long long nx, long long ny, long long nz) {
    size_t nyz = ny * nz;
    size_t ng = nx * nyz;
    size_t g = blockDim.x * (size_t)blockIdx.x + threadIdx.x;
    if (g >= ng) return;
    int ix = g / nyz;
    int iyz = g - nyz * ix;
    int iy = iyz / nz;
    int iz = iyz - nz * iy;
    // (-i Gv) * v
    double Gv = Gx[ix] + Gy[iy] + Gz[iz];
    cuDoubleComplex res = out[g];
    cuDoubleComplex v = vxcG[g];
    res.x += Gv * cuCimag(v);
    res.y -= Gv * cuCreal(v);
    out[g] = res;
}''')
        _kernel_registery[fn_name] = cp.RawKernel(kernel_code, fn_name)

    kernel = _kernel_registery[fn_name]
    kernel(((out.size + 1023) // 1024,), (1024,),
           (out, xc, Gx, Gy, Gz, len(Gx), len(Gy), len(Gz)))
    return out

def _get_coulomb_on_g_mesh(rhoG, Gv_bases, out=None):
    '''rhoG * 4pi/G^2'''
    fn_name = 'get_coulG'
    if fn_name not in _kernel_registery:
        kernel_code = ('''\
extern "C" __global__
void ''' + fn_name + r'''(double2* __restrict__ out, double2* __restrict__ rhoG,
    double *Gx, double *Gy, double *Gz, long long nx, long long ny, long long nz) {
    size_t nyz = ny * nz;
    size_t ng = nx * nyz;
    size_t g = blockDim.x * (size_t)blockIdx.x + threadIdx.x;
    if (g >= ng) return;
    int ix = g / nyz;
    int iyz = g - nyz * ix;
    int iy = iyz / nz;
    int iz = iyz - nz * iy;
    double GG = 0;
    for (int n = 0; n < 3; ++n) {
        double Gv = Gx[n*nx+ix] + Gy[n*ny+iy] + Gz[n*nz+iz];
        GG += Gv * Gv;
    }
    double2 coul = {0., 0.};
    if (GG != 0) {
        double fac = 12.566370614359172 / GG;
        double2 rho = rhoG[g];
        coul.x = fac * rho.x;
        coul.y = fac * rho.y;
    }
    out[g] = coul;
}''')
        _kernel_registery[fn_name] = cp.RawKernel(kernel_code, fn_name)

    kernel = _kernel_registery[fn_name]
    nx, ny, nz = [x.shape[1] for x in Gv_bases]
    ng = nx * ny * nz
    assert rhoG.size == ng
    out = ndarray((nx, ny, nz), dtype=np.complex128, buffer=out)
    kernel(((ng + 1023) // 1024,), (1024,),
           (out, rhoG, Gv_bases[0], Gv_bases[1], Gv_bases[2], nx, ny, nz))
    return out

def _density_to_real_space(rhoG, Gv_bases, tauG=None, out=None):
    '''
    Perform
        stack(ifft(rhoG), cp.einsum('g,gx->xg', rhoG, 1j*Gv), ifft(tauG)).real
    with reduced memory footprint
    '''
    assert rhoG.ndim == 3
    mesh = rhoG.shape
    nvar = 4
    if tauG is not None:
        nvar = 5

    work = None
    out = ndarray((nvar, *mesh), dtype=np.float64, buffer=out)
    if tauG is not None:
        tauR = ifft_in_place(tauG.reshape(mesh))
        out[4] = tauR.real
        work = tauG

    Gx, Gy, Gz = Gv_bases
    for n in range(3):
        work = _apply_Gv_1j(rhoG, Gx[n], Gy[n], Gz[n], work)
        out[n+1] = ifft_in_place(work).real

    out[0] = ifft_in_place(rhoG).real
    return out

def _vxc_to_reciprocal_space(vxc, vxcG, Gv_bases=None, work=None):
    '''
    Perform
        fft(vxc[0]) + cp.einsum('xg,gx->g', fft(vxc[0]), -1j*Gv)
    with reduced memory footprint
    '''
    assert vxc.ndim == 4
    mesh = vxc.shape[1:]
    is_mgga = len(vxc) == 5

    work = ndarray(mesh, dtype=np.complex128, buffer=work)

    work.real = vxc[0].reshape(mesh)
    work.imag.fill(0.)
    fft_in_place(work)
    vxcG += work

    if Gv_bases is not None:
        Gx, Gy, Gz = Gv_bases
        for n in range(3):
            work.real = vxc[n+1].reshape(mesh)
            work.imag.fill(0.)
            _contract_Gv_1j(vxcG, fft_in_place(work), Gx[n], Gy[n], Gz[n])

    if is_mgga:
        work.real = vxc[4].reshape(mesh)
        work.imag.fill(0.)
        vxcG_tau = fft_in_place(work)
        return vxcG, vxcG_tau
    else:
        return vxcG

def _real_dm_in_bvkcell(ni, dm_kpts, kpts, hermi=1):
    cell = ni.cell
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    n_dm, nkpts, nao = dms.shape[:3]
    dms = dms.reshape(n_dm*nkpts, nao, nao)

    dms = cell.apply_C_mat_CT(dms)
    if hermi == 1:
        dms *= 2
    else:
        #Only the tril part is processed by the integral code
        dms = transpose_sum(dms)
    nao = dms.shape[-1]
    dms = dms.reshape(n_dm, nkpts, nao, nao)

    if nkpts == 1:
        dm_sc = dms
    else:
        #expLk = cp.exp(1j*cp.asarray(ni.bvkmesh_Ls).dot(cp.asarray(kpts).T))
        expLk = fft_matrix(ni.kmesh)
        dm_sc = contract('nkpq,Lk->nqLp', dms, expLk)
        assert absmax(dm_sc.imag) < cell.precision*5e2
        dm_sc = cp.asarray(dm_sc.real, order='C')
    return dm_sc

def get_rho(ni, dm_kpts, kpts=None):
    '''Density in real space

    Args:
        ni:
            MultiGridNumInt instance
        dm:
            density matrix at a single k-point or density matrices for k-sampling

    Kwargs:
        kpts: (N, 3) ndarray
            k points. If not specified, gamma point is assumed
    '''
    assert dm_kpts.ndim < 4
    cell = ni.cell
    mesh = ni.mesh

    if ni.bvkcell is None:
        kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        ni.build(kmesh, 'LDA')
    assert len(dm_kpts) == np.prod(ni.kmesh)

    dm_sc = _real_dm_in_bvkcell(ni, dm_kpts, kpts)
    n_dm, nkpts, nao = dm_sc.shape[:3]
    assert n_dm == 1
    dm_sc = dm_sc[0]

    rhoG = _eval_density(ni, dm_sc)[0]
    rhoR = ifft_in_place(rhoG.reshape(mesh)).real.ravel()
    weight = cell.vol / np.prod(mesh)
    rhoR *= 1./weight
    return rhoR

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
    assert dm_kpts.ndim < 4
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

    dm_sc = _real_dm_in_bvkcell(ni, dm_kpts, kpts, hermi)
    assert len(dm_sc) == 1
    dm_sc = dm_sc[0]

    cell = ni.cell
    mesh = ni.mesh
    ngrids = np.prod(mesh)
    Gv_bases = _get_G_bases(mesh, cell.reciprocal_vectors())

    rhoG, tauG = _eval_density(ni, dm_sc, with_tau=xctype=='MGGA')

    rhoG = rhoG.reshape(mesh)
    n_electrons = float(rhoG[0,0,0].real.get())

    if xctype == 'HF':
        assert with_j
        coulomb_on_g_mesh = _get_coulomb_on_g_mesh(rhoG, Gv_bases)
        xc_for_fock = coulomb_on_g_mesh
        ecoul = .5 * float(_conj_dot(rhoG.ravel(), coulomb_on_g_mesh.ravel()).get())
        log.debug('Multigrid Coulomb energy %s', ecoul)
        rhoG = coulomb_on_g_mesh = None
        xc_energy_sum = None

    else:
        weight = cell.vol / ngrids
        if xctype == 'LDA':
            density = ifft_in_place(rhoG).real
        elif xctype == 'GGA':
            density = _density_to_real_space(rhoG, Gv_bases)
        else: # MGGA
            density = _density_to_real_space(rhoG, Gv_bases, tauG)
        density = density.reshape(nvar,ngrids)
        density = cp.asarray(density, dtype=np.float64, order='C')
        # *(1./weight) because rhoR is scaled by weight in _eval_density. If
        # computing rhoR with IFFT, the weight factor is not needed.
        density *= 1/weight
        t0 = log.timer_debug1("density", *t0)

        rho_sf = ndarray(ngrids, dtype=np.float64, buffer=tauG)
        rho_sf[:] = density[0].real

        # eval_xc_eff supports float64 only
        xc_for_energy, xc_for_fock = ni.eval_xc_eff(
            xc_code, density, deriv=1, xctype=xctype, spin=0, inplace=True)[:2]

        xc_for_fock *= weight
        xc_for_fock = xc_for_fock.reshape(nvar, *mesh)

        xc_energy_sum = float(rho_sf.dot(xc_for_energy.ravel()).get()) * weight
        xc_for_energy = density = rho_sf = None
        log.debug("Multigrid exc %s  nelec %s", xc_energy_sum, n_electrons)
        t0 = log.timer_debug1("eval_xc_eff", *t0)

        if with_j:
            coulomb_on_g_mesh = _get_coulomb_on_g_mesh(rhoG, Gv_bases, out=tauG)
            ecoul = .5 * float(_conj_dot(rhoG.ravel(), coulomb_on_g_mesh.ravel()).get())
            log.debug('Multigrid Coulomb energy %s', ecoul)
        else:
            ecoul = None
            coulomb_on_g_mesh = cp.zeros_like(rhoG)
        tauG = None

        if xctype == "LDA":
            # Now xc_for_fock represents xc on G space
            xc_for_fock = _vxc_to_reciprocal_space(
                xc_for_fock, coulomb_on_g_mesh, work=rhoG)
        else: # GGA or MGGA
            xc_for_fock = _vxc_to_reciprocal_space(
                xc_for_fock, coulomb_on_g_mesh, Gv_bases, work=rhoG)
        coulomb_on_g_mesh = rhoG = None

    nkpts = np.prod(ni.kmesh)
    if kpts_band is not None:
        raise NotImplementedError
        ni1 = ni.copy().reset().build(kmesh=kpts_band)
        veff = _eval_xc_mat(ni1, xc_for_fock)
    else:
        veff = _eval_xc_mat(ni, xc_for_fock, out=dm_sc)

    if nkpts != 1:
        #expLk = cp.exp(1j*cp.asarray(ni.bvkmesh_Ls).dot(cp.asarray(kpts).T))
        expLk = fft_matrix(ni.kmesh)
        expLkz = expLk.view(np.float64).reshape(-1, nkpts, 2)
        veff = contract('pLq,Lkz->kpqz', veff, expLkz)
        veff = veff.view(np.complex128)[:,:,:,0]

    veff = cell.apply_CT_mat_C(veff)
    veff = transpose_sum(veff)

    veff = _format_jks(veff, dm_kpts, kpts_band, kpts)
    veff = tag_array(veff, ecoul=ecoul, exc=xc_energy_sum)
    t0 = log.timer_debug1("xc matrix", *t0)
    return n_electrons, xc_energy_sum, veff

def nr_uks(ni, cell, grids, xc_code, dm_kpts, relativity=0, hermi=1,
           kpts=None, kpts_band=None, with_j=False, verbose=None):
    '''Compute the XC energy and UKS XC matrix at sampled k-points.
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
    if xctype == 'LDA':
        nvar = 1
    elif xctype == 'GGA':
        nvar = 4
    elif xctype == 'MGGA':
        nvar = 5
    elif xctype == 'HF':
        vj = ni.get_j(dm_kpts[0]+dm_kpts[1], hermi, kpts, kpts_band)
        veff = cp.stack([vj, vj])
        return lib.tag_array(veff, ecoul=vj.ecoul, exc=0)
    else:
        raise NotImplementedError(f'XC functional {xc_code}')

    assert kpts_band is None
    kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
    if ni.bvkcell is None or any(ni.kmesh != kmesh):
        ni.build(kmesh, xctype)

    dm_sc = _real_dm_in_bvkcell(ni, dm_kpts, kpts, hermi)
    assert len(dm_sc) == 2

    cell = ni.cell
    mesh = ni.mesh
    ngrids = np.prod(mesh)
    weight = cell.vol / ngrids

    Gv_bases = _get_G_bases(mesh, cell.reciprocal_vectors())

    density = cp.empty((2, nvar, *mesh))
    rhoG_sf = None
    for s in range(2):
        rhoG, tauG = _eval_density(ni, dm_sc[s], with_tau=xctype=='MGGA')
        rhoG = rhoG.reshape(mesh)
        if rhoG_sf is None:
            rhoG_sf = rhoG
        else:
            rhoG_sf += rhoG

        if xctype == 'LDA':
            density[s,0] = ifft_in_place(rhoG.reshape(mesh)).real
        elif xctype == 'GGA':
            _density_to_real_space(rhoG, Gv_bases, out=density[s])
        else: # MGGA
            _density_to_real_space(rhoG, Gv_bases, tauG, out=density[s])

        tauG = None # release memory

    n_electrons = float(rhoG_sf[0,0,0].real.get())

    density = density.reshape(2,nvar,ngrids)
    density = cp.asarray(density, dtype=np.float64, order='C')
    # *(1./weight) because rhoR is scaled by weight in _eval_density. If
    # computing rhoR with IFFT, the weight factor is not needed.
    density *= 1./weight
    t0 = log.timer_debug1("density", *t0)

    rho_sf = ndarray(ngrids, dtype=np.float64, buffer=rhoG)
    rho_sf[:] = density[0,0].real
    rho_sf[:] += density[1,0].real

    # eval_xc_eff supports float64 only
    xc_for_energy, xc_for_fock = ni.eval_xc_eff(
        xc_code, density, deriv=1, xctype=xctype, spin=1, inplace=True)[:2]

    xc_for_fock *= weight
    xc_for_fock = xc_for_fock.reshape(2, nvar, *mesh)

    xc_energy_sum = float(rho_sf.dot(xc_for_energy.ravel()).get()) * weight
    xc_for_energy = density = rho_sf = None
    log.debug("Multigrid exc %s  nelec %s", xc_energy_sum, n_electrons)
    t0 = log.timer_debug1("eval_xc_eff", *t0)

    if with_j:
        coulomb_on_g_mesh = _get_coulomb_on_g_mesh(rhoG_sf, Gv_bases, out=rhoG)
        ecoul = .5 * float(_conj_dot(rhoG_sf.ravel(), coulomb_on_g_mesh.ravel()).get())
        log.debug('Multigrid Coulomb energy %s', ecoul)

        coulomb_a = coulomb_on_g_mesh
        coulomb_b = rhoG_sf # reuse memory
        coulomb_b[:] = coulomb_a
    else:
        ecoul = None
        coulomb_a = cp.zeros_like(rhoG)
        coulomb_b = cp.zeros_like(rhoG)
    rhoG = rhoG_sf = None

    if kpts_band is not None:
        raise NotImplementedError

    # dm_sc and the output have the shape shape. Reuse its memory.
    veff = dm_sc

    if xctype == "LDA":
        # maximum memory usage = (2,ngrids) float64s + 3*ngrids complex128s
        # The 3*ngrids complex128s consist of coulomb_a, coulomb_b and the
        # workspace required by _vxc_to_reciprocal_space.
        vxc = _vxc_to_reciprocal_space(xc_for_fock[0], coulomb_a)
        _eval_xc_mat(ni, vxc, out=veff[0])
        vxc = coulomb_a = None # release memory
        vxc = _vxc_to_reciprocal_space(xc_for_fock[1], coulomb_b)
        _eval_xc_mat(ni, vxc, out=veff[1])
        vxc = coulomb_b = xc_for_fock = None

    else: # GGA or MGGA
        # maximum memory usage = (2,nvar,ngrids) float64s + 3*ngrids complex128s
        # The 3*ngrids complex128s consist of coulomb_a, coulomb_b and the
        # workspace required by _vxc_to_reciprocal_space.
        vxc = _vxc_to_reciprocal_space(xc_for_fock[0], coulomb_a, Gv_bases)
        # It's safe to reuse the memory of xc_for_fock[0] as the workspace.
        # The size of xc_for_fock[0] is 4*ngrids or 5*ngrids (float64), more
        # than the workspace required by _eval_xc_mat (2*ngrids float64).
        _eval_xc_mat(ni, vxc, out=veff[0], work=xc_for_fock[0])
        vxc = coulomb_a = None # release memory
        vxc = _vxc_to_reciprocal_space(xc_for_fock[1], coulomb_b, Gv_bases)
        _eval_xc_mat(ni, vxc, out=veff[1], work=xc_for_fock[0])
        vxc = coulomb_b = xc_for_fock = None

    nkpts = np.prod(ni.kmesh)
    if nkpts != 1:
        #expLk = cp.exp(1j*cp.asarray(ni.bvkmesh_Ls).dot(cp.asarray(kpts).T))
        expLk = fft_matrix(ni.kmesh)
        expLkz = expLk.view(np.float64).reshape(-1, nkpts, 2)
        veff = contract('spLq,Lkz->skpqz', veff, expLkz)
        veff = veff.view(np.complex128)[:,:,:,:,0]

    nao = veff.shape[-1]
    veff = cell.apply_CT_mat_C(veff.reshape(2*nkpts,nao,nao))
    veff = transpose_sum(veff)
    nao = veff.shape[-1]
    veff = veff.reshape(2,nkpts,nao,nao)

    veff = _format_jks(veff, dm_kpts, kpts_band, kpts)
    veff = tag_array(veff, ecoul=ecoul, exc=xc_energy_sum)
    t0 = log.timer_debug1("xc matrix", *t0)
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
            # PTR_BAS_COORD was not initialized in the super_cell function
            bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
            bvkmesh_Ls = translation_vectors_for_kmesh(cell, kmesh, wrap_around=True)
        self.bvkcell = bvkcell
        self.bvkmesh_Ls = bvkmesh_Ls
        bvk_ncells = len(bvkmesh_Ls)

        Ls = cp.asarray(bvkcell.get_lattice_Ls())
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
        nimgs = len(Ls)
        log.debug1('ft_ao bvk_ncells=%d, nimgs=%d', bvk_ncells, nimgs)
        _env = _scale_sp_ctr_coeff(bvkcell)
        ao_loc = bvkcell.ao_loc
        self.mg_envs = PBCIntEnvVars.new(
            cell.natm, cell.nbas, bvk_ncells, nimgs,
            bvkcell._atm, bvkcell._bas, _env, ao_loc, Ls)

        a = cell.lattice_vectors()
        b = cell.reciprocal_vectors(norm_to=1)
        libmgrid.update_lattice_vectors(a.ctypes, b.ctypes)

        precision = cell.precision
        bas_ij_idx = _non_trivial_bvk_pairs(self, precision)

        # Initialize buckets
        is_orth_lattice = abs(a - np.diag(a.diagonal())).max() < 1e-5
        self.aft_buckets = None
        self.fft_buckets = None

        # FIXME: ni.mesh and ni.ke_cutoff are coupled, might need only one of them
        mesh = self.mesh
        self.ke_cutoff = max(0.1, mesh_to_ke(a, mesh).min())

        if self.enable_aft and is_orth_lattice:
            # Estimate Ecut for AFT integrals. These can be potentially handled by
            # aft_eval_* functions.
            # Use self.ke_cutoff to limit the highest Ecut. This ensures to handle
            # shell-pairs even if their Ecuts are higher than ke_cutoff.
            aft_Ecut = _aft_Ecut_estimation(
                self, bas_ij_idx, self.ke_cutoff, precision, xctype)

            aft_init_ke = mesh_to_ke(a, [16]*3)
            aft_final_ke = aft_init_ke * 25
            log.debug1('aft initial/final ke_cutoff = %g, %g', aft_init_ke,
                       aft_final_ke)
            self.aft_buckets = _partition_ke_for_aft(
                self, bas_ij_idx, aft_Ecut, aft_init_ke, aft_final_ke,
                precision, xctype, log)

            # Filter shell pairs that are not handled by AFT. The remaining pairs
            # are handled by FFT.
            if self.aft_buckets:
                aft_ke_max = self.aft_buckets[-1]['ke_cutoff']
                if aft_ke_max < self.ke_cutoff:
                    bas_ij_idx = bas_ij_idx[aft_Ecut > aft_ke_max]
                else:
                    bas_ij_idx = None

            fft_init_ke = aft_final_ke * 1.5
        else:
            fft_init_ke = mesh_to_ke(a, [16]*3)
        log.debug1('fft initial ke_cutoff = %g', fft_init_ke)

        if bas_ij_idx is not None and len(bas_ij_idx) > 0:
            # bas_ij_idx are the effective paris between cell0 and bvkcell.
            # The FFT-MultiGrid code operates on cell0-supmol paris.
            # Every bvkcell shell in bas_ij_idx needs to be unpacked to several
            # primitive shells in supmol.
            self.fft_buckets = _partition_ke_for_fft(
                self, bas_ij_idx, fft_init_ke, precision, xctype, log)

            nimgs = cell.nimgs
            Tx = np.arange(-nimgs[0], nimgs[0]+1, dtype=np.float64)
            Ty = np.arange(-nimgs[1], nimgs[1]+1, dtype=np.float64)
            Tz = np.arange(-nimgs[2], nimgs[2]+1, dtype=np.float64)
            self.supmol_img_coords = cp.asarray(lib.cartesian_prod([Tx, Ty, Tz]).dot(a))

            # TODO: skip grid_tile_cache when memory is insufficient
            cache_tile_idx = True
            if cache_tile_idx:
                _cache_grid_range_to_tiles(self.fft_buckets, cell)

        # mesh of last bucket is likely smaller than the cell.mesh estimation
        #if self.fft_buckets:
        #    self.mesh = self.fft_buckets[-1]['mesh']
        #else:
        #    self.mesh = self.aft_buckets[-1]['mesh']
        #log.info('set MultiGrid maximum mesh %s', self.mesh)
        t0 = log.timer_debug1('Initialize buckets', *t0)
        return self

#    get_nuc = get_nuc
#    get_pp = get_pp

    get_j = get_j_kpts

    get_rho = get_rho
    nr_rks = nr_rks
    nr_uks = nr_uks

    def get_vxc(self, cell, grids, xc_code, dm_kpts, spin=0, hermi=1,
                kpts=None, kpts_band=None, with_j=False, verbose=None):
        fn = self.nr_rks if spin == 0 else self.nr_uks
        return fn(cell, grids, xc_code, dm_kpts, spin, hermi=hermi,
                  kpts=kpts, kpts_band=kpts_band, with_j=with_j, verbose=verbose)
    nr_vxc = get_vxc

    eval_xc_eff = numint.NumInt.eval_xc_eff
    _init_xcfuns = numint.NumInt._init_xcfuns

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
