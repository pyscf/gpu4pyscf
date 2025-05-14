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
Build GDF tensor using the range-separation integral algorithm.
'''

import os
import math
import ctypes
import warnings
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
from pyscf import lib
from pyscf.gto import ANG_OF, NPRIM_OF, NCTR_OF
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.df.rsdf_builder import (
    RCUT_THRESHOLD, estimate_ke_cutoff_for_omega)
from pyscf.pbc.df import aft as aft_cpu
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.lib.kpts_helper import kk_adapted_iter
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.gto.mole import cart2sph_by_l, extract_pgto_params
from gpu4pyscf.scf.jk import _scale_sp_ctr_coeff
from gpu4pyscf.pbc.df.int3c2e import (sr_aux_e2, estimate_rcut, libpbc,
                                      SRInt3c2eOpt, Int3c2eEnvVars)

OMEGA_MIN = 0.3

# In the ED of the j2c2e metric, the default LINEAR_DEP_THR setting in pyscf-2.8
# is too loose. The linear dependency truncation often leads to serious errors.
# PBC GDF very differs to the molecular GDF approximation where diffused
# functions typically have insignificant contributions. The diffused auxliary
# crystial orbitals have large impacts on the accuracy of Coulomb integrals. A
# tight linear dependency threshold have to be applied to control the error,
# even this may cause more numericial stability issues.
LINEAR_DEP_THR = 1e-11
# Use eigenvalue decomposition in decompose_j2c
PREFER_ED = False

THREADS = 256

def build_cderi(cell, auxcell, kpts=None, j_only=False,
                omega=None, linear_dep_threshold=LINEAR_DEP_THR):
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension >= 2
    with_long_range = cell.omega == 0
    if with_long_range:
        if omega is None:
            cell_exps, cs = extract_pgto_params(cell, 'diffused')
            omega = cell_exps.min()**.5
            logger.debug(cell, 'omega guess in rsdf_builder = %g', omega)
        omega = abs(omega)
    else:
        assert cell.omega < 0
        # Not supporting a custom omega for SR CDERI
        assert omega is None or omega == abs(cell.omega)
        omega = abs(cell.omega)

    if kpts is None or is_zero(kpts):
        return build_cderi_gamma_point(
            cell, auxcell, omega, with_long_range, linear_dep_threshold)
    elif j_only:
        return build_cderi_j_only(
            cell, auxcell, kpts, omega, with_long_range, linear_dep_threshold)
    else:
        return build_cderi_kk(
            cell, auxcell, kpts, omega, with_long_range, linear_dep_threshold)

def build_cderi_kk(cell, auxcell, kpts, omega=OMEGA_MIN, with_long_range=True,
                   linear_dep_threshold=LINEAR_DEP_THR):
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    if kpts is None:
        kpts = np.zeros((1, 3))
        bvk_kmesh = kmesh = np.ones(3, dtype=int)
    else:
        # The remote images may contribute to certain k-point mesh, contributing
        # to the finite-size effects in HFX. For sufficiently large number of
        # kpts, the truncation radious cell.rcut may cause finite-size errors.
        kpts = kpts.reshape(-1, 3)
        rcut = estimate_rcut(cell, auxcell, omega).max()
        bvk_kmesh = kmesh = kpts_to_kmesh(cell, kpts, rcut=rcut)
        if len(kpts) != np.prod(kmesh):
            # When targeting many kpts, num-kpts can be more than num-bvk-images.
            # Using a large radius to regenerate MP kmesh. The new MP kmesh
            # should cover all kpts.
            kmesh = kpts_to_kmesh(cell, kpts, rcut=rcut*20)
    j3c = sr_aux_e2(cell, auxcell, -omega, kpts, bvk_kmesh)
    t1 = log.timer('pass1: int3c2e', *t0)

    kpt_iters = list(kk_adapted_iter(kmesh))
    uniq_kpts = kpts[[x[0] for x in kpt_iters]]
    log.debug('Generate auxcell 2c2e integrals')
    j2c = _get_2c2e(auxcell, uniq_kpts, omega, with_long_range) # on CPU
    t1 = log.timer('int2c2e', *t1)

    if with_long_range:
        ft_ao_iter = _ft_ao_iter_generator(cell, auxcell, bvk_kmesh, omega, log)

    prefer_ed = PREFER_ED
    if cell.dimension == 2:
        prefer_ed = True
    cderi = {}
    cderip = {}
    for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        log.debug1('make_cderi for k-point %d %s', kp, kpts[kp])
        log.debug1('ki_idx = %s', ki_idx)
        log.debug1('kj_idx = %s', kj_idx)

        if with_long_range:
            '''exp(-i*(G + k) dot r) * Coulomb_kernel'''
            for pqG, auxG_conj in ft_ao_iter(kpts[kp], kpts[kj_idx]):
                # \sum_G coulG * ints(ij * exp(-i G * r)) * ints(P * exp(i G * r))
                # = \sum_G FT(ij, G) conj(FT(aux, G)) , where aux
                # functions |P> are assumed to be real
                j3c[ki_idx,kj_idx] += contract('kpqG,Gr->kpqr', pqG, auxG_conj)

        j2c_k = j2c[j2c_idx]
        if kp == kp_conj: # self conjugated
            # DF metric for self-conjugated k-point should be real
            j2c_k = j2c_k.real
        cd_j2c, cd_j2c_negative, j2ctag = decompose_j2c(
            j2c_k, prefer_ed, linear_dep_threshold)
        if cd_j2c.dtype != j3c.dtype:
            cd_j2c = cd_j2c.astype(j3c.dtype)

        for ki, kj in zip(ki_idx, kj_idx):
            j3c_k = j3c[ki,kj]
            cderi[ki,kj] = _solve_cderi(cd_j2c, j3c_k, j2ctag)
            if cd_j2c_negative is not None:
                assert cell.dimension == 2
                cderip[ki,kj] = _solve_cderi(cd_j2c_negative, j3c_k, j2ctag)
    t1 = log.timer('pass2: solve cderi', *t1)
    return cderi, cderip

def build_cderi_gamma_point(cell, auxcell, omega=OMEGA_MIN, with_long_range=True,
                            linear_dep_threshold=LINEAR_DEP_THR):
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    kmesh = None
    kpts = None

    j3c = sr_aux_e2(cell, auxcell, -omega)
    t1 = log.timer('pass1: int3c2e', *t0)

    log.debug('Generate auxcell 2c2e integrals')
    j2c = _get_2c2e(auxcell, kpts, omega, with_long_range) # on CPU
    j2c = j2c[0].real
    t1 = log.timer('int2c2e', *t1)

    cderi = {}
    cderip = {}
    if with_long_range:
        ft_ao_iter = _ft_ao_iter_generator(cell, auxcell, kmesh, omega, log)
        for pqG, auxG_conj in ft_ao_iter():
            # \sum_G coulG * ints(ij * exp(-i G * r)) * ints(P * exp(i G * r))
            # = \sum_G FT(ij, G) conj(FT(aux, G)) , where aux
            # functions |P> are assumed to be real
            j3c += contract('pqG,Gr->pqr', pqG[0], auxG_conj).real

    prefer_ed = PREFER_ED
    if cell.dimension == 2:
        prefer_ed = True
    cd_j2c, cd_j2c_negative, j2ctag = decompose_j2c(
        j2c, prefer_ed, linear_dep_threshold)

    cderi[0,0] = _solve_cderi(cd_j2c, j3c, j2ctag)
    if cd_j2c_negative is not None:
        assert cell.dimension == 2
        cderip[0,0] = _solve_cderi(cd_j2c_negative, j3c, j2ctag)
    t1 = log.timer('pass2: solve cderi', *t1)
    return cderi, cderip

def build_cderi_j_only(cell, auxcell, kpts, omega=OMEGA_MIN, with_long_range=True,
                       linear_dep_threshold=LINEAR_DEP_THR):
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    if kpts is None:
        kpts = np.zeros((1, 3))
        bvk_kmesh = np.ones(3, dtype=int)
    else:
        # Coulomb integrals requires smaller kmesh to converge finite-size effects.
        # A relatively small bvk_kmesh can be used for Coulomb integrals.
        kpts = kpts.reshape(-1, 3)
        bvk_kmesh = kpts_to_kmesh(cell, kpts)
    # TODO: time-reversal symmetry in j3c, j2c
    j3c = sr_aux_e2(cell, auxcell, -omega, kpts, bvk_kmesh, j_only=True)
    t1 = log.timer('pass1: int3c2e', *t0)

    log.debug('Generate auxcell 2c2e integrals')
    j2c = _get_2c2e(auxcell, None, omega, with_long_range) # on CPU
    j2c = j2c[0].real
    t1 = log.timer('int2c2e', *t1)

    # TODO: consider time-reversal symmetry
    cderi = {}
    cderip = {}
    if with_long_range:
        ft_ao_iter = _ft_ao_iter_generator(cell, auxcell, bvk_kmesh, omega, log)
        kpt = np.zeros(3)
        for pqG, auxG_conj in ft_ao_iter(kpt, kpts):
            # \sum_G coulG * ints(ij * exp(-i G * r)) * ints(P * exp(i G * r))
            # = \sum_G FT(ij, G) conj(FT(aux, G)) , where aux
            # functions |P> are assumed to be real
            j3c += contract('kpqG,Gr->kpqr', pqG, auxG_conj)

    prefer_ed = PREFER_ED
    if cell.dimension == 2:
        prefer_ed = True
    cd_j2c, cd_j2c_negative, j2ctag = decompose_j2c(
        j2c, prefer_ed, linear_dep_threshold)
    if cd_j2c.dtype != j3c.dtype:
        cd_j2c = cd_j2c.astype(j3c.dtype)

    nkpts = len(kpts)
    for k in range(nkpts):
        cderi[k, k] = _solve_cderi(cd_j2c, j3c[k], j2ctag)
        if cd_j2c_negative is not None:
            assert cell.dimension == 2
            cderip[k, k] = _solve_cderi(cd_j2c_negative, j3c[k], j2ctag)
    t1 = log.timer('pass2: solve cderi', *t1)
    return cderi, cderip

def _weighted_coulG_LR(cell, Gv, omega, kws, kpt=np.zeros(3)):
    coulG = pbctools.get_coulG(cell, kpt, exx=False, Gv=Gv, omega=abs(omega))
    coulG *= kws
    if is_zero(kpt):
        assert Gv[0].dot(Gv[0]) == 0
        coulG[0] -= np.pi / omega**2 / cell.vol
    return cp.asarray(coulG)

def _ft_ao_iter_generator(cell, auxcell, bvk_kmesh, omega, verbose=None):
    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    mesh = cell.symmetrize_mesh(mesh)
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    ngrids = len(Gv)
    nao = cell.nao

    ft_opt = ft_ao.FTOpt(cell, bvk_kmesh=bvk_kmesh)
    ft_kern = ft_opt.gen_ft_kernel(verbose=verbose)
    if bvk_kmesh is None:
        bvk_ncells = 1
    else:
        bvk_ncells = np.prod(bvk_kmesh)
    avail_mem = get_avail_mem() * .8
    Gblksize = max(16, int(avail_mem/(2*16*nao**2*bvk_ncells))//8*8)
    Gblksize = min(Gblksize, ngrids, 16384)
    #logger.debug1(cell, 'Gblksize = %d', Gblksize)
    def ft_ao_iter(kpt=np.zeros(3), kpts=None):
        coulG = _weighted_coulG_LR(auxcell, Gv, omega, kws, kpt)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            auxG_conj = cp.asarray(ft_ao.ft_ao(auxcell, Gv[p0:p1], kpt=kpt).conj(), order='C')
            auxG_conj *= cp.asarray(coulG[p0:p1,None])
            pqG = ft_kern(Gv[p0:p1], kpt, kpts).transpose(0,2,3,1)
            yield pqG, auxG_conj
    return ft_ao_iter

def decompose_j2c(j2c, prefer_ed=PREFER_ED, linear_dep_threshold=LINEAR_DEP_THR):
    if prefer_ed:
        return eigenvalue_decomposed_metric(j2c, linear_dep_threshold)
    else:
        return cholesky_decomposed_metric(j2c)

def cholesky_decomposed_metric(j2c):
    '''Return L for j2c = L L^T'''
    j2c_negative = None
    j2ctag = 'CD'
    # Cupy cholesky does not check positive-definite, seems returning nan in the
    # resultant CD matrix silently.
    j2c = cp.asarray(j2c)
    j2c = cp.linalg.cholesky(j2c)
    if cp.isnan(j2c[-1,-1]):
        raise RuntimeError('j2c is not positive definite')
    return j2c, j2c_negative, j2ctag

def eigenvalue_decomposed_metric(j2c, linear_dep_threshold=LINEAR_DEP_THR):
    j2c = cp.asarray(j2c)
    w, v = cp.linalg.eigh(j2c)
    mask = w > linear_dep_threshold
    v1 = v[:,mask].conj().T
    v1 *= w[mask, None]**-.5
    j2c = v1
    idx = cp.where(w < -linear_dep_threshold)[0]
    j2c_negative = None
    if len(idx) > 0:
        j2c_negative = (v[:,idx] * (-w[idx])**-.5).conj().T
    j2ctag = 'ED'
    return j2c, j2c_negative, j2ctag

# Create 2c2e, store on CPU
def _get_2c2e(auxcell, uniq_kpts, omega, with_long_range=True):
    # j2c ~ (-kpt_ji | kpt_ji) => hermi=1
    precision = auxcell.precision ** 1.5
    aux_exps, aux_cs = extract_pgto_params(auxcell, 'diffused')
    aux_exp = aux_exps.min()
    theta = 1./(2./aux_exp + omega**-2)
    rad = auxcell.vol**(-1./3) * auxcell.rcut + 1
    surface = 4*np.pi * rad**2
    lattice_sum_factor = 2*np.pi*auxcell.rcut/(auxcell.vol*theta) + surface
    rcut_sr = (np.log(lattice_sum_factor / precision + 1.) / theta)**.5
    logger.debug1(auxcell, 'auxcell  rcut_sr = %g', rcut_sr)
    auxcell_sr = auxcell.copy()
    auxcell_sr.rcut = rcut_sr
    with auxcell_sr.with_short_range_coulomb(omega):
        j2c = auxcell_sr.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)

    if not with_long_range:
        return j2c

    ke = estimate_ke_cutoff_for_omega(auxcell, omega, precision)
    mesh = auxcell.cutoff_to_mesh(ke)
    mesh = auxcell.symmetrize_mesh(mesh)
    logger.debug(auxcell, 'Set 2c2e integrals precision %g, mesh %s', precision, mesh)

    Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh)
    b = auxcell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = Gv.shape[0]
    naux = auxcell.nao
    max_memory = max(1000, auxcell.max_memory - lib.current_memory()[0])
    blksize = min(ngrids, int(max_memory*.4e6/16/naux), 200000)
    logger.debug2(auxcell, 'max_memory %s (MB)  blocksize %s', max_memory, blksize)

    if uniq_kpts is None:
        j2c = cp.asarray(j2c)
        coulG_LR = _weighted_coulG_LR(auxcell, Gv, omega, kws)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase).T
            j2c += (auxG.conj() * coulG_LR[p0:p1]).dot(auxG.T).real
            auxG = None
        j2c = [j2c.real.get()]
    else:
        for k, kpt in enumerate(uniq_kpts):
            j2c_k = cp.asarray(j2c[k])
            coulG_LR = _weighted_coulG_LR(auxcell, Gv, omega, kws, kpt)
            gamma_point = is_zero(kpt)

            for p0, p1 in lib.prange(0, ngrids, blksize):
                auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
                if gamma_point:
                    j2c_k += (auxG.conj() * coulG_LR[p0:p1]).dot(auxG.T).real
                else:
                    j2c_k += (auxG.conj() * coulG_LR[p0:p1]).dot(auxG.T)
                auxG = None
            j2c[k] = j2c_k.get()
    return j2c

def _solve_cderi(cd_j2c, j3c, j2ctag):
    if j2ctag == 'ED':
        return contract('Lr,pqr->Lpq', cd_j2c, j3c)
    else:
        nao, naux = j3c.shape[1:3]
        j3c = solve_triangular(cd_j2c, j3c.reshape(-1,naux).T, lower=True)
        return j3c.reshape(naux,nao,nao)

# Generate overlap masks using the int3c2e.overlap_img_counts function.
# This overlap mask will be used in the ft_aopair generation to generate the
# non-zero elements indices. This mask ensures that non-zero pairs in the
# int3c2e integrals are not overlooked by the ft_aopair kernel.
def _int3c2e_overlap_mask(int3c2e_opt, cutoff):
    cell = int3c2e_opt.cell
    pcell = int3c2e_opt.prim_cell
    p_nbas = pcell.nbas
    p2c_mapping = cp.asarray(int3c2e_opt.prim_to_ctr_mapping, dtype=np.int32)
    ovlp_img_counts = cp.zeros((p_nbas,p_nbas), dtype=np.int32)
    ls = pcell._bas[:,ANG_OF]
    exps, cs = extract_pgto_params(pcell, 'diffused')
    exps = cp.asarray(exps, dtype=np.float32)
    log_coeff = cp.log(abs(cp.asarray(cs, dtype=np.float32)))
    log_cutoff = math.log(cutoff)

    Ls = cp.asarray(pcell.get_lattice_Ls())
    Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
    nimgs = len(Ls)

    _atm = cp.array(pcell._atm)
    _bas = cp.array(pcell._bas)
    _env = cp.array(_scale_sp_ctr_coeff(pcell))
    int3c2e_envs = Int3c2eEnvVars(
        pcell.natm, p_nbas, 1, nimgs, _atm.data.ptr, _bas.data.ptr,
        _env.data.ptr, 0, Ls.data.ptr,
    )
    err = libpbc.bvk_overlap_img_counts(
        ctypes.cast(ovlp_img_counts.data.ptr, ctypes.c_void_p),
        ctypes.cast(p2c_mapping.data.ptr, ctypes.c_void_p),
        (ctypes.c_int*4)(0, p_nbas, 0, p_nbas),
        ctypes.byref(int3c2e_envs),
        ctypes.cast(exps.data.ptr, ctypes.c_void_p),
        ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
        ctypes.c_float(log_cutoff))
    if err != 0:
        raise RuntimeError('bvk_overlap_img_counts failed')
    p_ovlp_mask = np.asarray((ovlp_img_counts > 0).get(), dtype=np.int8)
    p2c_mapping = np.asarray(p2c_mapping.get(), dtype=np.int32)
    # Condense to contracted shells
    c_nbas = np.sum(int3c2e_opt.cell0_ctr_l_counts)
    c_ovlp_mask = np.zeros((c_nbas, c_nbas), dtype=np.int8)
    libpbc.condense_primitive_ovlp_mask(
        c_ovlp_mask.ctypes, p_ovlp_mask.ctypes, p2c_mapping.ctypes,
        ctypes.c_int(c_nbas), ctypes.c_int(p_nbas))

    lmax = cell._bas[:,ANG_OF].max()
    # generally contracted shells are convert to segement contracted shells in
    # either cases
    ls = np.repeat(cell._bas[:,ANG_OF], cell._bas[:,NCTR_OF])
    nprims = np.repeat(cell._bas[:,NPRIM_OF], cell._bas[:,NCTR_OF])

    #** Sort ovlp mask, to adapt the order in ft_aopair
    # sorted_idx indicates how the contracted shells of the original cell are
    # ordered in ft_aopair. See also the mole.group_basis function.
    l_ctrs = np.column_stack((ls, -nprims))
    _, inv_idx = np.unique(l_ctrs, return_inverse=True, axis=0)
    ft_sorting_idx = np.argsort(inv_idx.ravel(), kind='stable')

    idx = np.arange(len(ls))
    int3c_sorting_idx = np.hstack([idx[ls==l] for l in range(lmax+1)])
    rev_int3c_idx = np.empty_like(int3c_sorting_idx)
    # sorted_cell._bas[rev_int3c_idx] => cell._bas
    rev_int3c_idx[int3c_sorting_idx] = idx

    # int3c2e._sorted_cell[mapping] => ft._sorted_cell
    mapping = rev_int3c_idx[ft_sorting_idx]

    ovlp_mask = c_ovlp_mask[mapping[:,None],mapping]
    return ovlp_mask, mapping

def _build_aft_envs(ft_opt):
    sorted_cell = ft_opt.sorted_cell
    Ls = cp.asarray(sorted_cell.get_lattice_Ls())
    Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
    nimgs = len(Ls)
    nbas = sorted_cell.nbas

    _atm = cp.array(sorted_cell._atm)
    _bas = cp.array(sorted_cell._bas)
    _env = cp.array(_scale_sp_ctr_coeff(sorted_cell))
    ao_loc = cp.array(sorted_cell.ao_loc)
    aft_envs = ft_ao.AFTIntEnvVars(
        sorted_cell.natm, nbas, 1, nimgs, _atm.data.ptr,
        _bas.data.ptr, _env.data.ptr, ao_loc.data.ptr, Ls.data.ptr
    )
    # Keep a reference to these arrays, prevent releasing them upon returning the closure
    aft_envs._env_ref_holder = (_atm, _bas, _env, ao_loc, Ls)
    return aft_envs

def _make_img_idx_cache(ft_opt, aft_envs, cutoff, int3c2e_ovlp_mask, verbose):
    log = logger.new_logger(ft_opt.cell, verbose)
    sorted_cell = ft_opt.sorted_cell
    nbas = sorted_cell.nbas

    uniq_l_ctr = ft_opt.uniq_l_ctr
    l_ctr_offsets = ft_opt.l_ctr_offsets
    uniq_l = uniq_l_ctr[:,0]
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    n_groups = np.count_nonzero(uniq_l <= ft_ao.LMAX)

    exps, cs = extract_pgto_params(sorted_cell, 'diffused')
    exps = cp.asarray(exps, dtype=np.float32)
    log_coeff = cp.log(abs(cp.asarray(cs, dtype=np.float32)))
    log_cutoff = math.log(cutoff)

    permutation_symmetry = 1
    ij_tasks = [(i, j) for i in range(n_groups) for j in range(i+1)]

    bas_ij_cache = {}
    for i, j in ij_tasks:
        ll_pattern = f'{l_symb[i]}{l_symb[j]}'
        ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
        jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
        nish = ish1 - ish0
        njsh = jsh1 - jsh0
        img_counts = cp.zeros((nish,njsh), dtype=np.int32)
        err = libpbc.overlap_img_counts(
            ctypes.cast(img_counts.data.ptr, ctypes.c_void_p),
            (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
            ctypes.byref(aft_envs),
            ctypes.cast(exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff), ctypes.c_int(permutation_symmetry))
        if err != 0:
            raise RuntimeError(f'{ll_pattern} overlap_img_counts failed')
        mask = img_counts > 0
        mask |= int3c2e_ovlp_mask[ish0:ish1,jsh0:jsh1]
        bas_ij = cp.asarray(cp.where(mask.ravel())[0], dtype=np.int32)
        n_pairs = len(bas_ij)
        if n_pairs == 0:
            continue

        # Sort according to the number of images. In the CUDA kernel,
        # shell-pairs that have closed number of images are processed on
        # the same SM processor, ensuring the best parallel execution.
        img_counts = img_counts.ravel()
        counts_sorting = (-img_counts[bas_ij]).argsort()
        bas_ij = bas_ij[counts_sorting]
        img_counts = img_counts[bas_ij]
        img_offsets = cp.empty(n_pairs+1, dtype=np.int32)
        img_offsets[0] = 0
        cp.cumsum(img_counts, out=img_offsets[1:])
        tot_imgs = int(img_offsets[n_pairs])
        img_idx = cp.empty(tot_imgs, dtype=np.int32)
        err = libpbc.overlap_img_idx(
            ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_pairs),
            (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
            ctypes.byref(aft_envs),
            ctypes.cast(exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        if err != 0:
            raise RuntimeError(f'{ll_pattern} overlap_img_idx failed')
        img_counts = counts_sorting = None

        # bas_ij stores the non-negligible primitive-pair indices.
        ish, jsh = cp.unravel_index(bas_ij, (nish, njsh))
        ish += ish0
        jsh += jsh0
        bas_ij = cp.ravel_multi_index((ish, jsh), (nbas, nbas))
        bas_ij = cp.asarray(bas_ij, dtype=np.int32)
        bas_ij_cache[i, j] = (bas_ij, img_offsets, img_idx)
        log.debug1('task (%d, %d), n_pairs=%d', i, j, n_pairs)
    return bas_ij_cache

# The long-range part of the cderi for gamma point. The resultant 3-index tensor
# is compressed.
def _lr_int3c2e_gamma_point(int3c2e_opt):
    cell = int3c2e_opt.cell
    log = logger.new_logger(cell)
    t1 = log.init_timer()
    auxcell = int3c2e_opt.auxcell
    omega = abs(int3c2e_opt.omega)

    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    mesh = cell.symmetrize_mesh(mesh)
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    ngrids = len(Gv)
    nao = cell.nao
    naux = auxcell.nao

    # The cutoff from the int3c2e is utilized. This is generally smaller than
    # that created by the ft_aopair module. This is to ensure ft_aopair
    # producing more non-zero integrals than that in the int3c2e function.
    cutoff = int3c2e_opt.estimate_cutoff_with_penalty()
    int3c2e_ovlp_mask, mapping = _int3c2e_overlap_mask(int3c2e_opt, cutoff)
    int3c2e_ovlp_mask = cp.asarray(int3c2e_ovlp_mask, dtype=bool)

    # ft._sorted_cell._bas[rev_mapping] => int3c._sorted_cell._bas
    rev_mapping = np.empty_like(mapping)
    rev_mapping[mapping] = np.arange(len(mapping))

    ft_opt = ft_ao.FTOpt(cell)
    sorted_cell = ft_opt.sorted_cell
    nbas = sorted_cell.nbas

    # Save the indices of non-zero FT integrals in the aopair_offsets_lookup.
    # This lookup table will be used to generate the addresses for the
    # non-zere sr_int3c2e integrals.
    # aopair_offsets_lookup[ish,jsh] -> address in ft_aopair
    aopair_offsets_lookup = cp.zeros((nbas, nbas), dtype=np.int32)

    ao_pair_mapping = []
    # Given shell I in sorted_cell, this ao_loc maps shell I to the AO offset in
    # the original cell
    ao_loc = cp.asarray(ft_opt.ao_idx[ft_opt.sorted_cell.ao_loc[:-1]])

    aft_envs = _build_aft_envs(ft_opt)
    bas_ij_cache = _make_img_idx_cache(ft_opt, aft_envs, cutoff,
                                       int3c2e_ovlp_mask, log)
    t1 = log.timer_debug2('generating bas_ij indices', *t1)

    uniq_l_ctr = ft_opt.uniq_l_ctr
    l_ctr_offsets = ft_opt.l_ctr_offsets
    uniq_l = uniq_l_ctr[:,0]
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]

    ij_tasks = bas_ij_cache.keys()
    nf = nf_cart = (uniq_l + 1) * (uniq_l + 2) // 2
    if not cell.cart:
        nf = uniq_l * 2 + 1
        c2s = [cart2sph_by_l(l) for l in range(uniq_l.max()+1)]
        diag_addresses = [] # addresses wrt the compressed indices
    offset = 0
    p0 = p1 = 0
    for i, j in ij_tasks:
        nfi = nf[i]
        nfj = nf[j]
        nfij = nfi * nfj
        bas_ij = bas_ij_cache[i, j][0]
        n_pairs = len(bas_ij)
        p0, p1 = p1, p1 + nfij * n_pairs
        ish, jsh = divmod(bas_ij, nbas)
        aopair_offsets_lookup[jsh,ish] = \
                aopair_offsets_lookup[ish,jsh] = cp.arange(p0, p1, nfij)
        iaddr = ao_loc[ish,None] + cp.arange(nf[i])
        jaddr = ao_loc[jsh,None] + cp.arange(nf[j])
        # Note: in each <i|j> block, i is accessed in the inner loop
        ao_pair_mapping.append((iaddr[:,None,:] * nao + jaddr[:,:,None]).ravel())
        if i == j:
            idx = cp.where(ish == jsh)[0]
            addr = offset + idx[:,None] * (nfi*nfi) + cp.arange(nfi*nfi)
            diag_addresses.append(addr.ravel())
        offset += n_pairs * nfij
    non0_size = p1

    ao_pair_mapping = cp.hstack(ao_pair_mapping)
    rows, cols = divmod(ao_pair_mapping, nao)
    diag_addresses = cp.hstack(diag_addresses)
    cderi_idx = (rows.get(), cols.get(), diag_addresses.get())

    ft_ao.init_constant(sorted_cell)
    kern = libpbc.build_ft_ao

    avail_mem = get_avail_mem() * .8
    Gblksize = max(16, int(avail_mem/(2*16*nao**2))//8*8)
    Gblksize = min(Gblksize, ngrids, 16384)
    logger.debug1(cell, 'Gblksize = %d', Gblksize)

    j3c_compressed = cp.empty((non0_size,naux), dtype=np.float64)
    coulG = _weighted_coulG_LR(auxcell, Gv, omega, kws)
    pair0 = pair1 = 0
    for i, j in bas_ij_cache:
        li = uniq_l[i]
        lj = uniq_l[j]
        ll_pattern = f'{l_symb[i]}{l_symb[j]}'
        ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
        jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
        nfi = nf_cart[i]
        nfj = nf_cart[j]
        nfij = nfi * nfj
        bas_ij, img_offsets, img_idx = bas_ij_cache[i, j]
        n_pairs = len(bas_ij)
        j3c_tmp = cp.zeros((nfij*n_pairs,naux), dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            auxG_conj = cp.asarray(ft_ao.ft_ao(auxcell, Gv[p0:p1]).conj(), order='C')
            auxG_conj *= cp.asarray(coulG[p0:p1,None])
            GvT = cp.array(Gv[p0:p1].T, order='C', copy=True)
            # Padding zeros, allowing idle threads to access Gv over the bounds.
            GvT = cp.append(GvT, cp.zeros(THREADS))

            pqG = cp.zeros((nfij*n_pairs, nGv), dtype=np.complex128)
            scheme = ft_ao.ft_ao_scheme(cell, li, lj, nGv)
            log.debug2('ft_ao_scheme for %s: %s', ll_pattern, scheme)
            err = kern(
                ctypes.cast(pqG.data.ptr, ctypes.c_void_p),
                ctypes.c_int(1), # Compressing, remove zero elements
                ctypes.byref(aft_envs), (ctypes.c_int*3)(*scheme),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.c_int(n_pairs), ctypes.c_int(nGv),
                ctypes.cast(bas_ij.data.ptr, ctypes.c_void_p),
                ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                sorted_cell._atm.ctypes, ctypes.c_int(sorted_cell.natm),
                sorted_cell._bas.ctypes, ctypes.c_int(sorted_cell.nbas),
                sorted_cell._env.ctypes)
            if err != 0:
                raise RuntimeError(f'build_ft_ao_compressed kernel for {ll_pattern} failed')
            # \sum_G coulG * ints(ij * exp(-i G * r)) * ints(P * exp(i G * r))
            # = \sum_G FT(ij, G) conj(FT(aux, G)) , where aux
            # functions |P> are assumed to be real
            contract('pG,Gr->pr', pqG, auxG_conj, beta=1., out=j3c_tmp)
        t1 = log.timer_debug2(f'processing {ll_pattern}', *t1)

        j3c_tmp = j3c_tmp.real
        if not cell.cart:
            j3c_tmp = j3c_tmp.reshape(nfj,nfi,n_pairs,naux)
            j3c_tmp = contract('qj,qpmk->jpmk', c2s[lj], j3c_tmp)
            j3c_tmp = contract('pi,jpmk->mjik', c2s[li], j3c_tmp)
            j3c_tmp = j3c_tmp.reshape(-1,naux)

        pair0, pair1 = pair1, pair1 + n_pairs * nf[i] * nf[j]
        j3c_compressed[pair0:pair1] = j3c_tmp
        j3c_tmp = None
    return j3c_compressed, aopair_offsets_lookup, cp.asarray(rev_mapping), cderi_idx

def compressed_cderi_gamma_point(cell, auxcell, omega=OMEGA_MIN, with_long_range=True,
                                 linear_dep_threshold=LINEAR_DEP_THR):
    log = logger.new_logger(cell)
    t1 = log.init_timer()

    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega=-abs(omega)).build()
    c_shell_counts = np.asarray(int3c2e_opt.cell0_ctr_l_counts)
    lmax = cell._bas[:,ANG_OF].max()
    uniq_l = np.arange(lmax+1)
    if cell.cart:
        nf = (uniq_l + 1) * (uniq_l + 2) // 2
    else:
        nf = uniq_l * 2 + 1
    c_l_offsets = np.append(0, np.cumsum(c_shell_counts))
    lmax = cell._bas[:,ANG_OF].max()
    c2s = [cart2sph_by_l(l) for l in range(lmax+1)]

    naux = int3c2e_opt.aux_coeff.shape[1]
    if with_long_range:
        # LR int3c2e generally creates more non-negligible Coulomb integrals.
        # To add sr_int3c2e integrals to the corresponding elements in LR
        # tensor, bas_mapping and aopair_offsets_lookup are utilized for indexing.
        # bas_mapping[n] translates the shell n in sr_int3c2e.sorted_cell to
        # that in ft_aopair.sorted_cell. aopair_offsets_lookup convertes the
        # address in a dense tensor to compressed storage.
        j3c, aopair_offsets_lookup, bas_mapping, cderi_idx = \
                _lr_int3c2e_gamma_point(int3c2e_opt)
        t1 = log.timer_debug1('LR int3c2e', *t1)
    else:
        t1 = log.init_timer()
        img_idx_cache = int3c2e_opt.make_img_idx_cache()
        size = 0
        for (li, lj), img_idx in img_idx_cache.items():
            size += nf[li] * nf[lj] * len(img_idx[4])
        j3c = cp.zeros((size, naux), dtype=np.float64)
        # ao_pair_mapping stores AO-pair addresses in the nao x nao matrix,
        # which allows the decompression for the CUDA kernel generated compressed_eri3c:
        # sparse_eri3c[ao_pair_mapping] => compressed_eri3c
        ao_pair_mapping = []
        diag_addresses = [] # addresses wrt the compressed indices
        # Given shell Id in sorted_cell, this ao_loc maps shell to the AO offset
        # in the original cell
        ao_loc = cp.asarray(int3c2e_opt.ao_idx[int3c2e_opt.sorted_cell.ao_loc[:-1]])
        nao = cell.nao

    offset = 0
    p0 = p1 = 0
    for li, lj, c_pair_idx, j3c_tmp in int3c2e_opt.int3c2e_kernel():
        i0, i1 = c_l_offsets[li:li+2]
        j0, j1 = c_l_offsets[lj:lj+2]
        nctrj = c_shell_counts[lj]
        nfi = (li+1)*(li+2)//2
        nfj = (lj+1)*(lj+2)//2
        n_pairs = len(c_pair_idx)
        j3c_tmp = j3c_tmp.reshape(-1,nfi*nfj*n_pairs)
        j3c_tmp = j3c_tmp.T.dot(cp.asarray(int3c2e_opt.aux_coeff))
        if not cell.cart:
            j3c_tmp = j3c_tmp.reshape(nfj,nfi,n_pairs,naux)
            j3c_tmp = contract('qj,qpmk->jpmk', c2s[lj], j3c_tmp)
            j3c_tmp = contract('pi,jpmk->jimk', c2s[li], j3c_tmp)
            nfi = li * 2 + 1
            nfj = lj * 2 + 1
            j3c_tmp = j3c_tmp.reshape(-1,naux)

        ish, jsh = divmod(c_pair_idx, nctrj)
        ish += i0
        jsh += j0
        if with_long_range:
            ish = bas_mapping[ish]
            jsh = bas_mapping[jsh]
            ft_idx = aopair_offsets_lookup[ish,jsh]
            ij = cp.arange(nfi*nfj)
            idx = ij[:,None] + ft_idx
            # Due to the bas_mapping from int3c2e_opt.cell to ft_opt.cell,
            # the bas_ij pair for int3c2e_opt may correspond to the triu
            # bas-pair in ft_opt.cell. For these bas_ij, a transpose on <i|j>
            # should be applied to wrap the triu block to the tril block.
            triu_mask = ish < jsh
            ft_idx = ft_idx[triu_mask]
            if len(ft_idx) > 0:
                # Note: in each block, i is accessed in the inner loop
                ijT = ij.reshape(nfj,nfi).T
                idx[:,triu_mask] = ijT.reshape(-1,1) + ft_idx
            j3c[idx.ravel()] += j3c_tmp
            idx = ft_idx = ij = ijT = triu_mask = None
        else:
            p0, p1 = p1, p1 + nfi*nfj*n_pairs
            j3c[p0:p1] = j3c_tmp
            iaddr = ao_loc[ish] + cp.arange(nfi)[:,None]
            jaddr = ao_loc[jsh] + cp.arange(nfj)[:,None]
            # Note: address is computed in a different way than in the LR tensor.
            # The storage order here is [j,i,pair_id,aux].
            ao_pair_mapping.append((iaddr * nao + jaddr[:,None,:]).ravel())
            if li == lj:
                idx = cp.where(ish == jsh)[0]
                # The addresses for the compressed tensor
                addr = offset + idx[:,None] * (nfi*nfi) + cp.arange(nfi*nfi)
                diag_addresses.append(addr.ravel())
            offset += n_pairs * nfi * nfj
        j3c_tmp = ish = jsh = None

    if not with_long_range:
        ao_pair_mapping = cp.hstack(ao_pair_mapping)
        rows, cols = divmod(ao_pair_mapping, nao)
        diag_addresses = cp.hstack(diag_addresses)
        cderi_idx = (rows.get(), cols.get(), diag_addresses.get())
    t1 = log.timer_debug1('SR int3c2e', *t1)

    log.debug('Generate auxcell 2c2e integrals')
    j2c = _get_2c2e(auxcell, None, omega, with_long_range) # on CPU
    j2c = j2c[0].real
    t1 = log.timer('int2c2e', *t1)

    prefer_ed = PREFER_ED
    if cell.dimension == 2:
        prefer_ed = True
    cd_j2c, cd_j2c_negative, j2ctag = decompose_j2c(
        j2c, prefer_ed, linear_dep_threshold)

    if j2ctag == 'ED':
        cderi = contract('Lr,pr->Lp', cd_j2c, j3c)
    else:
        cderi = solve_triangular(cd_j2c, j3c.T, lower=True)

    cderip = None
    if cd_j2c_negative is not None:
        assert cell.dimension == 2
        cderip = contract('Lr,pr->Lp', cd_j2c_negative, j3c)

    t1 = log.timer_debug1('solving cderi', *t1)
    return cderi, cderip, cderi_idx


def get_pp_loc_part1(cell, kpts=None, with_pseudo=True, verbose=None):
    fakenuc = aft_cpu._fake_nuc(cell, with_pseudo=with_pseudo)
    cell_exps, cs = extract_pgto_params(cell, 'diffused')
    omega = (2*cell_exps.min())**.5
    logger.debug(cell, 'omega guess in get_pp_loc_part1 = %g', omega)

    if kpts is None or is_zero(kpts):
        kpts = None
        bvk_kmesh = np.ones(3, dtype=int)
    else:
        bvk_kmesh = kpts_to_kmesh(cell, kpts)
    nuc = sr_aux_e2(cell, fakenuc, -omega, kpts, bvk_kmesh, j_only=True)
    charges = -cp.asarray(cell.atom_charges())
    if kpts is None:
        nuc = contract('pqr,r->pq', nuc, charges)
    else:
        nuc = contract('kpqr,r->kpq', nuc, charges)

    # TODO: consider time-reversal symmetry
    ft_ao_iter = _ft_ao_iter_generator(cell, fakenuc, bvk_kmesh, omega, verbose)
    kpt = np.zeros(3)
    for i, (pqG, auxG_conj) in enumerate(ft_ao_iter(kpt, kpts)):
        ZG = auxG_conj.dot(charges)
        # contributions due to pseudo.pp_int.get_gth_vlocG_part1
        if (with_pseudo and i == 0 and
            (cell.dimension == 3 or
             (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
            exps = cp.asarray(np.hstack(fakenuc.bas_exps()))
            ZG[0] -= charges.dot(np.pi/exps) / cell.vol
        if kpts is None:
            nuc += contract('pqG,G->pq', pqG[0], ZG).real
        else:
            nuc += contract('kpqG,G->kpq', pqG, ZG)
    return nuc

def get_nuc(cell, kpts=None):
    '''Get the periodic nuc-el AO matrix, with G=0 removed.
    '''
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    nuc = get_pp_loc_part1(cell, kpts, with_pseudo=False, verbose=log)
    log.timer('get_nuc', *t0)
    return nuc

def get_pp(cell, kpts=None):
    '''Get the periodic pseudopotential nuc-el ao matrix, with G=0 removed.
    '''
    from pyscf.pbc.gto import pseudo
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    pp2builder = aft_cpu._IntPPBuilder(cell, kpts)
    vpp  = cp.asarray(pp2builder.get_pp_loc_part2())
    t1 = log.timer_debug1('get_pp_loc_part2', *t0)
    vpp += cp.asarray(pseudo.pp_int.get_pp_nl(cell, kpts))
    t1 = log.timer_debug1('get_pp_nl', *t1)

    vpp += get_pp_loc_part1(cell, kpts, with_pseudo=True, verbose=log)
    t1 = log.timer_debug1('get_pp_loc_part1', *t1)
    log.timer('get_pp', *t0)
    return vpp
