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
import ctypes
import warnings
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
from pyscf import lib
#from pyscf.pbc import gto as pbcgto
#from pyscf.pbc.gto import pseudo
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
from gpu4pyscf.gto.mole import extract_pgto_params
from gpu4pyscf.pbc.df.int3c2e import sr_aux_e2, estimate_rcut

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

def build_cderi(cell, auxcell, kpts=None, j_only=False,
                omega=None, linear_dep_threshold=LINEAR_DEP_THR):
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension >= 2
    if cell.omega != 0:
        assert cell.omega < 0
        omega = abs(cell.omega)
        with_long_range = False
    else:
        if omega is None:
            cell_exps, cs = extract_pgto_params(cell, 'diffused')
            omega = cell_exps.min()**.5
            logger.debug(cell, 'omega guess in rsdf_builder = %g', omega)
        omega = abs(omega)
        with_long_range = True

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
        auxG_conj = cp.asarray(ft_ao.ft_ao(auxcell, Gv, kpt=kpt).conj(), order='C')
        auxG_conj *= cp.asarray(coulG[:,None])
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            pqG = ft_kern(Gv[p0:p1], kpt, kpts).transpose(0,2,3,1)
            yield pqG, auxG_conj[p0:p1]
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
