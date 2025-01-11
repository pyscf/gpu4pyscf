# Copyright 2024 The PySCF Developers. All Rights Reserved.
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
import tempfile
import numpy as np
import cupy as cp
import scipy.linalg
from pyscf import lib
#from pyscf.pbc import gto as pbcgto
#from pyscf.pbc.gto import pseudo
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.df.rsdf_builder import (
    OMEGA_MIN, LINEAR_DEP_THR, RCUT_THRESHOLD, estimate_ke_cutoff_for_omega,
    _gaussian_int)
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.lib.kpts_helper import kk_adapted_iter
from gpu4pyscf.pbc.df.int3c2e import sr_aux_e2


def build_cderi(cell, auxcell, kmesh=None, omega=0.1, j_only=False,
                linear_dep_threshold=LINEAR_DEP_THR):
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension >= 2
    return build_cderi_kk(cell, auxcell, kmesh, omega, j_only,
                          linear_dep_threshold)

def build_cderi_kk(cell, auxcell, kmesh=None, omega=0.1,
                   linear_dep_threshold=LINEAR_DEP_THR):
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    if cell.omega != 0:
        assert cell.omega < 0
        omega = abs(cell.omega)
        has_long_range = False
    else:
        omega = abs(omega)
        has_long_range = True
        ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
        mesh = cell.cutoff_to_mesh(ke_cutoff)
        mesh = cell.symmetrize_mesh(mesh)

    j3c = sr_aux_e2(cell, auxcell, -omega, bvk_kmesh=kmesh)
    t1 = log.timer('pass1: int3c2e', *t0)

    log.debug('Generate auxcell 2c2e integrals')
    if kmesh is None:
        kpts = np.zeros((1, 3))
        kpt_iters = list(kk_adapted_iter([1, 1, 1]))
    else:
        kpts = cell.make_kpts(kmesh)
        kpt_iters = list(kk_adapted_iter(kmesh))
    nkpts = len(kpts)
    uniq_kpts = kpts[[x[0] for x in kpt_iters]]
    j2c = _get_2c2e(auxcell, uniq_kpts, omega, has_long_range) # on CPU
    t1 = log.timer('int2c2e', *t1)

    if has_long_range:
        _add_sr_G0(j3c, cell, auxcell, omega, kpts)
        ft_opt = ft_ao.FTOpt(cell, bvk_kmesh=kmesh)
        ft_kern = ft_opt.gen_ft_kernel(verbose=log)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        ngrids = len(Gv)
        nao = cell.nao
        bvk_ncells = nkpts
        avail_mem = get_avail_mem() * .8
        Gblksize = max(16, int(avail_mem/(2*16*nao**2*bvk_ncells))//8*8)
        Gblksize = min(Gblksize, ngrids, 16384)
        log.debug1('Gblksize = %d', Gblksize)

    out = {}
    out_neg = {}
    for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        log.debug1('make_cderi for k-point %d %s', kp, kpts[kp])
        log.debug1('ki_idx = %s', ki_idx)
        log.debug1('kj_idx = %s', kj_idx)

        if has_long_range:
            '''exp(-i*(G + k) dot r) * Coulomb_kernel'''
            # TODO: compute the long range part Coulomb kernel as the difference
            # between coulG(cell.omega) - coulG(df.omega). This allows the SR-
            # and regular integrals being treated in the same framework. See
            # more detail in pyscf.pbc.df.rsdf_builder.weighted_coulG_LR
            kpt = kpts[kp]
            coulG = pbctools.get_coulG(cell, kpt=kpt, exx=False, Gv=Gv, omega=omega)
            coulG *= kws
            auxG = ft_ao.ft_ao(auxcell, Gv, kpt=kpt).T
            auxG *= cp.asarray(coulG)
            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                Gpq = ft_kern(Gv[p0:p1], kpt, kpts)
                # \sum_G coulG * ints(ij * exp(-i G * r)) * ints(P * exp(i G * r))
                # = \sum_G FT(ij, G) conj(FT(aux, G)) , where aux
                # functions |P> are assumed to be real
                j3c[ki_idx,kj_idx] += contract('kGpq,rG->kpqr', Gpq, auxG.conj())

        j2c_k = j2c[j2c_idx]
        if self_conj:
            # DF metric for self-conjugated k-point should be real
            j2c_k = j2c_k.real
        cd_j2c, cd_j2c_negative, j2ctag = decompose_j2c(j2c_k, linear_dep_threshold)
        assert j2ctag == 'ED'

        for ki, kj in zip(ki_idx, kj_idx):
            j3c_k = j3c[ki,kj]
            out[ki,kj] = contract('pqr,Lr->pqL', j3c_k, cd_j2c)
            if cd_j2c_negative is not None:
                # for low-dimension systems
                out_neg[ki,kj] = contract('pqr,Lr->pqL', j3c_k, cd_j2c_negative)
    t1 = log.timer('pass2: solve cderi', *t1)
    return out, out_neg

def build_cderi_gamma_point(cell, auxcell, kmesh=None, omega=0.1,
                            linear_dep_threshold=LINEAR_DEP_THR):
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    if cell.omega != 0:
        assert cell.omega < 0
        omega = abs(cell.omega)
        has_long_range = False
    else:
        omega = abs(omega)
        has_long_range = True
        ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
        mesh = cell.cutoff_to_mesh(ke_cutoff)
        mesh = cell.symmetrize_mesh(mesh)
    kpts = np.zeros((1, 3))

    j3c = sr_aux_e2(cell, auxcell, -omega, bvk_kmesh=kmesh)
    t1 = log.timer('pass1: int3c2e', *t0)

    log.debug('Generate auxcell 2c2e integrals')
    j2c = _get_2c2e(auxcell, kpts, omega, has_long_range) # on CPU
    j2c = j2c[0].real
    t1 = log.timer('int2c2e', *t1)

    out = {}
    out_neg = {}
    if has_long_range:
        _add_sr_G0(j3c, cell, auxcell, omega, kpts)
        ft_opt = ft_ao.FTOpt(cell)
        ft_kern = ft_opt.gen_ft_kernel(verbose=log)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        ngrids = len(Gv)
        nao = cell.nao
        avail_mem = get_avail_mem() * .8
        Gblksize = max(16, int(avail_mem/(2*16*nao**2))//8*8)
        Gblksize = min(Gblksize, ngrids, 16384)
        log.debug1('Gblksize = %d', Gblksize)

        coulG = pbctools.get_coulG(cell, exx=False, Gv=Gv, omega=omega)
        coulG *= kws
        auxG = ft_ao.ft_ao(auxcell, Gv).T
        auxG *= cp.asarray(coulG)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            Gpq = ft_kern(Gv[p0:p1], kpt, kpts)
            # \sum_G coulG * ints(ij * exp(-i G * r)) * ints(P * exp(i G * r))
            # = \sum_G FT(ij, G) conj(FT(aux, G)) , where aux
            # functions |P> are assumed to be real
            j3c += contract('kGpq,rG->kpqr', Gpq, auxG.conj()).real

    if self_conj:
        # DF metric for self-conjugated k-point should be real
    cd_j2c, cd_j2c_negative, j2ctag = decompose_j2c(j2c, linear_dep_threshold)
    assert j2ctag == 'ED'

    out[0,0] = contract('pqr,Lr->pqL', j3c, cd_j2c)
    if cd_j2c_negative is not None:
        # for low-dimension systems
        out_neg[0,0] = contract('pqr,Lr->pqL', j3c, cd_j2c_negative)
    t1 = log.timer('pass2: solve cderi', *t1)
    return out, out_neg

def build_cderi_j_only(cell, auxcell, kmesh=None, omega=0.1,
                       linear_dep_threshold=LINEAR_DEP_THR):
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    if cell.omega != 0:
        assert cell.omega < 0
        omega = abs(cell.omega)
        has_long_range = False
    else:
        omega = abs(omega)
        has_long_range = True
        ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
        mesh = cell.cutoff_to_mesh(ke_cutoff)
        mesh = cell.symmetrize_mesh(mesh)

    # TODO: do not generate the entire array
    j3c = sr_aux_e2(cell, auxcell, -omega, bvk_kmesh=kmesh)
    t1 = log.timer('pass1: int3c2e', *t0)

    log.debug('Generate auxcell 2c2e integrals')
    if kmesh is None:
        kpts = np.zeros((1, 3))
        kpt_iters = list(kk_adapted_iter([1, 1, 1]))
    else:
        kpts = cell.make_kpts(kmesh)
        kpt_iters = list(kk_adapted_iter(kmesh))
    nkpts = len(kpts)
    uniq_kpts = kpts[[x[0] for x in kpt_iters]]
    j2c = _get_2c2e(auxcell, uniq_kpts, omega, has_long_range) # on CPU
    t1 = log.timer('int2c2e', *t1)

    ft_opt = ft_ao.FTOpt(cell, bvk_kmesh=kmesh)
    ft_kern = ft_opt.gen_ft_kernel(verbose=log)
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    ngrids = len(Gv)
    nao = cell.nao
    bvk_ncells = nkpts
    avail_mem = get_avail_mem() * .8
    Gblksize = max(16, int(avail_mem/(2*16*nao**2*bvk_ncells))//8*8)
    Gblksize = min(Gblksize, ngrids, 16384)
    log.debug1('Gblksize = %d', Gblksize)

    if has_long_range:
        _add_sr_G0(j3c, cell, auxcell, omega, kpts)

    out = {}
    out_neg = {}
    # TODO: consider time-reversal symmetry
    for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        log.debug1('make_cderi for k-point %d %s', kp, kpts[kp])
        log.debug1('ki_idx = %s', ki_idx)
        log.debug1('kj_idx = %s', kj_idx)

        if has_long_range:
            '''exp(-i*(G + k) dot r) * Coulomb_kernel'''
            coulG = pbctools.get_coulG(cell, exx=False, Gv=Gv, omega=omega)
            coulG *= kws
            auxG = ft_ao.ft_ao(auxcell, Gv).T
            auxG *= cp.asarray(coulG)
            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                Gpq = ft_kern(Gv[p0:p1], kpt, kpts)
                # \sum_G coulG * ints(ij * exp(-i G * r)) * ints(P * exp(i G * r))
                # = \sum_G FT(ij, G) conj(FT(aux, G)) , where aux
                # functions |P> are assumed to be real
                j3c[ki_idx,kj_idx] += contract('kGpq,rG->kpqr', Gpq, auxG.conj())

        j2c_k = j2c[j2c_idx]
        if kp == kp_conj: # self conjugated
            # DF metric for self-conjugated k-point should be real
            j2c_k = j2c_k.real
        cd_j2c, cd_j2c_negative, j2ctag = decompose_j2c(j2c_k, linear_dep_threshold)
        assert j2ctag == 'ED'

        for ki, kj in zip(ki_idx, kj_idx):
            j3c_k = j3c[ki,kj]
            out[ki,kj] = contract('pqr,Lr->pqL', j3c_k, cd_j2c)
            if cd_j2c_negative is not None:
                # for low-dimension systems
                out_neg[ki,kj] = contract('pqr,Lr->pqL', j3c_k, cd_j2c_negative)
    t1 = log.timer('pass2: solve cderi', *t1)
    return out, out_neg

def _add_sr_G0(j3c, cell, auxcell, omega, kpts):
    aux_chg = _gaussian_int(auxcell)
    vbar_idx = aux_chg != 0
    if np.any(vbar_idx):
        ovlp = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
        vbar = cp.asarray(np.pi / omega**2 / cell.vol * aux_chg[vbar_idx])
        if j3c.ndim == 3:
            vmod = cp.asarray(ovlp[0])[:,:,None] * vbar
            j3c[:,:,vbar_idx] -= vmod
        else:
            nkpts = len(kpts)
            for k in range(nkpts):
                j3c_k = j3c[k,k]
                vmod = cp.asarray(ovlp[k])[:,:,None] * vbar
                j3c_k[:,:,vbar_idx] -= vmod
    return j3c

def _weighted_coulG(cell, kpt=np.zeros(3), exx=False, mesh=None, omega=None):
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    coulG = pbctools.get_coulG(cell, kpt, exx, None, mesh, Gv, omega=omega)
    coulG *= kws
    return coulG

def decompose_j2c(j2c, linear_dep_threshold=LINEAR_DEP_THR):
    return eigenvalue_decomposed_metric(j2c, linear_dep_threshold)

def eigenvalue_decomposed_metric(j2c, linear_dep_threshold=LINEAR_DEP_THR):
    w, v = scipy.linalg.eigh(j2c)
    mask = w > linear_dep_threshold
    v1 = v[:,mask].conj().T
    v1 *= w[mask, None]**-.5
    j2c = v1
    idx = np.where(w < -linear_dep_threshold)[0]
    j2c_negative = None
    if len(idx) > 0:
        j2c_negative = (v[:,idx] * (-w[idx])**-.5).conj().T
    j2ctag = 'ED'
    return j2c, j2c_negative, j2ctag

# Create 2c2e, store on CPU
def _get_2c2e(auxcell, uniq_kpts, omega, has_long_range=True):
    # j2c ~ (-kpt_ji | kpt_ji) => hermi=1
    precision = auxcell.precision**1.5
    aux_exp = np.hstack(auxcell.bas_exps()).min()
    theta = 1./(2./aux_exp + omega**-2)
    lattice_sum_factor = max(2*np.pi*auxcell.rcut/(auxcell.vol*theta), 1)
    rcut_sr = (np.log(lattice_sum_factor / precision + 1.) / theta)**.5
    logger.debug1(auxcell, 'auxcell  rcut_sr = %g', rcut_sr)
    auxcell_sr = auxcell.copy()
    auxcell_sr.rcut = rcut_sr
    with auxcell_sr.with_short_range_coulomb(omega):
        j2c = auxcell_sr.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)

    if not has_long_range:
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

    for k, kpt in enumerate(uniq_kpts):
        j2c_k = cp.asarray(j2c[k])
        gamma_point = is_zero(kpt)
        if gamma_point:
            aux_chg = cp.asarray(_gaussian_int(auxcell))
            j2c_k -= cp.asarray(np.pi / omega**2 / auxcell.vol * aux_chg[:,None] * aux_chg)

        coulG_LR = cp.asarray(_weighted_coulG(auxcell, kpt, False, mesh, omega=omega))
        for p0, p1 in lib.prange(0, ngrids, blksize):
            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
            if gamma_point:
                j2c_k += (auxG.conj() * coulG_LR[p0:p1]).dot(auxG.T).real
            else:
                j2c_k += (auxG.conj() * coulG_LR[p0:p1]).dot(auxG.T)
            auxG = None
        j2c[k] = j2c_k.get()
    return j2c
