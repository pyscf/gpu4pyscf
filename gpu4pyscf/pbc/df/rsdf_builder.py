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
import numpy as np
import cupy as cp
import scipy.linalg
from pyscf import lib
#from pyscf.pbc import gto as pbcgto
#from pyscf.pbc.gto import pseudo
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.df.rsdf_builder import (
    OMEGA_MIN, LINEAR_DEP_THR, RCUT_THRESHOLD, estimate_ke_cutoff_for_omega)
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.lib.kpts_helper import kk_adapted_iter
from gpu4pyscf.pbc.df.int3c2e import sr_aux_e2


def build_cderi(cell, auxcell, kmesh=None, omega=0.1, j_only=False,
                linear_dep_threshold=LINEAR_DEP_THR):
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension >= 2
    if kmesh is None:
        return build_cderi_gamma_point(
            cell, auxcell, kmesh, omega, linear_dep_threshold)
    elif j_only:
        return build_cderi_j_only(
            cell, auxcell, kmesh, omega, linear_dep_threshold)
    else:
        return build_cderi_kk(
            cell, auxcell, kmesh, omega, linear_dep_threshold)

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
        ft_ao_iter = _ft_ao_iter_generator(cell, auxcell, kmesh, omega, log)

    cderi = {}
    cderip = {}
    for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        log.debug1('make_cderi for k-point %d %s', kp, kpts[kp])
        log.debug1('ki_idx = %s', ki_idx)
        log.debug1('kj_idx = %s', kj_idx)

        if has_long_range:
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
        cd_j2c, cd_j2c_negative, j2ctag = decompose_j2c(j2c_k, linear_dep_threshold)
        assert j2ctag == 'ED'

        for ki, kj in zip(ki_idx, kj_idx):
            j3c_k = j3c[ki,kj]
            cderi[ki,kj] = contract('Lr,pqr->Lpq', cd_j2c, j3c_k)
            if cd_j2c_negative is not None:
                # for low-dimension systems
                cderip[ki,kj] = contract('Lr,pqr->Lpq', cd_j2c_negative, j3c_k)
    t1 = log.timer('pass2: solve cderi', *t1)
    return cderi, cderip

def build_cderi_gamma_point(cell, auxcell, kmesh=None, omega=0.1,
                            linear_dep_threshold=LINEAR_DEP_THR):
    assert kmesh is None
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    if cell.omega != 0:
        assert cell.omega < 0
        omega = abs(cell.omega)
        has_long_range = False
    else:
        omega = abs(omega)
        has_long_range = True
    kpts = None

    j3c = sr_aux_e2(cell, auxcell, -omega)
    t1 = log.timer('pass1: int3c2e', *t0)

    log.debug('Generate auxcell 2c2e integrals')
    j2c = _get_2c2e(auxcell, kpts, omega, has_long_range) # on CPU
    j2c = j2c[0].real
    t1 = log.timer('int2c2e', *t1)

    cderi = {}
    cderip = {}
    if has_long_range:
        ft_ao_iter = _ft_ao_iter_generator(cell, auxcell, kmesh, omega, log)
        for pqG, auxG_conj in ft_ao_iter():
            # \sum_G coulG * ints(ij * exp(-i G * r)) * ints(P * exp(i G * r))
            # = \sum_G FT(ij, G) conj(FT(aux, G)) , where aux
            # functions |P> are assumed to be real
            j3c += contract('pqG,Gr->pqr', pqG[0], auxG_conj).real

    cd_j2c, cd_j2c_negative, j2ctag = decompose_j2c(j2c, linear_dep_threshold)
    assert j2ctag == 'ED'

    cderi[0,0] = contract('Lr,pqr->Lpq', cd_j2c, j3c)
    if cd_j2c_negative is not None:
        # for low-dimension systems
        cderip[0,0] = contract('Lr,pqr->Lpq', cd_j2c_negative, j3c)
    t1 = log.timer('pass2: solve cderi', *t1)
    return cderi, cderip

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

    # TODO: do not generate the entire array
    j3c = sr_aux_e2(cell, auxcell, -omega, bvk_kmesh=kmesh)
    t1 = log.timer('pass1: int3c2e', *t0)

    log.debug('Generate auxcell 2c2e integrals')
    if kmesh is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    j3c = j3c[np.arange(nkpts),np.arange(nkpts)] # FIXME
    j2c = _get_2c2e(auxcell, None, omega, has_long_range) # on CPU
    j2c = j2c[0].real
    t1 = log.timer('int2c2e', *t1)

    # TODO: consider time-reversal symmetry
    cderi = {}
    cderip = {}
    if has_long_range:
        ft_ao_iter = _ft_ao_iter_generator(cell, auxcell, kmesh, omega, log)
        kpt = np.zeros(3)
        for pqG, auxG_conj in ft_ao_iter(kpt, kpts):
            # \sum_G coulG * ints(ij * exp(-i G * r)) * ints(P * exp(i G * r))
            # = \sum_G FT(ij, G) conj(FT(aux, G)) , where aux
            # functions |P> are assumed to be real
            j3c += contract('kpqG,Gr->kpqr', pqG, auxG_conj)

    cd_j2c, cd_j2c_negative, j2ctag = decompose_j2c(j2c, linear_dep_threshold)
    assert j2ctag == 'ED'

    for k in range(nkpts):
        cderi[k, k] = contract('Lr,pqr->Lpq', cd_j2c, j3c[k])
        if cd_j2c_negative is not None:
            # for low-dimension systems
            cderip[k, k] = contract('Lr,pqr->Lpq', cd_j2c_negative, j3c[k])
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
        auxG_conj = ft_ao.ft_ao(auxcell, Gv, kpt=kpt).conj()
        auxG_conj *= cp.asarray(coulG[:,None])
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            pqG = ft_kern(Gv[p0:p1], kpt, kpts).transpose(0,2,3,1)
            yield pqG, auxG_conj[p0:p1]
    return ft_ao_iter

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

def get_nuc(cell):
    raise NotImplementedError

def get_pp(cell):
    raise NotImplementedError
