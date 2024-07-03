#!/usr/bin/env python
#
# This code was copied from the data generation program of Tencent Alchemy
# project (https://github.com/tencent-alchemy).
#

#
# Copyright 2019 Tencent America LLC. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#  Modified by:   Xiaojie Wu <wxj6000@gmail.com>

'''
Non-relativistic RHF analytical Hessian with density-fitting approximation
Ref:
[1] Efficient implementation of the analytic second derivatives of
    Hartree-Fock and hybrid DFT energies: a detailed analysis of different
    approximations.  Dmytro Bykov, Taras Petrenko, Robert Izsak, Simone
    Kossmann, Ute Becker, Edward Valeev, Frank Neese. Mol. Phys. 113, 1961 (2015)
'''



import numpy
import cupy
import numpy as np
from pyscf import lib
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.hessian import uhf as uhf_hess
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.lib.cupy_helper import (
    contract, tag_array, get_avail_mem, release_gpu_stack, take_last2d, pinv)
from gpu4pyscf.df import int3c2e, df
from gpu4pyscf.lib import logger
from gpu4pyscf import __config__
from gpu4pyscf.df.grad.rhf import _gen_metric_solver

LINEAR_DEP_THR = df.LINEAR_DEP_THR
BLKSIZE = 256
ALIGNED = getattr(__config__, 'ao_aligned', 32)
GB = 1024*1024*1024

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    e1, ej, ek = _partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                   atmlst, max_memory, verbose, True)
    return e1 + ej - ek

def _partial_hess_ejk(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None, with_k=True, omega=None):
    '''Partial derivative
    '''
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = log.init_timer()

    mol = hessobj.mol
    mf = hessobj.base
    mf.with_df._cderi = None
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    mo_coeff = cupy.asarray(mo_coeff, order='C')
    mo_energy = cupy.asarray(mo_energy, order='C')
    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = cupy.dot(mocca, mocca.T)
    dm0b = cupy.dot(moccb, moccb.T)
    dm0 = dm0a + dm0b
    mo_ea = mo_energy[0][mo_occ[0]>0]
    mo_eb = mo_energy[1][mo_occ[1]>0]
    auxmol = df.addons.make_auxmol(mol, auxbasis=mf.with_df.auxbasis)
    naux = auxmol.nao
    auxslices = auxmol.aoslice_by_atom()
    aoslices = mol.aoslice_by_atom()

    if omega and omega > 1e-10:
        with auxmol.with_range_coulomb(omega):
            int2c = auxmol.intor('int2c2e', aosym='s1')
            int2c_ip1 = auxmol.intor('int2c2e_ip1', aosym='s1')
    else:
        int2c = auxmol.intor('int2c2e', aosym='s1')
        int2c_ip1 = auxmol.intor('int2c2e_ip1', aosym='s1')
    t1 = log.timer_debug1('intermediate variables with int2c2e and int2c2e_ip1', *t1)

    # ================================ sorted AO begin ===============================================
    intopt = int3c2e.VHFOpt(mol, auxmol, 'int2e')
    intopt.build(mf.direct_scf_tol, diag_block_with_triu=True, aosym=False, group_size=BLKSIZE, group_size_aux=BLKSIZE)
    ao_idx = intopt.ao_idx
    aux_ao_idx = intopt.aux_ao_idx

    mocca = mocca[ao_idx, :]
    moccb = moccb[ao_idx, :]
    dm0a = take_last2d(dm0a, ao_idx)
    dm0b = take_last2d(dm0b, ao_idx)
    dm0a_tag = tag_array(dm0a, occ_coeff=mocca)
    dm0b_tag = tag_array(dm0b, occ_coeff=moccb)
    int2c = cupy.asarray(int2c, order='C')
    int2c = take_last2d(int2c, aux_ao_idx)
    int2c_inv = pinv(int2c, lindep=LINEAR_DEP_THR)
    solve_j2c = _gen_metric_solver(int2c)
    int2c = None

    int2c_ip1 = cupy.asarray(int2c_ip1, order='C')
    int2c_ip1 = take_last2d(int2c_ip1, aux_ao_idx)

    hj_ao_ao = cupy.zeros([nao,nao,3,3])
    hk_ao_ao = cupy.zeros([nao,nao,3,3])
    if hessobj.auxbasis_response:
        hj_ao_aux = cupy.zeros([nao,naux,3,3])
        hk_ao_aux = cupy.zeros([nao,naux,3,3])

    #  int3c contributions
    wja, wka_P__ = int3c2e.get_int3c2e_jk(mol, auxmol, dm0a_tag, omega=omega)
    wjb, wkb_P__ = int3c2e.get_int3c2e_jk(mol, auxmol, dm0b_tag, omega=omega)
    rhoj0_P = solve_j2c(wja + wjb)
    rhok0a_P__ = solve_j2c(wka_P__)
    rhok0b_P__ = solve_j2c(wkb_P__)
    wja = wjb = wka_P__ = wkb_P__ = None
    t1 = log.timer_debug1('intermediate variables with int3c2e', *t1)

    # int3c_ip2 contributions
    wja_ip2, wka_ip2_P__ = int3c2e.get_int3c2e_ip2_wjk(intopt, dm0a_tag, omega=omega)
    wjb_ip2, wkb_ip2_P__ = int3c2e.get_int3c2e_ip2_wjk(intopt, dm0b_tag, omega=omega)
    wj_ip2 = wja_ip2 + wjb_ip2
    t1 = log.timer_debug1('interdeidate variables with int3c2e_ip2', *t1)

    #  int3c_ip1 contributions
    wj1a_P, wk1a_Pko = int3c2e.get_int3c2e_ip1_wjk(intopt, dm0a_tag, omega=omega)
    wj1b_P, wk1b_Pko = int3c2e.get_int3c2e_ip1_wjk(intopt, dm0b_tag, omega=omega)
    wj1_P = wj1a_P + wj1b_P
    rhoj1_P = solve_j2c(wj1_P)

    hj_ao_ao += 4.0*contract('pix,pjy->ijxy', rhoj1_P, wj1_P)   # (10|0)(0|0)(0|01)
    wj1_P = None
    if hessobj.auxbasis_response:
        wj0_01 = contract('ypq,q->yp', int2c_ip1, rhoj0_P)
        wj1_01 = contract('yqp,pix->iqxy', int2c_ip1, rhoj1_P)
        hj_ao_aux += contract('pix,py->ipxy', rhoj1_P, wj_ip2)   # (10|0)(1|00)
        hj_ao_aux -= contract('pix,yp->ipxy', rhoj1_P, wj0_01)   # (10|0)(1|0)(0|00)
        hj_ao_aux -= contract('q,iqxy->iqxy', rhoj0_P, wj1_01)   # (10|0)(0|1)(0|00)
        wj1_01 = None
    rhoj1_P = None

    if with_k:
        mem_avail = get_avail_mem()
        nocc = mocca.shape[1] + moccb.shape[1]
        slice_size = naux*nocc*9   # largest slice of intermediate variables
        blksize = int(mem_avail*0.2/8/slice_size/ALIGNED) * ALIGNED
        blksize = min(blksize, int((mem_avail*0.2/8//9/naux)**.5/ALIGNED)*ALIGNED)
        log.debug(f'GPU Memory {mem_avail/GB:.1f} GB available, block size {blksize}')
        if blksize < ALIGNED:
            raise RuntimeError('Not enough memory for intermediate variables')
    
        for i0, i1 in lib.prange(0,nao,blksize):
            wk1a_Pko_islice = cupy.asarray(wk1a_Pko[:,i0:i1])
            wk1b_Pko_islice = cupy.asarray(wk1b_Pko[:,i0:i1])
            rhok1a_Pko = solve_j2c(wk1a_Pko_islice)
            rhok1b_Pko = solve_j2c(wk1b_Pko_islice)
            wk1a_Pko_islice = wk1b_Pko_islice = None
            for k0, k1 in lib.prange(0,nao,blksize):
                wk1a_Pko_kslice = cupy.asarray(wk1a_Pko[:,k0:k1])
                wk1b_Pko_kslice = cupy.asarray(wk1b_Pko[:,k0:k1])

                # (10|0)(0|10) without response of RI basis
                vk2_ip1_ip1 = contract('piox,pkoy->ikxy', rhok1a_Pko, wk1a_Pko_kslice)
                hk_ao_ao[i0:i1,k0:k1] += contract('ikxy,ik->ikxy', vk2_ip1_ip1, dm0a[i0:i1,k0:k1])
                vk2_ip1_ip1 = contract('piox,pkoy->ikxy', rhok1b_Pko, wk1b_Pko_kslice)
                hk_ao_ao[i0:i1,k0:k1] += contract('ikxy,ik->ikxy', vk2_ip1_ip1, dm0b[i0:i1,k0:k1])
                vk2_ip1_ip1 = None

                # (10|0)(0|01) without response of RI basis
                bra = contract('piox,ko->pikx', rhok1a_Pko, mocca[k0:k1])
                ket = contract('pkoy,io->pkiy', wk1a_Pko_kslice, mocca[i0:i1])
                hk_ao_ao[i0:i1,k0:k1] += contract('pikx,pkiy->ikxy', bra, ket)
                bra = contract('piox,ko->pikx', rhok1b_Pko, moccb[k0:k1])
                ket = contract('pkoy,io->pkiy', wk1b_Pko_kslice, moccb[i0:i1])
                hk_ao_ao[i0:i1,k0:k1] += contract('pikx,pkiy->ikxy', bra, ket)
                bra = ket = None
            wk1a_Pko_kslice = wk1a_Pko_kslice = None
            if hessobj.auxbasis_response:
                # (10|0)(1|00)
                wk_ip2_Ipo = contract('porx,io->pirx', wka_ip2_P__, mocca[i0:i1])
                hk_ao_aux[i0:i1] += contract('piox,pioy->ipxy', rhok1a_Pko, wk_ip2_Ipo)
                wk_ip2_Ipo = contract('porx,io->pirx', wkb_ip2_P__, moccb[i0:i1])
                hk_ao_aux[i0:i1] += contract('piox,pioy->ipxy', rhok1b_Pko, wk_ip2_Ipo)
                wk_ip2_Ipo = None

                # (10|0)(1|0)(0|00)
                rhok0a_P_I = contract('qor,ir->qoi', rhok0a_P__, mocca[i0:i1])
                wk1_P_I = contract('ypq,qoi->pioy', int2c_ip1, rhok0a_P_I)
                hk_ao_aux[i0:i1] -= contract("piox,pioy->ipxy", rhok1a_Pko, wk1_P_I)
                rhok0b_P_I = contract('qor,ir->qoi', rhok0b_P__, moccb[i0:i1])
                wk1_P_I = contract('ypq,qoi->pioy', int2c_ip1, rhok0b_P_I)
                hk_ao_aux[i0:i1] -= contract("piox,pioy->ipxy", rhok1b_Pko, wk1_P_I)
                wk1_P_I = None

                # (10|0)(0|1)(0|00)
                wk1_I = contract('yqp,piox->qioxy', int2c_ip1, rhok1a_Pko)
                hk_ao_aux[i0:i1] -= contract('qoi,qioxy->iqxy', rhok0a_P_I, wk1_I)
                wk1_I = contract('yqp,piox->qioxy', int2c_ip1, rhok1b_Pko)
                hk_ao_aux[i0:i1] -= contract('qoi,qioxy->iqxy', rhok0b_P_I, wk1_I)
                wk1_I = rhok0a_P_I = rhok0b_P_I = None
        rhok1a_Pko = rhok1b_Pko = None
    wk1a_Pko = wk1b_Pko = None
    t1 = log.timer_debug1('intermediate variables with int3c2e_ip1', *t1)

    cupy.get_default_memory_pool().free_all_blocks()
    #  int3c_ipip1 contributions
    fn = int3c2e.get_int3c2e_ipip1_hjk
    hja_ao_diag, hka_ao_diag = fn(intopt, rhoj0_P, rhok0a_P__, dm0a_tag, omega=omega, with_k=with_k)
    hjb_ao_diag, hkb_ao_diag = fn(intopt, rhoj0_P, rhok0b_P__, dm0b_tag, omega=omega, with_k=with_k)
    hj_ao_diag = 2.0 * (hja_ao_diag + hjb_ao_diag)
    if with_k:
        hk_ao_diag = 2.0 * (hka_ao_diag + hkb_ao_diag)
    t1 = log.timer_debug1('intermediate variables with int3c2e_ipip1', *t1)

    #  int3c_ipvip1 contributions
    # (11|0), (0|00) without response of RI basis
    fn = int3c2e.get_int3c2e_ipvip1_hjk
    hja, hka = fn(intopt, rhoj0_P, rhok0a_P__, dm0a_tag, omega=omega, with_k=with_k)
    hjb, hkb = fn(intopt, rhoj0_P, rhok0b_P__, dm0b_tag, omega=omega, with_k=with_k)
    hj_ao_ao += 2.0*(hja + hjb)
    if with_k:
        hk_ao_ao += (hka + hkb)
    hja = hjb = hka = hkb = None
    t1 = log.timer_debug1('intermediate variables with int3c2e_ipvip1', *t1)

    #  int3c_ip1ip2 contributions
    # (10|1), (0|0)(0|00)
    if hessobj.auxbasis_response:
        fn = int3c2e.get_int3c2e_ip1ip2_hjk
        hja, hka = fn(intopt, rhoj0_P, rhok0a_P__, dm0a_tag, omega=omega, with_k=with_k)
        hjb, hkb = fn(intopt, rhoj0_P, rhok0b_P__, dm0b_tag, omega=omega, with_k=with_k)
        hj_ao_aux += hja + hjb
        if with_k:
            hk_ao_aux += hka + hkb
        hja = hjb = hka = hkb = None
        t1 = log.timer_debug1('intermediate variables with int3c2e_ip1ip2', *t1)

    #  int3c_ipip2 contributions
    if hessobj.auxbasis_response > 1:
        # (00|2), (0|0)(0|00)
        fn = int3c2e.get_int3c2e_ipip2_hjk
        hja, hka = fn(intopt, rhoj0_P, rhok0a_P__, dm0a_tag, omega=omega, with_k=with_k)
        hjb, hkb = fn(intopt, rhoj0_P, rhok0b_P__, dm0b_tag, omega=omega, with_k=with_k)
        hj_aux_diag = hja + hjb
        if with_k:
            hk_aux_diag = (hka + hkb)
        hja = hjb = hka = hkb = None
        t1 = log.timer_debug1('intermediate variables with int3c2e_ipip2', *t1)

    # int2c contributions
    if hessobj.auxbasis_response > 1:
        if omega and omega > 1e-10:
            with auxmol.with_range_coulomb(omega):
                int2c_ipip1 = auxmol.intor('int2c2e_ipip1', aosym='s1')
        else:
            int2c_ipip1 = auxmol.intor('int2c2e_ipip1', aosym='s1')
        int2c_ipip1 = cupy.asarray(int2c_ipip1, order='C')
        int2c_ipip1 = take_last2d(int2c_ipip1, aux_ao_idx)
        rhoj2c_P = contract('xpq,q->xp', int2c_ipip1, rhoj0_P)
        # (00|0)(2|0)(0|00)
        # p,xp->px
        hj_aux_diag -= (rhoj0_P*rhoj2c_P).T.reshape(-1,3,3)
        if with_k:
            rho2c_0 = contract('pij,qji->pq', rhok0a_P__, rhok0a_P__)
            rho2c_0+= contract('pij,qji->pq', rhok0b_P__, rhok0b_P__)
            hk_aux_diag -= contract('pq,xpq->px', rho2c_0, int2c_ipip1).reshape(-1,3,3)
        int2c_ipip1 = None

        if omega and omega > 1e-10:
            with auxmol.with_range_coulomb(omega):
                int2c_ip1ip2 = auxmol.intor('int2c2e_ip1ip2', aosym='s1')
        else:
            int2c_ip1ip2 = auxmol.intor('int2c2e_ip1ip2', aosym='s1')
        int2c_ip1ip2 = cupy.asarray(int2c_ip1ip2, order='C')
        int2c_ip1ip2 = take_last2d(int2c_ip1ip2, aux_ao_idx)
        hj_aux_aux = -.5 * contract('p,xpq->pqx', rhoj0_P, int2c_ip1ip2*rhoj0_P).reshape(naux, naux,3,3)
        if with_k:
            hk_aux_aux = -.5 * contract('xpq,pq->pqx', int2c_ip1ip2, rho2c_0).reshape(naux,naux,3,3)
        t1 = log.timer_debug1('intermediate variables with int2c_*', *t1)
        int2c_ip1ip2 = None

    cupy.get_default_memory_pool().free_all_blocks()
    release_gpu_stack()
    # aux-aux pair
    if hessobj.auxbasis_response > 1:
        wj0_10 = contract('ypq,p->ypq', int2c_ip1, rhoj0_P)
        int2c_ip1_inv = contract('yqp,pr->yqr', int2c_ip1, int2c_inv)

        rhoj0_10 = contract('p,xpq->xpq', rhoj0_P, int2c_ip1_inv)     # (1|0)(0|00)
        hj_aux_aux += .5 * contract('xpr,yqr->pqxy', rhoj0_10, wj0_10)  # (00|0)(1|0), (0|1)(0|00)
        hj_aux_aux +=      contract('xpq,yq->pqxy',  rhoj0_10, wj0_01)  # (00|0)(1|0), (1|0)(0|00)
        rhoj0_10 = rhoj0_P = None

        rhoj1 = contract('px,pq->xpq', wj_ip2, int2c_inv)             # (0|0)(1|00)
        hj_aux_aux -=      contract('xpq,yq->pqxy',  rhoj1,    wj0_01)  # (00|1),      (1|0)(0|00)
        hj_aux_aux += .5 * contract('xpq,qy->pqxy',  rhoj1,    wj_ip2)  # (00|1),      (1|00)
        hj_aux_aux -=      contract('xpr,yqr->pqxy', rhoj1,    wj0_10)  # (00|1),      (0|1)(0|00)
        wj0_10 = rhoj1 = wj_ip2 = None

        rhoj0_01 = contract('xp,pq->xpq', wj0_01, int2c_inv)          # (0|1)(0|00)
        hj_aux_aux += .5 * contract('xpq,yq->pqxy',  rhoj0_01, wj0_01)  # (00|0)(0|1), (1|0)(0|00)
        wj0_01 = rhoj0_01 = None

        if with_k:
            rho2c_10 = contract('rijx,qij->rqx', wka_ip2_P__, rhok0a_P__)
            rho2c_10+= contract('rijx,qij->rqx', wkb_ip2_P__, rhok0b_P__)
            rhok0a_P__ = rhok0b_P__ = None


            rho2c_11 = contract('pijx,qijy->pqxy', wka_ip2_P__, wka_ip2_P__)
            rho2c_11+= contract('pijx,qijy->pqxy', wkb_ip2_P__, wkb_ip2_P__)
            wka_ip2_P__ = wkb_ip2_P__ = None
            hk_aux_aux += .5 * contract('pqxy,pq->pqxy', rho2c_11, int2c_inv)      # (00|1)(1|00)
            rho2c_11 = None

            rho2c0_10 = contract('xpq,qr->xpr', int2c_ip1, rho2c_0)              # (00|0)(0|1)_(0|00)
            hk_aux_aux +=      contract('xpq,yqp->pqxy', int2c_ip1_inv, rho2c0_10) # (00|0)(1|0)(1|0)(0|00)

            rho2c0_11 = contract('xpr,yqr->pqxy', rho2c0_10, int2c_ip1)          # (00|0)(0|1)_(1|0)(0|00)
            hk_aux_aux += .5 * contract('pqxy,pq->pqxy', rho2c0_11, int2c_inv)     # (00|0)(0|1)(1|0)(0|00)
            rho2c0_11 = rho2c0_10 = None

            rho2c1_10 = contract('xpr,qry->pqxy', int2c_ip1, rho2c_10)           # (00|1)_(1|0)(0|00)
            hk_aux_aux -=      contract('pqxy,pq->pqxy', rho2c1_10, int2c_inv)     # (00|1)(1|0)(0|00)
            rho2c1_10 = None

            int2c_ip_ip = contract('xpr,ysr->xyps', int2c_ip1_inv, int2c_ip1)    # (0|1)(0|0)(1|0)
            hk_aux_aux += .5 * contract('xypq,pq->pqxy', int2c_ip_ip, rho2c_0)     # (00|0)(1|0)(0|1)(0|00)
            int2c_ip_ip = rho2c_0 = None

            hk_aux_aux -=      contract('pqx,yqp->pqxy', rho2c_10, int2c_ip1_inv)  # (00|1)(0|1)(0|00)
            rho2c_10= int2c_ip1_inv = None
    t1 = log.timer_debug1('contract int2c_*', *t1)

    ao_idx = np.argsort(intopt.ao_idx)
    aux_idx = np.argsort(intopt.aux_ao_idx)
    rev_ao_ao = cupy.ix_(ao_idx, ao_idx)
    #dm0 = dm0[rev_ao_ao]
    hj_ao_diag = hj_ao_diag[ao_idx]
    hj_ao_ao = hj_ao_ao[rev_ao_ao]
    if hessobj.auxbasis_response:
        rev_ao_aux = cupy.ix_(ao_idx, aux_idx)
        hj_ao_aux = hj_ao_aux[rev_ao_aux]
    if hessobj.auxbasis_response > 1:
        rev_aux_aux = cupy.ix_(aux_idx, aux_idx)
        hj_aux_diag = hj_aux_diag[aux_idx]
        hj_aux_aux = hj_aux_aux[rev_aux_aux]

    if with_k:
        hk_ao_diag = hk_ao_diag[ao_idx]
        hk_ao_ao = hk_ao_ao[rev_ao_ao]
        if hessobj.auxbasis_response:
            hk_ao_aux = hk_ao_aux[rev_ao_aux]
        if hessobj.auxbasis_response > 1:
            hk_aux_diag = hk_aux_diag[aux_idx]
            hk_aux_aux = hk_aux_aux[rev_aux_aux]

    mocca = mocca[ao_idx]
    moccb = moccb[ao_idx]

    #======================================== sort AO end ===========================================
    # Energy weighted density matrix
    # pi,qi,i->pq
    dme0 = cupy.dot(mocca, (mocca * mo_ea).T)
    dme0+= cupy.dot(moccb, (moccb * mo_eb).T)
    hcore_deriv = rhf_hess.hcore_generator(hessobj, mol)
    hess_nuc_elec = rhf_hess.hess_nuc_elec(mol, dm0.get())

    # ------------------------------------
    #      overlap matrix contributions
    # ------------------------------------
    s1aa, s1ab, _ = rhf_hess.get_ovlp(mol)
    s1aa = cupy.asarray(s1aa, order='C')
    s1ab = cupy.asarray(s1ab, order='C')
    h1aa = 2.0*contract('xypq,pq->pxy', s1aa, dme0)
    h1ab = 2.0*contract('xypq,pq->pqxy', s1ab, dme0)
    #s1aa = s1ab = dme0 = None

    # -----------------------------------------
    #        collecting all
    # -----------------------------------------
    hk_ao_ao *= 2.0
    e1 = cupy.zeros([len(atmlst),len(atmlst),3,3])
    ej = cupy.zeros([len(atmlst),len(atmlst),3,3])
    ek = cupy.zeros([len(atmlst),len(atmlst),3,3])
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        e1[i0,i0] -= cupy.sum(h1aa[p0:p1], axis=0)
        ej[i0,i0] += cupy.sum(hj_ao_diag[p0:p1,:,:], axis=0)
        if with_k:
            ek[i0,i0] += cupy.sum(hk_ao_diag[p0:p1,:,:], axis=0)
        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            ej[i0,j0] += cupy.sum(hj_ao_ao[p0:p1,q0:q1], axis=[0,1])
            e1[i0,j0] -= cupy.sum(h1ab[p0:p1,q0:q1], axis=[0,1])
            if with_k:
                ek[i0,j0] += cupy.sum(hk_ao_ao[p0:p1,q0:q1], axis=[0,1])
            h1ao = hcore_deriv(ia, ja)
            e1[i0,j0] += contract('xypq,pq->xy', cupy.asarray(h1ao), dm0)
            e1[i0,j0] += hess_nuc_elec[:,:,ia,ja]

        #
        # The first order RI basis response
        #
        if hessobj.auxbasis_response:
            for j0, (q0, q1) in enumerate(auxslices[:,2:]):
                _ej = cupy.sum(hj_ao_aux[p0:p1,q0:q1], axis=[0,1])
                if hessobj.auxbasis_response > 1:
                    ej[i0,j0] += _ej * 2
                    ej[j0,i0] += _ej.T * 2
                else:
                    ej[i0,j0] += _ej
                    ej[j0,i0] += _ej.T
                if with_k:
                    _ek = cupy.sum(hk_ao_aux[p0:p1,q0:q1], axis=[0,1])
                    if hessobj.auxbasis_response > 1:
                        ek[i0,j0] += _ek * 2
                        ek[j0,i0] += _ek.T * 2
                    else:
                        ek[i0,j0] += _ek
                        ek[j0,i0] += _ek.T
        #
        # The second order RI basis response
        #
        if hessobj.auxbasis_response > 1:
            shl0, shl1, p0, p1 = auxslices[ia]
            ej[i0,i0] += cupy.sum(hj_aux_diag[p0:p1], axis=0)
            if with_k:
                ek[i0,i0] += cupy.sum(hk_aux_diag[p0:p1], axis=0)
            for j0, (q0, q1) in enumerate(auxslices[:,2:]):
                _ej = cupy.sum(hj_aux_aux[p0:p1,q0:q1], axis=[0,1])
                ej[i0,j0] += _ej
                ej[j0,i0] += _ej.T
                if with_k:
                    _ek = cupy.sum(hk_aux_aux[p0:p1,q0:q1], axis=[0,1])
                    ek[i0,j0] += _ek
                    ek[j0,i0] += _ek.T
    for i0, ia in enumerate(atmlst):
        for j0 in range(i0):
            e1[j0,i0] = e1[i0,j0].T
            ej[j0,i0] = ej[i0,j0].T
            ek[j0,i0] = ek[i0,j0].T
    t1 = log.timer_debug1('hcore contribution', *t1)
    log.timer('UHF partial hessian', *time0)
    return e1, ej, ek


def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    h1aoa = [None] * mol.natm
    h1aob = [None] * mol.natm
    for ia, h1, vj1, vk1 in _gen_jk(hessobj, mo_coeff, mo_occ, chkfile,
                                    atmlst, verbose, True):
        h1a, h1b = h1
        vj1a, vj1b = vj1
        vk1a, vk1b = vk1

        h1aoa[ia] = h1a + vj1a - vk1a
        h1aob[ia] = h1b + vj1b - vk1b
    return (h1aoa, h1aob)

def _gen_jk(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None,
            verbose=None, with_k=True, omega=None):
    log = logger.new_logger(hessobj, verbose)
    t0 = log.init_timer()
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    # FIXME
    with_k = True
    mo_coeff = cupy.asarray(mo_coeff, order='C')
    mo_occ = cupy.asarray(mo_occ, order='C')

    mf = hessobj.base
    #auxmol = hessobj.base.with_df.auxmol
    auxmol = df.addons.make_auxmol(mol, auxbasis=mf.with_df.auxbasis)
    naux = auxmol.nao
    aoslices = mol.aoslice_by_atom()
    auxslices = auxmol.aoslice_by_atom()

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = cupy.dot(mocca, mocca.T)
    dm0b = cupy.dot(moccb, moccb.T)

    if omega and omega > 1e-10:
        with auxmol.with_range_coulomb(omega):
            int2c = auxmol.intor('int2c2e', aosym='s1')
    else:
        int2c = auxmol.intor('int2c2e', aosym='s1')
    int2c = cupy.asarray(int2c, order='C')
    # ======================= sorted AO begin ======================================
    intopt = int3c2e.VHFOpt(mol, auxmol, 'int2e')
    intopt.build(mf.direct_scf_tol, 
                 diag_block_with_triu=True, 
                 aosym=False, 
                 group_size_aux=BLKSIZE, 
                 group_size=BLKSIZE)
    ao_idx = intopt.ao_idx
    aux_ao_idx = intopt.aux_ao_idx

    mocca = mocca[ao_idx, :]
    moccb = moccb[ao_idx, :]
    mo_coeff = mo_coeff[:, ao_idx,:]
    dm0a = take_last2d(dm0a, ao_idx)
    dm0b = take_last2d(dm0b, ao_idx)
    dm0 = dm0a + dm0b

    int2c = take_last2d(int2c, aux_ao_idx)
    solve_j2c = _gen_metric_solver(int2c)
    int2c = None

    fn = int3c2e.get_int3c2e_wjk
    dm0_tag = tag_array(dm0, occ_coeff=mocca)
    wj, wka_Pl_ = fn(mol, auxmol, dm0_tag, omega=omega)
    dm0_tag = tag_array(dm0, occ_coeff=moccb)
    wj, wkb_Pl_ = fn(mol, auxmol, dm0_tag, omega=omega)
    rhoj0 = solve_j2c(wj)
    wj = None

    if isinstance(wka_Pl_, cupy.ndarray):
        rhok0a_Pl_ = solve_j2c(wka_Pl_)
    else:
        rhok0a_Pl_ = np.empty_like(wka_Pl_)
        for p0, p1 in lib.prange(0,nao,64):
            wk_tmp = cupy.asarray(wka_Pl_[:,p0:p1])
            rhok0a_Pl_[:,p0:p1] = solve_j2c(wk_tmp).get()
        wk_tmp = None

    if isinstance(wkb_Pl_, cupy.ndarray):
        rhok0b_Pl_ = solve_j2c(wkb_Pl_)
    else:
        rhok0b_Pl_ = np.empty_like(wkb_Pl_)
        for p0, p1 in lib.prange(0,nao,64):
            wk_tmp = cupy.asarray(wkb_Pl_[:,p0:p1])
            rhok0b_Pl_[:,p0:p1] = solve_j2c(wk_tmp).get()
        wk_tmp = None
    wka_Pl_ = wkb_Pl_ = None

    # -----------------------------
    # int3c_ip1 contributions
    # ------------------------------
    cupy.get_default_memory_pool().free_all_blocks()
    fn = int3c2e.get_int3c2e_ip1_vjk
    dm0_tag = tag_array(dm0, occ_coeff=mocca)
    vj1_buf, vk1a_buf, vj1a_ao, vk1a_ao = fn(intopt, rhoj0, rhok0a_Pl_, dm0_tag, aoslices, omega=omega)
    dm0_tag = tag_array(dm0, occ_coeff=moccb)
    vj1_buf, vk1b_buf, vj1b_ao, vk1b_ao = fn(intopt, rhoj0, rhok0b_Pl_, dm0_tag, aoslices, omega=omega)
    rev_ao_idx = np.argsort(ao_idx)
    vj1_buf = take_last2d(vj1_buf, rev_ao_idx)
    vk1a_buf = take_last2d(vk1a_buf, rev_ao_idx)
    vk1b_buf = take_last2d(vk1b_buf, rev_ao_idx)

    vj1a_int3c = -contract('nxiq,ip->nxpq', vj1a_ao, mo_coeff[0])
    vj1b_int3c = -contract('nxiq,ip->nxpq', vj1b_ao, mo_coeff[1])
    vk1a_int3c = -contract('nxiq,ip->nxpq', vk1a_ao, mo_coeff[0])
    vk1b_int3c = -contract('nxiq,ip->nxpq', vk1b_ao, mo_coeff[1])
    vj1a_ao = vj1b_ao = vk1a_ao = vk1b_ao = None
    t0 = log.timer_debug1('Fock matrix due to int3c2e_ip1', *t0)

    # --------------------------
    #  int3c_ip2 contribution
    # --------------------------
    cupy.get_default_memory_pool().free_all_blocks()
    if hessobj.auxbasis_response:
        fn = int3c2e.get_int3c2e_ip2_vjk
        dm0_tag = tag_array(dm0, occ_coeff=mocca)
        vj1a_int3c_ip2, vk1a_int3c_ip2 = fn(intopt, rhoj0, rhok0a_Pl_, dm0_tag, auxslices, omega=omega)
        dm0_tag = tag_array(dm0, occ_coeff=moccb)
        vj1b_int3c_ip2, vk1b_int3c_ip2 = fn(intopt, rhoj0, rhok0b_Pl_, dm0_tag, auxslices, omega=omega)

        # Responses due to int2c2e_ip1
        if omega and omega > 1e-10:
            with auxmol.with_range_coulomb(omega):
                int2c_ip1 = auxmol.intor('int2c2e_ip1', aosym='s1')
        else:
            int2c_ip1 = auxmol.intor('int2c2e_ip1', aosym='s1')
        int2c_ip1 = cupy.asarray(int2c_ip1, order='C')
        int2c_ip1 = take_last2d(int2c_ip1, aux_ao_idx)

        # generate rhok0_P__
        if isinstance(rhok0a_Pl_, cupy.ndarray):
            rhok0a_P__ = contract('pio,ir->pro', rhok0a_Pl_, mocca)
        else:
            naux = len(aux_ao_idx)
            nocc = mocca.shape[1]
            rhok0a_P__ = cupy.empty([naux,nocc,nocc])
            for p0, p1 in lib.prange(0,naux,64):
                rhok0_Pl_tmp = cupy.asarray(rhok0a_Pl_[p0:p1])
                rhok0a_P__[p0:p1] = contract('pio,ir->pro', rhok0_Pl_tmp, mocca)
            rhok0_Pl_tmp = None

        # generate rhok0_P__
        if isinstance(rhok0b_Pl_, cupy.ndarray):
            rhok0b_P__ = contract('pio,ir->pro', rhok0b_Pl_, moccb)
        else:
            naux = len(aux_ao_idx)
            nocc = moccb.shape[1]
            rhok0b_P__ = cupy.empty([naux,nocc,nocc])
            for p0, p1 in lib.prange(0,naux,64):
                rhok0_Pl_tmp = cupy.asarray(rhok0b_Pl_[p0:p1])
                rhok0b_P__[p0:p1] = contract('pio,ir->pro', rhok0_Pl_tmp, moccb)
            rhok0_Pl_tmp = None

        wj0_10 = contract('xpq,q->xp', int2c_ip1, rhoj0)
        wk0a_10_P__ = contract('xqp,pro->xqro', int2c_ip1, rhok0a_P__)
        wk0b_10_P__ = contract('xqp,pro->xqro', int2c_ip1, rhok0b_P__)

        aux2atom = int3c2e.get_aux2atom(intopt, auxslices)
        mem_avail = get_avail_mem()
        nocc = mocca.shape[1] + moccb.shape[1]
        blksize = int(0.2*mem_avail/(3*naux*nocc*8)/ALIGNED) * ALIGNED
        log.debug(f'GPU Memory {mem_avail/GB:.1f} GB available, block size {blksize}')
        if blksize < ALIGNED:
            raise RuntimeError('Not enough memory to compute int3c2e_ip2')
        
        for p0, p1 in lib.prange(0,nao,64):
            rhoka_tmp = cupy.asarray(rhok0a_Pl_[:,p0:p1])
            rhokb_tmp = cupy.asarray(rhok0b_Pl_[:,p0:p1])
            vj1a_tmp = contract('pio,xp->xpio', rhoka_tmp, wj0_10)
            vj1b_tmp = contract('pio,xp->xpio', rhokb_tmp, wj0_10)

            wk0a_10_Pl_ = contract('xqp,pio->xqio', int2c_ip1, rhoka_tmp)
            wk0b_10_Pl_ = contract('xqp,pio->xqio', int2c_ip1, rhokb_tmp)
            vj1a_tmp += contract('xpio,p->xpio', wk0a_10_Pl_, rhoj0)
            vj1b_tmp += contract('xpio,p->xpio', wk0b_10_Pl_, rhoj0)
            vj1a_int3c_ip2[:,:,p0:p1] += contract('xpio,pa->axio', vj1a_tmp, aux2atom)
            vj1b_int3c_ip2[:,:,p0:p1] += contract('xpio,pa->axio', vj1b_tmp, aux2atom)
            vj1a_tmp = vj1b_tmp = None
            if with_k:
                vk1a_tmp = contract('xpio,pro->xpir', wk0a_10_Pl_, rhok0a_P__)
                vk1a_tmp += contract('xpro,pir->xpio', wk0a_10_P__, rhoka_tmp)
                vk1b_tmp = contract('xpio,pro->xpir', wk0b_10_Pl_, rhok0b_P__)
                vk1b_tmp += contract('xpro,pir->xpio', wk0b_10_P__, rhokb_tmp)

                vk1a_int3c_ip2[:,:,p0:p1] += contract('xpio,pa->axio', vk1a_tmp, aux2atom)
                vk1b_int3c_ip2[:,:,p0:p1] += contract('xpio,pa->axio', vk1b_tmp, aux2atom)
                vk1a_tmp = vk1b_tmp = None
            wk0a_10_Pl_ = wk0b_10_Pl_ = rhoka_tmp = rhokb_tmp = None
        wj0_10 = wk0a_10_P__ = wk0b_10_P__ = rhok0a_P__ =rhok0b_P__ = int2c_ip1 = None
        rhoj0 = rhok0a_Pl_ = rhok0b_Pl_ = None
        aux2atom = None

        vj1a_int3c += contract('nxiq,ip->nxpq', vj1a_int3c_ip2, mo_coeff[0])
        vj1b_int3c += contract('nxiq,ip->nxpq', vj1b_int3c_ip2, mo_coeff[1])
        if with_k:
            vk1a_int3c += contract('nxiq,ip->nxpq', vk1a_int3c_ip2, mo_coeff[0])
            vk1b_int3c += contract('nxiq,ip->nxpq', vk1b_int3c_ip2, mo_coeff[1])
        vk1a_int3c_ip2 = vk1b_int3c_ip2 = None
        t0 = log.timer_debug1('Fock matrix due to int3c2e_ip2', *t0)

    mocca = mocca[rev_ao_idx]
    moccb = moccb[rev_ao_idx]
    mo_coeff = mo_coeff[:,rev_ao_idx]
    release_gpu_stack()

    # ========================== sorted AO end ================================
    def _ao2mo(mat, mocc, mo):
        tmp = contract('xij,jo->xio', mat, mocc)
        return contract('xik,ip->xpk', tmp, mo)

    gobj = hessobj.base.nuc_grad_method()
    grad_hcore_a = rhf_grad.get_grad_hcore(gobj, mo_coeff[0], mo_occ[0])
    grad_hcore_b = rhf_grad.get_grad_hcore(gobj, mo_coeff[1], mo_occ[1])
    cupy.get_default_memory_pool().free_all_blocks()

    vk1a = vk1b = None
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        vj1_ao = cupy.zeros([3,nao,nao])
        vk1a_ao = cupy.zeros([3,nao,nao])
        vk1b_ao = cupy.zeros([3,nao,nao])

        vj1_ao[:,p0:p1,:] -= vj1_buf[:,p0:p1,:]
        vj1_ao[:,:,p0:p1] -= vj1_buf[:,p0:p1,:].transpose(0,2,1)
        if with_k:
            vk1a_ao[:,p0:p1,:] -= vk1a_buf[:,p0:p1,:]
            vk1a_ao[:,:,p0:p1] -= vk1a_buf[:,p0:p1,:].transpose(0,2,1)
            vk1b_ao[:,p0:p1,:] -= vk1b_buf[:,p0:p1,:]
            vk1b_ao[:,:,p0:p1] -= vk1b_buf[:,p0:p1,:].transpose(0,2,1)

        h1a = grad_hcore_a[:,i0]
        h1b = grad_hcore_b[:,i0]
        vj1a = vj1a_int3c[ia] + _ao2mo(vj1_ao, mocca, mo_coeff[0])
        vj1b = vj1b_int3c[ia] + _ao2mo(vj1_ao, moccb, mo_coeff[1])
        if with_k:
            vk1a = vk1a_int3c[ia] + _ao2mo(vk1a_ao, mocca, mo_coeff[0])
            vk1b = vk1b_int3c[ia] + _ao2mo(vk1b_ao, moccb, mo_coeff[1])
        yield ia, (h1a, h1b), (vj1a, vj1b), (vk1a, vk1b)

class Hessian(uhf_hess.Hessian):
    '''Non-relativistic restricted Hartree-Fock hessian'''

    from gpu4pyscf.lib.utils import to_gpu, device

    def __init__(self, mf):
        uhf_hess.Hessian.__init__(self, mf)

    auxbasis_response = 1
    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1
    kernel = rhf_hess.kernel
    hess = kernel
