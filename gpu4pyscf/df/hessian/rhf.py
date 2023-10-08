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
from pyscf import lib, df
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.lib.cupy_helper import contract, tag_array, release_gpu_stack
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib import logger

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
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    mf.with_df._cderi = None
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    mo_coeff = cupy.asarray(mo_coeff)
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    mocc_2 = cupy.einsum('pi,i->pi', mocc, mo_occ[mo_occ>0]**.5)
    dm0 = cupy.dot(mocc, mocc.T) * 2

    hcore_deriv = hessobj.hcore_generator(mol)

    # ------------------------------------
    #      overlap matrix contributions
    # ------------------------------------
    s1aa, s1ab, _ = rhf_hess.get_ovlp(mol)
    s1aa = cupy.asarray(s1aa)
    s1ab = cupy.asarray(s1ab)

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
    intopt.build(mf.direct_scf_tol, diag_block_with_triu=True, aosym=False, group_size_aux=128, group_size=128)
    sph_ao_idx = intopt.sph_ao_idx
    sph_aux_idx = intopt.sph_aux_idx

    mocc_2 = mocc_2[sph_ao_idx, :]
    dm0 = dm0[cupy.ix_(sph_ao_idx, sph_ao_idx)]
    dm0_tag = tag_array(dm0, occ_coeff=mocc_2)

    int2c = cupy.asarray(int2c)
    int2c = int2c[cupy.ix_(sph_aux_idx, sph_aux_idx)]
    int2c_inv = cupy.linalg.pinv(int2c, rcond=1e-12)

    int2c_ip1 = cupy.asarray(int2c_ip1)
    int2c_ip1 = int2c_ip1[cupy.ix_(np.arange(3), sph_aux_idx, sph_aux_idx)]
    int2c_ip1_inv = contract('yqp,pr->yqr', int2c_ip1, int2c_inv)

    hj_ao_ao = cupy.zeros([nao,nao,3,3])
    hk_ao_ao = cupy.zeros([nao,nao,3,3])
    if hessobj.auxbasis_response:
        hj_ao_aux = cupy.zeros([nao,naux,3,3])
        hk_ao_aux = cupy.zeros([nao,naux,3,3])

    #  int3c contributions
    wj, wk_Pl_ = int3c2e.get_int3c2e_wjk(mol, auxmol, dm0_tag, omega=omega)
    rhoj0_P = contract('pq,q->p', int2c_inv, wj)
    wk_P__ = contract('Lio,ir->Lro', wk_Pl_, mocc_2)
    rhok0_P__ = contract('pq,qij->pij', int2c_inv, wk_P__)
    wj = wk_P__ = wk_Pl_ = None
    t1 = log.timer_debug1('intermediate variables with int3c2e', *t1)

    # int3c_ip2 contributions
    wj_ip2, wk_ip2_P__ = int3c2e.get_int3c2e_ip2_wjk(intopt, dm0_tag, omega=omega)
    t1 = log.timer_debug1('interdeidate variables with int3c2e_ip2', *t1)

    #  int3c_ip1 contributions
    wj1_P, wk1_Pko = int3c2e.get_int3c2e_ip1_wjk(intopt, dm0_tag, omega=omega)
    rhoj1_P = contract('pq,ipx->iqx', int2c_inv, wj1_P)

    hj_ao_ao += 4.0*contract('ipx,jpy->ijxy', rhoj1_P, wj1_P)   # (10|0)(0|0)(0|01)
    wj1_P = None
    if hessobj.auxbasis_response:
        wj0_01 = contract('ypq,q->yp', int2c_ip1, rhoj0_P)
        wj1_01 = contract('yqp,ipx->iqxy', int2c_ip1, rhoj1_P)
        hj_ao_aux += contract('ipx,py->ipxy', rhoj1_P, wj_ip2)   # (10|0)(1|00)
        hj_ao_aux -= contract('ipx,yp->ipxy', rhoj1_P, wj0_01)   # (10|0)(1|0)(0|00)
        hj_ao_aux -= contract('q,iqxy->iqxy', rhoj0_P, wj1_01)   # (10|0)(0|1)(0|00)
        wj1_01 = None

    if with_k:
        for p0, p1 in lib.prange(0,naux,64):
            rhok1_Pko = contract('pq,iqox->pxio', int2c_inv[p0:p1], wk1_Pko)
            # (10|0)(0|10) without response of RI basis
            vk2_ip1_ip1 = cupy.einsum('ipox,pyko->kixy', wk1_Pko[:,p0:p1], rhok1_Pko)
            hk_ao_ao += cupy.einsum('kixy,ki->ikxy', vk2_ip1_ip1, dm0)
            vk2_ip1_ip1 = None
            # (10|0)(0|01) without response of RI basis
            bra = cupy.einsum('pyko,io->ikpy', rhok1_Pko, mocc_2)
            ket = cupy.einsum('ipox,ko->ipkx', wk1_Pko[:,p0:p1], mocc_2)
            hk_ao_ao += cupy.einsum('ikpy,ipkx->ikxy', bra, ket)
            bra = ket = None
            if hessobj.auxbasis_response:
                # (10|0)(1|00)
                wk_ip2_Ipo = cupy.einsum('porx,io->ipxr', wk_ip2_P__[p0:p1], mocc_2)
                hk_ao_aux[:,p0:p1] += cupy.einsum('pxio,ipyo->ipxy', rhok1_Pko, wk_ip2_Ipo)
                wk_ip2_Ipo = None
                # (10|0)(1|0)(0|00)
                wk1_P__ = cupy.einsum('ypq,qor->ypor', int2c_ip1[:,p0:p1], rhok0_P__)
                wk1_P_I = cupy.einsum('ypor,ir->ypoi', wk1_P__, mocc_2)
                hk_ao_aux[:,p0:p1] -= cupy.einsum('pxio,ypoi->ipxy', rhok1_Pko, wk1_P_I)
                wk1_P_I = wk1_P__ = None
                # (10|0)(0|1)(0|00)
                int2c_tmp = cupy.asarray(int2c_ip1_inv[:,p0:p1], order='C')
                wk1_I = contract('yqp,ipox->qxyio', int2c_tmp, wk1_Pko)
                rhok0_tmp = cupy.einsum('qor,ir->qoi', rhok0_P__[p0:p1], mocc_2)
                hk_ao_aux[:,p0:p1] -= cupy.einsum('qoi,qxyio->iqxy', rhok0_tmp, wk1_I)
                wk1_I = rhok0_tmp = None
        wk1_Pko = rhok1_Pko = int2c_tmp = None
    t1 = log.timer_debug1('intermediate variables with int3c2e_ip1', *t1)

    cupy.get_default_memory_pool().free_all_blocks()
    #  int3c_ipip1 contributions
    hj_ao_diag, hk_ao_diag = int3c2e.get_int3c2e_ipip1_hjk(intopt, rhoj0_P, rhok0_P__, dm0_tag, omega=omega)
    hj_ao_diag *= 2.0
    t1 = log.timer_debug1('intermediate variables with int3c2e_ipip1', *t1)

    #  int3c_ipvip1 contributions
    # (11|0), (0|00) without response of RI basis
    hj, hk = int3c2e.get_int3c2e_ipvip1_hjk(intopt, rhoj0_P, rhok0_P__, dm0_tag, omega=omega)
    hj_ao_ao += 2.0*hj
    hk_ao_ao += hk
    t1 = log.timer_debug1('intermediate variables with int3c2e_ipvip1', *t1)

    #  int3c_ip1ip2 contributions
    # (10|1), (0|0)(0|00)
    if hessobj.auxbasis_response:
        hj, hk = int3c2e.get_int3c2e_ip1ip2_hjk(intopt, rhoj0_P, rhok0_P__, dm0_tag, omega=omega)
        hj_ao_aux += hj
        hk_ao_aux += hk
        t1 = log.timer_debug1('intermediate variables with int3c2e_ip1ip2', *t1)

    #  int3c_ipip2 contributions
    if hessobj.auxbasis_response > 1:
        # (00|2), (0|0)(0|00)
        hj, hk = int3c2e.get_int3c2e_ipip2_hjk(intopt, rhoj0_P, rhok0_P__, dm0_tag, omega=omega)
        hj_aux_diag = hj
        hk_aux_diag = .5*hk
        t1 = log.timer_debug1('intermediate variables with int3c2e_ipip2', *t1)

    # int2c contributions
    if hessobj.auxbasis_response > 1:
        aux_aux_9 = cupy.ix_(np.arange(9), sph_aux_idx, sph_aux_idx)
        if omega and omega > 1e-10:
            with auxmol.with_range_coulomb(omega):
                int2c_ipip1 = auxmol.intor('int2c2e_ipip1', aosym='s1')
        else:
            int2c_ipip1 = auxmol.intor('int2c2e_ipip1', aosym='s1')
        int2c_ipip1 = cupy.asarray(int2c_ipip1)
        int2c_ipip1 = int2c_ipip1[aux_aux_9]
        rhoj2c_P = cupy.einsum('xpq,q->xp', int2c_ipip1, rhoj0_P)
        # (00|0)(2|0)(0|00)
        hj_aux_diag -= cupy.einsum('p,xp->px', rhoj0_P, rhoj2c_P).reshape(-1,3,3)
        if with_k:
            rho2c_0 = cupy.einsum('pij,qji->pq', rhok0_P__, rhok0_P__)
            hk_aux_diag -= .5 * cupy.einsum('pq,xpq->px', rho2c_0, int2c_ipip1).reshape(-1,3,3)
        int2c_ipip1 = None

        if omega and omega > 1e-10:
            with auxmol.with_range_coulomb(omega):
                int2c_ip1ip2 = auxmol.intor('int2c2e_ip1ip2', aosym='s1')
        else:
            int2c_ip1ip2 = auxmol.intor('int2c2e_ip1ip2', aosym='s1')
        int2c_ip1ip2 = cupy.asarray(int2c_ip1ip2)
        int2c_ip1ip2 = int2c_ip1ip2[aux_aux_9]
        hj_aux_aux = -.5 * cupy.einsum('p,xpq,q->pqx', rhoj0_P, int2c_ip1ip2, rhoj0_P).reshape(naux, naux,3,3)
        if with_k:
            hk_aux_aux = -.5 * cupy.einsum('xpq,pq->pqx', int2c_ip1ip2, rho2c_0).reshape(naux,naux,3,3)
        t1 = log.timer_debug1('intermediate variables with int2c_*', *t1)
        int2c_ip1ip2 = aux_aux_9 = None

    # aux-aux pair
    if hessobj.auxbasis_response > 1:
        wj0_10 = cupy.einsum('ypq,p->ypq', int2c_ip1, rhoj0_P)
        rhoj1 = cupy.einsum('px,pq->xpq', wj_ip2, int2c_inv)             # (0|0)(1|00)
        rhoj0_01 = cupy.einsum('xp,pq->xpq', wj0_01, int2c_inv)          # (0|1)(0|00)
        rhoj0_10 = cupy.einsum('p,xpq->xpq', rhoj0_P, int2c_ip1_inv)     # (1|0)(0|00)

        hj_aux_aux += .5 * cupy.einsum('xpr,yqr->pqxy', rhoj0_10, wj0_10)  # (00|0)(1|0), (0|1)(0|00)
        hj_aux_aux -=      cupy.einsum('xpq,yq->pqxy',  rhoj1,    wj0_01)  # (00|1),      (1|0)(0|00)
        hj_aux_aux += .5 * cupy.einsum('xpq,qy->pqxy',  rhoj1,    wj_ip2)  # (00|1),      (1|00)
        hj_aux_aux -=      cupy.einsum('xpr,yqr->pqxy', rhoj1,    wj0_10)  # (00|1),      (0|1)(0|00)
        hj_aux_aux += .5 * cupy.einsum('xpq,yq->pqxy',  rhoj0_01, wj0_01)  # (00|0)(0|1), (1|0)(0|00)
        hj_aux_aux +=      cupy.einsum('xpq,yq->pqxy',  rhoj0_10, wj0_01)  # (00|0)(1|0), (1|0)(0|00)
        wj0_01 = wj0_10 = rhoj1 = rhoj0_01 = rhoj0_10 = rhoj0_P = wj_ip2 = None

        if with_k:
            rho2c_10 = cupy.einsum('rijx,qij->rqx', wk_ip2_P__, rhok0_P__)
            rho2c_11 = cupy.einsum('pijx,qijy->pqxy', wk_ip2_P__, wk_ip2_P__)
            rho2c0_10 = cupy.einsum('xpq,qr->xpr', int2c_ip1, rho2c_0)              # (00|0)(0|1)_(0|00)
            rho2c1_10 = cupy.einsum('xpr,qry->pqxy', int2c_ip1, rho2c_10)           # (00|1)_(1|0)(0|00)
            rho2c0_11 = cupy.einsum('xpr,yqr->pqxy', rho2c0_10, int2c_ip1)          # (00|0)(0|1)_(1|0)(0|00)
            int2c_ip_ip = cupy.einsum('xpr,ysr->xyps', int2c_ip1_inv, int2c_ip1)    # (0|1)(0|0)(1|0)

            hk_aux_aux += .5 * cupy.einsum('xypq,pq->pqxy', int2c_ip_ip, rho2c_0)     # (00|0)(1|0)(0|1)(0|00)
            hk_aux_aux += .5 * cupy.einsum('pqxy,pq->pqxy', rho2c0_11, int2c_inv)     # (00|0)(0|1)(1|0)(0|00)
            hk_aux_aux +=      cupy.einsum('xpq,yqp->pqxy', int2c_ip1_inv, rho2c0_10) # (00|0)(1|0)(1|0)(0|00)
            hk_aux_aux -=      cupy.einsum('pqxy,pq->pqxy', rho2c1_10, int2c_inv)     # (00|1)(1|0)(0|00)
            hk_aux_aux -=      cupy.einsum('pqx,yqp->pqxy', rho2c_10, int2c_ip1_inv)  # (00|1)(0|1)(0|00)
            hk_aux_aux += .5 * cupy.einsum('pqxy,pq->pqxy', rho2c_11, int2c_inv)      # (00|1)(1|00)
            rho2c_0 = rho2c_10 = rho2c_11 = rho2c0_10 = rho2c1_10 = rho2c0_11 = int2c_ip_ip = None
            wk_ip2_P__ = int2c_ip1_inv = None
    ao_idx = np.argsort(intopt.sph_ao_idx)
    aux_idx = np.argsort(intopt.sph_aux_idx)
    rev_ao_ao = cupy.ix_(ao_idx, ao_idx)
    dm0 = dm0[rev_ao_ao]
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

    #======================================== sort AO end ===========================================
    # Energy weighted density matrix
    dme0 = cupy.einsum('pi,qi,i->pq', mocc, mocc, mo_energy[mo_occ>0]) * 2.0
    # -----------------------------------------
    #        collecting all
    # -----------------------------------------
    e1 = cupy.zeros([len(atmlst),len(atmlst),3,3])
    ej = cupy.zeros([len(atmlst),len(atmlst),3,3])
    ek = cupy.zeros([len(atmlst),len(atmlst),3,3])
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        e1[i0,i0] -= cupy.einsum('xypq,pq->xy', s1aa[:,:,p0:p1], dme0[p0:p1]) * 2.0
        ej[i0,i0] += cupy.sum(hj_ao_diag[p0:p1,:,:], axis=0)
        if with_k:
            ek[i0,i0] += cupy.sum(hk_ao_diag[p0:p1,:,:], axis=0)
        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            ej[i0,j0] += cupy.sum(hj_ao_ao[p0:p1,q0:q1], axis=[0,1])
            e1[i0,j0] -= 2.0 * cupy.einsum('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], dme0[p0:p1,q0:q1])
            if with_k:
                ek[i0,j0] += cupy.sum(hk_ao_ao[p0:p1,q0:q1], axis=[0,1])
            h1ao = hcore_deriv(ia, ja)
            e1[i0,j0] += cupy.einsum('xypq,pq->xy', h1ao, dm0)
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
                        ek[i0,j0] += _ek
                        ek[j0,i0] += _ek.T
                    else:
                        ek[i0,j0] += _ek * .5
                        ek[j0,i0] += _ek.T * .5
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
                    ek[i0,j0] += _ek * .5
                    ek[j0,i0] += _ek.T * .5
    for i0, ia in enumerate(atmlst):
        for j0 in range(i0):
            e1[j0,i0] = e1[i0,j0].T
            ej[j0,i0] = ej[i0,j0].T
            ek[j0,i0] = ek[i0,j0].T

    log.timer('RHF partial hessian', *time0)
    return e1, ej, ek


def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    h1ao = [None] * mol.natm
    for ia, h1, vj1, vk1 in _gen_jk(hessobj, mo_coeff, mo_occ, chkfile,
                                    atmlst, verbose, True):
        h1 += vj1 - vk1 * .5
        if chkfile is None:
            h1ao[ia] = h1
        else:
            key = 'scf_f1ao/%d' % ia
            lib.chkfile.save(chkfile, key, h1)
    if chkfile is None:
        return h1ao
    else:
        return chkfile

def _gen_jk(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None,
            verbose=None, with_k=True, omega=None):
    log = logger.new_logger(hessobj, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    # FIXME
    with_k = True

    mf = hessobj.base
    #auxmol = hessobj.base.with_df.auxmol
    auxmol = df.addons.make_auxmol(mol, auxbasis=mf.with_df.auxbasis)
    aoslices = mol.aoslice_by_atom()
    auxslices = auxmol.aoslice_by_atom()

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = cupy.dot(mocc, mocc.T) * 2

    if omega and omega > 1e-10:
        with auxmol.with_range_coulomb(omega):
            int2c = auxmol.intor('int2c2e', aosym='s1')
    else:
        int2c = auxmol.intor('int2c2e', aosym='s1')
    int2c = cupy.asarray(int2c)
    # ======================= sorted AO begin ======================================
    intopt = int3c2e.VHFOpt(mol, auxmol, 'int2e')
    intopt.build(mf.direct_scf_tol, diag_block_with_triu=True, aosym=False, group_size_aux=64, group_size=64)
    sph_ao_idx = intopt.sph_ao_idx
    sph_aux_idx = intopt.sph_aux_idx
    rev_ao_idx = np.argsort(intopt.sph_ao_idx)

    mocc = mocc[sph_ao_idx, :]
    mo_coeff = mo_coeff[sph_ao_idx,:]
    dm0 = dm0[cupy.ix_(sph_ao_idx, sph_ao_idx)]
    dm0_tag = tag_array(dm0, occ_coeff=mocc)

    int2c = int2c[cupy.ix_(sph_aux_idx, sph_aux_idx)]
    int2c_inv = cupy.linalg.pinv(int2c, rcond=1e-12)

    wj, wk_Pl_ = int3c2e.get_int3c2e_wjk(mol, auxmol, dm0_tag, omega=omega)
    wk_P__ = contract('pio,ir->pro', wk_Pl_, mocc)
    rhoj0 = contract('pq,q->p', int2c_inv, wj)
    rhok0_Pl_ = contract('pq,qio->pio', int2c_inv, wk_Pl_)
    if with_k:
        rhok0_P__ = contract('pq,qij->pij', int2c_inv, wk_P__)
    wj = wk_Pl_ = wk_P__ = int2c_inv = int2c = None

    # int3c_ip1 contributions
    vj1_buf, vk1_buf, vj1_ao, vk1_ao = int3c2e.get_int3c2e_ip1_vjk(intopt, rhoj0, rhok0_Pl_, dm0_tag, aoslices, omega=omega)
    vj1_buf = vj1_buf[cupy.ix_(numpy.arange(3), rev_ao_idx, rev_ao_idx)]
    vk1_buf = vk1_buf[cupy.ix_(numpy.arange(3), rev_ao_idx, rev_ao_idx)]

    vj1_int3c_ip1 = -cupy.einsum('nxiq,ip->nxpq', vj1_ao, mo_coeff)
    vk1_int3c_ip1 = -cupy.einsum('nxiq,ip->nxpq', vk1_ao, mo_coeff)
    vj1_ao = vk1_ao = None
    t0 = log.timer_debug1('Fock matrix due to int3c2e_ip1', *t0)

    # --------------------------
    #  int3c_ip2 contribution
    # --------------------------
    cupy.get_default_memory_pool().free_all_blocks()
    if hessobj.auxbasis_response:
        aux2atom = int3c2e.get_aux2atom(intopt, auxslices)
        vj1_int3c_ip2, vk1_int3c_ip2 = int3c2e.get_int3c2e_ip2_vjk(intopt, rhoj0, rhok0_Pl_, dm0_tag, auxslices, omega=omega)
        # Responses due to int2c2e_ip1
        if omega and omega > 1e-10:
            with auxmol.with_range_coulomb(omega):
                int2c_ip1 = auxmol.intor('int2c2e_ip1', aosym='s1')
        else:
            int2c_ip1 = auxmol.intor('int2c2e_ip1', aosym='s1')
        int2c_ip1 = cupy.asarray(int2c_ip1)
        int2c_ip1 = int2c_ip1[cupy.ix_(np.arange(3), sph_aux_idx, sph_aux_idx)]

        wj0_10 = contract('xpq,q->xp', int2c_ip1, rhoj0)
        wk0_10_P__ = contract('xqp,pro->xqro', int2c_ip1, rhok0_P__)

        for p0, p1 in lib.prange(0,nao,64):
            vj1_tmp = cupy.einsum('pio,xp->xpio', rhok0_Pl_[:,p0:p1], wj0_10)

            wk0_10_Pl_ = cupy.einsum('xqp,pio->xqio', int2c_ip1, rhok0_Pl_[:,p0:p1])
            vj1_tmp += cupy.einsum('xpio,p->xpio', wk0_10_Pl_, rhoj0)
            vj1_int3c_ip2[:,:,p0:p1] += cupy.einsum('xpio,pa->axio', vj1_tmp, aux2atom)
            if with_k:
                vk1_tmp = 2.0 * cupy.einsum('xpio,pro->xpir', wk0_10_Pl_, rhok0_P__)
                vk1_tmp += 2.0 * cupy.einsum('xpro,pir->xpio', wk0_10_P__, rhok0_Pl_[:,p0:p1])
                vk1_int3c_ip2[:,:,p0:p1] += cupy.einsum('xpio,pa->axio', vk1_tmp, aux2atom)
        wj0_10 = wk0_10_P__ = rhok0_P__ = int2c_ip1 = None
        vj1_tmp = vk1_tmp = wk0_10_Pl_ = rhoj0 = rhok0_Pl_ = None
        aux2atom = None

        vj1_int3c_ip2 = contract('nxiq,ip->nxpq', vj1_int3c_ip2, mo_coeff)
        vk1_int3c_ip2 = contract('nxiq,ip->nxpq', vk1_int3c_ip2, mo_coeff)
        t0 = log.timer_debug1('Fock matrix due to int3c2e_ip2', *t0)

    mocc = mocc[rev_ao_idx]
    mo_coeff = mo_coeff[rev_ao_idx]
    release_gpu_stack()

    # ========================== sorted AO end ================================
    def _ao2mo(mat):
        tmp = cupy.einsum('xij,jo->xio', mat, mocc)
        return cupy.einsum('xik,ip->xpk', tmp, mo_coeff)

    vj1_int3c = vj1_int3c_ip1 + vj1_int3c_ip2
    vj1_int3c_ip1 = vj1_int3c_ip2 = None
    if with_k:
        vk1_int3c = vk1_int3c_ip1 + vk1_int3c_ip2
        vk1_int3c_ip1 = vk1_int3c_ip2 = None

    cupy.get_default_memory_pool().free_all_blocks()
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)
    vk1 = None
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        vj1_ao = cupy.zeros([3,nao,nao])
        vk1_ao = cupy.zeros([3,nao,nao])

        vj1_ao[:,p0:p1,:] -= vj1_buf[:,p0:p1,:]
        vj1_ao[:,:,p0:p1] -= vj1_buf[:,p0:p1,:].transpose(0,2,1)
        if with_k:
            vk1_ao[:,p0:p1,:] -= vk1_buf[:,p0:p1,:]
            vk1_ao[:,:,p0:p1] -= vk1_buf[:,p0:p1,:].transpose(0,2,1)

        h1 = hcore_deriv(ia)
        h1 = _ao2mo(h1)
        vj1 = vj1_int3c[ia] + _ao2mo(vj1_ao)
        if with_k:
            vk1 = vk1_int3c[ia] + _ao2mo(vk1_ao)
        yield ia, h1, vj1, vk1

class Hessian(rhf_hess.Hessian):
    '''Non-relativistic restricted Hartree-Fock hessian'''

    from gpu4pyscf.lib.utils import to_cpu, to_gpu, device

    def __init__(self, mf):
        self.auxbasis_response = 1
        rhf_hess.Hessian.__init__(self, mf)

    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1
