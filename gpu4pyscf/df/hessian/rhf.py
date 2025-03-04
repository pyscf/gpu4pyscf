# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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


import cupy
from pyscf import lib
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.lib.cupy_helper import (
    contract, tag_array, get_avail_mem, release_gpu_stack, pinv, copy_array)
from gpu4pyscf.df import int3c2e, df
from gpu4pyscf.lib import logger
from gpu4pyscf import __config__
from gpu4pyscf.df.grad.rhf import _gen_metric_solver
from gpu4pyscf.df.hessian import jk

LINEAR_DEP_THR = df.LINEAR_DEP_THR
BLKSIZE = 256
ALIGNED = getattr(__config__, 'ao_aligned', 32)
GB = 1024*1024*1024

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    e1, ej, ek = _partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                   atmlst, max_memory, verbose, True)
    return e1 + ej - ek

def _hk_ip1_ip1(rhok1_Pko, dm0, mocc_2):
    ''' hk contributions due to (10|0)(0|10) + (10|0)(0|01)
    '''
    nnz = rhok1_Pko.shape[0]
    nao = dm0.shape[0]
    hk_ao_ao = cupy.zeros([nao,nao,3,3])
    cupy.get_default_memory_pool().free_all_blocks()
    mem_avail = get_avail_mem()
    blksize = int(((mem_avail-hk_ao_ao.nbytes)*0.4/(nao*nao*3*8)/ALIGNED))*ALIGNED
    for k0, k1 in lib.prange(0,nnz,blksize):
        #rhok1_Pko_kslice = cupy.asarray(rhok1_Pko[k0:k1])
        rhok1_Pko_kslice = copy_array(rhok1_Pko[k0:k1])

        # (10|0)(0|10) without response of RI basis
        vk2_ip1_ip1 = contract('piox,pkoy->ikxy', rhok1_Pko_kslice, rhok1_Pko_kslice)
        hk_ao_ao += contract('ikxy,ik->ikxy', vk2_ip1_ip1, dm0)
        vk2_ip1_ip1 = None

        # (10|0)(0|01) without response of RI basis
        rhok1_Pkl_kslice = contract('piox,ko->pikx', rhok1_Pko_kslice, mocc_2)
        rhok1_Pko_kslice = None
        hk_ao_ao += contract('pikx,pkiy->ikxy', rhok1_Pkl_kslice, rhok1_Pkl_kslice)
        rhok1_Pkl_kslice = None
    return hk_ao_ao

def _partial_hess_ejk(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None,
                      max_memory=None, verbose=None, with_j=True, with_k=True, omega=None):
    '''Partial derivative
    '''
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = log.init_timer()
    mem_avail = get_avail_mem()
    log.debug('Partial Hessian with density fitting approximation')
    log.debug(f'Memory available {mem_avail/GB} GB')

    mol = hessobj.mol
    mf = hessobj.base
    mf.with_df._cderi = None
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    mo_coeff = cupy.asarray(mo_coeff, order='C')
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    mocc_2 = mocc * mo_occ[mo_occ>0]**.5
    dm0 = cupy.dot(mocc, mocc.T) * 2

    auxmol = df.addons.make_auxmol(mol, auxbasis=mf.with_df.auxbasis)
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
    intopt.build(mf.direct_scf_tol, diag_block_with_triu=True, aosym=False, verbose=0,
                 group_size=BLKSIZE, group_size_aux=BLKSIZE)
    naux = auxmol.nao
    mocc_2 = intopt.sort_orbitals(mocc_2, axis=[0])
    dm0 = intopt.sort_orbitals(dm0, axis=[0,1])
    dm0_tag = tag_array(dm0, occ_coeff=mocc_2)

    int2c = cupy.asarray(int2c, order='C')
    int2c = intopt.sort_orbitals(int2c, aux_axis=[0,1])
    solve_j2c = _gen_metric_solver(int2c)
    
    #  int3c contributions
    wj, wk_P__ = int3c2e.get_int3c2e_jk(mol, auxmol, dm0_tag, omega=omega)
    rhoj0_P = rhok0_P__ = None

    if with_j:
        rhoj0_P = solve_j2c(wj)
        wj = None
    if with_k:
        rhok0_P__ = solve_j2c(wk_P__)
        wk_P__ = None
    t1 = log.timer_debug1('intermediate variables with int3c2e', *t1)

    hj_ipip, hk_ipip = jk.get_int3c2e_hjk(intopt, rhoj0_P, rhok0_P__, dm0_tag,
                                          with_j=with_j, with_k=with_k, omega=omega,
                                          auxbasis_response=hessobj.auxbasis_response)
    t1 = log.timer_debug1('intermediate variables with int3c2e_ipip', *t1)

    # int3c_ip2 contributions
    wj_ip2, wk_ip2_P__ = int3c2e.get_int3c2e_ip2_wjk(intopt, dm0_tag, omega=omega)
    t1 = log.timer_debug1('intermediate variables with int3c2e_ip2', *t1)

    #  int3c_ip1 contributions
    wj1_P, wk1_Pko = int3c2e.get_int3c2e_ip1_wjk(intopt, dm0_tag, omega=omega)
    t1 = log.timer_debug1('intermediate variables with int3c2e_ip1', *t1)
    
    cupy.get_default_memory_pool().free_all_blocks()
    release_gpu_stack()

    #rhoj1_P = contract('pq,pix->qix', int2c_inv, wj1_P)
    int2c_ip1 = cupy.asarray(int2c_ip1, order='C')
    int2c_ip1 = intopt.sort_orbitals(int2c_ip1, aux_axis=[1,2])

    if with_j:
        rhoj1_P = solve_j2c(wj1_P)
        hj_ao_ao = 4.0*contract('pix,pjy->ijxy', rhoj1_P, wj1_P)   # (10|0)(0|0)(0|01)
        wj1_P = None
        if hessobj.auxbasis_response:
            wj0_01 = contract('ypq,q->yp', int2c_ip1, rhoj0_P)
            wj1_01 = contract('yqp,pix->iqxy', int2c_ip1, rhoj1_P)
            hj_ao_aux = contract('pix,py->ipxy', rhoj1_P, wj_ip2)   # (10|0)(1|00)
            hj_ao_aux -= contract('pix,yp->ipxy', rhoj1_P, wj0_01)   # (10|0)(1|0)(0|00)
            hj_ao_aux -= contract('q,iqxy->iqxy', rhoj0_P, wj1_01)   # (10|0)(0|1)(0|00)
            wj1_01 = None
        rhoj1_P = None

    if with_k:
        cupy.get_default_memory_pool().free_all_blocks()
        mem_avail = get_avail_mem()
        nocc = mocc.shape[1]
        slice_size = naux*nocc*9   # largest slice of intermediate variables
        blksize = int(mem_avail*0.2/8/slice_size)
        log.debug(f'GPU Memory {mem_avail/GB:.1f} GB available, {blksize} aux AOs per block')
        assert blksize > 0
        if hessobj.auxbasis_response:
            hk_ao_aux = cupy.zeros([nao,naux,3,3])
            for i0, i1 in lib.prange(0,nao,blksize):
                #wk1_Pko_islice = cupy.asarray(wk1_Pko[:,i0:i1])
                wk1_Pko_islice = copy_array(wk1_Pko[:,i0:i1])

                #rhok1_Pko = contract('pq,qiox->piox', int2c_inv, wk1_Pko_islice)
                rhok1_Pko = solve_j2c(wk1_Pko_islice)
                wk1_Pko_islice = None

                # (10|0)(1|00)
                wk_ip2_Ipo = contract('porx,io->pirx', wk_ip2_P__, mocc_2[i0:i1])
                hk_ao_aux[i0:i1] += contract('piox,pioy->ipxy', rhok1_Pko, wk_ip2_Ipo)
                wk_ip2_Ipo = None

                # (10|0)(1|0)(0|00)
                rhok0_P_I = contract('qor,ir->qoi', rhok0_P__, mocc_2[i0:i1])
                wk1_P_I = contract('ypq,qoi->pioy', int2c_ip1, rhok0_P_I)
                hk_ao_aux[i0:i1] -= contract("piox,pioy->ipxy", rhok1_Pko, wk1_P_I)
                wk1_P_I = None

                # (10|0)(0|1)(0|00)
                wk1_I = contract('yqp,piox->qioxy', int2c_ip1, rhok1_Pko)
                hk_ao_aux[i0:i1] -= contract('qoi,qioxy->iqxy', rhok0_P_I, wk1_I)
                wk1_I = rhok0_P_I = None
                rhok1_Pko = None
        t1 = log.timer_debug1('contract int3c2e_ip1 with int2c_ip1', *t1)
        
        rho2c_0 = contract('pij,qji->pq', rhok0_P__, rhok0_P__)
        rho2c_10 = contract('rijx,qij->rqx', wk_ip2_P__, rhok0_P__)
        rho2c_11 = contract('pijx,qijy->pqxy', wk_ip2_P__, wk_ip2_P__)
        rhok0_P__ = wk_ip2_P__ = None

        w, v = cupy.linalg.eigh(int2c)
        idx = w > LINEAR_DEP_THR
        cd_low = (v[:,idx] / cupy.sqrt(w[idx]))
        nnz = cd_low.shape[1]
        w = v = None

        rhok1_Pko = wk1_Pko[:nnz]  # Reuse the same memory
        for i0, i1 in lib.prange(0,nao,blksize):
            #wk1_tmp = cupy.asarray(wk1_Pko[:,i0:i1])
            wk1_tmp = copy_array(wk1_Pko[:,i0:i1])
            if isinstance(rhok1_Pko, cupy.ndarray):
                rhok1_Pko[:,i0:i1] = contract('qp,qiox->piox', cd_low, wk1_tmp)
            else:
                #rhok1_Pko[:,i0:i1] = contract('qp,qiox->piox', cd_low, wk1_tmp).get()
                wk1_tmp = contract('qp,qiox->piox', cd_low, wk1_tmp)
                copy_array(wk1_tmp, rhok1_Pko[:,i0:i1])
            wk1_tmp = None
        cd_low = None
        hk_ao_ao = _hk_ip1_ip1(rhok1_Pko, dm0, mocc_2)
    wk1_Pko = rhok1_Pko = None
    solve_j2c = None
    t1 = log.timer_debug1('contract int3c2e_ip1 with int3c2e_ip1', *t1)

    # int2c contributions
    if hessobj.auxbasis_response > 1:
        cupy.get_default_memory_pool().free_all_blocks()
        if omega and omega > 1e-10:
            with auxmol.with_range_coulomb(omega):
                int2c_ipip1 = auxmol.intor('int2c2e_ipip1', aosym='s1')
        else:
            int2c_ipip1 = auxmol.intor('int2c2e_ipip1', aosym='s1')
        int2c_ipip1 = cupy.asarray(int2c_ipip1, order='C')
        int2c_ipip1 = intopt.sort_orbitals(int2c_ipip1, aux_axis=[1,2])

        # (00|0)(2|0)(0|00)
        # p,xp->px
        if with_j:
            rhoj2c_P = contract('xpq,q->xp', int2c_ipip1, rhoj0_P)
            hj_aux_diag = -(rhoj0_P*rhoj2c_P).T.reshape(-1,3,3)
        if with_k:
            hk_aux_diag = -.5 * contract('pq,xpq->px', rho2c_0, int2c_ipip1).reshape(-1,3,3)
        int2c_ipip1 = None

        if omega and omega > 1e-10:
            with auxmol.with_range_coulomb(omega):
                int2c_ip1ip2 = auxmol.intor('int2c2e_ip1ip2', aosym='s1')
        else:
            int2c_ip1ip2 = auxmol.intor('int2c2e_ip1ip2', aosym='s1')
        int2c_ip1ip2 = cupy.asarray(int2c_ip1ip2, order='C')
        int2c_ip1ip2 = intopt.sort_orbitals(int2c_ip1ip2, aux_axis=[1,2])
        if with_j:
            hj_aux_aux = -.5 * contract('p,xpq->pqx', rhoj0_P, int2c_ip1ip2*rhoj0_P).reshape(naux, naux,3,3)
        if with_k:
            hk_aux_aux = -.5 * contract('xpq,pq->pqx', int2c_ip1ip2, rho2c_0).reshape(naux,naux,3,3)
        t1 = log.timer_debug1('intermediate variables with int2c_*', *t1)
        int2c_ip1ip2 = None

        # aux-aux pair
        int2c_inv = pinv(int2c, lindep=LINEAR_DEP_THR)
        int2c_ip1_inv = contract('yqp,pr->yqr', int2c_ip1, int2c_inv)
        if with_j:
            wj0_10 = contract('ypq,p->ypq', int2c_ip1, rhoj0_P)
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

    dm0 = intopt.unsort_orbitals(dm0, axis=[0,1])
    if with_j:
        hj_ao_ao = intopt.unsort_orbitals(hj_ao_ao, axis=[0,1])
        if hessobj.auxbasis_response:
            hj_ao_aux = intopt.unsort_orbitals(hj_ao_aux, axis=[0], aux_axis=[1])
        if hessobj.auxbasis_response > 1:
            hj_aux_aux = intopt.unsort_orbitals(hj_aux_aux, aux_axis=[0,1])
    if with_k:
        hk_ao_ao = intopt.unsort_orbitals(hk_ao_ao, axis=[0,1])
        if hessobj.auxbasis_response:
            hk_ao_aux = intopt.unsort_orbitals(hk_ao_aux, axis=[0], aux_axis=[1])
        if hessobj.auxbasis_response > 1:
            hk_aux_aux = intopt.unsort_orbitals(hk_aux_aux, aux_axis=[0,1])
    #======================================== sort AO end ===========================================
    # Energy weighted density matrix
    # pi,qi,i->pq
    dme0 = cupy.dot(mocc, (mocc * mo_energy[mo_occ>0] * 2).T)
    de_hcore = rhf_hess._e_hcore_generator(hessobj, dm0)
    t1 = log.timer_debug1('hcore generate', *t1)

    # ------------------------------------
    #      overlap matrix contributions
    # ------------------------------------
    s1aa, s1ab, _ = rhf_hess.get_ovlp(mol)
    s1aa = cupy.asarray(s1aa, order='C')
    s1ab = cupy.asarray(s1ab, order='C')
    h1aa = 2.0*contract('xypq,pq->pxy', s1aa, dme0)
    h1ab = 2.0*contract('xypq,pq->pqxy', s1ab, dme0)
    s1aa = s1ab = dme0 = None
    # -----------------------------------------
    #        collecting all
    # -----------------------------------------
    natm = len(atmlst)
    e1 = cupy.zeros([natm,natm,3,3])
    ej = hj_ipip
    ek = hk_ipip

    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        e1[i0,i0] -= cupy.sum(h1aa[p0:p1], axis=0)
        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            e1[i0,j0] -= cupy.sum(h1ab[p0:p1,q0:q1], axis=[0,1])
            if with_j:
                ej[i0,j0] += cupy.sum(hj_ao_ao[p0:p1,q0:q1], axis=[0,1])
            if with_k:
                ek[i0,j0] += cupy.sum(hk_ao_ao[p0:p1,q0:q1], axis=[0,1])
            e1[i0,j0] += de_hcore(ia, ja)
        #
        # The first order RI basis response
        #
        if hessobj.auxbasis_response:
            for j0, (q0, q1) in enumerate(auxslices[:,2:]):
                if with_j:
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
            for j0, (q0, q1) in enumerate(auxslices[:,2:]):
                if with_j:
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
            if with_j:
                ej[j0,i0] = ej[i0,j0].T
            if with_k:
                ek[j0,i0] = ek[i0,j0].T
    t1 = log.timer_debug1('hcore contribution', *t1)

    aux2atom = int3c2e.get_aux2atom(intopt, auxslices)

    natm = mol.natm
    idx = range(natm)
    # Diagonal contributions
    if hessobj.auxbasis_response > 1:
        if with_j:
            ej[idx, idx] += contract('ia,ixy->axy', aux2atom, hj_aux_diag)
        if with_k:
            ek[idx, idx] += contract('ia,ixy->axy', aux2atom, hk_aux_diag)

    log.timer('RHF partial hessian', *time0)
    return e1, ej, ek


def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    natm = mol.natm
    assert atmlst is None or atmlst ==range(natm)
    vj, vk = _get_jk_ip(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose, True)
    # h1mo = h1 + vj - 0.5 * vk
    h1mo = vk
    h1mo *= -.5
    h1mo += vj
    h1mo += rhf_grad.get_grad_hcore(hessobj.base.nuc_grad_method())
    return h1mo

def _get_jk_ip(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None,
            verbose=None, with_j=True, with_k=True, omega=None):
    '''
    Derivatives of J, K matrices in MO bases
    '''
    log = logger.new_logger(hessobj, verbose)
    t0 = log.init_timer()
    log.debug("Generate JK for RHF partial hessian")
    mem_avail = get_avail_mem()
    log.debug(f'GPU memory {mem_avail/GB:.1f} GB available')
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    mo_coeff = cupy.asarray(mo_coeff, order='C')
    mo_occ = cupy.asarray(mo_occ, order='C')

    mf = hessobj.base
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
    int2c = cupy.asarray(int2c, order='C')
    # ======================= sorted AO begin ======================================
    intopt = int3c2e.VHFOpt(mol, auxmol, 'int2e')
    mem_avail = get_avail_mem()
    blksize = int(mem_avail*0.2/(nao*nao*8*3)/ALIGNED) * ALIGNED
    log.debug(f'GPU Memory {mem_avail/GB:.1f} GB available, {blksize} aux AOs per block')

    intopt.build(mf.direct_scf_tol,
                 diag_block_with_triu=True,
                 aosym=False, verbose=0,
                 group_size_aux=BLKSIZE,
                 group_size=BLKSIZE)
    naux = auxmol.nao
    mocc = intopt.sort_orbitals(mocc, axis=[0])
    nocc = mocc.shape[1]
    mo_coeff = intopt.sort_orbitals(mo_coeff, axis=[0])
    dm0 = intopt.sort_orbitals(dm0, axis=[0,1])
    dm0_tag = tag_array(dm0, occ_coeff=mocc)

    int2c = intopt.sort_orbitals(int2c, aux_axis=[0,1])
    solve_j2c = _gen_metric_solver(int2c)
    wj, wk_Pl_ = int3c2e.get_int3c2e_wjk(mol, auxmol, dm0_tag,
                                         with_j=with_j, with_k=True, omega=omega)
    rhoj0 = None
    if with_j:
        rhoj0 = solve_j2c(wj)
        wj = None

    if isinstance(wk_Pl_, cupy.ndarray):
        rhok0_Pl_ = solve_j2c(wk_Pl_)
    else:
        rhok0_Pl_ = wk_Pl_ # reuse the memory
        for p0, p1 in lib.prange(0,nao,64):
            #wk_tmp = cupy.asarray(wk_Pl_[:,p0:p1])
            #rhok0_Pl_[:,p0:p1] = solve_j2c(wk_tmp).get()
            wk_tmp = copy_array(wk_Pl_[:,p0:p1])
            wk_tmp = solve_j2c(wk_tmp)
            copy_array(wk_tmp, rhok0_Pl_[:,p0:p1])
        wk_tmp = None
    wk_Pl_ = None
    solve_j2c = None
    t0 = log.timer_debug1('Fock matrix due to int3c2e', *t0)
    vj1_int3c = vk1_int3c = None

    # --------------------------
    #  int3c_ip2 contribution
    # --------------------------
    cupy.get_default_memory_pool().free_all_blocks()
    if hessobj.auxbasis_response:
        fn = int3c2e.get_int3c2e_ip2_vjk
        vj1_int3c, vk1_int3c = fn(intopt, rhoj0, rhok0_Pl_, dm0_tag, auxslices,
                                  with_j=with_j, with_k=with_k, omega=omega)
        t0 = log.timer_debug1('Fock matrix due to int3c2e_ip2', *t0)

        # Responses due to int2c2e_ip1
        if omega and omega > 1e-10:
            with auxmol.with_range_coulomb(omega):
                int2c_ip1 = auxmol.intor('int2c2e_ip1', aosym='s1')
        else:
            int2c_ip1 = auxmol.intor('int2c2e_ip1', aosym='s1')
        int2c_ip1 = cupy.asarray(int2c_ip1, order='C')
        int2c_ip1 = intopt.sort_orbitals(int2c_ip1, aux_axis=[1,2])

        if with_j:
            wj0_10 = contract('xpq,q->xp', int2c_ip1, rhoj0)
        if with_k:
            # Generate rhok0_P__
            if isinstance(rhok0_Pl_, cupy.ndarray):
                rhok0_P__ = contract('pio,ir->pro', rhok0_Pl_, mocc)
            else:
                rhok0_P__ = cupy.empty([naux,nocc,nocc])
                for p0, p1 in lib.prange(0,naux,64):
                    #rhok0_Pl_tmp = cupy.asarray(rhok0_Pl_[p0:p1])
                    rhok0_Pl_tmp = copy_array(rhok0_Pl_[p0:p1])
                    rhok0_P__[p0:p1] = contract('pio,ir->pro', rhok0_Pl_tmp, mocc)
                rhok0_Pl_tmp = None
            wk0_10_P__ = contract('xqp,pro->xqro', int2c_ip1, rhok0_P__)

        aux2atom = int3c2e.get_aux2atom(intopt, auxslices)
        mem_avail = get_avail_mem()
        blksize = int(0.2*mem_avail/(3*naux*nocc*8)/ALIGNED) * ALIGNED
        log.debug(f'GPU Memory {mem_avail/GB:.1f} GB available, {blksize} AOs per block')
        if blksize < ALIGNED:
            raise RuntimeError('Not enough memory to compute int2c2e_ip2')

        for p0, p1 in lib.prange(0,nao,blksize):
            #rhok_tmp = cupy.asarray(rhok0_Pl_[:,p0:p1])
            rhok_tmp = copy_array(rhok0_Pl_[:,p0:p1])
            wk0_10_Pl_ = contract('xqp,pio->xqio', int2c_ip1, rhok_tmp)
            if with_j:
                vj1_tmp = contract('pio,xp->xpio', rhok_tmp, wj0_10)
                vj1_tmp += contract('xpio,p->xpio', wk0_10_Pl_, rhoj0)
                vj1_int3c[:,:,p0:p1] += contract('xpio,pa->axio', vj1_tmp, aux2atom)
                vj1_tmp = None
            if with_k:
                vk1_tmp = contract('xpio,pro->xpir', wk0_10_Pl_, rhok0_P__)
                vk1_tmp += contract('xpro,pir->xpio', wk0_10_P__, rhok_tmp)
                vk1_int3c[:,:,p0:p1] += contract('xpio,pa->axio', vk1_tmp, aux2atom)
                vk1_tmp = None
            wk0_10_Pl_ = rhok_tmp = None
        wj0_10 = wk0_10_P__ = rhok0_P__ = int2c_ip1 = None
        aux2atom = None
        t0 = log.timer_debug1('Fock matrix due to int2c2e_ip1', *t0)

    # -----------------------------
    # int3c_ip1 contributions
    # ------------------------------
    cupy.get_default_memory_pool().free_all_blocks()
    fn = int3c2e.get_int3c2e_ip1_vjk
    vj1_buf, vk1_buf, vj1_ao, vk1_ao = fn(intopt, rhoj0, rhok0_Pl_, dm0_tag, aoslices,
                                          omega=omega, with_j=with_j, with_k=with_k)
    rhoj0 = rhok0_Pl_ = dm0_tag = None
    if with_j:
        vj1_buf = intopt.unsort_orbitals(vj1_buf, axis=[1,2])
        if vj1_int3c is None:
            vj1_int3c = -vj1_ao
        else:
            vj1_int3c -= vj1_ao
        vj1_ao = None
        # NOTE: vj1_int3c and vk1_int3c are in [natm,3,nao,nocc]
        #       axis=2 in AO, axis=3 in MO
        #       convert axis=2 into MO now
        vj1_int3c = contract('nxiq,ip->nxpq', vj1_int3c, mo_coeff)

    if with_k:
        vk1_buf = intopt.unsort_orbitals(vk1_buf, axis=[1,2])
        if vk1_int3c is None:
            vk1_int3c = -vk1_ao
        else:
            vk1_int3c -= vk1_ao
        vk1_ao = None
        # * 2.0 due to the contraction with mocc
        vk1_buf *= 2.0
        vk1_int3c = 2.0 * contract('nxiq,ip->nxpq', vk1_int3c, mo_coeff)
    t0 = log.timer_debug1('Fock matrix due to int3c2e_ip1', *t0)

    mocc = intopt.unsort_orbitals(mocc, axis=[0])
    mo_coeff = intopt.unsort_orbitals(mo_coeff, axis=[0])
    release_gpu_stack()

    # ========================== sorted AO end ================================
    def _ao2mo(mat):
        tmp = contract('xij,jo->xio', mat, mocc)
        return contract('xik,ip->xpk', tmp, mo_coeff)

    cupy.get_default_memory_pool().free_all_blocks()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        if with_j:
            vj1_ao = cupy.zeros([3,nao,nao])
            vj1_ao[:,p0:p1,:] -= vj1_buf[:,p0:p1,:]
            vj1_ao[:,:,p0:p1] -= vj1_buf[:,p0:p1,:].transpose(0,2,1)
            vj1_int3c[ia] += _ao2mo(vj1_ao)
        if with_k:
            vk1_ao = cupy.zeros([3,nao,nao])
            vk1_ao[:,p0:p1,:] -= vk1_buf[:,p0:p1,:]
            vk1_ao[:,:,p0:p1] -= vk1_buf[:,p0:p1,:].transpose(0,2,1)
            vk1_int3c[ia] += _ao2mo(vk1_ao)
    return vj1_int3c, vk1_int3c

def _get_jk_mo(hessobj, mol, dms, mo_coeff, mocc,
           hermi=1, with_j=True, with_k=True, omega=None):
    mf = hessobj.base
    dfobj = mf.with_df
    if omega is None:
        return jk.get_jk(dfobj, dms, mo_coeff, mocc,
                         hermi=hermi, with_j=with_j, with_k=with_k)

    # A temporary treatment for RSH-DF integrals
    key = '%.6f' % omega
    if key in dfobj._rsh_df:
        rsh_df = dfobj._rsh_df[key]
    else:
        rsh_df = dfobj._rsh_df[key] = dfobj.copy().reset()
        logger.info(dfobj, 'Create RSH-DF object %s for omega=%s', rsh_df, omega)

    with rsh_df.mol.with_range_coulomb(omega):
        return jk.get_jk(rsh_df, dms, mo_coeff, mocc,
                         hermi=hermi, with_j=with_j, with_k=with_k, omega=omega)


class Hessian(rhf_hess.Hessian):
    '''Non-relativistic restricted Hartree-Fock hessian'''

    from gpu4pyscf.lib.utils import to_gpu, device

    auxbasis_response = 1
    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1
    get_jk_mo = _get_jk_mo
