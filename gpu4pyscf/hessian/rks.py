#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
#  modified by Xiaojie Wu <wxj6000@gmail.com>

'''
Non-relativistic RKS analytical Hessian
'''


import numpy
import cupy
from pyscf import lib
from pyscf.hessian import rks as rks_hess
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import contract, add_sparse, take_last2d
from gpu4pyscf.lib import logger

# import pyscf.grad.rks to activate nuc_grad_method method
from gpu4pyscf.grad import rks  # noqa


def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = cupy.dot(mocc, mocc.T) * 2

    if mf.do_nlc():
        raise NotImplementedError
    #enabling range-separated hybrids
    omega, alpha, beta = mf._numint.rsh_coeff(mf.xc)
    if abs(omega) > 1e-10:
        hyb = alpha + beta
    else:
        hyb = mf._numint.hybrid_coeff(mf.xc, spin=mol.spin)
    de2, ej, ek = rhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                             atmlst, max_memory, verbose,
                                             abs(hyb) > 1e-10)
    de2 += ej - hyb * ek  # (A,B,dR_A,dR_B)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    veff_diag = _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)
    if abs(omega) > 1e-10:
        with mol.with_range_coulomb(omega):
            vk1 = rhf_hess._get_jk(mol, 'int2e_ipip1', 9, 's2kl',
                                   ['jk->s1il', dm0])[0]
        veff_diag -= (alpha-hyb)*.5 * vk1.reshape(3,3,nao,nao)
    vk1 = None
    t1 = log.timer_debug1('contracting int2e_ipip1', *t1)

    aoslices = mol.aoslice_by_atom()
    vxc = _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]

        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        veff = vxc[ia]
        if abs(omega) > 1e-10:
            with mol.with_range_coulomb(omega):
                vk1, vk2 = rhf_hess._get_jk(mol, 'int2e_ip1ip2', 9, 's1',
                                            ['li->s1kj', dm0[:,p0:p1],  # vk1
                                             'lj->s1ki', dm0         ], # vk2
                                            shls_slice=shls_slice)
            veff -= (alpha-hyb)*.5 * vk1.reshape(3,3,nao,nao)
            veff[:,:,:,p0:p1] -= (alpha-hyb)*.5 * vk2.reshape(3,3,nao,p1-p0)
            t1 = log.timer_debug1('range-separated int2e_ip1ip2 for atom %d'%ia, *t1)
            with mol.with_range_coulomb(omega):
                vk1 = rhf_hess._get_jk(mol, 'int2e_ipvip1', 9, 's2kl',
                                       ['li->s1kj', dm0[:,p0:p1]], # vk1
                                       shls_slice=shls_slice)[0]
            veff -= (alpha-hyb)*.5 * vk1.transpose(0,2,1).reshape(3,3,nao,nao)
            t1 = log.timer_debug1('range-separated int2e_ipvip1 for atom %d'%ia, *t1)
            vk1 = vk2 = None
        de2[i0,i0] += contract('xypq,pq->xy', veff_diag[:,:,p0:p1], dm0[p0:p1])*2
        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            de2[i0,j0] += contract('xypq,pq->xy', veff[:,:,q0:q1], dm0[q0:q1])*2

        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    log.timer('RKS partial hessian', *time0)
    return de2

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    if isinstance(mo_coeff, cupy.ndarray):
        mo_coeff = mo_coeff.get()
    if isinstance(mo_occ, cupy.ndarray):
        mo_occ = mo_occ.get()

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)

    mf = hessobj.base
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    h1ao = _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory).get()
    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        if abs(hyb) > 1e-10:
            vj1, vj2, vk1, vk2 = \
                    rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                     ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                      'lk->s1ij', -dm0         ,  # vj2
                                      'li->s1kj', -dm0[:,p0:p1],  # vk1
                                      'jk->s1il', -dm0         ], # vk2
                                     shls_slice=shls_slice)
            veff = vj1 - hyb * .5 * vk1
            veff[:,p0:p1] += vj2 - hyb * .5 * vk2
            if abs(omega) > 1e-10:
                with mol.with_range_coulomb(omega):
                    vk1, vk2 = \
                        rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                         ['li->s1kj', -dm0[:,p0:p1],  # vk1
                                          'jk->s1il', -dm0         ], # vk2
                                         shls_slice=shls_slice)
                veff -= (alpha-hyb) * .5 * vk1
                veff[:,p0:p1] -= (alpha-hyb) * .5 * vk2
        else:
            vj1, vj2 = rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                        ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                         'lk->s1ij', -dm0         ], # vj2
                                        shls_slice=shls_slice)
            veff = vj1
            veff[:,p0:p1] += vj2

        veff = hcore_deriv(ia) + veff + veff.transpose(0,2,1)
        veff = numpy.einsum('xij,ip,jq->xpq', veff, mo_coeff, mocc)
        #h1ao[ia] += veff + veff.transpose(0,2,1)
        #h1ao[ia] += hcore_deriv(ia)
        h1ao[ia] += veff

    if chkfile is None:
        return h1ao
    else:
        for ia in atmlst:
            lib.chkfile.save(chkfile, 'scf_f1ao/%d'%ia, h1ao[ia])
        return chkfile

XX, XY, XZ = 4, 5, 6
YX, YY, YZ = 5, 7, 8
ZX, ZY, ZZ = 6, 8, 9
XXX, XXY, XXZ, XYY, XYZ, XZZ = 10, 11, 12, 13, 14, 15
YYY, YYZ, YZZ, ZZZ = 16, 17, 18, 19

def _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=False)

    # move data to GPU
    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)

    nao_sph, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    coeff = cupy.asarray(opt.coeff)
    mo_coeff = coeff @ mo_coeff
    nao = mo_coeff.shape[0]

    vmat = cupy.zeros((6,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            mo_coeff_mask = mo_coeff[mask,:]
            rho = numint.eval_rho2(_sorted_mol, ao[0], mo_coeff_mask, mo_occ, mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
            wv = weight * vxc[0]
            aow = numint._scale_ao(ao[0], wv)
            for i in range(6):
                vmat_tmp = numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)
                add_sparse(vmat[i], vmat_tmp, mask)
            aow = None

    elif xctype == 'GGA':
        def contract_(ao, aoidx, wv, mask):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1])
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2])
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3])
            return numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            mo_coeff_mask = mo_coeff[mask,:]
            rho = numint.eval_rho2(_sorted_mol, ao[:4], mo_coeff_mask, mo_occ, mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
            wv = weight * vxc
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
            aow = numint._scale_ao(ao[:4], wv[:4])

            vmat_tmp = [0]*6
            for i in range(6):
                vmat_tmp[i] = numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)

            vmat_tmp[0] += contract_(ao, [XXX,XXY,XXZ], wv, mask)
            vmat_tmp[1] += contract_(ao, [XXY,XYY,XYZ], wv, mask)
            vmat_tmp[2] += contract_(ao, [XXZ,XYZ,XZZ], wv, mask)
            vmat_tmp[3] += contract_(ao, [XYY,YYY,YYZ], wv, mask)
            vmat_tmp[4] += contract_(ao, [XYZ,YYZ,YZZ], wv, mask)
            vmat_tmp[5] += contract_(ao, [XZZ,YZZ,ZZZ], wv, mask)
            for i in range(6):
                add_sparse(vmat[i], vmat_tmp[i], mask)
            rho = vxc = wv = aow = None
    elif xctype == 'MGGA':
        def contract_(ao, aoidx, wv, mask):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1])
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2])
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3])
            return numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            mo_coeff_mask = mo_coeff[mask,:]
            rho = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff_mask, mo_occ, mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
            wv = weight * vxc
            wv[4] *= .5  # for the factor 1/2 in tau
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
            vmat_tmp = [0]*6
            aow = numint._scale_ao(ao[:4], wv[:4])
            for i in range(6):
                vmat_tmp[i] = numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)

            vmat_tmp[0] += contract_(ao, [XXX,XXY,XXZ], wv, mask)
            vmat_tmp[1] += contract_(ao, [XXY,XYY,XYZ], wv, mask)
            vmat_tmp[2] += contract_(ao, [XXZ,XYZ,XZZ], wv, mask)
            vmat_tmp[3] += contract_(ao, [XYY,YYY,YYZ], wv, mask)
            vmat_tmp[4] += contract_(ao, [XYZ,YYZ,YZZ], wv, mask)
            vmat_tmp[5] += contract_(ao, [XZZ,YZZ,ZZZ], wv, mask)

            aow = [numint._scale_ao(ao[i], wv[4]) for i in range(1, 4)]
            for i, j in enumerate([XXX, XXY, XXZ, XYY, XYZ, XZZ]):
                vmat_tmp[i] += numint._dot_ao_ao(mol, ao[j], aow[0], mask, shls_slice, ao_loc)

            for i, j in enumerate([XXY, XYY, XYZ, YYY, YYZ, YZZ]):
                vmat_tmp[i] += numint._dot_ao_ao(mol, ao[j], aow[1], mask, shls_slice, ao_loc)

            for i, j in enumerate([XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ]):
                vmat_tmp[i] += numint._dot_ao_ao(mol, ao[j], aow[2], mask, shls_slice, ao_loc)

            for i in range(6):
                add_sparse(vmat[i], vmat_tmp[i], mask)

    vmat = vmat[[0,1,2,
                 1,3,4,
                 2,4,5]]

    vmat = contract('npq,qj->npj', vmat, coeff)
    vmat = contract('pi,npj->nij', coeff, vmat)
    return vmat.reshape(3,3,nao_sph,nao_sph)

def _make_dR_rho1(ao, ao_dm0, atm_id, aoslices, xctype):
    p0, p1 = aoslices[atm_id][2:]
    ngrids = ao[0].shape[1]
    if xctype == 'GGA':
        rho1 = cupy.zeros((3,4,ngrids))
    elif xctype == 'MGGA':
        rho1 = cupy.zeros((3,5,ngrids))
        ao_dm0_x = ao_dm0[1][p0:p1]
        ao_dm0_y = ao_dm0[2][p0:p1]
        ao_dm0_z = ao_dm0[3][p0:p1]
        # (d_X \nabla mu) dot \nalba nu DM_{mu,nu}
        rho1[0,4] += numint._contract_rho(ao[XX,p0:p1], ao_dm0_x)
        rho1[0,4] += numint._contract_rho(ao[XY,p0:p1], ao_dm0_y)
        rho1[0,4] += numint._contract_rho(ao[XZ,p0:p1], ao_dm0_z)
        rho1[1,4] += numint._contract_rho(ao[YX,p0:p1], ao_dm0_x)
        rho1[1,4] += numint._contract_rho(ao[YY,p0:p1], ao_dm0_y)
        rho1[1,4] += numint._contract_rho(ao[YZ,p0:p1], ao_dm0_z)
        rho1[2,4] += numint._contract_rho(ao[ZX,p0:p1], ao_dm0_x)
        rho1[2,4] += numint._contract_rho(ao[ZY,p0:p1], ao_dm0_y)
        rho1[2,4] += numint._contract_rho(ao[ZZ,p0:p1], ao_dm0_z)
        rho1[:,4] *= .5
    else:
        raise RuntimeError

    ao_dm0_0 = ao_dm0[0][p0:p1]
    # (d_X \nabla_x mu) nu DM_{mu,nu}
    rho1[:,0] = numint._contract_rho1(ao[1:4,p0:p1], ao_dm0_0)
    rho1[0,1]+= numint._contract_rho(ao[XX,p0:p1], ao_dm0_0)
    rho1[0,2]+= numint._contract_rho(ao[XY,p0:p1], ao_dm0_0)
    rho1[0,3]+= numint._contract_rho(ao[XZ,p0:p1], ao_dm0_0)
    rho1[1,1]+= numint._contract_rho(ao[YX,p0:p1], ao_dm0_0)
    rho1[1,2]+= numint._contract_rho(ao[YY,p0:p1], ao_dm0_0)
    rho1[1,3]+= numint._contract_rho(ao[YZ,p0:p1], ao_dm0_0)
    rho1[2,1]+= numint._contract_rho(ao[ZX,p0:p1], ao_dm0_0)
    rho1[2,2]+= numint._contract_rho(ao[ZY,p0:p1], ao_dm0_0)
    rho1[2,3]+= numint._contract_rho(ao[ZZ,p0:p1], ao_dm0_0)
    # (d_X mu) (\nabla_x nu) DM_{mu,nu}
    rho1[:,1] += numint._contract_rho1(ao[1:4,p0:p1], ao_dm0[1][p0:p1])
    rho1[:,2] += numint._contract_rho1(ao[1:4,p0:p1], ao_dm0[2][p0:p1])
    rho1[:,3] += numint._contract_rho1(ao[1:4,p0:p1], ao_dm0[3][p0:p1])

    # *2 for |mu> DM <d_X nu|
    return rho1 * 2

def _d1d2_dot_(vmat, mol, ao1, ao2, mask, ao_loc, dR1_on_bra=True):
    shls_slice = None
    if dR1_on_bra:  # (d/dR1 bra) * (d/dR2 ket)
        for d1 in range(3):
            for d2 in range(3):
                vmat[d1,d2] += numint._dot_ao_ao(mol, ao1[d1], ao2[d2], mask,
                                                 shls_slice, ao_loc)
        #vmat += contract('xig,yjg->xyij', ao1, ao2)
    else:  # (d/dR2 bra) * (d/dR1 ket)
        for d1 in range(3):
            for d2 in range(3):
                vmat[d1,d2] += numint._dot_ao_ao(mol, ao1[d2], ao2[d1], mask,
                                                 shls_slice, ao_loc)
        #vmat += contract('yig,xjg->xyij', ao1, ao2)

def _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj.base
    log = logger.new_logger(mol, mol.verbose)
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    # move data to GPU
    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    coeff = cupy.asarray(opt.coeff)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dm0_sorted = take_last2d(dm0, opt.ao_idx)
    vmat_dm = cupy.zeros((_sorted_mol.natm,3,3,nao))
    ipip = cupy.zeros((3,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        t1 = log.init_timer()
        for ao_mask, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            nao_non0 = len(mask)
            ao = contract('nip,ij->njp', ao_mask, coeff[mask])
            rho = numint.eval_rho2(_sorted_mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            t1 = log.timer_debug2('eval rho', *t1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            t1 = log.timer_debug2('eval vxc', *t1)
            wv = weight * vxc[0]
            aow = [numint._scale_ao(ao[i], wv) for i in range(1, 4)]
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)
            dm0_mask = dm0_sorted[numpy.ix_(mask, mask)]

            ao_dm_mask = contract('nig,ij->njg', ao_mask[:4], dm0_mask)
            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            wf = weight * fxc[0,0]
            for ia in range(_sorted_mol.natm):
                p0, p1 = aoslices[ia][2:]
                # *2 for \nabla|ket> in rho1
                rho1 = contract('xig,ig->xg', ao[1:,p0:p1,:], ao_dm0[p0:p1,:]) * 2
                # aow ~ rho1 ~ d/dR1
                wv = wf * rho1
                aow = cupy.empty_like(ao_dm_mask[1:4])
                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dm_mask[0], wv[i])
                vmat_dm[ia][:,:,mask] += contract('yjg,xjg->xyj', ao_mask[1:4], aow)
            ao_dm0 = aow = None
            t1 = log.timer_debug2('integration', *t1)
        for ia in range(_sorted_mol.natm):
            vmat_dm[ia] = vmat_dm[ia][:,:,opt.rev_ao_idx]
            p0, p1 = aoslices[ia][2:]
            vmat_dm[ia] += contract('xypq,pq->xyp', ipip[:,:,:,p0:p1], dm0[:,p0:p1])
    elif xctype == 'GGA':
        ao_deriv = 2
        t1 = log.init_timer()
        for ao_mask, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            nao_non0 = len(mask)
            ao = contract('nip,ij->njp', ao_mask, coeff[mask])
            rho = numint.eval_rho2(_sorted_mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
            t1 = log.timer_debug2('eval rho', *t1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            t1 = log.timer_debug2('eval vxc', *t1)
            wv = weight * vxc
            wv[0] *= .5
            aow = rks_grad._make_dR_dao_w(ao, wv)
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)
            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            dm0_mask = dm0_sorted[numpy.ix_(mask, mask)]
            ao_dm_mask = contract('nig,ij->njg', ao_mask[:4], dm0_mask)
            vmat_dm_tmp = cupy.empty([3,3,nao_non0])
            for ia in range(_sorted_mol.natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv = contract('xyg,sxg->syg', wf, dR_rho1)
                wv[:,0] *= .5
                for i in range(3):
                    aow = rks_grad._make_dR_dao_w(ao_mask, wv[i])
                    vmat_dm_tmp[i] = contract('xjg,jg->xj', aow, ao_dm_mask[0])
                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dm_mask[:4], wv[i,:4])
                vmat_dm_tmp += contract('yjg,xjg->xyj', ao_mask[1:4], aow)
                vmat_dm[ia][:,:,mask] += vmat_dm_tmp
            ao_dm0 = aow = None
            t1 = log.timer_debug2('integration', *t1)
        for ia in range(_sorted_mol.natm):
            vmat_dm[ia] = vmat_dm[ia][:,:,opt.rev_ao_idx]
            p0, p1 = aoslices[ia][2:]
            vmat_dm[ia] += contract('xypq,pq->xyp', ipip[:,:,:,p0:p1], dm0[:,p0:p1])
            vmat_dm[ia] += contract('yxqp,pq->xyp', ipip[:,:,p0:p1], dm0[:,p0:p1])
    elif xctype == 'MGGA':
        XX, XY, XZ = 4, 5, 6
        YX, YY, YZ = 5, 7, 8
        ZX, ZY, ZZ = 6, 8, 9
        ao_deriv = 2
        t1 = log.init_timer()
        for ao_mask, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            nao_non0 = len(mask)
            ao = contract('nip,ij->njp', ao_mask, coeff[mask])
            rho = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
            t1 = log.timer_debug2('eval rho', *t1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            t1 = log.timer_debug2('eval vxc', *t1)
            wv = weight * vxc
            wv[0] *= .5
            wv[4] *= .25
            aow = rks_grad._make_dR_dao_w(ao, wv)
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            aow = [numint._scale_ao(ao[i], wv[4]) for i in range(4, 10)]
            _d1d2_dot_(ipip, mol, [aow[0], aow[1], aow[2]], [ao[XX], ao[XY], ao[XZ]], mask, ao_loc, False)
            _d1d2_dot_(ipip, mol, [aow[1], aow[3], aow[4]], [ao[YX], ao[YY], ao[YZ]], mask, ao_loc, False)
            _d1d2_dot_(ipip, mol, [aow[2], aow[4], aow[5]], [ao[ZX], ao[ZY], ao[ZZ]], mask, ao_loc, False)
            dm0_mask = dm0_sorted[numpy.ix_(mask, mask)]
            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            ao_dm_mask = contract('nig,ij->njg', ao_mask[:4], dm0_mask)
            wf = weight * fxc
            for ia in range(_sorted_mol.natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv = contract('xyg,sxg->syg', wf, dR_rho1)
                wv[:,0] *= .5
                wv[:,4] *= .5  # for the factor 1/2 in tau
                vmat_dm_tmp = cupy.empty([3,3,nao_non0])
                for i in range(3):
                    aow = rks_grad._make_dR_dao_w(ao_mask, wv[i])
                    vmat_dm_tmp[i] = contract('xjg,jg->xj', aow, ao_dm_mask[0])

                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dm_mask[:4], wv[i,:4])
                vmat_dm_tmp += contract('yjg,xjg->xyj', ao_mask[1:4], aow)

                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dm_mask[1], wv[i,4])
                vmat_dm_tmp[:,0] += contract('jg,xjg->xj', ao_mask[XX], aow)
                vmat_dm_tmp[:,1] += contract('jg,xjg->xj', ao_mask[XY], aow)
                vmat_dm_tmp[:,2] += contract('jg,xjg->xj', ao_mask[XZ], aow)

                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dm_mask[2], wv[i,4])
                vmat_dm_tmp[:,0] += contract('jg,xjg->xj', ao_mask[YX], aow)
                vmat_dm_tmp[:,1] += contract('jg,xjg->xj', ao_mask[YY], aow)
                vmat_dm_tmp[:,2] += contract('jg,xjg->xj', ao_mask[YZ], aow)

                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dm_mask[3], wv[i,4])
                vmat_dm_tmp[:,0] += contract('jg,xjg->xj', ao_mask[ZX], aow)
                vmat_dm_tmp[:,1] += contract('jg,xjg->xj', ao_mask[ZY], aow)
                vmat_dm_tmp[:,2] += contract('jg,xjg->xj', ao_mask[ZZ], aow)

                vmat_dm[ia][:,:,mask] += vmat_dm_tmp
            t1 = log.timer_debug2('integration', *t1)
        for ia in range(_sorted_mol.natm):
            vmat_dm[ia] = vmat_dm[ia][:,:,opt.rev_ao_idx]
            p0, p1 = aoslices[ia][2:]
            vmat_dm[ia] += contract('xypq,pq->xyp', ipip[:,:,:,p0:p1], dm0[:,p0:p1])
            vmat_dm[ia] += contract('yxqp,pq->xyp', ipip[:,:,p0:p1], dm0[:,p0:p1])
    return vmat_dm

def _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj.base
    log = logger.new_logger(mol, mol.verbose)
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids

    if grids.coords is None:
        grids.build(with_non0tab=True)

    # move data to GPU
    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    mol = None
    _sorted_mol = opt._sorted_mol
    coeff = cupy.asarray(opt.coeff)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    v_ip = cupy.zeros((3,nao,nao))
    vmat = cupy.zeros((_sorted_mol.natm,3,nao,nocc))
    max_memory = max(2000, max_memory-vmat.size*8/1e6)
    if xctype == 'LDA':
        ao_deriv = 1
        t1 = t0 = log.init_timer()
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            ao = contract('nip,ij->njp', ao, coeff[mask])
            rho = numint.eval_rho2(_sorted_mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            t1 = log.timer_debug2('eval rho', *t1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            t1 = log.timer_debug2('eval vxc', *t1)
            wv = weight * vxc[0]
            aow = numint._scale_ao(ao[0], wv)
            v_ip += rks_grad._d1_dot_(ao[1:4], aow.T)
            mo = contract('xig,ip->xpg', ao, mocc)
            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            wf = weight * fxc[0,0]
            for ia in range(_sorted_mol.natm):
                p0, p1 = aoslices[ia][2:]
# First order density = rho1 * 2.  *2 is not applied because + c.c. in the end
                rho1 = contract('xig,ig->xg', ao[1:,p0:p1,:], ao_dm0[p0:p1,:])
                wv = wf * rho1
                aow = [numint._scale_ao(ao[0], wv[i]) for i in range(3)]
                mow = [numint._scale_ao(mo[0], wv[i]) for i in range(3)]
                vmat[ia] += rks_grad._d1_dot_(aow, mo[0].T)
                vmat[ia] += rks_grad._d1_dot_(mow, ao[0].T).transpose([0,2,1])
            ao_dm0 = aow = None
            t1 = log.timer_debug2('integration', *t1)
    elif xctype == 'GGA':
        ao_deriv = 2
        t1 = t0 = log.init_timer()
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            ao = contract('nip,ij->njp', ao, coeff[mask])
            rho = numint.eval_rho2(_sorted_mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
            t1 = log.timer_debug2('eval rho', *t1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            t1 = log.timer_debug2('eval vxc', *t1)
            wv = weight * vxc
            wv[0] *= .5
            v_ip += rks_grad._gga_grad_sum_(ao, wv)
            mo = contract('xig,ip->xpg', ao, mocc)
            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc)
                      for i in range(4)]
            wf = weight * fxc
            for ia in range(_sorted_mol.natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv = contract('xyg,sxg->syg', wf, dR_rho1)
                wv[:,0] *= .5
                aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                mow = [numint._scale_ao(mo[:4], wv[i,:4]) for i in range(3)]
                vmat[ia] += rks_grad._d1_dot_(aow, mo[0].T)
                vmat[ia] += rks_grad._d1_dot_(mow, ao[0].T).transpose([0,2,1])
            t1 = log.timer_debug2('integration', *t1)
            ao_dm0 = aow = None
    elif xctype == 'MGGA':
        if grids.level < 5:
            log.warn('MGGA Hessian is sensitive to dft grids.')
        ao_deriv = 2
        t1 = t0 = log.init_timer()
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            ao = contract('nip,ij->njp', ao, coeff[mask])
            rho = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
            t1 = log.timer_debug2('eval rho', *t1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            t1 = log.timer_debug2('eval vxc', *t0)
            wv = weight * vxc
            wv[0] *= .5
            wv[4] *= .5  # for the factor 1/2 in tau
            v_ip += rks_grad._gga_grad_sum_(ao, wv)
            v_ip += rks_grad._tau_grad_dot_(ao, wv[4])
            mo = contract('xig,ip->xpg', ao, mocc)
            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            for ia in range(_sorted_mol.natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv = contract('xyg,sxg->syg', wf, dR_rho1)
                wv[:,0] *= .5
                wv[:,4] *= .25
                aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                mow = [numint._scale_ao(mo[:4], wv[i,:4]) for i in range(3)]
                vmat[ia] += rks_grad._d1_dot_(aow, mo[0].T)
                vmat[ia] += rks_grad._d1_dot_(mow, ao[0].T).transpose([0,2,1])
                for j in range(1, 4):
                    aow = [numint._scale_ao(ao[j], wv[i,4]) for i in range(3)]
                    mow = [numint._scale_ao(mo[j], wv[i,4]) for i in range(3)]
                    vmat[ia] += rks_grad._d1_dot_(aow, mo[j].T)
                    vmat[ia] += rks_grad._d1_dot_(mow, ao[j].T).transpose([0,2,1])
            ao_dm0 = aow = None
            t1 = log.timer_debug2('integration', *t1)

    vmat = -contract("kxiq,ip->kxpq", vmat, mo_coeff)

    for ia in range(_sorted_mol.natm):
        p0, p1 = aoslices[ia][2:]
        vmat_tmp = cupy.zeros([3,nao,nao])
        vmat_tmp[:,p0:p1] += v_ip[:,p0:p1]
        vmat_tmp[:,:,p0:p1] += v_ip[:,p0:p1].transpose(0,2,1)

        vmat_tmp = contract('xij,jq->xiq', vmat_tmp, mocc)
        vmat_tmp = contract('xiq,ip->xpq', vmat_tmp, mo_coeff)
        vmat[ia] -= vmat_tmp
    return vmat


class Hessian(rhf_hess.HessianBase):
    '''Non-relativistic RKS hessian'''

    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'grids', 'grid_response'}

    def __init__(self, mf):
        rhf_hess.Hessian.__init__(self, mf)
        self.grids = None
        self.grid_response = False

    partial_hess_elec = partial_hess_elec
    hess_elec = rhf_hess.hess_elec
    make_h1 = make_h1
    hess = NotImplemented
    kernel = NotImplemented

from gpu4pyscf import dft
dft.rks.RKS.Hessian = lib.class_as_method(Hessian)
