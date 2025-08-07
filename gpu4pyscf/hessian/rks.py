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
#  modified by Xiaojie Wu <wxj6000@gmail.com>

'''
Non-relativistic RKS analytical Hessian
'''

from concurrent.futures import ThreadPoolExecutor
import numpy
import cupy
from pyscf import lib
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import (contract, add_sparse, get_avail_mem,
                                       reduce_to_device, transpose_sum)
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import _streams, num_devices, min_grid_blksize
from gpu4pyscf.hessian import jk
from gpu4pyscf.dft.numint import NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD
import ctypes

libgdft = numint.libgdft

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    mocc = mo_coeff[:,mo_occ>0]
    dm0 = cupy.dot(mocc, mocc.T) * 2

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    j_factor = 1.
    k_factor = 0.
    if with_k:
        if omega == 0:
            k_factor = hyb
        elif alpha == 0: # LR=0, only SR exchange
            pass
        elif hyb == 0: # SR=0, only LR exchange
            k_factor = alpha
        else: # SR and LR exchange with different ratios
            k_factor = alpha
    de2, ejk = rhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                          atmlst, max_memory, verbose,
                                          j_factor, k_factor)
    de2 += ejk  # (A,B,dR_A,dR_B)
    if with_k and omega != 0:
        j_factor = 0.
        omega = -omega # Prefer computing the SR part
        if alpha == 0: # LR=0, only SR exchange
            k_factor = hyb
        elif hyb == 0: # SR=0, only LR exchange
            # full range exchange was computed in the previous step
            k_factor = -alpha
        else: # SR and LR exchange with different ratios
            k_factor = hyb - alpha # =beta
        vhfopt = mf._opt_gpu.get(omega, None)
        with mol.with_range_coulomb(omega):
            de2 += rhf_hess._partial_ejk_ip2(
                mol, dm0, vhfopt, j_factor, k_factor, verbose=verbose)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    veff_diag = _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)
    t1 = log.timer_debug1('hessian of 2e part', *t1)

    aoslices = mol.aoslice_by_atom()
    vxc_dm = _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)
    if atmlst is None:
        atmlst = range(mol.natm)
    for i0, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        veff = vxc_dm[ia]
        de2[i0,i0] += contract('xypq,pq->xy', veff_diag[:,:,p0:p1], dm0[p0:p1])*2
        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            de2[i0,j0] += 2.0 * veff[:,:,q0:q1].sum(axis=2)

        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    if mf.do_nlc():
        de2 += _get_enlc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)

    log.timer('RKS partial hessian', *time0)
    return de2

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    natm = mol.natm
    assert atmlst is None or atmlst == range(natm)
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    avail_mem = get_avail_mem()
    max_memory = avail_mem * .8e-6

    mf = hessobj.base
    ni = mf._numint

    h1mo = _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    if mf.do_nlc():
        h1mo += _get_vnlc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    h1mo += rhf_grad.get_grad_hcore(hessobj.base.Gradients())

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)

    # Estimate the size of intermediate variables
    # dm, vj, and vk in [natm,3,nao_cart,nao_cart]
    nao_cart = mol.nao_cart()
    avail_mem -= 8 * h1mo.size
    slice_size = int(avail_mem*0.5) // (8*3*nao_cart*nao_cart*3)
    for atoms_slice in lib.prange(0, natm, slice_size):
        vj, vk = rhf_hess._get_jk_ip1(mol, dm0, with_k=with_k,
                                      atoms_slice=atoms_slice, verbose=verbose)
        veff = vj
        if with_k:
            vk *= .5 * hyb
            veff -= vk
        vj = vk = None
        if abs(omega) > 1e-10 and abs(alpha-hyb) > 1e-10:
            with mol.with_range_coulomb(omega):
                vk_lr = rhf_hess._get_jk_ip1(
                    mol, dm0, with_j=False, atoms_slice=atoms_slice, verbose=verbose)[1]
                vk_lr *= (alpha-hyb) * .5
                veff -= vk_lr
        atom0, atom1 = atoms_slice
        for i, ia in enumerate(range(atom0, atom1)):
            for ix in range(3):
                h1mo[ia,ix] += mo_coeff.T.dot(veff[i,ix].dot(mocc))
        vk_lr = veff = None
    return h1mo

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

    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])
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

    vmat = opt.unsort_orbitals(vmat, axis=[1,2])
    return vmat.reshape(3,3,nao,nao)

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

def _get_vxc_deriv2_task(hessobj, grids, mo_coeff, mo_occ, max_memory, device_id=0, verbose=0):
    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint
    nao = mol.nao
    opt = ni.gdftopt

    _sorted_mol = opt._sorted_mol
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    ngrids_glob = grids.coords.shape[0]
    grid_start, grid_end = numint.gen_grid_range(ngrids_glob, device_id)

    with cupy.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(mol, verbose)
        t1 = t0 = log.init_timer()
        mo_occ = cupy.asarray(mo_occ)
        mo_coeff = cupy.asarray(mo_coeff)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        dm0_sorted = opt.sort_orbitals(dm0, axis=[0,1])
        coeff = cupy.asarray(opt.coeff)
        vmat_dm = cupy.zeros((_sorted_mol.natm,3,3,nao))
        ipip = cupy.zeros((3,3,nao,nao))
        if xctype == 'LDA':
            ao_deriv = 1
            t1 = log.init_timer()
            for ao_mask, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                          grid_range=(grid_start, grid_end)):
                nao_non0 = len(mask)
                ao = contract('nip,ij->njp', ao_mask, coeff[mask])
                rho = numint.eval_rho2(_sorted_mol, ao[0], mo_coeff, mo_occ, mask, xctype)
                t1 = log.timer_debug2('eval rho', *t1)
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
                t1 = log.timer_debug2('eval vxc', *t1)
                wv = weight * vxc[0]
                aow = [numint._scale_ao(ao[i], wv) for i in range(1, 4)]
                _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)
                dm0_mask = dm0_sorted[mask[:,None], mask]

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

            vmat_dm = opt.unsort_orbitals(vmat_dm, axis=[3])
            for ia in range(_sorted_mol.natm):
                p0, p1 = aoslices[ia][2:]
                vmat_dm[ia] += contract('xypq,pq->xyp', ipip[:,:,:,p0:p1], dm0[:,p0:p1])
        elif xctype == 'GGA':
            ao_deriv = 2
            t1 = log.init_timer()
            for ao_mask, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                          grid_range=(grid_start, grid_end)):
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
                dm0_mask = dm0_sorted[mask[:,None], mask]
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
            vmat_dm = opt.unsort_orbitals(vmat_dm, axis=[3])
            for ia in range(_sorted_mol.natm):
                p0, p1 = aoslices[ia][2:]
                vmat_dm[ia] += contract('xypq,pq->xyp', ipip[:,:,:,p0:p1], dm0[:,p0:p1])
                vmat_dm[ia] += contract('yxqp,pq->xyp', ipip[:,:,p0:p1], dm0[:,p0:p1])

        elif xctype == 'MGGA':
            XX, XY, XZ = 4, 5, 6
            YX, YY, YZ = 5, 7, 8
            ZX, ZY, ZZ = 6, 8, 9
            ao_deriv = 2
            t1 = log.init_timer()
            for ao_mask, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                          grid_range=(grid_start, grid_end)):
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
                dm0_mask = dm0_sorted[mask[:,None], mask]
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
            vmat_dm = opt.unsort_orbitals(vmat_dm, axis=[3])
            for ia in range(_sorted_mol.natm):
                p0, p1 = aoslices[ia][2:]
                vmat_dm[ia] += contract('xypq,pq->xyp', ipip[:,:,:,p0:p1], dm0[:,p0:p1])
                vmat_dm[ia] += contract('yxqp,pq->xyp', ipip[:,:,p0:p1], dm0[:,p0:p1])
        t0 = log.timer_debug1(f'vxc_deriv2 on Device {device_id}', *t0)
    return vmat_dm

def _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory):
    '''Partially contracted vxc*dm'''
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids

    if grids.coords is None:
        grids.build(with_non0tab=True)

    ni = mf._numint
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _get_vxc_deriv2_task,
                hessobj, grids, mo_coeff, mo_occ, max_memory,
                device_id=device_id, verbose=mol.verbose)
            futures.append(future)
    vmat_dm_dist = [future.result() for future in futures]
    vmat_dm = reduce_to_device(vmat_dm_dist, inplace=True)
    return vmat_dm

def _get_enlc_deriv2_numerical(hessobj, mo_coeff, mo_occ, max_memory):
    """
        Attention: Numerical nlc energy 2nd derivative includes grid response.
    """
    mol = hessobj.mol
    mf = hessobj.base
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2

    de2 = cupy.empty([mol.natm, mol.natm, 3, 3])

    def get_nlc_de(grad_obj, dm):
        from gpu4pyscf.grad.rks import _get_denlc
        mol = grad_obj.mol
        denlc_orbital, denlc_grid = _get_denlc(grad_obj, mol, dm, max_memory = 500)
        denlc = 2 * denlc_orbital
        if grad_obj.grid_response:
            assert denlc_grid is not None
            denlc += denlc_grid
        return denlc

    dx = 1e-3
    mol_copy = mol.copy()
    grad_obj = mf.Gradients()
    grad_obj.grid_response = True
    if not grad_obj.grid_response:
        from gpu4pyscf.lib.cupy_helper import tag_array
        dm0 = tag_array(dm0, mo_coeff = mo_coeff, mo_occ = mo_occ)

    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            grad_obj.reset(mol_copy)
            de_p = get_nlc_de(grad_obj, dm0)

            xyz_m = mol.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            grad_obj.reset(mol_copy)
            de_m = get_nlc_de(grad_obj, dm0)

            de2[i_atom, :, i_xyz, :] = (de_p - de_m) / (2 * dx)
    grad_obj.reset(mol)

    return de2

def get_d2mu_dr2(ao):
    assert ao.ndim == 3
    nao = ao.shape[1]
    ngrids = ao.shape[2]

    d2mu_dr2 = cupy.empty([3, 3, nao, ngrids])
    d2mu_dr2[0,0,:,:] = ao[XX, :, :]
    d2mu_dr2[0,1,:,:] = ao[XY, :, :]
    d2mu_dr2[1,0,:,:] = ao[XY, :, :]
    d2mu_dr2[0,2,:,:] = ao[XZ, :, :]
    d2mu_dr2[2,0,:,:] = ao[XZ, :, :]
    d2mu_dr2[1,1,:,:] = ao[YY, :, :]
    d2mu_dr2[1,2,:,:] = ao[YZ, :, :]
    d2mu_dr2[2,1,:,:] = ao[YZ, :, :]
    d2mu_dr2[2,2,:,:] = ao[ZZ, :, :]
    return d2mu_dr2

def get_d3mu_dr3(ao):
    assert ao.ndim == 3
    nao = ao.shape[1]
    ngrids = ao.shape[2]

    d3mu_dr3 = cupy.empty([3, 3, 3, nao, ngrids])
    d3mu_dr3[0,0,0,:,:] = ao[XXX,:,:]
    d3mu_dr3[0,0,1,:,:] = ao[XXY,:,:]
    d3mu_dr3[0,1,0,:,:] = ao[XXY,:,:]
    d3mu_dr3[1,0,0,:,:] = ao[XXY,:,:]
    d3mu_dr3[0,0,2,:,:] = ao[XXZ,:,:]
    d3mu_dr3[0,2,0,:,:] = ao[XXZ,:,:]
    d3mu_dr3[2,0,0,:,:] = ao[XXZ,:,:]
    d3mu_dr3[0,1,1,:,:] = ao[XYY,:,:]
    d3mu_dr3[1,0,1,:,:] = ao[XYY,:,:]
    d3mu_dr3[1,1,0,:,:] = ao[XYY,:,:]
    d3mu_dr3[0,1,2,:,:] = ao[XYZ,:,:]
    d3mu_dr3[1,0,2,:,:] = ao[XYZ,:,:]
    d3mu_dr3[1,2,0,:,:] = ao[XYZ,:,:]
    d3mu_dr3[0,2,1,:,:] = ao[XYZ,:,:]
    d3mu_dr3[2,0,1,:,:] = ao[XYZ,:,:]
    d3mu_dr3[2,1,0,:,:] = ao[XYZ,:,:]
    d3mu_dr3[0,2,2,:,:] = ao[XZZ,:,:]
    d3mu_dr3[2,0,2,:,:] = ao[XZZ,:,:]
    d3mu_dr3[2,2,0,:,:] = ao[XZZ,:,:]
    d3mu_dr3[1,1,1,:,:] = ao[YYY,:,:]
    d3mu_dr3[1,1,2,:,:] = ao[YYZ,:,:]
    d3mu_dr3[1,2,1,:,:] = ao[YYZ,:,:]
    d3mu_dr3[2,1,1,:,:] = ao[YYZ,:,:]
    d3mu_dr3[1,2,2,:,:] = ao[YZZ,:,:]
    d3mu_dr3[2,1,2,:,:] = ao[YZZ,:,:]
    d3mu_dr3[2,2,1,:,:] = ao[YZZ,:,:]
    d3mu_dr3[2,2,2,:,:] = ao[ZZZ,:,:]

    return d3mu_dr3

def get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    natm = len(aoslices)
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert dm0.shape == (nao, nao)

    d2rho_dAdr = cupy.zeros([natm, 3, 3, ngrids])
    for i_atom in range(natm):
        p0, p1 = aoslices[i_atom][2:]
        # d2rho_dAdr[i_atom, :, :, :] += cupy.einsum('dDig,jg,ij->dDg', -d2mu_dr2[:, :, p0:p1, :], mu, dm0[p0:p1, :])
        # d2rho_dAdr[i_atom, :, :, :] += cupy.einsum('dDig,jg,ij->dDg', -d2mu_dr2[:, :, p0:p1, :], mu, dm0[:, p0:p1].T)
        # d2rho_dAdr[i_atom, :, :, :] += cupy.einsum('dig,Djg,ij->dDg', -dmu_dr[:, p0:p1, :], dmu_dr, dm0[p0:p1, :])
        # d2rho_dAdr[i_atom, :, :, :] += cupy.einsum('dig,Djg,ij->dDg', -dmu_dr[:, p0:p1, :], dmu_dr, dm0[:, p0:p1].T)
        nu_dot_dm = dm0[p0:p1, :] @ mu
        d2rho_dAdr[i_atom, :, :, :] += contract('dDig,ig->dDg', -d2mu_dr2[:, :, p0:p1, :], nu_dot_dm)
        nu_dot_dm = None
        mu_dot_dm = dm0[:, p0:p1].T @ mu
        d2rho_dAdr[i_atom, :, :, :] += contract('dDig,ig->dDg', -d2mu_dr2[:, :, p0:p1, :], mu_dot_dm)
        mu_dot_dm = None
        dnudr_dot_dm = contract('djg,ij->dig', dmu_dr, dm0[p0:p1, :])
        d2rho_dAdr[i_atom, :, :, :] += contract('dig,Dig->dDg', -dmu_dr[:, p0:p1, :], dnudr_dot_dm)
        dnudr_dot_dm = None
        dmudr_dot_dm = contract('djg,ij->dig', dmu_dr, dm0[:, p0:p1].T)
        d2rho_dAdr[i_atom, :, :, :] += contract('dig,Dig->dDg', -dmu_dr[:, p0:p1, :], dmudr_dot_dm)
        dmudr_dot_dm = None
    return d2rho_dAdr

def get_d2rho_dAdr_grid_response(d2mu_dr2, dmu_dr, mu, dm0, atom_to_grid_index_map = None, i_atom = None):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert dm0.shape == (nao, nao)

    if i_atom is None:
        assert atom_to_grid_index_map is not None
        natm = len(atom_to_grid_index_map)

        d2rho_dAdr_grid_response = cupy.zeros([natm, 3, 3, ngrids])
        for i_atom in range(natm):
            associated_grid_index = atom_to_grid_index_map[i_atom]
            # d2rho_dAdr_response  = cupy.einsum('dDig,jg,ij->dDg', d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index], dm0)
            # d2rho_dAdr_response += cupy.einsum('dDig,jg,ij->dDg', d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index], dm0.T)
            # d2rho_dAdr_response += cupy.einsum('dig,Djg,ij->dDg', dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], dm0)
            # d2rho_dAdr_response += cupy.einsum('dig,Djg,ij->dDg', dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], dm0.T)
            dm_dot_mu_and_nu = (dm0 + dm0.T) @ mu[:, associated_grid_index]
            d2rho_dAdr_response  = contract('dDig,ig->dDg', d2mu_dr2[:, :, :, associated_grid_index], dm_dot_mu_and_nu)
            dm_dot_mu_and_nu = None
            dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr[:, :, associated_grid_index], dm0 + dm0.T)
            d2rho_dAdr_response += contract('dig,Dig->dDg', dmu_dr[:, :, associated_grid_index], dm_dot_dmu_and_dnu)
            dm_dot_dmu_and_dnu = None

            d2rho_dAdr_grid_response[i_atom][:, :, associated_grid_index] = d2rho_dAdr_response
    else:
        assert atom_to_grid_index_map is None

        # Here we assume all grids belong to atom i
        dm_dot_mu_and_nu = (dm0 + dm0.T) @ mu
        d2rho_dAdr_grid_response  = contract('dDig,ig->dDg', d2mu_dr2, dm_dot_mu_and_nu)
        dm_dot_mu_and_nu = None
        dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr, dm0 + dm0.T)
        d2rho_dAdr_grid_response += contract('dig,Dig->dDg', dmu_dr, dm_dot_dmu_and_dnu)
        dm_dot_dmu_and_dnu = None

    return d2rho_dAdr_grid_response

def get_drhodA_dgammadA_orbital_response(d2mu_dr2, dmu_dr, mu, drho_dr, dm0, aoslices):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    natm = len(aoslices)
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert drho_dr.shape == (3, ngrids)
    assert dm0.shape == (nao, nao)

    drhodr_dot_dmudr = contract('Djg,Dg->jg', dmu_dr, drho_dr)

    drho_dA = cupy.zeros([natm, 3, ngrids])
    dgamma_dA = cupy.zeros([natm, 3, ngrids])
    for i_atom in range(natm):
        p0, p1 = aoslices[i_atom][2:]

        # drho_dA[i_atom, :, :] += cupy.einsum('dig,jg,ij->dg', -dmu_dr[:, p0:p1, :], mu, dm0[p0:p1, :])
        # drho_dA[i_atom, :, :] += cupy.einsum('dig,jg,ij->dg', -dmu_dr[:, p0:p1, :], mu, dm0[:, p0:p1].T)
        nu_dot_dm = dm0[p0:p1, :] @ mu
        drho_dA[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], nu_dot_dm)
        mu_dot_dm = dm0[:, p0:p1].T @ mu
        drho_dA[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], mu_dot_dm)

        # dgamma_dA[i_atom, :, :] += cupy.einsum('dDig,jg,Dg,ij->dg', -d2mu_dr2[:, :, p0:p1, :], mu, drho_dr, dm0[p0:p1, :])
        # dgamma_dA[i_atom, :, :] += cupy.einsum('dDig,jg,Dg,ij->dg', -d2mu_dr2[:, :, p0:p1, :], mu, drho_dr, dm0[:, p0:p1].T)
        # dgamma_dA[i_atom, :, :] += cupy.einsum('dig,Djg,Dg,ij->dg', -dmu_dr[:, p0:p1, :], dmu_dr, drho_dr, dm0[p0:p1, :])
        # dgamma_dA[i_atom, :, :] += cupy.einsum('dig,Djg,Dg,ij->dg', -dmu_dr[:, p0:p1, :], dmu_dr, drho_dr, dm0[:, p0:p1].T)
        d2mudAdr_dot_drhodr = contract('dDig,Dg->dig', -d2mu_dr2[:, :, p0:p1, :], drho_dr)
        dgamma_dA[i_atom, :, :] += contract('dig,ig->dg', d2mudAdr_dot_drhodr, nu_dot_dm)
        dgamma_dA[i_atom, :, :] += contract('dig,ig->dg', d2mudAdr_dot_drhodr, mu_dot_dm)
        d2mudAdr_dot_drhodr = None
        nu_dot_dm = None
        mu_dot_dm = None
        drhodr_dot_dnudr_dot_dm = dm0[p0:p1, :] @ drhodr_dot_dmudr
        dgamma_dA[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], drhodr_dot_dnudr_dot_dm)
        drhodr_dot_dnudr_dot_dm = None
        drhodr_dot_dmudr_dot_dm = dm0[:, p0:p1].T @ drhodr_dot_dmudr
        dgamma_dA[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], drhodr_dot_dmudr_dot_dm)
        drhodr_dot_dmudr_dot_dm = None
    dgamma_dA *= 2

    return drho_dA, dgamma_dA

def get_drhodA_dgammadA_grid_response(d2mu_dr2, dmu_dr, mu, drho_dr, dm0, atom_to_grid_index_map = None, i_atom = None):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert drho_dr.shape == (3, ngrids)
    assert dm0.shape == (nao, nao)

    if i_atom is None:
        assert atom_to_grid_index_map is not None

        natm = len(atom_to_grid_index_map)
        drho_dA_grid_response   = cupy.zeros([natm, 3, ngrids])
        dgamma_dA_grid_response = cupy.zeros([natm, 3, ngrids])
        for i_atom in range(natm):
            associated_grid_index = atom_to_grid_index_map[i_atom]
            # rho_response  = cupy.einsum('dig,jg,ij->dg', dmu_dr[:, :, associated_grid_index], mu[:, associated_grid_index], dm0)
            # rho_response += cupy.einsum('dig,jg,ij->dg', dmu_dr[:, :, associated_grid_index], mu[:, associated_grid_index], dm0.T)
            dm_dot_mu_and_nu = (dm0 + dm0.T) @ mu[:, associated_grid_index]
            rho_response = contract('dig,ig->dg', dmu_dr[:, :, associated_grid_index], dm_dot_mu_and_nu)
            drho_dA_grid_response[i_atom][:, associated_grid_index] = rho_response
            rho_response = None

            # gamma_response  = cupy.einsum('dDig,jg,Dg,ij->dg',
            #     d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index], drho_dr[:, associated_grid_index], dm0)
            # gamma_response += cupy.einsum('dDig,jg,Dg,ij->dg',
            #     d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index], drho_dr[:, associated_grid_index], dm0.T)
            # gamma_response += cupy.einsum('dig,Djg,Dg,ij->dg',
            #     dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], drho_dr[:, associated_grid_index], dm0)
            # gamma_response += cupy.einsum('dig,Djg,Dg,ij->dg',
            #     dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], drho_dr[:, associated_grid_index], dm0.T)
            d2mudr2_dot_drhodr = contract('dDig,Dg->dig', d2mu_dr2[:, :, :, associated_grid_index], drho_dr[:, associated_grid_index])
            gamma_response  = contract('dig,ig->dg', d2mudr2_dot_drhodr, dm_dot_mu_and_nu)
            d2mudr2_dot_drhodr = None
            dm_dot_mu_and_nu = None
            dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr[:, :, associated_grid_index], dm0 + dm0.T)
            dmudr_dot_drhodr = contract('dig,dg->ig', dmu_dr[:, :, associated_grid_index], drho_dr[:, associated_grid_index])
            gamma_response += contract('dig,ig->dg', dm_dot_dmu_and_dnu, dmudr_dot_drhodr)
            dmudr_dot_drhodr = None
            dm_dot_dmu_and_dnu = None
            dgamma_dA_grid_response[i_atom][:, associated_grid_index] = gamma_response
            gamma_response = None
    else:
        assert atom_to_grid_index_map is None

        # Here we assume all grids belong to atom i
        dm_dot_mu_and_nu = (dm0 + dm0.T) @ mu
        drho_dA_grid_response = contract('dig,ig->dg', dmu_dr, dm_dot_mu_and_nu)

        d2mudr2_dot_drhodr = contract('dDig,Dg->dig', d2mu_dr2, drho_dr)
        dgamma_dA_grid_response = contract('dig,ig->dg', d2mudr2_dot_drhodr, dm_dot_mu_and_nu)
        d2mudr2_dot_drhodr = None
        dm_dot_mu_and_nu = None
        dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr, dm0 + dm0.T)
        dmudr_dot_drhodr = contract('dig,dg->ig', dmu_dr, drho_dr)
        dgamma_dA_grid_response += contract('dig,ig->dg', dm_dot_dmu_and_dnu, dmudr_dot_drhodr)
        dmudr_dot_drhodr = None
        dm_dot_dmu_and_dnu = None

    dgamma_dA_grid_response *= 2

    return drho_dA_grid_response, dgamma_dA_grid_response

def get_d2rhodAdB_d2gammadAdB(mol, grids_coords, dm0):
    """
        This function should never be used in practice. It requires crazy amount of memory,
        and it's left for debug purpose only. Use the contract function instead.
    """
    natm = mol.natm
    ngrids = grids_coords.shape[0]

    ao = numint.eval_ao(mol, grids_coords, deriv = 3, gdftopt = None, transpose = False)
    rho_drho = numint.eval_rho(mol, ao[:4, :], dm0, xctype = "GGA", hermi = 1, with_lapl = False)
    drho = rho_drho[1:4, :]
    mu = ao[0, :, :]
    dmu_dr = ao[1:4, :, :]
    d2mu_dr2 = get_d2mu_dr2(ao)
    d3mu_dr3 = get_d3mu_dr3(ao)

    aoslices = mol.aoslice_by_atom()
    d2rho_dAdB = cupy.zeros([natm, natm, 3, 3, ngrids])
    d2gamma_dAdB = cupy.zeros([natm, natm, 3, 3, ngrids])
    for i_atom in range(natm):
        pi0, pi1 = aoslices[i_atom][2:]
        d2rho_dAdB[i_atom, i_atom, :, :, :] += cupy.einsum('dDig,jg,ij->dDg', d2mu_dr2[:, :, pi0:pi1, :], mu, dm0[pi0:pi1, :])
        d2rho_dAdB[i_atom, i_atom, :, :, :] += cupy.einsum('dDig,jg,ij->dDg', d2mu_dr2[:, :, pi0:pi1, :], mu, dm0[:, pi0:pi1].T)
        d2gamma_dAdB[i_atom, i_atom, :, :, :] += cupy.einsum('dDPig,jg,Pg,ij->dDg', d3mu_dr3[:, :, :, pi0:pi1, :], mu, drho, dm0[pi0:pi1, :])
        d2gamma_dAdB[i_atom, i_atom, :, :, :] += cupy.einsum('dDPig,jg,Pg,ij->dDg', d3mu_dr3[:, :, :, pi0:pi1, :], mu, drho, dm0[:, pi0:pi1].T)
        d2gamma_dAdB[i_atom, i_atom, :, :, :] += cupy.einsum('dDig,Pjg,Pg,ij->dDg', d2mu_dr2[:, :, pi0:pi1, :], dmu_dr, drho, dm0[pi0:pi1, :])
        d2gamma_dAdB[i_atom, i_atom, :, :, :] += cupy.einsum('dDig,Pjg,Pg,ij->dDg', d2mu_dr2[:, :, pi0:pi1, :], dmu_dr, drho, dm0[:, pi0:pi1].T)
        for j_atom in range(natm):
            pj0, pj1 = aoslices[j_atom][2:]
            d2rho_dAdB[i_atom, j_atom, :, :, :] += cupy.einsum('dig,Djg,ij->dDg',
                dmu_dr[:, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], dm0[pi0:pi1, pj0:pj1])
            d2rho_dAdB[i_atom, j_atom, :, :, :] += cupy.einsum('dig,Djg,ij->dDg',
                dmu_dr[:, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], dm0[pj0:pj1, pi0:pi1].T)
            d2gamma_dAdB[i_atom, j_atom, :, :, :] += cupy.einsum('dPig,Djg,Pg,ij->dDg',
                d2mu_dr2[:, :, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], drho, dm0[pi0:pi1, pj0:pj1])
            d2gamma_dAdB[i_atom, j_atom, :, :, :] += cupy.einsum('dPig,Djg,Pg,ij->dDg',
                d2mu_dr2[:, :, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], drho, dm0[pj0:pj1, pi0:pi1].T)
            d2gamma_dAdB[i_atom, j_atom, :, :, :] += cupy.einsum('dig,DPjg,Pg,ij->dDg',
                dmu_dr[:, pi0:pi1, :], d2mu_dr2[:, :, pj0:pj1, :], drho, dm0[pi0:pi1, pj0:pj1])
            d2gamma_dAdB[i_atom, j_atom, :, :, :] += cupy.einsum('dig,DPjg,Pg,ij->dDg',
                dmu_dr[:, pi0:pi1, :], d2mu_dr2[:, :, pj0:pj1, :], drho, dm0[pj0:pj1, pi0:pi1].T)

    d2rho_dAdr = get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices)
    d2gamma_dAdB += cupy.einsum('AdPg,BDPg->ABdDg', d2rho_dAdr, d2rho_dAdr)
    d2gamma_dAdB *= 2
    return d2rho_dAdB, d2gamma_dAdB

def contract_d2rhodAdB_d2gammadAdB(d3mu_dr3, d2mu_dr2, dmu_dr, mu, drho_dr, dm0, aoslices, fw_rho, fw_gamma):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    natm = len(aoslices)
    assert d3mu_dr3.shape == (3, 3, 3, nao, ngrids)
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert drho_dr.shape == (3, ngrids)
    assert dm0.shape == (nao, nao)

    drhodr_dot_dmudr = contract('djg,dg->jg', dmu_dr, drho_dr)

    d2e_rho_dAdB = cupy.zeros([natm, natm, 3, 3])
    d2e_gamma_dAdB = cupy.zeros([natm, natm, 3, 3])
    for i_atom in range(natm):
        pi0, pi1 = aoslices[i_atom][2:]

        nu_dot_dm = dm0[pi0:pi1, :] @ mu
        d2rho_dA2  = contract('dDig,ig->dDg', d2mu_dr2[:, :, pi0:pi1, :], nu_dot_dm)
        mu_dot_dm = dm0[:, pi0:pi1].T @ mu
        d2rho_dA2 += contract('dDig,ig->dDg', d2mu_dr2[:, :, pi0:pi1, :], mu_dot_dm)
        d2e_rho_dAdB[i_atom, i_atom, :, :] += contract('dDg,g->dD', d2rho_dA2, fw_rho)
        d2rho_dA2 = None

        d3mudA2dr_dot_drhodr = contract('dDPig,Pg->dDig', d3mu_dr3[:, :, :, pi0:pi1, :], drho_dr)
        d2gamma_dA2  = contract('dDig,ig->dDg', d3mudA2dr_dot_drhodr, nu_dot_dm)
        d2gamma_dA2 += contract('dDig,ig->dDg', d3mudA2dr_dot_drhodr, mu_dot_dm)
        d3mudA2dr_dot_drhodr = None
        nu_dot_dm = None
        mu_dot_dm = None
        drhodr_dot_dmudr_dot_dm = dm0[pi0:pi1, :] @ drhodr_dot_dmudr
        d2gamma_dA2 += contract('dDig,ig->dDg', d2mu_dr2[:, :, pi0:pi1, :], drhodr_dot_dmudr_dot_dm)
        drhodr_dot_dmudr_dot_dm = None
        drhodr_dot_dnudr_dot_dm = dm0[:, pi0:pi1].T @ drhodr_dot_dmudr
        d2gamma_dA2 += contract('dDig,ig->dDg', d2mu_dr2[:, :, pi0:pi1, :], drhodr_dot_dnudr_dot_dm)
        drhodr_dot_dnudr_dot_dm = None
        d2e_gamma_dAdB[i_atom, i_atom, :, :] += contract('dDg,g->dD', d2gamma_dA2, fw_gamma)
        d2gamma_dA2 = None

        for j_atom in range(natm):
            pj0, pj1 = aoslices[j_atom][2:]
            dnudr_dot_dm = contract('djg,ij->dig', dmu_dr[:, pj0:pj1, :], dm0[pi0:pi1, pj0:pj1])
            d2rho_dAdB  = contract('dig,Dig->dDg', dmu_dr[:, pi0:pi1, :], dnudr_dot_dm)
            dmudr_dot_dm = contract('djg,ij->dig', dmu_dr[:, pj0:pj1, :], dm0[pj0:pj1, pi0:pi1].T)
            d2rho_dAdB += contract('dig,Dig->dDg', dmu_dr[:, pi0:pi1, :], dmudr_dot_dm)
            d2e_rho_dAdB[i_atom, j_atom, :, :] += contract('dDg,g->dD', d2rho_dAdB, fw_rho)
            d2rho_dAdB = None

            drhodr_dot_d2mudAdr = contract('dDig,Dg->dig', d2mu_dr2[:, :, pi0:pi1, :], drho_dr)
            d2gamma_dAdB  = contract('dig,Dig->dDg', drhodr_dot_d2mudAdr, dnudr_dot_dm)
            dnudr_dot_dm = None
            d2gamma_dAdB += contract('dig,Dig->dDg', drhodr_dot_d2mudAdr, dmudr_dot_dm)
            dmudr_dot_dm = None
            drhodr_dot_d2mudAdr = None
            d2gamma_dAdB = contract('dDg,g->dD', d2gamma_dAdB, fw_gamma)
            d2e_gamma_dAdB[i_atom, j_atom, :, :] += d2gamma_dAdB
            d2e_gamma_dAdB[j_atom, i_atom, :, :] += d2gamma_dAdB.T
            d2gamma_dAdB = None

    d2rho_dAdr = get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices)
    d2e_gamma_dAdB += contract('AdPg,BDPg->ABdD', d2rho_dAdr, d2rho_dAdr * fw_gamma)

    return d2e_rho_dAdB + 2 * d2e_gamma_dAdB

def _get_enlc_deriv2(hessobj, mo_coeff, mo_occ, max_memory):
    """
        Equation notation follows:
        Liang J, Feng X, Liu X, Head-Gordon M. Analytical harmonic vibrational frequencies with
        VV10-containing density functionals: Theory, efficient implementation, and
        benchmark assessments. J Chem Phys. 2023 May 28;158(20):204109. doi: 10.1063/5.0152838.
    """

    mol = hessobj.mol
    mf = hessobj.base

    mocc = mo_coeff[:,mo_occ>0]
    dm0 = 2 * mocc @ mocc.T

    grids = mf.nlcgrids
    if grids.coords is None:
        grids.build()

    if numint.libxc.is_nlc(mf.xc):
        xc_code = mf.xc
    else:
        xc_code = mf.nlc
    nlc_coefs = mf._numint.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]

    kappa_prefactor = nlc_pars[0] * 1.5 * numpy.pi * (9 * numpy.pi)**(-1.0/6.0)
    C_in_omega = nlc_pars[1]
    beta = 0.03125 * (3.0 / nlc_pars[0]**2)**0.75

    # ao = numint.eval_ao(mol, grids.coords, deriv = 3, gdftopt = None, transpose = False)
    # rho_drho = numint.eval_rho(mol, ao, dm0, xctype = "NLC", hermi = 1, with_lapl = False)

    ngrids_full = grids.coords.shape[0]
    rho_drho = cupy.empty([4, ngrids_full])

    available_gpu_memory = get_avail_mem()
    available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
    ao_nbytes_per_grid = ((4*2) * mol.nao + 4) * 8 # factor of 2 from the ao sorting inside numint.eval_ao()
    ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
    if ngrids_per_batch < 16:
        raise MemoryError(f"Out of GPU memory for NLC energy second derivative, available gpu memory = {get_avail_mem()}"
                          f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids = {ngrids_full}")
    ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
    ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

    for g0 in range(0, ngrids_full, ngrids_per_batch):
        g1 = min(g0 + ngrids_per_batch, ngrids_full)
        split_grids_coords = grids.coords[g0:g1, :]
        split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 1, gdftopt = None, transpose = False)
        split_rho_drho = numint.eval_rho(mol, split_ao, dm0, xctype = "NLC", hermi = 1, with_lapl = False)
        rho_drho[:, g0:g1] = split_rho_drho

    rho_i = rho_drho[0,:]

    rho_nonzero_mask = (rho_i >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD)

    rho_i = rho_i[rho_nonzero_mask]
    nabla_rho_i = rho_drho[1:4, rho_nonzero_mask]
    grids_coords = cupy.ascontiguousarray(grids.coords[rho_nonzero_mask, :])
    grids_weights = grids.weights[rho_nonzero_mask]
    ngrids = grids_coords.shape[0]

    gamma_i = nabla_rho_i[0,:]**2 + nabla_rho_i[1,:]**2 + nabla_rho_i[2,:]**2
    omega_i = cupy.sqrt(C_in_omega * gamma_i**2 / rho_i**4 + (4.0/3.0*numpy.pi) * rho_i)
    kappa_i = kappa_prefactor * rho_i**(1.0/6.0)

    U_i = cupy.empty(ngrids)
    W_i = cupy.empty(ngrids)
    A_i = cupy.empty(ngrids)
    B_i = cupy.empty(ngrids)
    C_i = cupy.empty(ngrids)
    E_i = cupy.empty(ngrids)

    stream = cupy.cuda.get_current_stream()
    libgdft.VXC_vv10nlc_hess_eval_UWABCE(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(U_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(W_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(B_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(C_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(E_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids)
    )

    domega_drho_i         = cupy.empty(ngrids)
    domega_dgamma_i       = cupy.empty(ngrids)
    d2omega_drho2_i       = cupy.empty(ngrids)
    d2omega_dgamma2_i     = cupy.empty(ngrids)
    d2omega_drho_dgamma_i = cupy.empty(ngrids)
    libgdft.VXC_vv10nlc_hess_eval_omega_derivative(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(domega_drho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_drho2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_dgamma2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_drho_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(gamma_i.data.ptr, ctypes.c_void_p),
        ctypes.c_double(C_in_omega),
        ctypes.c_int(ngrids)
    )
    dkappa_drho_i   = kappa_prefactor * (1.0/6.0) * rho_i**(-5.0/6.0)
    d2kappa_drho2_i = kappa_prefactor * (-5.0/36.0) * rho_i**(-11.0/6.0)

    f_rho_i = beta + E_i + rho_i * (dkappa_drho_i * U_i + domega_drho_i * W_i)
    f_gamma_i = rho_i * domega_dgamma_i * W_i
    f_rho_i   =   f_rho_i * grids_weights
    f_gamma_i = f_gamma_i * grids_weights

    aoslices = mol.aoslice_by_atom()
    natm = mol.natm

    # ao = numint.eval_ao(mol, grids.coords, deriv = 3, gdftopt = None, transpose = False)
    # ao_nonzero_rho = ao[:, :, rho_nonzero_mask]
    # mu = ao_nonzero_rho[0, :, :]
    # dmu_dr = ao_nonzero_rho[1:4, :, :]
    # d2mu_dr2 = get_d2mu_dr2(ao_nonzero_rho)
    # d3mu_dr3 = get_d3mu_dr3(ao_nonzero_rho)

    # drho_dA, dgamma_dA = get_drhodA_dgammadA_orbital_response(d2mu_dr2, dmu_dr, mu, nabla_rho_i, dm0, aoslices)
    # d2e = contract_d2rhodAdB_d2gammadAdB(d3mu_dr3, d2mu_dr2, dmu_dr, mu, nabla_rho_i, dm0, aoslices,
    #                                      f_rho_i, f_gamma_i)

    drho_dA   = cupy.empty([natm, 3, ngrids], order = "C")
    dgamma_dA = cupy.empty([natm, 3, ngrids], order = "C")
    d2e = cupy.zeros([natm, natm, 3, 3])

    available_gpu_memory = get_avail_mem()
    available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
    ao_nbytes_per_grid = ((20 + 1*2 + 3*2 + 9 + 27) * mol.nao + (3*2 + 9) * mol.natm) * 8
    ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
    if ngrids_per_batch < 16:
        raise MemoryError(f"Out of GPU memory for NLC energy second derivative, available gpu memory = {get_avail_mem()}"
                          f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (nonzero rho) = {ngrids}")
    ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
    ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

    for g0 in range(0, ngrids, ngrids_per_batch):
        g1 = min(g0 + ngrids_per_batch, ngrids)
        split_grids_coords = grids_coords[g0:g1, :]
        split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 3, gdftopt = None, transpose = False)

        mu = split_ao[0, :, :]
        dmu_dr = split_ao[1:4, :, :]
        d2mu_dr2 = get_d2mu_dr2(split_ao)
        d3mu_dr3 = get_d3mu_dr3(split_ao)
        split_drho_dr = nabla_rho_i[:, g0:g1]

        split_drho_dA, split_dgamma_dA = get_drhodA_dgammadA_orbital_response(d2mu_dr2, dmu_dr, mu, split_drho_dr, dm0, aoslices)
        drho_dA  [:, :, g0:g1] = split_drho_dA
        dgamma_dA[:, :, g0:g1] = split_dgamma_dA

        split_fw_rho   = f_rho_i  [g0:g1]
        split_fw_gamma = f_gamma_i[g0:g1]
        d2e += contract_d2rhodAdB_d2gammadAdB(d3mu_dr3, d2mu_dr2, dmu_dr, mu, split_drho_dr, dm0, aoslices, split_fw_rho, split_fw_gamma)

        split_ao = None
        mu = None
        dmu_dr = None
        d2mu_dr2 = None
        d3mu_dr3 = None
        split_drho_dA = None
        split_dgamma_dA = None

    drho_dA   = cupy.ascontiguousarray(drho_dA)
    dgamma_dA = cupy.ascontiguousarray(dgamma_dA)
    f_rho_A_i   = cupy.empty([mol.natm, 3, ngrids], order = "C")
    f_gamma_A_i = cupy.empty([mol.natm, 3, ngrids], order = "C")

    libgdft.VXC_vv10nlc_hess_eval_f_t(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(f_rho_A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(f_gamma_A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(U_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(W_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(B_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(C_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_drho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(dkappa_drho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_drho2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_dgamma2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_drho_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2kappa_drho2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(drho_dA.data.ptr, ctypes.c_void_p),
        ctypes.cast(dgamma_dA.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(3 * mol.natm),
    )

    d2e += contract("Adg,BDg->ABdD",   drho_dA,   f_rho_A_i * grids_weights)
    d2e += contract("Adg,BDg->ABdD", dgamma_dA, f_gamma_A_i * grids_weights)

    return d2e

def _get_vxc_deriv1_task(hessobj, grids, mo_coeff, mo_occ, max_memory, device_id=0):
    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint
    nao, nmo = mo_coeff.shape
    natm = mol.natm
    opt = ni.gdftopt

    _sorted_mol = opt._sorted_mol
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    ngrids_glob = grids.coords.shape[0]
    grid_start, grid_end = numint.gen_grid_range(ngrids_glob, device_id)
    with cupy.cuda.Device(device_id), _streams[device_id]:
        mo_occ = cupy.asarray(mo_occ)
        mo_coeff = cupy.asarray(mo_coeff)
        coeff = cupy.asarray(opt.coeff)
        mocc = mo_coeff[:,mo_occ>0]
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        nocc = mocc.shape[1]

        log = logger.new_logger(mol, mol.verbose)
        v_ip = cupy.zeros((3,nao,nao))
        vmat = cupy.zeros((natm,3,nao,nocc))
        max_memory = max(2000, max_memory-vmat.size*8/1e6)
        t1 = t0 = log.init_timer()
        if xctype == 'LDA':
            ao_deriv = 1
            for ao, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                     grid_range=(grid_start, grid_end)):
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
                for ia in range(natm):
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
            for ao, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                     grid_range=(grid_start, grid_end)):
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
                for ia in range(natm):
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
            for ao, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                     grid_range=(grid_start, grid_end)):
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
                for ia in range(natm):
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
        t0 = log.timer_debug1(f'vxc_deriv1 on Device {device_id}', *t0)

        # Inplace transform the AO to MO.
        v_mo = cupy.ndarray((natm,3,nmo,nocc), dtype=vmat.dtype, memptr=vmat.data)
        vmat_tmp = cupy.empty([3,nao,nao])
        for ia in range(natm):
            p0, p1 = aoslices[ia][2:]
            vmat_tmp[:] = 0.
            vmat_tmp[:,p0:p1] += v_ip[:,p0:p1]
            vmat_tmp[:,:,p0:p1] += v_ip[:,p0:p1].transpose(0,2,1)
            tmp = contract('xij,jq->xiq', vmat_tmp, mocc)
            tmp += vmat[ia]
            contract('xiq,ip->xpq', tmp, mo_coeff, alpha=-1., out=v_mo[ia])
    return v_mo

def _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
    '''
    Derivatives of Vxc matrix in MO bases
    '''
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids

    if grids.coords is None:
        grids.build(with_non0tab=True)

    ni = mf._numint
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _get_vxc_deriv1_task,
                hessobj, grids, mo_coeff, mo_occ, max_memory,
                device_id=device_id)
            futures.append(future)
    vmat_dist = [future.result() for future in futures]
    vmat = reduce_to_device(vmat_dist, inplace=True)
    return vmat

def _get_vnlc_deriv1_numerical(hessobj, mo_coeff, mo_occ, max_memory):
    """
        Attention: Numerical nlc Fock matrix 1st derivative includes grid response.
    """
    mol = hessobj.mol
    mf = hessobj.base
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2

    nao = mol.nao
    vmat = cupy.empty([mol.natm, 3, nao, nao])

    def get_nlc_vmat(mol, mf, dm):
        ni = mf._numint
        if ni.libxc.is_nlc(mf.xc):
            xc = mf.xc
        else:
            assert ni.libxc.is_nlc(mf.nlc)
            xc = mf.nlc
        mf.nlcgrids.build()
        _, _, vnlc = ni.nr_nlc_vxc(mol, mf.nlcgrids, xc, dm)
        return vnlc

    dx = 1e-3
    mol_copy = mol.copy()
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            mol_copy.build()
            mf.reset(mol_copy)
            vmat_p = get_nlc_vmat(mol_copy, mf, dm0)

            xyz_m = mol.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            mf.reset(mol_copy)
            vmat_m = get_nlc_vmat(mol_copy, mf, dm0)

            vmat[i_atom, i_xyz, :, :] = (vmat_p - vmat_m) / (2 * dx)
    mf.reset(mol)

    vmat = contract('Adij,jq->Adiq', vmat, mocc)
    vmat = contract('Adiq,ip->Adpq', vmat, mo_coeff)
    return vmat

def get_dweight_dA(mol, grids):
    ngrids = grids.coords.shape[0]
    assert grids.atm_idx.shape[0] == ngrids
    assert grids.quadrature_weights.shape[0] == ngrids
    atm_coords = cupy.asarray(mol.atom_coords(), order = "C")

    from gpu4pyscf.dft import radi
    a_factor = radi.get_treutler_fac(mol, grids.atomic_radii)

    dweight_dA = cupy.zeros([mol.natm, 3, ngrids], order = "C")
    libgdft.GDFTbecke_partition_weight_derivative(
        ctypes.cast(dweight_dA.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids.coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids.quadrature_weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(atm_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(a_factor.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids.atm_idx.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(mol.natm),
    )
    dweight_dA[grids.atm_idx, 0, cupy.arange(ngrids)] = -cupy.sum(dweight_dA[:, 0, :], axis=[0])
    dweight_dA[grids.atm_idx, 1, cupy.arange(ngrids)] = -cupy.sum(dweight_dA[:, 1, :], axis=[0])
    dweight_dA[grids.atm_idx, 2, cupy.arange(ngrids)] = -cupy.sum(dweight_dA[:, 2, :], axis=[0])

    return dweight_dA

def get_d2weight_dAdB(mol, grids):
    ngrids = grids.coords.shape[0]
    assert grids.atm_idx.shape[0] == ngrids
    assert grids.quadrature_weights.shape[0] == ngrids
    atm_coords = cupy.asarray(mol.atom_coords(), order = "C")

    from gpu4pyscf.dft import radi
    a_factor = radi.get_treutler_fac(mol, grids.atomic_radii)

    d2weight_dAdB = cupy.zeros([mol.natm, mol.natm, 3, 3, ngrids], order = "C")
    libgdft.GDFTbecke_partition_weight_second_derivative(
        ctypes.cast(d2weight_dAdB.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids.coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids.quadrature_weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(atm_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(a_factor.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids.atm_idx.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(mol.natm),
    )

    range_ngrids = cupy.arange(ngrids)
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            for j_xyz in range(3):
                d2weight_dAdB[i_atom, grids.atm_idx, i_xyz, j_xyz, range_ngrids] = -cupy.sum(d2weight_dAdB[i_atom, :, i_xyz, j_xyz, :], axis=[0])

    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            for j_xyz in range(3):
                d2weight_dAdB[grids.atm_idx, i_atom, i_xyz, j_xyz, range_ngrids] = -cupy.sum(d2weight_dAdB[:, i_atom, i_xyz, j_xyz, :], axis=[0])

    return d2weight_dAdB

def _get_vnlc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
    """
        Equation notation follows:
        Liang J, Feng X, Liu X, Head-Gordon M. Analytical harmonic vibrational frequencies with
        VV10-containing density functionals: Theory, efficient implementation, and
        benchmark assessments. J Chem Phys. 2023 May 28;158(20):204109. doi: 10.1063/5.0152838.
    """

    # Note (Henry Wang 20250428):
    # We observed that in several very simple systems, for example H2O2, H2CO, C2H4,
    # if we do not include the grid response term, the analytical and numerical Fock matrix
    # derivative, although only diff by else than 1e-7 (norm 1), can cause a 1e-3 error in hessian,
    # likely because the CPHF converged to a different solution.
    grid_response = True

    mol = hessobj.mol
    mf = hessobj.base
    natm = mol.natm

    mocc = mo_coeff[:,mo_occ>0]
    dm0 = 2 * mocc @ mocc.T

    grids = mf.nlcgrids
    if grids.coords is None:
        grids.build()

    if numint.libxc.is_nlc(mf.xc):
        xc_code = mf.xc
    else:
        xc_code = mf.nlc
    nlc_coefs = mf._numint.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]

    kappa_prefactor = nlc_pars[0] * 1.5 * numpy.pi * (9 * numpy.pi)**(-1.0/6.0)
    C_in_omega = nlc_pars[1]
    beta = 0.03125 * (3.0 / nlc_pars[0]**2)**0.75

    # ao = numint.eval_ao(mol, grids.coords, deriv = 2, gdftopt = None, transpose = False)
    # rho_drho = numint.eval_rho(mol, ao[:4, :], dm0, xctype = "NLC", hermi = 1, with_lapl = False)

    ngrids_full = grids.coords.shape[0]
    rho_drho = cupy.empty([4, ngrids_full])

    available_gpu_memory = get_avail_mem()
    available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
    ao_nbytes_per_grid = ((4*2) * mol.nao + 4) * 8 # factor of 2 from the ao sorting inside numint.eval_ao()
    ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
    if ngrids_per_batch < 16:
        raise MemoryError(f"Out of GPU memory for NLC Fock first derivative, available gpu memory = {get_avail_mem()}"
                          f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids = {ngrids_full}")
    ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
    ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

    for g0 in range(0, ngrids_full, ngrids_per_batch):
        g1 = min(g0 + ngrids_per_batch, ngrids_full)
        split_grids_coords = grids.coords[g0:g1, :]
        split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 1, gdftopt = None, transpose = False)
        split_rho_drho = numint.eval_rho(mol, split_ao, dm0, xctype = "NLC", hermi = 1, with_lapl = False)
        rho_drho[:, g0:g1] = split_rho_drho

    rho_i = rho_drho[0,:]

    rho_nonzero_mask = (rho_i >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD)

    rho_i = rho_i[rho_nonzero_mask]
    nabla_rho_i = rho_drho[1:4, rho_nonzero_mask]
    grids_coords = cupy.ascontiguousarray(grids.coords[rho_nonzero_mask, :])
    grids_weights = grids.weights[rho_nonzero_mask]
    ngrids = grids_coords.shape[0]

    gamma_i = nabla_rho_i[0,:]**2 + nabla_rho_i[1,:]**2 + nabla_rho_i[2,:]**2
    omega_i = cupy.sqrt(C_in_omega * gamma_i**2 / rho_i**4 + (4.0/3.0*numpy.pi) * rho_i)
    kappa_i = kappa_prefactor * rho_i**(1.0/6.0)

    U_i = cupy.empty(ngrids)
    W_i = cupy.empty(ngrids)
    A_i = cupy.empty(ngrids)
    B_i = cupy.empty(ngrids)
    C_i = cupy.empty(ngrids)
    E_i = cupy.empty(ngrids)

    stream = cupy.cuda.get_current_stream()
    libgdft.VXC_vv10nlc_hess_eval_UWABCE(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(U_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(W_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(B_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(C_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(E_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids)
    )

    domega_drho_i         = cupy.empty(ngrids)
    domega_dgamma_i       = cupy.empty(ngrids)
    d2omega_drho2_i       = cupy.empty(ngrids)
    d2omega_dgamma2_i     = cupy.empty(ngrids)
    d2omega_drho_dgamma_i = cupy.empty(ngrids)
    libgdft.VXC_vv10nlc_hess_eval_omega_derivative(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(domega_drho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_drho2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_dgamma2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_drho_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(gamma_i.data.ptr, ctypes.c_void_p),
        ctypes.c_double(C_in_omega),
        ctypes.c_int(ngrids)
    )
    dkappa_drho_i   = kappa_prefactor * (1.0/6.0) * rho_i**(-5.0/6.0)
    d2kappa_drho2_i = kappa_prefactor * (-5.0/36.0) * rho_i**(-11.0/6.0)

    f_rho_i = beta + E_i + rho_i * (dkappa_drho_i * U_i + domega_drho_i * W_i)
    f_gamma_i = rho_i * domega_dgamma_i * W_i

    aoslices = mol.aoslice_by_atom()
    if grid_response:
        assert grids.atm_idx.shape[0] == grids.coords.shape[0]
        grid_to_atom_index_map = grids.atm_idx[rho_nonzero_mask]
        atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]

    # ao = numint.eval_ao(mol, grids.coords, deriv = 2, gdftopt = None, transpose = False)
    # ao_nonzero_rho = ao[:,:,rho_nonzero_mask]
    # mu = ao_nonzero_rho[0, :, :]
    # dmu_dr = ao_nonzero_rho[1:4, :, :]
    # d2mu_dr2 = get_d2mu_dr2(ao_nonzero_rho)

    # drho_dA, dgamma_dA = get_drhodA_dgammadA_orbital_response(d2mu_dr2, dmu_dr, mu, nabla_rho_i, dm0, aoslices)
    # if grid_response:
    #     drho_dA_grid_response, dgamma_dA_grid_response = \
    #         get_drhodA_dgammadA_grid_response(d2mu_dr2, dmu_dr, mu, nabla_rho_i, dm0, atom_to_grid_index_map = atom_to_grid_index_map)
    #     drho_dA   += drho_dA_grid_response
    #     dgamma_dA += dgamma_dA_grid_response
    #     drho_dA_grid_response = None
    #     dgamma_dA_grid_response = None

    drho_dA   = cupy.empty([natm, 3, ngrids], order = "C")
    dgamma_dA = cupy.empty([natm, 3, ngrids], order = "C")

    available_gpu_memory = get_avail_mem()
    available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
    ao_nbytes_per_grid = ((10 + 1*2 + 3*2 + 9) * mol.nao + (3*2) * mol.natm) * 8
    ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
    if ngrids_per_batch < 16:
        raise MemoryError(f"Out of GPU memory for NLC Fock first derivative, available gpu memory = {get_avail_mem()}"
                          f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (nonzero rho) = {ngrids}")
    ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
    ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

    for g0 in range(0, ngrids, ngrids_per_batch):
        g1 = min(g0 + ngrids_per_batch, ngrids)
        split_grids_coords = grids_coords[g0:g1, :]
        split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2, gdftopt = None, transpose = False)

        mu = split_ao[0, :, :]
        dmu_dr = split_ao[1:4, :, :]
        d2mu_dr2 = get_d2mu_dr2(split_ao)
        split_drho_dr = nabla_rho_i[:, g0:g1]

        split_drho_dA, split_dgamma_dA = get_drhodA_dgammadA_orbital_response(d2mu_dr2, dmu_dr, mu, split_drho_dr, dm0, aoslices)
        drho_dA  [:, :, g0:g1] = split_drho_dA
        dgamma_dA[:, :, g0:g1] = split_dgamma_dA
        split_drho_dA   = None
        split_dgamma_dA = None

    if grid_response:
        for i_atom in range(natm):
            associated_grid_index = atom_to_grid_index_map[i_atom]
            associated_grids_coords = grids_coords[associated_grid_index, :]
            ngrids_per_atom = associated_grids_coords.shape[0]

            associated_drho_dr = nabla_rho_i[:, associated_grid_index]

            drho_dA_grid_response   = cupy.empty([3, ngrids_per_atom])
            dgamma_dA_grid_response = cupy.empty([3, ngrids_per_atom])
            for g0 in range(0, ngrids_per_atom, ngrids_per_batch):
                g1 = min(g0 + ngrids_per_batch, ngrids_per_atom)

                split_grids_coords = associated_grids_coords[g0:g1, :]
                split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2, gdftopt = None, transpose = False)

                mu = split_ao[0, :, :]
                dmu_dr = split_ao[1:4, :, :]
                d2mu_dr2 = get_d2mu_dr2(split_ao)
                split_drho_dr = associated_drho_dr[:, g0:g1]
                split_drho_dA_grid_response, split_dgamma_dA_grid_response = \
                    get_drhodA_dgammadA_grid_response(d2mu_dr2, dmu_dr, mu, split_drho_dr, dm0, i_atom = i_atom)

                drho_dA_grid_response  [:, g0:g1] =   split_drho_dA_grid_response
                dgamma_dA_grid_response[:, g0:g1] = split_dgamma_dA_grid_response

            drho_dA  [i_atom][:, associated_grid_index] += drho_dA_grid_response
            dgamma_dA[i_atom][:, associated_grid_index] += dgamma_dA_grid_response
            drho_dA_grid_response   = None
            dgamma_dA_grid_response = None

    drho_dA   = cupy.ascontiguousarray(drho_dA)
    dgamma_dA = cupy.ascontiguousarray(dgamma_dA)
    f_rho_A_i   = cupy.empty([natm, 3, ngrids], order = "C")
    f_gamma_A_i = cupy.empty([natm, 3, ngrids], order = "C")

    libgdft.VXC_vv10nlc_hess_eval_f_t(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(f_rho_A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(f_gamma_A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(U_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(W_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(B_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(C_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_drho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(dkappa_drho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_drho2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_dgamma2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_drho_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2kappa_drho2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(drho_dA.data.ptr, ctypes.c_void_p),
        ctypes.cast(dgamma_dA.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(3 * natm),
    )
    drho_dA = None
    dgamma_dA = None

    vmat_mo = cupy.zeros([natm, 3, mo_coeff.shape[1], mocc.shape[1]])

    # ao = numint.eval_ao(mol, grids.coords, deriv = 2, gdftopt = None, transpose = False)
    # ao_nonzero_rho = ao[:,:,rho_nonzero_mask]
    # mu = ao_nonzero_rho[0, :, :]
    # dmu_dr = ao_nonzero_rho[1:4, :, :]
    # d2mu_dr2 = get_d2mu_dr2(ao_nonzero_rho)

    # d2rho_dAdr = get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices)
    # if grid_response:
    #     d2rho_dAdr_grid_response = get_d2rho_dAdr_grid_response(d2mu_dr2, dmu_dr, mu, dm0, atom_to_grid_index_map = atom_to_grid_index_map)
    #     d2rho_dAdr += d2rho_dAdr_grid_response
    #     d2rho_dAdr_grid_response = None

    available_gpu_memory = get_avail_mem()
    available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
    ao_nbytes_per_grid = ((10 + 1*2 + 3*2 + 9) * mol.nao + (9*2)) * 8
    ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
    if ngrids_per_batch < 16:
        raise MemoryError(f"Out of GPU memory for NLC Fock first derivative, available gpu memory = {get_avail_mem()}"
                          f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (nonzero rho) = {ngrids}")
    ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
    ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

    for i_atom in range(natm):
        aoslice_one_atom = [aoslices[i_atom]]
        d2rho_dAdr = cupy.empty([3, 3, ngrids])

        for g0 in range(0, ngrids, ngrids_per_batch):
            g1 = min(g0 + ngrids_per_batch, ngrids)
            split_grids_coords = grids_coords[g0:g1, :]
            split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2, gdftopt = None, transpose = False)

            mu = split_ao[0, :, :]
            dmu_dr = split_ao[1:4, :, :]
            d2mu_dr2 = get_d2mu_dr2(split_ao)
            split_drho_dr = nabla_rho_i[:, g0:g1]

            split_d2rho_dAdr = get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslice_one_atom)
            d2rho_dAdr[:, :, g0:g1] = split_d2rho_dAdr
            split_d2rho_dAdr = None

        if grid_response:
            associated_grid_index = atom_to_grid_index_map[i_atom]
            associated_grids_coords = grids_coords[associated_grid_index, :]
            ngrids_per_atom = associated_grids_coords.shape[0]

            d2rho_dAdr_grid_response = cupy.empty([3, 3, ngrids_per_atom])
            for g0 in range(0, ngrids_per_atom, ngrids_per_batch):
                g1 = min(g0 + ngrids_per_batch, ngrids_per_atom)

                split_grids_coords = associated_grids_coords[g0:g1, :]
                split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2, gdftopt = None, transpose = False)

                mu = split_ao[0, :, :]
                dmu_dr = split_ao[1:4, :, :]
                d2mu_dr2 = get_d2mu_dr2(split_ao)

                split_d2rho_dAdr_grid_response = get_d2rho_dAdr_grid_response(d2mu_dr2, dmu_dr, mu, dm0, i_atom = i_atom)
                d2rho_dAdr_grid_response[:, :, g0:g1] = split_d2rho_dAdr_grid_response

            d2rho_dAdr[:, :, associated_grid_index] += d2rho_dAdr_grid_response
            split_d2rho_dAdr_grid_response = None

        for g0 in range(0, ngrids, ngrids_per_batch):
            g1 = min(g0 + ngrids_per_batch, ngrids)
            split_grids_coords = grids_coords[g0:g1, :]
            split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2, gdftopt = None, transpose = False)

            mu = split_ao[0, :, :]
            dmu_dr = split_ao[1:4, :, :]
            d2mu_dr2 = get_d2mu_dr2(split_ao)
            split_drho_dr = nabla_rho_i[:, g0:g1]

            # # w_i 2 f_i^\gamma \nabla_A \nabla\rho \cdot \nabla(\phi_\mu \phi_nu)_i
            # vmat[i_atom, :, :, :] += 2 * cupy.einsum('dDg,Dig,jg,g->dij', d2rho_dAdr[i_atom, :, :, :], dmu_dr, mu, f_gamma_i * grids_weights)
            # vmat[i_atom, :, :, :] += 2 * cupy.einsum('dDg,Dig,jg,g->dji', d2rho_dAdr[i_atom, :, :, :], dmu_dr, mu, f_gamma_i * grids_weights)
            d2rhodAdr_dot_dmudr = contract('dDg,Dig->dig', d2rho_dAdr[:, :, g0:g1], dmu_dr)
            dF  = contract('dig,jg->dij', d2rhodAdr_dot_dmudr, mu * f_gamma_i[g0:g1] * grids_weights[g0:g1])
            d2rhodAdr_dot_dmudr = None

            # # w_i 2 (\nabla\rho)_i \cdot (\nabla(\phi_\mu \phi_nu))_i f_i^{\gamma, A}
            # vmat[i_atom, :, :, :] += 2 * cupy.einsum('dg,Dig,jg,Dg->dij', f_gamma_A_i[i_atom, :, :], dmu_dr, mu, nabla_rho_i * grids_weights)
            # vmat[i_atom, :, :, :] += 2 * cupy.einsum('dg,Dig,jg,Dg->dji', f_gamma_A_i[i_atom, :, :], dmu_dr, mu, nabla_rho_i * grids_weights)
            f_gamma_A_i_mu = contract('dg,ig->dig', f_gamma_A_i[i_atom, :, g0:g1], mu)
            drhodr_dot_dmudr = contract('dig,dg->ig', dmu_dr, split_drho_dr * grids_weights[g0:g1])
            dF += contract('dig,jg->dij', f_gamma_A_i_mu, drhodr_dot_dmudr)
            drhodr_dot_dmudr = None
            f_gamma_A_i_mu = None

            dF += dF.transpose(0,2,1)
            dF *= 2

            # # w_i \phi_{\mu i} \phi_{\nu i} f_i^{\rho, A}
            # vmat[i_atom, :, :, :] += cupy.einsum('dg,ig,jg,g->dij', f_rho_A_i[i_atom, :, :], mu, mu, grids_weights)
            f_rho_A_i_mu = contract('dg,ig->dig', f_rho_A_i[i_atom, :, g0:g1], mu)
            dF += contract('dig,jg->dij', f_rho_A_i_mu, mu * grids_weights[g0:g1])
            f_rho_A_i_mu = None

            vmat_mo[i_atom, :, :, :] += jk._ao2mo(dF, mocc, mo_coeff)
            dF = None

            p0, p1 = aoslices[i_atom][2:]
            # # w_i f_i^\rho \nabla_A (\phi_\mu \phi_nu)_i
            # vmat[i_atom, :, p0:p1, :] += cupy.einsum('dig,jg->dij', -dmu_dr[:, p0:p1, :], mu * f_rho_i * grids_weights)
            # vmat[i_atom, :, :, p0:p1] += cupy.einsum('dig,jg->dji', -dmu_dr[:, p0:p1, :], mu * f_rho_i * grids_weights)
            f_rho_dmudA_nu = contract('dig,jg->dij', -dmu_dr[:, p0:p1, :], mu * f_rho_i[g0:g1] * grids_weights[g0:g1])

            # # w_i 2 f_i^\gamma \nabla\rho \cdot \nabla_A \nabla(\phi_\mu \phi_nu)_i
            # vmat[i_atom, :, p0:p1, :] += 2 * cupy.einsum('dDig,jg,Dg->dij', -d2mu_dr2[:, :, p0:p1, :], mu, nabla_rho_i * f_gamma_i * grids_weights)
            # vmat[i_atom, :, :, p0:p1] += 2 * cupy.einsum('dDig,jg,Dg->dji', -d2mu_dr2[:, :, p0:p1, :], mu, nabla_rho_i * f_gamma_i * grids_weights)
            # vmat[i_atom, :, p0:p1, :] += 2 * cupy.einsum('dig,Djg,Dg->dij', -dmu_dr[:, p0:p1, :], dmu_dr, nabla_rho_i * f_gamma_i * grids_weights)
            # vmat[i_atom, :, :, p0:p1] += 2 * cupy.einsum('dig,Djg,Dg->dji', -dmu_dr[:, p0:p1, :], dmu_dr, nabla_rho_i * f_gamma_i * grids_weights)
            mu_dot_drhodr = contract('ig,dg->dig', mu, split_drho_dr * f_gamma_i[g0:g1] * grids_weights[g0:g1])
            f_gamma_d2mudr2_nu = contract('dDig,Djg->dij', -d2mu_dr2[:, :, p0:p1, :], mu_dot_drhodr)
            mu_dot_drhodr = None
            dmudr_dot_drhodr = contract('dig,dg->ig', dmu_dr, split_drho_dr * f_gamma_i[g0:g1] * grids_weights[g0:g1])
            f_gamma_dmudr_dnudr = contract('dig,jg->dij', -dmu_dr[:, p0:p1, :], dmudr_dot_drhodr)
            dmudr_dot_drhodr = None

            dF_ao = f_rho_dmudA_nu + 2 * (f_gamma_d2mudr2_nu + f_gamma_dmudr_dnudr)
            f_rho_dmudA_nu = None
            f_gamma_d2mudr2_nu = None
            f_gamma_dmudr_dnudr = None

            dF_mo = dF_ao @ mocc
            dF_mo = contract('diq,ip->dpq', dF_mo, mo_coeff[p0:p1, :])
            vmat_mo[i_atom, :, :, :] += dF_mo
            dF_mo = dF_ao.transpose(0,2,1) @ mocc[p0:p1, :]
            dF_mo = contract('diq,ip->dpq', dF_mo, mo_coeff)
            vmat_mo[i_atom, :, :, :] += dF_mo
            dF_ao = None
            dF_mo = None

        d2rho_dAdr = None

        if grid_response:
            associated_grid_index = atom_to_grid_index_map[i_atom]
            associated_grids_coords = grids_coords[associated_grid_index, :]
            ngrids_per_atom = associated_grids_coords.shape[0]

            associated_drho_dr = nabla_rho_i[:, associated_grid_index]
            fw_rho_associated_grids   =   f_rho_i[associated_grid_index] * grids_weights[associated_grid_index]
            fw_gamma_associated_grids = f_gamma_i[associated_grid_index] * grids_weights[associated_grid_index]

            for g0 in range(0, ngrids_per_atom, ngrids_per_batch):
                g1 = min(g0 + ngrids_per_batch, ngrids_per_atom)

                split_grids_coords = associated_grids_coords[g0:g1, :]
                split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2, gdftopt = None, transpose = False)

                mu = split_ao[0, :, :]
                dmu_dr = split_ao[1:4, :, :]
                d2mu_dr2 = get_d2mu_dr2(split_ao)
                split_drho_dr = associated_drho_dr[:, g0:g1]

                # # w_i f_i^\rho \nabla_A (\phi_\mu \phi_nu)_i
                # vmat[i_atom, :, :, :] += cupy.einsum('dig,jg->dij',
                #     dmu_dr[:, :, associated_grid_index],
                #     mu[:, associated_grid_index] * f_rho_i[associated_grid_index] * grids_weights[associated_grid_index])
                # vmat[i_atom, :, :, :] += cupy.einsum('dig,jg->dji',
                #     dmu_dr[:, :, associated_grid_index],
                #     mu[:, associated_grid_index] * f_rho_i[associated_grid_index] * grids_weights[associated_grid_index])
                f_rho_dmudA_nu = contract('dig,jg->dij', dmu_dr, mu * fw_rho_associated_grids[g0:g1])

                # # w_i 2 f_i^\gamma \nabla\rho \cdot \nabla_A \nabla(\phi_\mu \phi_nu)_i
                # vmat[i_atom, :, :, :] += 2 * cupy.einsum('dDig,jg,Dg->dij',
                #     d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index],
                #     nabla_rho_i[:, associated_grid_index] * f_gamma_i[associated_grid_index] * grids_weights[associated_grid_index])
                # vmat[i_atom, :, :, :] += 2 * cupy.einsum('dDig,jg,Dg->dji',
                #     d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index],
                #     nabla_rho_i[:, associated_grid_index] * f_gamma_i[associated_grid_index] * grids_weights[associated_grid_index])
                # vmat[i_atom, :, :, :] += 2 * cupy.einsum('dig,Djg,Dg->dij',
                #     dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index],
                #     nabla_rho_i[:, associated_grid_index] * f_gamma_i[associated_grid_index] * grids_weights[associated_grid_index])
                # vmat[i_atom, :, :, :] += 2 * cupy.einsum('dig,Djg,Dg->dji',
                #     dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index],
                #     nabla_rho_i[:, associated_grid_index] * f_gamma_i[associated_grid_index] * grids_weights[associated_grid_index])
                d2mudr2_dot_drhodr = contract('dDig,Dg->dig', d2mu_dr2, split_drho_dr * fw_gamma_associated_grids[g0:g1])
                f_gamma_d2mudr2_nu = contract('dig,jg->dij', d2mudr2_dot_drhodr, mu)
                d2mudr2_dot_drhodr = None
                dmudr_dot_drhodr = contract('dig,dg->ig', dmu_dr, split_drho_dr * fw_gamma_associated_grids[g0:g1])
                f_gamma_dmudr_dnudr = contract('dig,jg->dij', dmu_dr, dmudr_dot_drhodr)
                dmudr_dot_drhodr = None

                dF_ao = f_rho_dmudA_nu + 2 * (f_gamma_d2mudr2_nu + f_gamma_dmudr_dnudr)
                f_rho_dmudA_nu = None
                f_gamma_d2mudr2_nu = None
                f_gamma_dmudr_dnudr = None

                dF_ao += dF_ao.transpose(0,2,1)

                vmat_mo[i_atom, :, :, :] += jk._ao2mo(dF_ao, mocc, mo_coeff)
                dF_ao = None

    if grid_response:
        E_Bgr_i = cupy.empty([natm, 3, ngrids], order = "C")
        U_Bgr_i = cupy.empty([natm, 3, ngrids], order = "C")
        W_Bgr_i = cupy.empty([natm, 3, ngrids], order = "C")
        libgdft.VXC_vv10nlc_hess_eval_EUW_grid_response(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(E_Bgr_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(U_Bgr_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(W_Bgr_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_weights.data.ptr, ctypes.c_void_p),
            ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grid_to_atom_index_map.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(natm),
        )

        grids_weights_1 = get_dweight_dA(mol, grids)
        grids_weights_1 = grids_weights_1[:, :, rho_nonzero_mask]
        grids_weights_1 = cupy.ascontiguousarray(grids_weights_1)

        E_Bw_i = cupy.empty([natm, 3, ngrids], order = "C")
        U_Bw_i = cupy.empty([natm, 3, ngrids], order = "C")
        W_Bw_i = cupy.empty([natm, 3, ngrids], order = "C")
        libgdft.VXC_vv10nlc_hess_eval_EUW_with_weight1(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(E_Bw_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(U_Bw_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(W_Bw_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_weights_1.data.ptr, ctypes.c_void_p),
            ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(natm * 3),
        )

        f_rho_grid_response_i = (E_Bw_i + E_Bgr_i) + ((U_Bw_i + U_Bgr_i) * dkappa_drho_i + (W_Bw_i + W_Bgr_i) * domega_drho_i) * rho_i
        f_gamma_grid_response_i = (W_Bw_i + W_Bgr_i) * domega_dgamma_i * rho_i
        E_Bw_i = None
        U_Bw_i = None
        W_Bw_i = None
        E_Bgr_i = None
        U_Bgr_i = None
        W_Bgr_i = None

        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
        ao_nbytes_per_grid = ((4 + 1*2 + 3*2) * mol.nao) * 8
        ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
        if ngrids_per_batch < 16:
            raise MemoryError(f"Out of GPU memory for NLC Fock first derivative, available gpu memory = {get_avail_mem()}"
                            f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (nonzero rho) = {ngrids}")
        ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
        ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

        for g0 in range(0, ngrids, ngrids_per_batch):
            g1 = min(g0 + ngrids_per_batch, ngrids)
            split_grids_coords = grids_coords[g0:g1, :]
            split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2, gdftopt = None, transpose = False)

            mu = split_ao[0, :, :]
            dmu_dr = split_ao[1:4, :, :]
            d2mu_dr2 = get_d2mu_dr2(split_ao)
            split_drho_dr = nabla_rho_i[:, g0:g1]

            for i_atom in range(natm):
                # # \nabla_A w_i term
                # vmat[i_atom, :, :, :] += cupy.einsum('dg,ig,jg->dij', grids_weights_1[i_atom, :, :], mu, mu * f_rho_i)
                # vmat[i_atom, :, :, :] += 2 * cupy.einsum('dg,Dig,jg,Dg->dij', grids_weights_1[i_atom, :, :], dmu_dr, mu, nabla_rho_i * f_gamma_i)
                # vmat[i_atom, :, :, :] += 2 * cupy.einsum('dg,Dig,jg,Dg->dji', grids_weights_1[i_atom, :, :], dmu_dr, mu, nabla_rho_i * f_gamma_i)
                dwdr_dot_mu = contract('dg,ig->dig', grids_weights_1[i_atom, :, g0:g1], mu)
                f_rho_dwdr  = contract('dig,jg->dij', dwdr_dot_mu, mu * f_rho_i[g0:g1])
                dmudr_dot_drhodr = contract('dig,dg->ig', dmu_dr, split_drho_dr * f_gamma_i[g0:g1])
                f_gamma_dwdr  = contract('dig,jg->dij', dwdr_dot_mu, dmudr_dot_drhodr)
                dmudr_dot_drhodr = None
                dwdr_dot_mu = None

                # # E_i^{Aw} and E_i^{Agr} terms combined
                # vmat[i_atom, :, :, :] += cupy.einsum('dg,ig,jg->dij', f_rho_grid_response_i[i_atom, :, :], mu, mu * grids_weights)
                # vmat[i_atom, :, :, :] += 2 * cupy.einsum('dg,Dig,jg,Dg->dij', f_gamma_grid_response_i[i_atom, :, :], dmu_dr, mu, nabla_rho_i * grids_weights)
                # vmat[i_atom, :, :, :] += 2 * cupy.einsum('dg,Dig,jg,Dg->dji', f_gamma_grid_response_i[i_atom, :, :], dmu_dr, mu, nabla_rho_i * grids_weights)
                dfrhodr_dot_mu = contract('dg,ig->dig', f_rho_grid_response_i[i_atom, :, g0:g1], mu)
                f_rho_dwdr += contract('dig,jg->dij', dfrhodr_dot_mu, mu * grids_weights[g0:g1])
                dfrhodr_dot_mu = None
                dfgammadr_dot_mu = contract('dg,ig->dig', f_gamma_grid_response_i[i_atom, :, g0:g1], mu)
                dmudr_dot_drhodr = contract('dig,dg->ig', dmu_dr, split_drho_dr * grids_weights[g0:g1])
                f_gamma_dwdr += contract('dig,jg->dij', dfgammadr_dot_mu, dmudr_dot_drhodr)
                dmudr_dot_drhodr = None
                dfgammadr_dot_mu = None

                f_gamma_dwdr += f_gamma_dwdr.transpose(0,2,1)
                dF_ao = f_rho_dwdr + 2 * f_gamma_dwdr
                f_rho_dwdr = None
                f_gamma_dwdr = None

                vmat_mo[i_atom, :, :, :] += jk._ao2mo(dF_ao, mocc, mo_coeff)
                dF_ao = None

    return vmat_mo

def _nr_rks_fxc_mo_task(ni, mol, grids, xc_code, fxc, mo_coeff, mo1, mocc,
                        verbose=None, hermi=1, device_id=0):
    with cupy.cuda.Device(device_id), _streams[device_id]:
        if mo_coeff is not None: mo_coeff = cupy.asarray(mo_coeff)
        if mo1 is not None: mo1 = cupy.asarray(mo1)
        if mocc is not None: mocc = cupy.asarray(mocc)
        if fxc is not None: fxc = cupy.asarray(fxc)

        assert isinstance(verbose, int)
        log = logger.new_logger(mol, verbose)
        xctype = ni._xc_type(xc_code)
        opt = getattr(ni, 'gdftopt', None)

        _sorted_mol = opt.mol
        nao = mol.nao
        nset = mo1.shape[0]
        vmat = cupy.zeros((nset, nao, nao))

        if xctype in ['LDA', 'HF']:
            ao_deriv = 0
        else:
            ao_deriv = 1

        ngrids_glob = grids.coords.shape[0]
        grid_start, grid_end = numint.gen_grid_range(ngrids_glob, device_id)
        ngrids_local = grid_end - grid_start
        log.debug(f"{ngrids_local} grids on Device {device_id}")

        p0 = p1 = grid_start
        t1 = t0 = log.init_timer()
        for ao, mask, weights, coords in ni.block_loop(_sorted_mol, grids, nao, ao_deriv,
                                                       max_memory=None, blksize=None,
                                                       grid_range=(grid_start, grid_end)):
            p0, p1 = p1, p1+len(weights)
            occ_coeff_mask = mocc[mask]
            rho1 = numint.eval_rho4(_sorted_mol, ao, 2.0*occ_coeff_mask, mo1[:,mask],
                                    xctype=xctype, hermi=hermi)
            t1 = log.timer_debug2('eval rho', *t1)
            if xctype == 'HF':
                continue
            # precompute fxc_w
            if xctype == 'LDA':
                fxc_w = fxc[0,0,p0:p1] * weights
                wv = rho1 * fxc_w
            else:
                fxc_w = fxc[:,:,p0:p1] * weights
                wv = contract('axg,xyg->ayg', rho1, fxc_w)

            for i in range(nset):
                if xctype == 'LDA':
                    vmat_tmp = ao.dot(numint._scale_ao(ao, wv[i]).T)
                elif xctype == 'GGA':
                    wv[i,0] *= .5
                    aow = numint._scale_ao(ao, wv[i])
                    vmat_tmp = aow.dot(ao[0].T)
                elif xctype == 'NLC':
                    raise NotImplementedError('NLC')
                else:
                    wv[i,0] *= .5
                    wv[i,4] *= .5
                    vmat_tmp = ao[0].dot(numint._scale_ao(ao[:4], wv[i,:4]).T)
                    vmat_tmp+= numint._tau_dot(ao, ao, wv[i,4])
                add_sparse(vmat[i], vmat_tmp, mask)

            t1 = log.timer_debug2('integration', *t1)
            ao = rho1 = None
        t0 = log.timer_debug1(f'vxc on Device {device_id} ', *t0)
        if xctype != 'LDA':
            transpose_sum(vmat)
        vmat = jk._ao2mo(vmat, mocc, mo_coeff)
    return vmat

def nr_rks_fxc_mo(ni, mol, grids, xc_code, dm0=None, dms=None, mo_coeff=None, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    if fxc is None:
        raise RuntimeError('fxc was not initialized')
    #xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None or mol not in [opt.mol, opt._sorted_mol]:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    nao = mol.nao
    dms = cupy.asarray(dms)
    dm_shape = dms.shape
    # AO basis -> gdftopt AO basis
    with_mocc = hasattr(dms, 'mo1')
    mo1 = mocc = None
    if with_mocc:
        mo1 = opt.sort_orbitals(dms.mo1, axis=[1])
        mocc = opt.sort_orbitals(dms.occ_coeff, axis=[0])
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])
    dms = opt.sort_orbitals(dms.reshape(-1,nao,nao), axis=[1,2])

    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _nr_rks_fxc_mo_task,
                ni, mol, grids, xc_code, fxc, mo_coeff, mo1, mocc,
                verbose=log.verbose, hermi=hermi, device_id=device_id)
            futures.append(future)
    dms = None
    vmat_dist = []
    for future in futures:
        vmat_dist.append(future.result())
    vmat = reduce_to_device(vmat_dist, inplace=True)

    if len(dm_shape) == 2:
        vmat = vmat[0]
    t0 = log.timer_debug1('nr_rks_fxc', *t0)
    return cupy.asarray(vmat)

def nr_rks_fnlc_mo(mf, mol, mo_coeff, mo_occ, dm1s, return_in_mo = True):
    """
        Equation notation follows:
        Liang J, Feng X, Liu X, Head-Gordon M. Analytical harmonic vibrational frequencies with
        VV10-containing density functionals: Theory, efficient implementation, and
        benchmark assessments. J Chem Phys. 2023 May 28;158(20):204109. doi: 10.1063/5.0152838.

        mo_coeff, mo_occ are 0-th order
        dm1s is first order

        TODO: check the effect of different grid, using mf.nlcgrids right now
    """
    if mo_coeff.ndim == 2:
        mocc = mo_coeff[:,mo_occ>0]
        mo_occ = mo_occ[mo_occ > 0]
        dm0 = (mocc * mo_occ) @ mocc.T
    else:
        assert mo_coeff.ndim == 3 # unrestricted case
        assert mo_coeff.shape[0] == 2
        assert mo_occ.shape[0] == 2
        assert not return_in_mo # Only support gen_response() for now
        mocc_a = mo_coeff[0][:, mo_occ[0] > 0]
        mocc_b = mo_coeff[1][:, mo_occ[1] > 0]
        mo_occ_a = mo_occ[0, mo_occ[0] > 0]
        mo_occ_b = mo_occ[1, mo_occ[1] > 0]
        dm0 = (mocc_a * mo_occ_a) @ mocc_a.T + (mocc_b * mo_occ_b) @ mocc_b.T

    output_in_2d = False
    if dm1s.ndim == 2:
        assert dm1s.shape == (mol.nao, mol.nao)
        dm1s = dm1s.reshape((1, mol.nao, mol.nao))
        output_in_2d = True
    assert dm1s.ndim == 3

    grids = mf.nlcgrids
    if grids.coords is None:
        grids.build()

    n_dm1 = dm1s.shape[0]

    ni = mf._numint
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol

    if numint.libxc.is_nlc(mf.xc):
        xc_code = mf.xc
    else:
        xc_code = mf.nlc
    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]

    kappa_prefactor = nlc_pars[0] * 1.5 * numpy.pi * (9 * numpy.pi)**(-1.0/6.0)
    C_in_omega = nlc_pars[1]

    # ao = numint.eval_ao(mol, grids.coords, deriv = 1, gdftopt = None, transpose = False)
    # rho_drho = numint.eval_rho(mol, ao, dm0, xctype = "NLC", hermi = 1, with_lapl = False)

    dm0_sorted = opt.sort_orbitals(dm0, axis=[0,1])
    dm0 = None
    ngrids_full = grids.coords.shape[0]
    rho_drho = cupy.empty([4, ngrids_full])
    g1 = 0
    for split_ao, ao_mask_index, split_weights, split_coords in ni.block_loop(_sorted_mol, grids, deriv = 1):
        g0, g1 = g1, g1 + split_weights.size
        dm0_masked = dm0_sorted[ao_mask_index[:,None], ao_mask_index]
        rho_drho[:, g0:g1] = numint.eval_rho(_sorted_mol, split_ao, dm0_masked, xctype = "NLC", hermi = 1)
    dm0_sorted = None

    rho_i = rho_drho[0,:]

    rho_nonzero_mask = (rho_i >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD)

    rho_i = rho_i[rho_nonzero_mask]
    nabla_rho_i = rho_drho[1:4, rho_nonzero_mask]
    grids_coords = cupy.ascontiguousarray(grids.coords[rho_nonzero_mask, :])
    grids_weights = grids.weights[rho_nonzero_mask]
    ngrids = grids_coords.shape[0]

    gamma_i = nabla_rho_i[0,:]**2 + nabla_rho_i[1,:]**2 + nabla_rho_i[2,:]**2
    omega_i = cupy.sqrt(C_in_omega * gamma_i**2 / rho_i**4 + (4.0/3.0*numpy.pi) * rho_i)
    kappa_i = kappa_prefactor * rho_i**(1.0/6.0)

    U_i = cupy.empty(ngrids)
    W_i = cupy.empty(ngrids)
    A_i = cupy.empty(ngrids)
    B_i = cupy.empty(ngrids)
    C_i = cupy.empty(ngrids)
    E_i = cupy.empty(ngrids) # Not used

    stream = cupy.cuda.get_current_stream()
    libgdft.VXC_vv10nlc_hess_eval_UWABCE(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(U_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(W_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(B_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(C_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(E_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids)
    )
    E_i = None

    domega_drho_i         = cupy.empty(ngrids)
    domega_dgamma_i       = cupy.empty(ngrids)
    d2omega_drho2_i       = cupy.empty(ngrids)
    d2omega_dgamma2_i     = cupy.empty(ngrids)
    d2omega_drho_dgamma_i = cupy.empty(ngrids)
    libgdft.VXC_vv10nlc_hess_eval_omega_derivative(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(domega_drho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_drho2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_dgamma2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_drho_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(gamma_i.data.ptr, ctypes.c_void_p),
        ctypes.c_double(C_in_omega),
        ctypes.c_int(ngrids)
    )
    dkappa_drho_i   = kappa_prefactor * (1.0/6.0) * rho_i**(-5.0/6.0)
    d2kappa_drho2_i = kappa_prefactor * (-5.0/36.0) * rho_i**(-11.0/6.0)

    f_gamma_i = rho_i * domega_dgamma_i * W_i

    # ao = numint.eval_ao(mol, grids.coords, deriv = 1, gdftopt = None, transpose = False)
    # rho_drho_t = cupy.empty([n_dm1, 4, ngrids])
    # for i_dm in range(n_dm1):
    #     dm1 = dm1s[i_dm, :, :]
    #     rho_drho_1 = numint.eval_rho(mol, ao, dm1, xctype = "NLC", hermi = 0, with_lapl = False)
    #     rho_drho_t[i_dm, :, :] = rho_drho_1[:, rho_nonzero_mask]

    dm1s_sorted = opt.sort_orbitals(dm1s, axis=[1,2])
    dm1s = None

    if return_in_mo:
        vmat = cupy.zeros([n_dm1, mo_coeff.shape[1], mocc.shape[1]])
        mocc = opt.sort_orbitals(mocc, axis=[0])
        mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])
    else:
        vmat = cupy.zeros([n_dm1, mol.nao, mol.nao])

    available_gpu_memory = get_avail_mem()
    available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
    fxc_nbytes_per_dm1 = ((1*6 + 3*2) * ngrids + (1*2 + 3*2) * ngrids_full) * 8
    ndm1_per_batch = int(available_gpu_memory / fxc_nbytes_per_dm1)
    if ndm1_per_batch < 6:
        raise MemoryError(f"Out of GPU memory for NLC response (orbital hessian), available gpu memory = {get_avail_mem()}"
                          f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (nonzero rho) = {ngrids}")
    ndm1_per_batch = (ndm1_per_batch + 6 - 1) // 6 * 6

    for i_dm1_batch in range(0, n_dm1, ndm1_per_batch):
        n_dm1_batch = min(ndm1_per_batch, n_dm1 - i_dm1_batch)

        rho_drho_t = cupy.empty([n_dm1_batch, 4, ngrids_full])
        g1 = 0
        for split_ao, ao_mask_index, split_weights, split_coords in ni.block_loop(_sorted_mol, grids, deriv = 1):
            g0, g1 = g1, g1 + split_weights.size
            for i_dm in range(n_dm1_batch):
                dm1_sorted = dm1s_sorted[i_dm + i_dm1_batch, :, :]
                dm1_masked = dm1_sorted[ao_mask_index[:,None], ao_mask_index]
                rho_drho_t[i_dm, :, g0:g1] = numint.eval_rho(_sorted_mol, split_ao, dm1_masked, xctype = "NLC", hermi = 0)
                dm1_sorted = None
                dm1_masked = None
        rho_drho_t = rho_drho_t[:, :, rho_nonzero_mask]

        rho_t_i = rho_drho_t[:, 0, :]
        nabla_rho_t_i = rho_drho_t[:, 1:4, :]
        gamma_t_i = nabla_rho_i[0, :] * nabla_rho_t_i[:, 0, :] \
                    + nabla_rho_i[1, :] * nabla_rho_t_i[:, 1, :] \
                    + nabla_rho_i[2, :] * nabla_rho_t_i[:, 2, :]
        gamma_t_i *= 2 # Account for the factor of 2 before gamma_j^t term in equation (22)
        rho_drho_t = None

        rho_t_i   = cupy.ascontiguousarray(rho_t_i)
        gamma_t_i = cupy.ascontiguousarray(gamma_t_i)
        f_rho_t_i   = cupy.empty([n_dm1_batch, ngrids], order = "C")
        f_gamma_t_i = cupy.empty([n_dm1_batch, ngrids], order = "C")

        libgdft.VXC_vv10nlc_hess_eval_f_t(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(f_rho_t_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(f_gamma_t_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_weights.data.ptr, ctypes.c_void_p),
            ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(U_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(W_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(A_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(B_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(C_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(domega_drho_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(domega_dgamma_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(dkappa_drho_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(d2omega_drho2_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(d2omega_dgamma2_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(d2omega_drho_dgamma_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(d2kappa_drho2_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(rho_t_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(gamma_t_i.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(n_dm1_batch),
        )
        rho_t_i = None
        gamma_t_i = None

        fxc_rho = f_rho_t_i * grids_weights
        f_rho_t_i = None
        fxc_gamma  = contract("dg,tg->tdg", nabla_rho_i, f_gamma_t_i)
        f_gamma_t_i = None
        fxc_gamma += nabla_rho_t_i * f_gamma_i
        nabla_rho_t_i = None
        fxc_gamma = 2 * fxc_gamma * grids_weights

        fxc_rho_full = cupy.zeros([n_dm1_batch, ngrids_full])
        fxc_rho_full[:, rho_nonzero_mask] = fxc_rho
        fxc_rho = None
        fxc_gamma_full = cupy.zeros([n_dm1_batch, 3, ngrids_full])
        fxc_gamma_full[:, :, rho_nonzero_mask] = fxc_gamma
        fxc_gamma = None

        g1 = 0
        for split_ao, ao_mask_index, split_weights, split_coords in ni.block_loop(_sorted_mol, grids, deriv = 1):
            g0, g1 = g1, g1 + split_weights.size
            split_fxc_rho = fxc_rho_full[:, g0:g1]
            split_fxc_gamma = fxc_gamma_full[:, :, g0:g1]

            for i_dm in range(n_dm1_batch):
                # \mu \nu
                V_munu = contract("ig,jg->ij", split_ao[0], split_ao[0] * split_fxc_rho[i_dm, :])

                # \mu \nabla\nu + \nabla\mu \nu
                nabla_fxc_dot_nabla_ao = contract("dg,dig->ig", split_fxc_gamma[i_dm, :, :], split_ao[1:4])
                V_munu_gamma = contract("ig,jg->ij", split_ao[0], nabla_fxc_dot_nabla_ao)
                nabla_fxc_dot_nabla_ao = None
                V_munu += V_munu_gamma
                V_munu += V_munu_gamma.T
                V_munu_gamma = None

                vmat_ao = cupy.zeros([mol.nao, mol.nao])
                add_sparse(vmat_ao, V_munu, ao_mask_index)
                V_munu = None

                if return_in_mo:
                    vmat[i_dm + i_dm1_batch, :, :] += mo_coeff.T @ vmat_ao @ mocc
                else:
                    vmat[i_dm + i_dm1_batch, :, :] += opt.unsort_orbitals(vmat_ao, axis=[0,1])
                vmat_ao = None

    if output_in_2d:
        vmat = vmat.reshape((mol.nao, mol.nao))

    return vmat

def get_veff_resp_mo(hessobj, mol, dms, mo_coeff, mo_occ, hermi=1, omega=None):
    mol = hessobj.mol
    mf = hessobj.base
    grids = getattr(mf, 'cphf_grids', None)
    if grids is not None:
        logger.info(mf, 'Secondary grids defined for CPHF in Hessian')
    else:
        # If cphf_grids is not defined, e.g object defined from CPU
        grids = getattr(mf, 'grids', None)
        logger.info(mf, 'Primary grids is used for CPHF in Hessian')

    if grids and grids.coords is None:
        grids.build(mol=mol, with_non0tab=False, sort_grids=True)

    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    hermi = 1

    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    nao, nmo = mo_coeff.shape
    # TODO: evaluate v1 in MO
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, mf.xc,
                                        mo_coeff, mo_occ, 0)
    v1 = nr_rks_fxc_mo(ni, mol, grids, mf.xc, None, dms, mo_coeff, 0, hermi,
                                    rho0, vxc, fxc, max_memory=None)
    v1 = v1.reshape(-1,nmo*nocc)

    if mf.do_nlc():
        vnlc = nr_rks_fnlc_mo(mf, mol, mo_coeff, mo_occ, dms)
        v1 += vnlc.reshape(-1,nmo*nocc)

    if hybrid:
        vj, vk = hessobj.get_jk_mo(mol, dms, mo_coeff, mo_occ, hermi=1)
        vk *= hyb
        if omega > 1e-10:  # For range separated Coulomb
            _, vk_lr = hessobj.get_jk_mo(mol, dms, mo_coeff, mo_occ, hermi,
                                        with_j=False, omega=omega)
            vk_lr *= (alpha-hyb)
            vk += vk_lr
        v1 += vj - .5 * vk
    else:
        v1 += hessobj.get_jk_mo(mol, dms, mo_coeff, mo_occ, hermi=1,
                                with_k=False)[0]

    return v1


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
    gen_vind = rhf_hess.gen_vind
    get_jk_mo = rhf_hess._get_jk_mo
    get_veff_resp_mo = get_veff_resp_mo

from gpu4pyscf import dft
dft.rks.RKS.Hessian = lib.class_as_method(Hessian)
