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
from gpu4pyscf.hessian.rhf import _ao2mo
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import (
    contract, add_sparse, get_avail_mem, reduce_to_device, transpose_sum,
    take_last2d, batched_vec_norm2)
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import num_devices, min_grid_blksize
from gpu4pyscf.dft.numint import NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD, _contract_rho1_fxc
import ctypes
from pyscf import __config__
MIN_BLK_SIZE = getattr(__config__, 'min_grid_blksize', 4096)
ALIGNED = getattr(__config__, 'grid_aligned', 16*16)

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
        vhfopt = mf._opt_gpu.get(omega)
        with mol.with_range_coulomb(omega):
            de2 += rhf_hess._partial_ejk_ip2(
                mol, dm0, vhfopt, j_factor, k_factor, verbose=verbose)

    t1 = log.timer_debug1('hessian of JK part', *t1)

    de2 += _get_exc_deriv2(hessobj, mo_coeff, mo_occ, dm0, max_memory, atmlst, log)
    if mf.do_nlc():
        de2 += _get_enlc_deriv2(hessobj, mo_coeff, mo_occ, max_memory, log)

    t1 = log.timer_debug1('hessian of XC part', *t1)
    log.timer('RKS partial hessian', *time0)
    return de2

def _get_exc_deriv2(hessobj, mo_coeff, mo_occ, dm0, max_memory, atmlst = None, log = None):
    if log is None:
        log = logger.new_logger(hessobj)

    if hessobj.grid_response:
        log.info("Calculating grid response for DFT Hessian")
        return _get_exc_deriv2_grid_response(hessobj, mo_coeff, mo_occ, max_memory)

    mol = hessobj.mol
    mf = hessobj.base

    de2 = cupy.zeros([mol.natm, mol.natm, 3, 3])

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    veff_diag = _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)

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

    return de2

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    natm = mol.natm
    assert atmlst is None or atmlst == range(natm)
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = cupy.dot(mocc, mocc.T) * 2
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
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]

    ni = mf._numint
    xctype = ni._xc_type(mf.xc)

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])
    nao = mo_coeff.shape[0]

    if xctype == 'LDA':
        ncomp = 1
    elif xctype == 'GGA':
        ncomp = 4
    else:
        ncomp = 5
    rho_buf = cupy.empty(ncomp*MIN_BLK_SIZE)
    vtmp_buf = cupy.empty((6, nao, nao))
    vmat = cupy.zeros((6,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 2
        aow_buf = cupy.empty(MIN_BLK_SIZE * max(nao,1*nocc))
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            blk_size = len(weight)
            nao_sub = len(mask)
            rho = cupy.ndarray((blk_size), memptr=rho_buf.data)
            vtmp = cupy.ndarray((6, nao_sub, nao_sub), memptr=vtmp_buf.data)

            mo_coeff_mask = mo_coeff[mask,:]
            rho = numint.eval_rho2(_sorted_mol, ao[0], mo_coeff_mask, mo_occ, mask, xctype, buf=aow_buf, out=rho)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype, spin=0)[1][0]
            wv = cupy.multiply(weight, vxc, out=vxc)
            aow  = cupy.ndarray((nao_sub, blk_size), memptr=aow_buf.data)
            aow = numint._scale_ao(ao[0], wv, out=aow)
            vtmp = contract('bik,lk->bil', ao[4:10], aow, out=vtmp)
            for i in range(6):                                     ### 4 XX, 5 XY, 6 XZ, 7 YX, 8 YY, 9 ZZ
                add_sparse(vmat[i], vtmp[i], mask)
            aow = None

    elif xctype == 'GGA':
        def contract_ao(ao, aoidx, wv, buf, aow, out):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1], out=aow)
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2], out=buf)
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3], out=buf)
            return contract('ik,lk->il', aow, ao[0], beta=1, out=out)

        ao_deriv = 3
        aow_buf = cupy.empty(MIN_BLK_SIZE * max(nao,2*nocc))
        buf = cupy.empty(MIN_BLK_SIZE * nao)
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            blk_size = len(weight)
            nao_sub = len(mask)
            rho = cupy.ndarray((ncomp, blk_size), memptr=rho_buf.data)
            vtmp = cupy.ndarray((6, nao_sub, nao_sub), memptr=vtmp_buf.data)
            buf = cupy.ndarray((nao_sub, blk_size), memptr=buf.data)

            mo_coeff_mask = mo_coeff[mask,:]
            rho = numint.eval_rho2(_sorted_mol, ao[:4], mo_coeff_mask, mo_occ, mask, xctype, buf=aow_buf, out=rho)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype, spin=0)[1]
            wv = cupy.multiply(weight, vxc, out=vxc)
            aow  = cupy.ndarray((nao_sub, blk_size), memptr=aow_buf.data)
            aow = numint._scale_ao(ao[:4], wv[:4], out=aow)

            vtmp = contract('bik,lk->bil', ao[4:10], aow, out=vtmp)

            contract_ao(ao, [XXX,XXY,XXZ], wv, buf, aow, vtmp[0])
            contract_ao(ao, [XXY,XYY,XYZ], wv, buf, aow, vtmp[1])
            contract_ao(ao, [XXZ,XYZ,XZZ], wv, buf, aow, vtmp[2])
            contract_ao(ao, [XYY,YYY,YYZ], wv, buf, aow, vtmp[3])
            contract_ao(ao, [XYZ,YYZ,YZZ], wv, buf, aow, vtmp[4])
            contract_ao(ao, [XZZ,YZZ,ZZZ], wv, buf, aow, vtmp[5])

            for i in range(6):
                add_sparse(vmat[i], vtmp[i], mask)

    elif xctype == 'MGGA':
        def contract_ao(ao, aoidx, wv, buf, aow, out):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1], out=aow)
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2], out=buf)
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3], out=buf)
            return contract('ik,lk->il', aow, ao[0], beta=1, out=out)

        ao_deriv = 3
        aow_buf = cupy.empty(MIN_BLK_SIZE * max(3 *nao,2*nocc))
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            blk_size = len(weight)
            nao_sub = len(mask)
            rho = cupy.ndarray((ncomp, blk_size), memptr=rho_buf.data)
            vtmp = cupy.ndarray((6, nao_sub, nao_sub), memptr=vtmp_buf.data)

            mo_coeff_mask = mo_coeff[mask,:]
            rho = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff_mask, mo_occ, mask, xctype, buf=aow_buf, out=rho)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype, spin=0)[1]
            wv = cupy.multiply(weight, vxc, out=vxc)
            wv[4] *= .5  # for the factor 1/2 in tau
            aow  = cupy.ndarray((3, nao_sub, blk_size), memptr=aow_buf.data)
            numint._scale_ao(ao[:4], wv[:4], out=aow[0])
            vtmp = contract('bik,lk->bil', ao[4:10], aow[0], out=vtmp)

            contract_ao(ao, [XXX,XXY,XXZ], wv, aow[0], aow[1], vtmp[0])
            contract_ao(ao, [XXY,XYY,XYZ], wv, aow[0], aow[1], vtmp[1])
            contract_ao(ao, [XXZ,XYZ,XZZ], wv, aow[0], aow[1], vtmp[2])
            contract_ao(ao, [XYY,YYY,YYZ], wv, aow[0], aow[1], vtmp[3])
            contract_ao(ao, [XYZ,YYZ,YZZ], wv, aow[0], aow[1], vtmp[4])
            contract_ao(ao, [XZZ,YZZ,ZZZ], wv, aow[0], aow[1], vtmp[5])

            for i in range(0, 3):
                numint._scale_ao(ao[i+1], wv[4], out=aow[i])
            for i, j in enumerate([XXX, XXY, XXZ, XYY, XYZ, XZZ]):
                contract('ik,lk->il', ao[j], aow[0], beta=1, out=vtmp[i])
            for i, j in enumerate([XXY, XYY, XYZ, YYY, YYZ, YZZ]):
                contract('ik,lk->il', ao[j], aow[1], beta=1, out=vtmp[i])
            for i, j in enumerate([XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ]):
                contract('ik,lk->il', ao[j], aow[2], beta=1, out=vtmp[i])

            for i in range(6):
                add_sparse(vmat[i], vtmp[i], mask)

    vmat = vmat[[0,1,2,
                 1,3,4,
                 2,4,5]]

    vmat = opt.unsort_orbitals(vmat, axis=[1,2])
    return vmat.reshape(3,3,nao,nao)

def _make_dR_rho1(ao, ao_dm0, atm_id, aoslices, xctype, buf=None, out=None):
    p0, p1 = aoslices[atm_id][2:]
    ngrids = ao[0].shape[1]

    if xctype == 'GGA':
        ncomp = 4
    elif xctype == 'MGGA':
        ncomp = 5
    else:
        raise RuntimeError
    if buf is None:
        buf = cupy.empty(ngrids)
    if out is None:
        rho1 = cupy.zeros((3, ncomp,ngrids))
    else:
        rho1 = out
        rho1.fill(0)
    if xctype == 'MGGA':
        ao_dm0_x = ao_dm0[1][p0:p1]
        ao_dm0_y = ao_dm0[2][p0:p1]
        ao_dm0_z = ao_dm0[3][p0:p1]
        # (d_X \nabla mu) dot \nalba nu DM_{mu,nu}
        rho1[0,4] += numint._contract_rho(ao[XX,p0:p1], ao_dm0_x, rho=buf)
        rho1[0,4] += numint._contract_rho(ao[XY,p0:p1], ao_dm0_y, rho=buf)
        rho1[0,4] += numint._contract_rho(ao[XZ,p0:p1], ao_dm0_z, rho=buf)
        rho1[1,4] += numint._contract_rho(ao[YX,p0:p1], ao_dm0_x, rho=buf)
        rho1[1,4] += numint._contract_rho(ao[YY,p0:p1], ao_dm0_y, rho=buf)
        rho1[1,4] += numint._contract_rho(ao[YZ,p0:p1], ao_dm0_z, rho=buf)
        rho1[2,4] += numint._contract_rho(ao[ZX,p0:p1], ao_dm0_x, rho=buf)
        rho1[2,4] += numint._contract_rho(ao[ZY,p0:p1], ao_dm0_y, rho=buf)
        rho1[2,4] += numint._contract_rho(ao[ZZ,p0:p1], ao_dm0_z, rho=buf)
        rho1[:,4] *= .5

    ao_dm0_0 = ao_dm0[0][p0:p1]
    # (d_X \nabla_x mu) nu DM_{mu,nu}
    rho1[0,1]+= numint._contract_rho(ao[XX,p0:p1], ao_dm0_0, rho=buf)
    rho1[0,2]+= numint._contract_rho(ao[XY,p0:p1], ao_dm0_0, rho=buf)
    rho1[0,3]+= numint._contract_rho(ao[XZ,p0:p1], ao_dm0_0, rho=buf)
    rho1[1,1]+= numint._contract_rho(ao[YX,p0:p1], ao_dm0_0, rho=buf)
    rho1[1,2]+= numint._contract_rho(ao[YY,p0:p1], ao_dm0_0, rho=buf)
    rho1[1,3]+= numint._contract_rho(ao[YZ,p0:p1], ao_dm0_0, rho=buf)
    rho1[2,1]+= numint._contract_rho(ao[ZX,p0:p1], ao_dm0_0, rho=buf)
    rho1[2,2]+= numint._contract_rho(ao[ZY,p0:p1], ao_dm0_0, rho=buf)
    rho1[2,3]+= numint._contract_rho(ao[ZZ,p0:p1], ao_dm0_0, rho=buf)
    # (d_X mu) (\nabla_x nu) DM_{mu,nu}
    for i in range(3):
        rho1[i,0] += numint._contract_rho(ao[i+1,p0:p1], ao_dm0_0, rho=buf)
        rho1[i,1] += numint._contract_rho(ao[i+1,p0:p1], ao_dm0[1][p0:p1], rho=buf)
        rho1[i,2] += numint._contract_rho(ao[i+1,p0:p1], ao_dm0[2][p0:p1], rho=buf)
        rho1[i,3] += numint._contract_rho(ao[i+1,p0:p1], ao_dm0[3][p0:p1], rho=buf)

    # *2 for |mu> DM <d_X nu|
    rho1 *= 2
    return rho1

def _d1d2_dot_(vmat, mol, ao1, ao2, mask, ao_loc, dR1_on_bra=True):
    if dR1_on_bra:  # (d/dR1 bra) * (d/dR2 ket)
        for d1 in range(3):
            for d2 in range(3):
                contract('ik,lk->il', ao1[d1], ao2[d2], beta=1, out=vmat[d1,d2])
        #vmat += contract('xig,yjg->xyij', ao1, ao2)
    else:  # (d/dR2 bra) * (d/dR1 ket)
        for d1 in range(3):
            for d2 in range(3):
                contract('ik,lk->il', ao1[d2], ao2[d1], beta=1, out=vmat[d1,d2])
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
    ao_loc = mol.ao_loc_nr()

    ngrids_glob = grids.coords.shape[0]
    grid_start, grid_end = numint.gen_grid_range(ngrids_glob, device_id)

    with cupy.cuda.Device(device_id):
        log = logger.new_logger(mol, verbose)
        t1 = t0 = log.init_timer()
        mo_occ = cupy.asarray(mo_occ)
        mo_coeff = cupy.asarray(mo_coeff)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        dm0_sorted = opt.sort_orbitals(dm0, axis=[0,1])
        coeff = cupy.asarray(opt.coeff)
        mocc = mo_coeff[:,mo_occ>0]
        nocc = mocc.shape[1]

        vmat_dm = cupy.zeros((_sorted_mol.natm,3,3,nao))
        ipip = cupy.zeros((3,3,nao,nao))

        if xctype == 'LDA':
            ncomp = 1
        elif xctype == 'GGA':
            ncomp = 4
        else:
            ncomp = 5
        rho_buf = cupy.empty(ncomp*MIN_BLK_SIZE)

        if xctype == 'LDA':
            ao_deriv = 1
            nd = (ao_deriv+1)*(ao_deriv+2)*(ao_deriv+3)//6
            aow_buf = cupy.empty(max(3*nao,1*nocc)* MIN_BLK_SIZE)
            wv_buf = cupy.empty(3* MIN_BLK_SIZE)
            ao1_buf = cupy.empty(nd*nao*MIN_BLK_SIZE)
            ao_dm_mask_buf = cupy.empty(4 * nao * MIN_BLK_SIZE)
            ao_dm0_buf = cupy.empty(nao * MIN_BLK_SIZE)
            dm_mask_buf = cupy.empty(nao*nao)

            t1 = log.init_timer()
            for ao_mask, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                          grid_range=(grid_start, grid_end)):
                nao_sub = len(mask)
                blk_size = len(weight)
                ao1 = cupy.ndarray((nd, nao, blk_size), memptr=ao1_buf.data)
                rho = cupy.ndarray((blk_size), memptr=rho_buf.data)
                ao_dm_mask = cupy.ndarray((4, nao_sub, blk_size), memptr=ao_dm_mask_buf.data)
                ao_dm0 = cupy.ndarray((nao, blk_size), memptr=ao_dm0_buf.data)
                wv = cupy.ndarray((3, blk_size), memptr=wv_buf.data)

                ao1 = contract('nip,ij->njp', ao_mask, coeff[mask], out=ao1)

                rho = numint.eval_rho2(_sorted_mol, ao1[0], mo_coeff, mo_occ, mask, xctype, buf=aow_buf, out=rho)
                t1 = log.timer_debug2('eval rho', *t1)
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype, spin=0)[1:3]
                t1 = log.timer_debug2('eval vxc', *t1)
                wv1 = cupy.multiply(weight, vxc[0], out=vxc[0])
                wf = cupy.multiply(weight, fxc[0,0], out=fxc[0,0])
                aow  = cupy.ndarray((3, nao, blk_size), memptr=aow_buf.data)
                for i in range(1, 4):
                    numint._scale_ao(ao1[i], wv1, out=aow[i-1])
                _d1d2_dot_(ipip, mol, aow, ao1[1:4], mask, ao_loc, False)
                dm_mask = dm_mask_buf[:nao_sub**2].reshape(nao_sub,nao_sub)
                dm_mask = take_last2d(dm0_sorted, mask, out=dm_mask)
                ao_dm_mask = contract('nig,ij->njg', ao_mask[:4], dm_mask, out=ao_dm_mask)
                ao_dm0 = contract('ik,il->kl', dm0, ao1[0], out=ao_dm0)
                aow = aow[:,:nao_sub]
                for ia in range(_sorted_mol.natm):
                    p0, p1 = aoslices[ia][2:]
                    # *2 for \nabla|ket> in rho1
                    wv = contract('xig,ig->xg', ao1[1:,p0:p1,:], ao_dm0[p0:p1,:], out=wv)
                    wv *= 2
                    # aow ~ rho1 ~ d/dR1
                    wv = cupy.multiply(wf, wv, out=wv)
                    for i in range(3):
                        numint._scale_ao(ao_dm_mask[0], wv[i], out=aow[i])
                    vmat_dm[ia][:,:,mask] += contract('yjg,xjg->xyj', ao_mask[1:4], aow)
                t1 = log.timer_debug2('integration', *t1)

            vmat_dm = opt.unsort_orbitals(vmat_dm, axis=[3])
            for ia in range(_sorted_mol.natm):
                p0, p1 = aoslices[ia][2:]
                contract('xypq,pq->xyp', ipip[:,:,:,p0:p1], dm0[:,p0:p1], beta=1, out=vmat_dm[ia])
        elif xctype == 'GGA':
            ao_deriv = 2
            t1 = log.init_timer()
            nd = (ao_deriv+1)*(ao_deriv+2)*(ao_deriv+3)//6
            aow_buf = cupy.empty(max(3*nao,2*nocc)* MIN_BLK_SIZE)
            wv_buf = cupy.empty(3* ncomp* MIN_BLK_SIZE)
            ao1_buf = cupy.empty(nd*nao*MIN_BLK_SIZE)
            ao_dm_mask_buf = cupy.empty(4 * nao * MIN_BLK_SIZE)
            ao_dm0_buf = cupy.empty(4 * nao * MIN_BLK_SIZE)
            dm_mask_buf = cupy.empty(nao*nao)
            vmat_dm_buf = cupy.empty(3*3*nao)
            dR_rho1_buf = cupy.empty(3*ncomp*MIN_BLK_SIZE)
            for ao_mask, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                          grid_range=(grid_start, grid_end)):
                nao_sub = len(mask)
                blk_size = len(weight)
                ao1 = cupy.ndarray((nd, nao, blk_size), memptr=ao1_buf.data)
                rho = cupy.ndarray((ncomp, blk_size), memptr=rho_buf.data)

                ao_dm_mask = cupy.ndarray((4, nao_sub, blk_size), memptr=ao_dm_mask_buf.data)
                ao_dm0 = cupy.ndarray((4, nao, blk_size), memptr=ao_dm0_buf.data)
                wv = cupy.ndarray((3, ncomp, blk_size), memptr=wv_buf.data)
                dR_rho1 =  cupy.ndarray((3, ncomp, blk_size), memptr=dR_rho1_buf.data)
                vmat_dm_tmp = cupy.ndarray((3,3,nao_sub), memptr=vmat_dm_buf.data)

                ao = contract('nip,ij->njp', ao_mask, coeff[mask], out=ao1)
                rho = numint.eval_rho2(_sorted_mol, ao[:4], mo_coeff, mo_occ, mask, xctype, buf=aow_buf, out=rho)
                t1 = log.timer_debug2('eval rho', *t1)
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype, spin=0)[1:3]
                t1 = log.timer_debug2('eval vxc', *t1)
                wv1 = cupy.multiply(weight, vxc, out=vxc)
                wf = cupy.multiply(weight, fxc, out=fxc)
                wv1[0] *= .5
                aow  = cupy.ndarray((3, nao, blk_size), memptr=aow_buf.data)
                aow = rks_grad._make_dR_dao_w(ao, wv1, out=aow)
                _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)
                contract('ki,nkl->nil', dm0, ao[:4], out=ao_dm0)
                dm_mask = dm_mask_buf[:nao_sub**2].reshape(nao_sub,nao_sub)
                dm_mask = take_last2d(dm0_sorted, mask, out=dm_mask)
                ao_dm_mask = contract('nig,ij->njg', ao_mask[:4], dm_mask, out=ao_dm_mask)
                aow  = cupy.ndarray((3, nao_sub, blk_size), memptr=aow_buf.data)
                for ia in range(_sorted_mol.natm):
                    dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype,
                                            buf=wv[0,0], out=dR_rho1)
                    wv = _contract_rho1_fxc(dR_rho1, wf)
                    wv[:,0] *= .5
                    for i in range(3):
                        aow = rks_grad._make_dR_dao_w(ao_mask, wv[i], out=aow)
                        contract('xjg,jg->xj', aow, ao_dm_mask[0], out=vmat_dm_tmp[i])
                    for i in range(3):
                        numint._scale_ao(ao_dm_mask[:4], wv[i,:4], out=aow[i])
                    vmat_dm_tmp = contract('yjg,xjg->xyj', ao_mask[1:4], aow, beta=1, out=vmat_dm_tmp)
                    vmat_dm[ia][:,:,mask] += vmat_dm_tmp
                t1 = log.timer_debug2('integration', *t1)
            vmat_dm = opt.unsort_orbitals(vmat_dm, axis=[3])
            for ia in range(_sorted_mol.natm):
                p0, p1 = aoslices[ia][2:]
                contract('xypq,pq->xyp', ipip[:,:,:,p0:p1], dm0[:,p0:p1], beta=1, out=vmat_dm[ia])
                contract('yxqp,pq->xyp', ipip[:,:,p0:p1], dm0[:,p0:p1],  beta=1, out=vmat_dm[ia])
        elif xctype == 'MGGA':
            ao_deriv = 2
            t1 = log.init_timer()

            nd = (ao_deriv+1)*(ao_deriv+2)*(ao_deriv+3)//6
            aow_buf = cupy.empty(max(6*nao,2*nocc)* MIN_BLK_SIZE)
            wv_buf = cupy.empty(3* ncomp* MIN_BLK_SIZE)
            ao1_buf = cupy.empty(nd*nao*MIN_BLK_SIZE)
            ao_dm_mask_buf = cupy.empty(4 * nao * MIN_BLK_SIZE)
            ao_dm0_buf = cupy.empty(4 * nao * MIN_BLK_SIZE)
            dm_mask_buf = cupy.empty(nao*nao)
            vmat_dm_buf = cupy.empty(3*3*nao)
            dR_rho1_buf = cupy.empty(3*ncomp*MIN_BLK_SIZE)

            for ao_mask, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                          grid_range=(grid_start, grid_end)):
                nao_sub = len(mask)
                blk_size = len(weight)
                ao1 = cupy.ndarray((nd, nao, blk_size), memptr=ao1_buf.data)
                rho = cupy.ndarray((ncomp, blk_size), memptr=rho_buf.data)
                ao_dm_mask = cupy.ndarray((4, nao_sub, blk_size), memptr=ao_dm_mask_buf.data)
                ao_dm0 = cupy.ndarray((4, nao, blk_size), memptr=ao_dm0_buf.data)
                wv = cupy.ndarray((3, ncomp, blk_size), memptr=wv_buf.data)
                dR_rho1 =  cupy.ndarray((3, ncomp, blk_size), memptr=dR_rho1_buf.data)
                vmat_dm_tmp = cupy.ndarray((3,3,nao_sub), memptr=vmat_dm_buf.data)

                ao = contract('nip,ij->njp', ao_mask, coeff[mask], out=ao1)
                rho = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff, mo_occ, mask, xctype, buf=aow_buf, out=rho)
                t1 = log.timer_debug2('eval rho', *t1)
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype, spin=0)[1:3]
                t1 = log.timer_debug2('eval vxc', *t1)
                wv1 = cupy.multiply(weight, vxc, out=vxc)
                wf = cupy.multiply(weight, fxc, out=fxc)
                wv1[0] *= .5
                wv1[4] *= .25
                aow  = cupy.ndarray((3, nao, blk_size), memptr=aow_buf.data)
                aow = rks_grad._make_dR_dao_w(ao, wv1, out=aow)
                _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)
                aow  = cupy.ndarray((6, nao, blk_size), memptr=aow_buf.data)
                for i in range(4,10):
                    numint._scale_ao(ao[i], wv1[4], out=aow[i-4])
                _d1d2_dot_(ipip, mol, [aow[0], aow[1], aow[2]], [ao[XX], ao[XY], ao[XZ]], mask, ao_loc, False)
                _d1d2_dot_(ipip, mol, [aow[1], aow[3], aow[4]], [ao[YX], ao[YY], ao[YZ]], mask, ao_loc, False)
                _d1d2_dot_(ipip, mol, [aow[2], aow[4], aow[5]], [ao[ZX], ao[ZY], ao[ZZ]], mask, ao_loc, False)
                dm_mask = dm_mask_buf[:nao_sub**2].reshape(nao_sub,nao_sub)
                dm_mask = take_last2d(dm0_sorted, mask, out=dm_mask)
                contract('ki,nkl->nil', dm0, ao[:4], out=ao_dm0)
                ao_dm_mask = contract('nig,ij->njg', ao_mask[:4], dm_mask, out=ao_dm_mask)
                aow  = cupy.ndarray((3, nao_sub, blk_size), memptr=aow_buf.data)
                for ia in range(_sorted_mol.natm):
                    dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype,
                                            buf=wv[0,0], out=dR_rho1)
                    wv = _contract_rho1_fxc(dR_rho1, wf)
                    wv[:,0] *= .5
                    wv[:,4] *= .5  # for the factor 1/2 in tau
                    for i in range(3):
                        aow = rks_grad._make_dR_dao_w(ao_mask, wv[i], out=aow)
                        contract('xjg,jg->xj', aow, ao_dm_mask[0], out=vmat_dm_tmp[i])

                    for i in range(3):
                        numint._scale_ao(ao_dm_mask[:4], wv[i,:4], out=aow[i])
                    contract('yjg,xjg->xyj', ao_mask[1:4], aow, beta=1, out=vmat_dm_tmp)

                    for i in range(3):
                        numint._scale_ao(ao_dm_mask[1], wv[i,4], out=aow[i])
                    contract('jg,xjg->xj', ao_mask[XX], aow, beta=1, out=vmat_dm_tmp[:,0])
                    contract('jg,xjg->xj', ao_mask[XY], aow, beta=1, out=vmat_dm_tmp[:,1])
                    contract('jg,xjg->xj', ao_mask[XZ], aow, beta=1, out=vmat_dm_tmp[:,2])

                    for i in range(3):
                        numint._scale_ao(ao_dm_mask[2], wv[i,4], out=aow[i])
                    contract('jg,xjg->xj', ao_mask[YX], aow, beta=1, out=vmat_dm_tmp[:,0])
                    contract('jg,xjg->xj', ao_mask[YY], aow, beta=1, out=vmat_dm_tmp[:,1])
                    contract('jg,xjg->xj', ao_mask[YZ], aow, beta=1, out=vmat_dm_tmp[:,2])

                    for i in range(3):
                        numint._scale_ao(ao_dm_mask[3], wv[i,4], out=aow[i])
                    contract('jg,xjg->xj', ao_mask[ZX], aow, beta=1, out=vmat_dm_tmp[:,0])
                    contract('jg,xjg->xj', ao_mask[ZY], aow, beta=1, out=vmat_dm_tmp[:,1])
                    contract('jg,xjg->xj', ao_mask[ZZ], aow, beta=1, out=vmat_dm_tmp[:,2])

                    vmat_dm[ia][:,:,mask] += vmat_dm_tmp
                t1 = log.timer_debug2('integration', *t1)
            vmat_dm = opt.unsort_orbitals(vmat_dm, axis=[3])
            for ia in range(_sorted_mol.natm):
                p0, p1 = aoslices[ia][2:]
                contract('xypq,pq->xyp', ipip[:,:,:,p0:p1], dm0[:,p0:p1], beta=1, out=vmat_dm[ia])
                contract('yxqp,pq->xyp', ipip[:,:,p0:p1], dm0[:,p0:p1], beta=1, out=vmat_dm[ia])
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

def _get_enlc_deriv2(hessobj, mo_coeff, mo_occ, max_memory, log = None):
    """
        Equation notation follows:
        Liang J, Feng X, Liu X, Head-Gordon M. Analytical harmonic vibrational frequencies with
        VV10-containing density functionals: Theory, efficient implementation, and
        benchmark assessments. J Chem Phys. 2023 May 28;158(20):204109. doi: 10.1063/5.0152838.
    """

    if log is None:
        log = logger.new_logger(hessobj)
    mol = hessobj.mol
    mf = hessobj.base

    grids = mf.nlcgrids
    if not hessobj.grid_response:
        if grids.coords is None:
            grids.build()

        ni = mf._numint
        opt = getattr(ni, 'gdftopt', None)
        if opt is None:
            ni.build(mol, grids.coords)
            opt = ni.gdftopt
    else:
        grids = grids.copy()
        grids.build(sort_grids_of_each_atom = True)

        ni = numint.NumInt()
        ni.gdftopt = None
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    natm = _sorted_mol.natm
    mol = None

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    mo_coeff = None
    mo_occ = None
    if dm0.ndim == 3:
        assert dm0.shape[0] == 2
        dm0 = dm0[0] + dm0[1]
    dm0_sorted = opt.sort_orbitals(dm0, axis=[0,1])
    dm_mask_buf = cupy.empty(nao * nao)
    dm0 = None

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
    beta = 0.03125 * (3.0 / nlc_pars[0]**2)**0.75

    ngrids_full = grids.coords.shape[0]
    rho_drho = cupy.empty([4, ngrids_full])
    g1 = 0
    for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 1, strict_grid_order = True):
        g0, g1 = g1, g1 + weight.size
        dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)
        rho_drho[:, g0:g1] = numint.eval_rho(_sorted_mol, ao, dm0_masked, xctype = "NLC", hermi = 1)
    assert g1 == ngrids_full

    rho_i = rho_drho[0,:]

    rho_nonzero_mask = cupy.logical_and(
        rho_i >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD,
        cupy.abs(grids.weights) > 1e-14,
    )

    rho_i = rho_i[rho_nonzero_mask]
    grids_coords = cupy.ascontiguousarray(grids.coords[rho_nonzero_mask, :])
    grids_weights = grids.weights[rho_nonzero_mask]
    ngrids = grids_coords.shape[0]

    nabla_rho_i = rho_drho[1:4, rho_nonzero_mask]
    gamma_i = batched_vec_norm2(nabla_rho_i.T)

    stream = cupy.cuda.get_current_stream()

    omega_i               = cupy.empty(ngrids)
    domega_drho_i         = cupy.empty(ngrids)
    domega_dgamma_i       = cupy.empty(ngrids)
    d2omega_drho2_i       = cupy.empty(ngrids)
    d2omega_dgamma2_i     = cupy.empty(ngrids)
    d2omega_drho_dgamma_i = cupy.empty(ngrids)
    libgdft.VXC_vv10nlc_hess_eval_omega_derivative(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
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
    kappa_i         = kappa_prefactor * rho_i**(1.0/6.0)
    dkappa_drho_i   = (kappa_prefactor * (1.0/6.0)) * rho_i**(-5.0/6.0)
    d2kappa_drho2_i = (kappa_prefactor * (-5.0/36.0)) * rho_i**(-11.0/6.0)

    rho_weight_i = rho_i * grids_weights
    U_i = cupy.empty(ngrids)
    W_i = cupy.empty(ngrids)
    A_i = cupy.empty(ngrids)
    B_i = cupy.empty(ngrids)
    C_i = cupy.empty(ngrids)
    E_i = cupy.empty(ngrids)

    libgdft.VXC_vv10nlc_hess_eval_UWABCE(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(U_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(W_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(B_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(C_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(E_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_weight_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids)
    )
    del rho_weight_i

    f_rho_i = beta + E_i + rho_i * (dkappa_drho_i * U_i + domega_drho_i * W_i)
    f_gamma_i = rho_i * domega_dgamma_i * W_i
    fw_rho_i   =   f_rho_i * grids_weights
    fw_gamma_i = f_gamma_i * grids_weights

    ao_loc_sorted = _sorted_mol.ao_loc
    ao_expand = ao_loc_sorted[1:] - ao_loc_sorted[:-1]
    from pyscf.gto.mole import ATOM_OF
    i_atom_of_aos = numpy.repeat(_sorted_mol._bas[:,ATOM_OF], ao_expand)
    i_atom_of_aos = cupy.asarray(i_atom_of_aos, dtype = cupy.int32)

    d2e = cupy.zeros([natm, natm, 3, 3])

    if not hessobj.grid_response:
        drho_dA   = cupy.empty([natm, 3, ngrids], order = "C")
        dgamma_dA = cupy.empty([natm, 3, ngrids], order = "C")

        g0_full = 0
        g0_nonzero = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv = 3, strict_grid_order = True):
            g1_full = g0_full + weight.shape[0]

            ao = ao[:, :, rho_nonzero_mask[g0_full : g1_full]]

            if ao.size == 0:
                g0_full = g1_full
                continue

            g1_nonzero = g0_nonzero + ao.shape[-1]

            mu = ao[0, :, :]
            dmu_dr = ao[1:4, :, :]
            d2mu_dr2 = get_d2mu_dr2(ao)
            d3mu_dr3 = get_d3mu_dr3(ao)

            split_drho_dr = nabla_rho_i[:, g0_nonzero : g1_nonzero]

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)
            masked_i_atom_of_aos = i_atom_of_aos[idx]

            drho_dA_orbital_response, _ = \
                get_drho_dA_sparse(dm0_masked, "GGA", natm, masked_i_atom_of_aos, None, mu, dmu_dr, d2mu_dr2, with_grid_response = False)
            drho_dA[:, :, g0_nonzero : g1_nonzero] = drho_dA_orbital_response[:, :, 0, :]
            dgamma_dA[:, :, g0_nonzero : g1_nonzero] = 2 * contract("Adxg,xg->Adg", drho_dA_orbital_response[:, :, 1:4, :], split_drho_dr)

            split_fw_rho   =   fw_rho_i[g0_nonzero : g1_nonzero]
            split_fw_gamma = fw_gamma_i[g0_nonzero : g1_nonzero]

            # d2gamma/dAdB = 2 * d_nabla_rho/dA * d_nabla_rho/dB + 2 * nabla_rho * d2_nabla_rho/dAdB
            d2e += 2 * cupy.einsum("Adxg,BDxg,g->ABdD", drho_dA_orbital_response[:, :, 1:4, :], drho_dA_orbital_response[:, :, 1:4, :], split_fw_gamma)
            del drho_dA_orbital_response

            d2e += contract_d2rho_dAdB_sparse(dm0_masked, "GGA", natm, masked_i_atom_of_aos, None,
                                              mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                                              split_fw_rho, 2 * split_drho_dr * split_fw_gamma,
                                              with_grid_response = False)

            del split_fw_rho, split_fw_gamma, split_drho_dr
            del mu, dmu_dr, d2mu_dr2, d3mu_dr3

            g0_nonzero = g1_nonzero
            g0_full = g1_full
        assert g1_full == ngrids_full
        assert g1_nonzero == ngrids

        del fw_rho_i, fw_gamma_i

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

        d2e += contract("Adg,BDg->ABdD",   drho_dA,   f_rho_A_i * grids_weights)
        d2e += contract("Adg,BDg->ABdD", dgamma_dA, f_gamma_A_i * grids_weights)

    else:
        log.info("Calculating grid response for VV10 Hessian")

        # First half of E_{w,w}^{AB} in Eq 32
        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
        ao_nbytes_per_grid = ((9 * 2) * natm * natm + 3 * natm + 2) * 8
        ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
        if ngrids_per_batch < 16:
            raise MemoryError(f"Out of GPU memory for NLC energy second derivative, available gpu memory = {get_avail_mem()}"
                              f" bytes, nao = {nao}, natm = {natm}, ngrids = {ngrids_full}")
        ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
        ### Don't split the batch too small for get_d2weight_dAdB()
        # ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)
        ngrids_per_batch = min(ngrids_per_batch, numpy.iinfo(numpy.int32).max // (natm * natm * 9)) # Avoid int32 overflow

        g0_nonzero = 0
        for g0_full in range(0, ngrids_full, ngrids_per_batch):
            g1_full = min(g0_full + ngrids_per_batch, ngrids_full)
            d2w_dAdB = get_d2weight_dAdB(_sorted_mol, grids, (g0_full, g1_full))
            d2w_dAdB = d2w_dAdB[:, :, :, :, rho_nonzero_mask[g0_full : g1_full]]
            g1_nonzero = g0_nonzero + d2w_dAdB.shape[4]
            d2e += contract("ABdDg,g->ABdD", d2w_dAdB, rho_i[g0_nonzero : g1_nonzero] * (beta + E_i[g0_nonzero : g1_nonzero]))
            g0_nonzero = g1_nonzero
        assert g0_nonzero == ngrids

        grids_weights_1 = get_dweight_dA(_sorted_mol, grids)
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

        # Second half of E_{w,w}^{AB} in Eq 32
        d2e += contract("Adg,BDg->ABdD", grids_weights_1, E_Bw_i * rho_i)

        drho_dA_full_response   = cupy.empty([natm, 3, ngrids], order = "C")
        dgamma_dA_full_response = cupy.empty([natm, 3, ngrids], order = "C")

        g0_full = 0
        g0_nonzero = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv = 2, strict_grid_order = True):
            g1_full = g0_full + weight.shape[0]

            ao = ao[:, :, rho_nonzero_mask[g0_full : g1_full]]

            if ao.size == 0:
                g0_full = g1_full
                continue

            g1_nonzero = g0_nonzero + ao.shape[-1]

            mu = ao[0, :, :]
            dmu_dr = ao[1:4, :, :]
            d2mu_dr2 = get_d2mu_dr2(ao)

            split_drho_dr = nabla_rho_i[:, g0_nonzero : g1_nonzero]

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)
            i_atom_of_grids = int(grids.atm_idx[g0_full])
            assert cupy.max(cupy.abs(grids.atm_idx[g0_full : g1_full] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)
            masked_i_atom_of_aos = i_atom_of_aos[idx]

            drho_dA_orbital_response, drho_dA_grid_response = \
                get_drho_dA_sparse(dm0_masked, "GGA", natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr, d2mu_dr2)
            drho_dA_full = drho_dA_orbital_response + drho_dA_grid_response
            del drho_dA_orbital_response, drho_dA_grid_response
            drho_dA_full_response[:, :, g0_nonzero : g1_nonzero] = drho_dA_full[:, :, 0, :]
            dgamma_dA_full_response[:, :, g0_nonzero : g1_nonzero] = 2 * contract("Adxg,xg->Adg", drho_dA_full[:, :, 1:4, :], split_drho_dr)

            # First two terms in E_{G,G}^{AB} in Eq 37, first piece:
            # d2gamma/dAdB = 2 * d_nabla_rho/dA * d_nabla_rho/dB + 2 * nabla_rho * d2_nabla_rho/dAdB
            split_fw_gamma = fw_gamma_i[g0_nonzero : g1_nonzero]
            d2e += 2 * cupy.einsum("Adxg,BDxg,g->ABdD", drho_dA_full[:, :, 1:4, :], drho_dA_full[:, :, 1:4, :], split_fw_gamma)
            del split_fw_gamma, drho_dA_full

            del split_drho_dr
            del mu, dmu_dr, d2mu_dr2

            g0_nonzero = g1_nonzero
            g0_full = g1_full
        assert g1_full == ngrids_full
        assert g1_nonzero == ngrids

        # First term of E_{G,w}^{AB} in Eq 34, and its transpose
        E_Gw_AB_term_1_right = drho_dA_full_response * f_rho_i + dgamma_dA_full_response * f_gamma_i
        E_Gw_AB_term_1 = contract("Adg,BDg->ABdD", grids_weights_1, E_Gw_AB_term_1_right)
        del E_Gw_AB_term_1_right
        d2e += E_Gw_AB_term_1 + E_Gw_AB_term_1.transpose(1,0,3,2)
        del E_Gw_AB_term_1
        # Second term of E_{G,w}^{AB} in Eq 34, and its transpose
        E_Gw_AB_term_2_right = (E_Bw_i + (U_Bw_i * dkappa_drho_i + W_Bw_i * domega_drho_i) * rho_i) * grids_weights
        E_Gw_AB_term_2 = contract("Adg,BDg->ABdD", drho_dA_full_response, E_Gw_AB_term_2_right)
        del E_Gw_AB_term_2_right
        d2e += E_Gw_AB_term_2 + E_Gw_AB_term_2.transpose(1,0,3,2)
        del E_Gw_AB_term_2
        # Third term of E_{G,w}^{AB} in Eq 34, and its transpose
        E_Gw_AB_term_3_right = W_Bw_i * domega_dgamma_i * rho_i * grids_weights
        E_Gw_AB_term_3 = contract("Adg,BDg->ABdD", dgamma_dA_full_response, E_Gw_AB_term_3_right)
        del E_Gw_AB_term_3_right
        d2e += E_Gw_AB_term_3 + E_Gw_AB_term_3.transpose(1,0,3,2)
        del E_Gw_AB_term_3

        del E_Bw_i, U_Bw_i, W_Bw_i

        assert grids.atm_idx.shape[0] == grids.coords.shape[0]
        grid_to_atom_index_map = grids.atm_idx[rho_nonzero_mask]

        grid_offsets_of_atom = cupy.r_[0, cupy.flatnonzero(cupy.diff(grid_to_atom_index_map)) + 1]
        if grid_to_atom_index_map[-1] < 0:
            pass # There's padded grids whose index < 0, and the first index of padded grids is the number of valid grids
        else:
            grid_offsets_of_atom = cupy.append(grid_offsets_of_atom, grid_to_atom_index_map.shape[0])
        grid_offsets_of_atom = cupy.asarray(grid_offsets_of_atom, dtype = cupy.int32)

        assert grid_offsets_of_atom.shape == (natm + 1,)
        for i_atom in range(natm):
            assert cupy.all(grid_to_atom_index_map[grid_offsets_of_atom[i_atom] : grid_offsets_of_atom[i_atom + 1]] == i_atom)
        assert cupy.all(grid_to_atom_index_map[grid_offsets_of_atom[natm] : ] < 0)

        rho_weight_i = rho_i * grids_weights
        E_Bgr_i = cupy.empty([natm, 3, ngrids], order = "C")
        U_Bgr_i = cupy.empty([natm, 3, ngrids], order = "C")
        W_Bgr_i = cupy.empty([natm, 3, ngrids], order = "C")
        libgdft.VXC_vv10nlc_hess_eval_EUW_grid_response_offdiagonal(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(E_Bgr_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(U_Bgr_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(W_Bgr_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
            ctypes.cast(rho_weight_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grid_to_atom_index_map.data.ptr, ctypes.c_void_p),
            ctypes.cast(grid_offsets_of_atom.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(natm),
        )
        del rho_weight_i
        for i_atom in range(natm):
            g0, g1 = grid_offsets_of_atom[i_atom], grid_offsets_of_atom[i_atom + 1]

            E_Bgr_i[i_atom, :, g0:g1] = -cupy.sum(E_Bgr_i[:, :, g0:g1], axis = 0)
            U_Bgr_i[i_atom, :, g0:g1] = -cupy.sum(U_Bgr_i[:, :, g0:g1], axis = 0)
            W_Bgr_i[i_atom, :, g0:g1] = -cupy.sum(W_Bgr_i[:, :, g0:g1], axis = 0)

        # E_{w,gr}^{AB} in Eq 33, and its transpose
        E_wgr_AB_term = contract("Adg,BDg->ABdD", grids_weights_1, E_Bgr_i * rho_i)
        d2e += E_wgr_AB_term
        d2e += E_wgr_AB_term.transpose(1,0,3,2)
        del grids_weights_1
        del E_wgr_AB_term

        # First term in E_{G,gr}^{AB} in Eq 35, and its transpose
        E_Ggr_AB_term_1_right = (E_Bgr_i + (U_Bgr_i * dkappa_drho_i + W_Bgr_i * domega_drho_i) * rho_i) * grids_weights
        E_Ggr_AB_term_1 = contract("Adg,BDg->ABdD", drho_dA_full_response, E_Ggr_AB_term_1_right)
        del E_Ggr_AB_term_1_right
        d2e += E_Ggr_AB_term_1 + E_Ggr_AB_term_1.transpose(1,0,3,2)
        del E_Ggr_AB_term_1
        # Second term in E_{G,gr}^{AB} in Eq 35, and its transpose
        E_Ggr_AB_term_2_right = W_Bgr_i * domega_dgamma_i * rho_i * grids_weights
        E_Ggr_AB_term_2 = contract("Adg,BDg->ABdD", dgamma_dA_full_response, E_Ggr_AB_term_2_right)
        del E_Ggr_AB_term_2_right
        d2e += E_Ggr_AB_term_2 + E_Ggr_AB_term_2.transpose(1,0,3,2)
        del E_Ggr_AB_term_2

        del E_Bgr_i, U_Bgr_i, W_Bgr_i

        # First two terms in E_{G,G}^{AB} in Eq 37, second piece
        g0_full = 0
        g0_nonzero = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv = 3, strict_grid_order = True):
            g1_full = g0_full + weight.shape[0]

            ao = ao[:, :, rho_nonzero_mask[g0_full : g1_full]]

            if ao.size == 0:
                g0_full = g1_full
                continue

            g1_nonzero = g0_nonzero + ao.shape[-1]

            mu = ao[0, :, :]
            dmu_dr = ao[1:4, :, :]
            d2mu_dr2 = get_d2mu_dr2(ao)
            d3mu_dr3 = get_d3mu_dr3(ao)

            split_drho_dr = nabla_rho_i[:, g0_nonzero : g1_nonzero]

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)
            i_atom_of_grids = int(grids.atm_idx[g0_full])
            assert cupy.max(cupy.abs(grids.atm_idx[g0_full : g1_full] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)
            masked_i_atom_of_aos = i_atom_of_aos[idx]

            split_fw_rho   =   fw_rho_i[g0_nonzero : g1_nonzero]
            split_fw_gamma = fw_gamma_i[g0_nonzero : g1_nonzero]

            d2e += contract_d2rho_dAdB_sparse(dm0_masked, "GGA", natm, masked_i_atom_of_aos, i_atom_of_grids,
                                              mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                                              split_fw_rho, 2 * split_drho_dr * split_fw_gamma)

            del split_fw_rho, split_fw_gamma, split_drho_dr
            del mu, dmu_dr, d2mu_dr2, d3mu_dr3

            g0_nonzero = g1_nonzero
            g0_full = g1_full
        assert g1_full == ngrids_full
        assert g1_nonzero == ngrids

        # Last two terms in E_{G,G}^{AB} in Eq 37
        drho_dA_full_response   = cupy.ascontiguousarray(drho_dA_full_response)
        dgamma_dA_full_response = cupy.ascontiguousarray(dgamma_dA_full_response)
        f_rho_A_i_full_response   = cupy.empty([natm, 3, ngrids], order = "C")
        f_gamma_A_i_full_response = cupy.empty([natm, 3, ngrids], order = "C")

        libgdft.VXC_vv10nlc_hess_eval_f_t(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(f_rho_A_i_full_response.data.ptr, ctypes.c_void_p),
            ctypes.cast(f_gamma_A_i_full_response.data.ptr, ctypes.c_void_p),
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
            ctypes.cast(drho_dA_full_response.data.ptr, ctypes.c_void_p),
            ctypes.cast(dgamma_dA_full_response.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(3 * natm),
        )

        d2e += contract("Adg,BDg->ABdD",   drho_dA_full_response,   f_rho_A_i_full_response * grids_weights)
        d2e += contract("Adg,BDg->ABdD", dgamma_dA_full_response, f_gamma_A_i_full_response * grids_weights)

        del f_rho_A_i_full_response, f_gamma_A_i_full_response
        del drho_dA_full_response, dgamma_dA_full_response

        # E_{gr,gr}^{AB} in Eq 36
        rho_weight_i = rho_i * grids_weights
        D_B_i = cupy.empty([natm, 3, 3, ngrids], order = "C")
        libgdft.VXC_vv10nlc_hess_eval_D_B_in_double_grid_response_offdiagonal(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(D_B_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
            ctypes.cast(rho_weight_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grid_to_atom_index_map.data.ptr, ctypes.c_void_p),
            ctypes.cast(grid_offsets_of_atom.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(natm),
        )
        del rho_weight_i
        for i_atom in range(natm):
            g0, g1 = grid_offsets_of_atom[i_atom], grid_offsets_of_atom[i_atom + 1]
            D_B_i[i_atom, :, :, g0:g1] = -cupy.sum(D_B_i[:, :, :, g0:g1], axis = 0)

        atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]
        for i_atom in range(natm):
            g_i_with_response = atom_to_grid_index_map[i_atom]
            if len(g_i_with_response) == 0:
                continue
            d2e[i_atom, :, :, :] += contract("BdDg,g->BdD", D_B_i[:, :, :, g_i_with_response], rho_i[g_i_with_response] * grids_weights[g_i_with_response])
        D_B_i = None

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

    ngrids_glob = grids.coords.shape[0]
    grid_start, grid_end = numint.gen_grid_range(ngrids_glob, device_id)
    with cupy.cuda.Device(device_id):
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
            ncomp = 1
        elif xctype == 'GGA':
            ncomp = 4
        else:
            ncomp = 5
        rho_buf = cupy.empty(ncomp*MIN_BLK_SIZE)
        if xctype == 'LDA':
            ao_deriv = 1
            nd = (ao_deriv+1)*(ao_deriv+2)*(ao_deriv+3)//6
            aow_buf = cupy.empty(max(3*nao,1*nocc)* MIN_BLK_SIZE)
            wv_buf = cupy.empty(3* MIN_BLK_SIZE)
            ao1_buf = cupy.empty(nd*nao*MIN_BLK_SIZE)
            mo_buf = cupy.empty(nd*nocc*MIN_BLK_SIZE)
            mow_buf = cupy.empty(3*nocc*MIN_BLK_SIZE)
            ao_dm0_buf = cupy.empty(nao * MIN_BLK_SIZE)
            for ao, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                     grid_range=(grid_start, grid_end)):
                blk_size = len(weight)
                # nao_sub = len(mask)
                rho = cupy.ndarray((blk_size), memptr=rho_buf.data)
                aow  = cupy.ndarray((3, nao, blk_size), memptr=aow_buf.data)
                wv = cupy.ndarray((3, blk_size), memptr=wv_buf.data)
                ao1 = cupy.ndarray((nd, nao, blk_size), memptr=ao1_buf.data)
                mo = cupy.ndarray((nd, nocc, blk_size), memptr=mo_buf.data)
                mow = cupy.ndarray((3, nocc, blk_size), memptr=mow_buf.data)
                ao_dm0 = cupy.ndarray((nao, blk_size), memptr=ao_dm0_buf.data)

                ao1 = contract('nip,ij->njp', ao, coeff[mask], out=ao1)
                rho = numint.eval_rho2(_sorted_mol, ao1[0], mo_coeff, mo_occ, mask, xctype, buf=aow_buf, out=rho)

                t1 = log.timer_debug2('eval rho', *t1)
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype, spin=0)[1:3]
                t1 = log.timer_debug2('eval vxc', *t1)
                wv1 = cupy.multiply(weight, vxc[0], out=vxc[0])
                wf = cupy.multiply(weight, fxc[0,0], out=fxc[0,0])

                numint._scale_ao(ao1[0], wv1, out=aow[0])
                v_ip = rks_grad._d1_dot_(ao1[1:4], aow[0].T, beta=1.0, out=v_ip)
                mo = contract('xig,ip->xpg', ao1, mocc, out=mo)
                ao_dm0 =  contract('ik,il->kl', dm0, ao1[0], out=ao_dm0)
                for ia in range(natm):
                    p0, p1 = aoslices[ia][2:]
                # First order density = rho1 * 2.  *2 is not applied because + c.c. in the end
                    rho1 = contract('xig,ig->xg', ao1[1:,p0:p1,:], ao_dm0[p0:p1,:], out=wv)
                    wv = cupy.multiply(wf, rho1,  out=wv)
                    for i in range(3):
                        numint._scale_ao(ao1[0], wv[i],  out=aow[i])
                        numint._scale_ao(mo[0], wv[i], out=mow[i])
                    rks_grad._d1_dot_(aow, mo[0].T, beta=1.0, out=vmat[ia])
                    rks_grad._d1_dot_(mow, ao1[0].T, beta=1.0, transpose=True, out=vmat[ia])
                t1 = log.timer_debug2('integration', *t1)
        elif xctype == 'GGA':
            ao_deriv = 2
            nd = (ao_deriv+1)*(ao_deriv+2)*(ao_deriv+3)//6
            aow_buf = cupy.empty(max(3*nao,2*nocc)* MIN_BLK_SIZE)
            wv_buf = cupy.empty(3* MIN_BLK_SIZE)
            ao1_buf = cupy.empty(nd*nao*MIN_BLK_SIZE)
            mo_buf = cupy.empty(nd*nocc*MIN_BLK_SIZE)
            mow_buf = cupy.empty(3*nocc*MIN_BLK_SIZE)
            ao_dm0_buf = cupy.empty(4*nao * MIN_BLK_SIZE)
            dR_rho1_buf = cupy.empty(3* ncomp * MIN_BLK_SIZE)

            for ao, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                     grid_range=(grid_start, grid_end)):
                blk_size = len(weight)
                # nao_sub = len(mask)
                rho = cupy.ndarray((ncomp, blk_size), memptr=rho_buf.data)
                aow  = cupy.ndarray((3, nao, blk_size), memptr=aow_buf.data)
                wv = cupy.ndarray((3, blk_size), memptr=wv_buf.data)
                ao1 = cupy.ndarray((nd, nao, blk_size), memptr=ao1_buf.data)
                mo = cupy.ndarray((nd, nocc, blk_size), memptr=mo_buf.data)
                mow = cupy.ndarray((3, nocc, blk_size), memptr=mow_buf.data)
                ao_dm0 = cupy.ndarray((4, nao, blk_size), memptr=ao_dm0_buf.data)
                dR_rho1 =  cupy.ndarray((3, ncomp, blk_size), memptr=dR_rho1_buf.data)


                ao1 = contract('nip,ij->njp', ao, coeff[mask], out=ao1)
                rho = numint.eval_rho2(_sorted_mol, ao1[:4], mo_coeff, mo_occ, mask, xctype, buf=aow_buf, out=rho)
                t1 = log.timer_debug2('eval rho', *t1)
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype, spin=0)[1:3]
                t1 = log.timer_debug2('eval vxc', *t1)
                wv = cupy.multiply(weight, vxc, out=vxc)
                wv[0] *= .5
                wf = cupy.multiply(weight, fxc, out=fxc)
                v_ip = rks_grad._gga_grad_sum_(ao1, wv,  accumulate=True, buf=aow, out=v_ip)
                mo = contract('xig,ip->xpg', ao1, mocc, out=mo)
                ao_dm0 =  contract('ik,pil->pkl', dm0, ao1[:4], out=ao_dm0)
                for ia in range(natm):
                    dR_rho1 = _make_dR_rho1(ao1, ao_dm0, ia, aoslices, xctype,
                                            buf=wv[0], out=dR_rho1)
                    wv2 = contract('xyg,sxg->syg', wf, dR_rho1, out=dR_rho1)
                    wv2[:,0] *= .5
                    for i in range(3):
                        numint._scale_ao(ao1[:4], wv2[i,:4],  out=aow[i])
                        numint._scale_ao(mo[:4], wv2[i,:4], out=mow[i])
                    rks_grad._d1_dot_(aow, mo[0].T, beta=1.0, out=vmat[ia])
                    rks_grad._d1_dot_(mow, ao1[0].T, beta=1.0, transpose=True, out=vmat[ia])
                t1 = log.timer_debug2('integration', *t1)

        elif xctype == 'MGGA':
            if grids.level < 5:
                log.warn('MGGA Hessian is sensitive to dft grids.')
            ao_deriv = 2
            nd = (ao_deriv+1)*(ao_deriv+2)*(ao_deriv+3)//6
            aow_buf = cupy.empty(max(3*nao,2*nocc)* MIN_BLK_SIZE)
            wv_buf = cupy.empty(3* MIN_BLK_SIZE)
            ao1_buf = cupy.empty(nd*nao*MIN_BLK_SIZE)
            mo_buf = cupy.empty(nd*nocc*MIN_BLK_SIZE)
            mow_buf = cupy.empty(3*nocc*MIN_BLK_SIZE)
            ao_dm0_buf = cupy.empty(4*nao * MIN_BLK_SIZE)
            dR_rho1_buf = cupy.empty(3* ncomp * MIN_BLK_SIZE)
            for ao, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                     grid_range=(grid_start, grid_end)):
                blk_size = len(weight)
                # nao_sub = len(mask)
                rho = cupy.ndarray((ncomp, blk_size), memptr=rho_buf.data)
                aow  = cupy.ndarray((3, nao, blk_size), memptr=aow_buf.data)
                wv = cupy.ndarray((3, blk_size), memptr=wv_buf.data)
                ao1 = cupy.ndarray((nd, nao, blk_size), memptr=ao1_buf.data)
                mo = cupy.ndarray((nd, nocc, blk_size), memptr=mo_buf.data)
                mow = cupy.ndarray((3, nocc, blk_size), memptr=mow_buf.data)
                ao_dm0 = cupy.ndarray((4, nao, blk_size), memptr=ao_dm0_buf.data)
                dR_rho1 =  cupy.ndarray((3, ncomp, blk_size), memptr=dR_rho1_buf.data)


                ao1 = contract('nip,ij->njp', ao, coeff[mask], out=ao1)
                rho = numint.eval_rho2(_sorted_mol, ao1[:10], mo_coeff, mo_occ, mask, xctype, buf=aow_buf, out=rho)
                t1 = log.timer_debug2('eval rho', *t1)
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype, spin=0)[1:3]
                t1 = log.timer_debug2('eval vxc', *t0)
                wv = cupy.multiply(weight, vxc, out=vxc)
                wf = cupy.multiply(weight, fxc, out=fxc)
                wv[0] *= .5
                wv[4] *= .5  # for the factor 1/2 in tau
                v_ip = rks_grad._gga_grad_sum_(ao1, wv,  accumulate=True, buf=aow, out=v_ip)
                v_ip = rks_grad._tau_grad_dot_(ao1, wv[4], accumulate=True, buf=aow[0], out=v_ip)
                mo = contract('xig,ip->xpg', ao1, mocc, out=mo)
                ao_dm0 = contract('ik,pil->pkl', dm0, ao1[:4], out=ao_dm0)
                for ia in range(natm):
                    dR_rho1 = _make_dR_rho1(ao1, ao_dm0, ia, aoslices, xctype,
                                            buf=wv[0], out=dR_rho1)
                    wv2 = contract('xyg,sxg->syg', wf, dR_rho1, out=dR_rho1)
                    wv2[:,0] *= .5
                    wv2[:,4] *= .25
                    for i in range(3):
                        numint._scale_ao(ao1[:4], wv2[i,:4], out=aow[i])
                        numint._scale_ao(mo[:4], wv2[i,:4], out=mow[i])
                    rks_grad._d1_dot_(aow, mo[0].T, beta=1.0, out=vmat[ia])
                    rks_grad._d1_dot_(mow, ao1[0].T, beta=1.0, transpose=True, out=vmat[ia])

                    for j in range(1, 4):
                        for i in range(3):
                            numint._scale_ao(ao1[j], wv2[i,4], out=aow[i])
                            numint._scale_ao(mo[j], wv2[i,4], out=mow[i])
                        rks_grad._d1_dot_(aow, mo[j].T,  beta=1.0, out=vmat[ia])
                        rks_grad._d1_dot_(mow, ao1[j].T, beta=1.0, transpose=True, out=vmat[ia])
                t1 = log.timer_debug2('integration', *t1)

        elif xctype == 'HF':
            pass
        else:
            raise NotImplementedError(f"xctype = {xctype} not supported")

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
    if hessobj.grid_response:
        return _get_vxc_deriv1_grid_response(hessobj, mo_coeff, mo_occ, max_memory)

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

def _get_vxc_deriv1_grid_response(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj.base
    ni = numint.NumInt()
    xctype = ni._xc_type(mf.xc)

    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    grids = grids.copy()
    grids.build(sort_grids_of_each_atom = True)
    ngrids = grids.coords.shape[0]

    ni.gdftopt = None
    ni.build(mol, grids.coords)
    opt = ni.gdftopt

    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    natm = mol.natm
    mol = None

    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)
    assert mo_coeff.ndim == 2
    mo_coeff_sorted = opt.sort_orbitals(mo_coeff, axis=[0])
    mocc_sorted = mo_coeff_sorted[:, mo_occ > 0]
    nao = _sorted_mol.nao
    # nmo = mo_coeff_sorted.shape[1]
    nocc = mocc_sorted.shape[1]

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    assert dm0.ndim == 2
    dm0_sorted = opt.sort_orbitals(dm0, axis=[0,1])
    dm_mask_buf = cupy.empty(nao * nao)
    dm0 = None
    mo_coeff = None

    nonzero_weight_mask = cupy.abs(grids.weights) > 1e-20 # There are d2rho/dAdB terms with very low weight but very big contribution

    ao_loc_sorted = _sorted_mol.ao_loc
    ao_expand = ao_loc_sorted[1:] - ao_loc_sorted[:-1]
    from pyscf.gto.mole import ATOM_OF
    i_atom_of_aos = numpy.repeat(_sorted_mol._bas[:,ATOM_OF], ao_expand)
    i_atom_of_aos = cupy.asarray(i_atom_of_aos, dtype = cupy.int32)

    dFock_orbital_response_dmudA_nu_term = cupy.zeros((3, nao, nao))
    dFock_ao_occ = cupy.zeros((natm, 3, nao, nocc))

    if xctype == 'LDA':
        g0 = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 1, strict_grid_order = True):
            g1 = g0 + weight.shape[0]

            ao = ao[:, :, nonzero_weight_mask[g0:g1]]

            if ao.size == 0:
                g0 = g1
                continue

            mu = ao[0]
            dmu_dr = ao[1:4]

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)
            # mo_coeff_masked = mo_coeff_sorted[idx, :]
            mocc_masked = mocc_sorted[idx, :]

            rho = numint.eval_rho(_sorted_mol, ao[0], dm0_masked, xctype = xctype, hermi = 1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype=xctype, spin=0)[1:3]
            del rho

            depsilon_drho = vxc[0] # Just of shape (ngrids,)
            d2epsilon_drho2 = fxc[0,0] # Just of shape (ngrids,)
            del vxc, fxc

            dw_dA = get_dweight_dA(_sorted_mol, grids, (g0,g1))
            dw_dA = dw_dA[:, :, nonzero_weight_mask[g0:g1]]

            # dFock_mo_occ += cupy.einsum("pi,Adg,g,pg,qg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_drho, mu, mu, mocc_masked)

            i_atom_of_grids = int(grids.atm_idx[g0])
            assert cupy.max(cupy.abs(grids.atm_idx[g0:g1] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)

            masked_i_atom_of_aos = i_atom_of_aos[idx]

            drho_dA_orbital_response, drho_dA_grid_response = \
                get_drho_dA_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr)
            drho_dA_full_response = drho_dA_orbital_response + drho_dA_grid_response
            del drho_dA_orbital_response, drho_dA_grid_response

            weight = weight[nonzero_weight_mask[g0:g1]]

            # dFock_mo_occ += cupy.einsum("pi,g,Adg,pg,qg,qj->Adij", mo_coeff_masked, d2epsilon_drho2 * weight, drho_dA_full_response, mu, mu, mocc_masked)
            mu_nu_term_prefactor = drho_dA_full_response * (d2epsilon_drho2 * weight) + dw_dA * depsilon_drho
            del drho_dA_full_response, d2epsilon_drho2, dw_dA

            dFock_sparse_ao_occ = cupy.zeros((natm, 3, mu.shape[0], nocc))

            mu_mocc = mu.T @ mocc_masked
            for i_atom in range(natm):
                prefactor_with_nu_mocc = contract("dg,gj->dgj", mu_nu_term_prefactor[i_atom, :, :], mu_mocc)
                dFock_sparse_ao_occ[i_atom, :, :, :] += contract("pg,dgj->dpj", mu, prefactor_with_nu_mocc)
                del prefactor_with_nu_mocc
            del mu_mocc
            del mu_nu_term_prefactor

            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += cupy.einsum("g,dpg,qg->dpq", depsilon_drho * weight, dmu_dr, mu)
            dmudA_nu_ao = contract("dpg,qg->dpq", dmu_dr, mu * (depsilon_drho * weight))
            dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += dmudA_nu_ao

            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum("pi,g,dpg,qg,qj->dij", mo_coeff_masked, depsilon_drho * weight, dmu_dr, mu, mocc_masked)
            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum("pi,g,dqg,pg,qj->dij", mo_coeff_masked, depsilon_drho * weight, dmu_dr, mu, mocc_masked)
            dmudA_nu_ao_ao = dmudA_nu_ao + dmudA_nu_ao.transpose(0,2,1)
            del dmudA_nu_ao
            dFock_sparse_ao_occ[i_atom_of_grids, :, :, :] += contract("dpq,qj->dpj", dmudA_nu_ao_ao, mocc_masked)
            del dmudA_nu_ao_ao

            dFock_ao_occ[:, :, idx, :] += dFock_sparse_ao_occ
            del dFock_sparse_ao_occ

            g0 = g1
        assert g1 == ngrids

    elif xctype == 'GGA':
        g0 = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 2, strict_grid_order = True):
            g1 = g0 + weight.shape[0]

            ao = ao[:, :, nonzero_weight_mask[g0:g1]]

            if ao.size == 0:
                g0 = g1
                continue

            mu = ao[0]
            dmu_dr = ao[1:4]
            d2mu_dr2 = get_d2mu_dr2(ao)

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)
            # mo_coeff_masked = mo_coeff_sorted[idx, :]
            mocc_masked = mocc_sorted[idx, :]

            rho = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked, xctype = xctype, hermi = 1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype=xctype, spin=0)[1:3]
            del rho

            dw_dA = get_dweight_dA(_sorted_mol, grids, (g0,g1))
            dw_dA = dw_dA[:, :, nonzero_weight_mask[g0:g1]]

            # dFock_mo_occ += cupy.einsum("pi,Adg,g,pg,qg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_drho, mu, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adg,xg,xpg,qg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_dnablarho, dmu_dr, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adg,xg,xqg,pg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_dnablarho, dmu_dr, mu, mocc_masked)
            dwdA_vxc = contract("Adg,xg->Adxg", dw_dA, vxc)
            del dw_dA

            i_atom_of_grids = int(grids.atm_idx[g0])
            assert cupy.max(cupy.abs(grids.atm_idx[g0:g1] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)

            masked_i_atom_of_aos = i_atom_of_aos[idx]

            drho_dA_orbital_response, drho_dA_grid_response = \
                get_drho_dA_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr, d2mu_dr2)
            drho_dA_full_response = drho_dA_orbital_response + drho_dA_grid_response
            del drho_dA_orbital_response, drho_dA_grid_response

            weight = weight[nonzero_weight_mask[g0:g1]]

            fwxc = fxc * weight
            del fxc
            drhodA_fwxc = contract("xyg,Adyg->Adxg", fwxc, drho_dA_full_response)
            del drho_dA_full_response
            del fwxc

            # dFock_mo_occ += cupy.einsum("pi,Adg,pg,qg,qj->Adij", mo_coeff_masked, drhodA_fwxc[:, :, 0, :], mu, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adxg,xpg,qg,qj->Adij", mo_coeff_masked, drhodA_fwxc[:, :, 1:4, :], dmu_dr, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adxg,xqg,pg,qj->Adij", mo_coeff_masked, drhodA_fwxc[:, :, 1:4, :], dmu_dr, mu, mocc_masked)
            mu_nu_term_prefactor = drhodA_fwxc + dwdA_vxc
            del drhodA_fwxc, dwdA_vxc

            dFock_sparse_ao_occ = cupy.zeros((natm, 3, mu.shape[0], nocc))

            mu_mocc = mu.T @ mocc_masked
            dmudr_mocc = contract("dqg,qj->djg", dmu_dr, mocc_masked)
            for i_atom in range(natm):
                mu_on_right_term = contract("xpg,dxg->dpg", ao[0:4], mu_nu_term_prefactor[i_atom, :, 0:4, :])
                dFock_sparse_ao_occ[i_atom, :, :, :] += contract("dpg,gj->dpj", mu_on_right_term, mu_mocc)
                del mu_on_right_term
                nablarho_dmudr_on_right_term = contract("dxg,xjg->djg", mu_nu_term_prefactor[i_atom, :, 1:4, :], dmudr_mocc)
                dFock_sparse_ao_occ[i_atom, :, :, :] += contract("pg,djg->dpj", mu, nablarho_dmudr_on_right_term)
                del nablarho_dmudr_on_right_term
            del mu_mocc, dmudr_mocc
            del mu_nu_term_prefactor

            wv = vxc * weight
            del vxc
            weight_depsilon_drho = wv[0]
            weight_depsilon_dnablarho = wv[1:4]
            del wv

            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += cupy.einsum("g,dpg,qg->dpq", depsilon_drho * weight, dmu_dr, mu)
            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += cupy.einsum("xg,dxpg,qg->dpq", depsilon_dnablarho * weight, d2mu_dr2, mu)
            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += cupy.einsum("xg,dpg,xqg->dpq", depsilon_dnablarho * weight, dmu_dr, dmu_dr)
            dmudr_weight_depsilondnablarho = contract("xg,xpg->pg", weight_depsilon_dnablarho, dmu_dr)
            dmudA_nu_ao = contract("dpg,qg->dpq", dmu_dr, mu * weight_depsilon_drho + dmudr_weight_depsilondnablarho)
            del dmudr_weight_depsilondnablarho
            d2mudr2_weight_depsilondnablarho = contract("dxpg,xg->dpg", d2mu_dr2, weight_depsilon_dnablarho)
            dmudA_nu_ao += contract("dpg,qg->dpq", d2mudr2_weight_depsilondnablarho, mu)
            del d2mudr2_weight_depsilondnablarho

            dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += dmudA_nu_ao

            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum(
            #     "pi,g,dpg,qg,qj->dij", mo_coeff_masked, depsilon_drho * weight, dmu_dr, mu, mocc_masked)
            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum(
            #     "pi,g,dqg,pg,qj->dij", mo_coeff_masked, depsilon_drho * weight, dmu_dr, mu, mocc_masked)
            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum(
            #     "pi,xg,dxpg,qg,qj->dij", mo_coeff_masked, depsilon_dnablarho * weight, d2mu_dr2, mu, mocc_masked)
            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum(
            #     "pi,xg,dxqg,pg,qj->dij", mo_coeff_masked, depsilon_dnablarho * weight, d2mu_dr2, mu, mocc_masked)
            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum(
            #     "pi,xg,dpg,xqg,qj->dij", mo_coeff_masked, depsilon_dnablarho * weight, dmu_dr, dmu_dr, mocc_masked)
            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum(
            #     "pi,xg,dqg,xpg,qj->dij", mo_coeff_masked, depsilon_dnablarho * weight, dmu_dr, dmu_dr, mocc_masked)
            dmudA_nu_ao_ao = dmudA_nu_ao + dmudA_nu_ao.transpose(0,2,1)
            del dmudA_nu_ao
            dFock_sparse_ao_occ[i_atom_of_grids, :, :, :] += contract("dpq,qj->dpj", dmudA_nu_ao_ao, mocc_masked)
            del dmudA_nu_ao_ao

            dFock_ao_occ[:, :, idx, :] += dFock_sparse_ao_occ
            del dFock_sparse_ao_occ

            g0 = g1
        assert g1 == ngrids

    elif xctype == 'MGGA':
        g0 = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 2, strict_grid_order = True):
            g1 = g0 + weight.shape[0]

            ao = ao[:, :, nonzero_weight_mask[g0:g1]]

            if ao.size == 0:
                g0 = g1
                continue

            mu = ao[0]
            dmu_dr = ao[1:4]
            d2mu_dr2 = get_d2mu_dr2(ao)

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)
            # mo_coeff_masked = mo_coeff_sorted[idx, :]
            mocc_masked = mocc_sorted[idx, :]

            rho = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked, xctype = xctype, hermi = 1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype=xctype, spin=0)[1:3]
            del rho

            dw_dA = get_dweight_dA(_sorted_mol, grids, (g0,g1))
            dw_dA = dw_dA[:, :, nonzero_weight_mask[g0:g1]]

            # dFock_mo_occ += cupy.einsum("pi,Adg,g,pg,qg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_drho, mu, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adg,xg,xpg,qg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_dnablarho, dmu_dr, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adg,xg,xqg,pg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_dnablarho, dmu_dr, mu, mocc_masked)
            # dFock_mo_occ += 0.5 * cupy.einsum("pi,Adg,g,xpg,xqg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_dtau, dmu_dr, dmu_dr, mocc_masked)
            dwdA_vxc = contract("Adg,xg->Adxg", dw_dA, vxc)
            del dw_dA

            i_atom_of_grids = int(grids.atm_idx[g0])
            assert cupy.max(cupy.abs(grids.atm_idx[g0:g1] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)

            masked_i_atom_of_aos = i_atom_of_aos[idx]

            drho_dA_orbital_response, drho_dA_grid_response = \
                get_drho_dA_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr, d2mu_dr2)
            drho_dA_full_response = drho_dA_orbital_response + drho_dA_grid_response
            del drho_dA_orbital_response, drho_dA_grid_response

            weight = weight[nonzero_weight_mask[g0:g1]]

            fwxc = fxc * weight
            del fxc
            drhodA_fwxc = contract("xyg,Adyg->Adxg", fwxc, drho_dA_full_response)
            del drho_dA_full_response
            del fwxc

            # dFock_mo_occ += cupy.einsum("pi,Adg,pg,qg,qj->Adij", mo_coeff_masked, drhodA_fwxc[:, :, 0, :], mu, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adxg,xpg,qg,qj->Adij", mo_coeff_masked, drhodA_fwxc[:, :, 1:4, :], dmu_dr, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adxg,xqg,pg,qj->Adij", mo_coeff_masked, drhodA_fwxc[:, :, 1:4, :], dmu_dr, mu, mocc_masked)
            # dFock_mo_occ += 0.5 * cupy.einsum("pi,Adg,xpg,xqg,qj->Adij", mo_coeff_masked, drhodA_fwxc[:, :, 4, :], dmu_dr, dmu_dr, mocc_masked)
            mu_nu_term_prefactor = drhodA_fwxc + dwdA_vxc
            del drhodA_fwxc, dwdA_vxc

            dFock_sparse_ao_occ = cupy.zeros((natm, 3, mu.shape[0], nocc))

            mu_mocc = mu.T @ mocc_masked
            dmudr_mocc = contract("dqg,qj->djg", dmu_dr, mocc_masked)
            for i_atom in range(natm): # Henry 20260519: This is the most performance critical loop
                mu_on_right_term = contract("xpg,dxg->dpg", ao[0:4], mu_nu_term_prefactor[i_atom, :, 0:4, :])
                dFock_sparse_ao_occ[i_atom, :, :, :] += contract("dpg,gj->dpj", mu_on_right_term, mu_mocc)
                del mu_on_right_term
                nablarho_dmudr_on_right_term = contract("dxg,xjg->djg", mu_nu_term_prefactor[i_atom, :, 1:4, :], dmudr_mocc)
                dFock_sparse_ao_occ[i_atom, :, :, :] += contract("pg,djg->dpj", mu, nablarho_dmudr_on_right_term)
                del nablarho_dmudr_on_right_term
                tau_term = contract("dg,xjg->dxjg", mu_nu_term_prefactor[i_atom, :, 4, :], dmudr_mocc)
                dFock_sparse_ao_occ[i_atom, :, :, :] += 0.5 * contract("xpg,dxjg->dpj", dmu_dr, tau_term)
                del tau_term
            del mu_mocc, dmudr_mocc
            del mu_nu_term_prefactor

            wv = vxc * weight
            del vxc
            weight_depsilon_drho = wv[0]
            weight_depsilon_dnablarho = wv[1:4]
            weight_depsilon_dtau = wv[4]
            del wv

            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += cupy.einsum(
            #     "g,dpg,qg->dpq", depsilon_drho * weight, dmu_dr, mu)
            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += cupy.einsum(
            #     "xg,dxpg,qg->dpq", depsilon_dnablarho * weight, d2mu_dr2, mu)
            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += cupy.einsum(
            #     "xg,dpg,xqg->dpq", depsilon_dnablarho * weight, dmu_dr, dmu_dr)
            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += 0.5 * cupy.einsum(
            #     "g,dxpg,xqg->dpq", depsilon_dtau * weight, d2mu_dr2, dmu_dr)
            dmudr_weight_depsilondnablarho = contract("xg,xpg->pg", weight_depsilon_dnablarho, dmu_dr)
            dmudA_nu_ao = contract("dpg,qg->dpq", dmu_dr, mu * weight_depsilon_drho + dmudr_weight_depsilondnablarho)
            del dmudr_weight_depsilondnablarho
            mu_weight_depsilondnablarho = contract("xg,qg->xqg", weight_depsilon_dnablarho, mu)
            dmudA_nu_ao += contract("dxpg,xqg->dpq", d2mu_dr2, 0.5 * dmu_dr * weight_depsilon_dtau + mu_weight_depsilondnablarho)
            del mu_weight_depsilondnablarho

            dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += dmudA_nu_ao

            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum(
            #     "pi,g,dpg,qg,qj->dij", mo_coeff_masked, depsilon_drho * weight, dmu_dr, mu, mocc_masked)
            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum(
            #     "pi,g,dqg,pg,qj->dij", mo_coeff_masked, depsilon_drho * weight, dmu_dr, mu, mocc_masked)
            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum(
            #     "pi,xg,dxpg,qg,qj->dij", mo_coeff_masked, depsilon_dnablarho * weight, d2mu_dr2, mu, mocc_masked)
            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum(
            #     "pi,xg,dxqg,pg,qj->dij", mo_coeff_masked, depsilon_dnablarho * weight, d2mu_dr2, mu, mocc_masked)
            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum(
            #     "pi,xg,dpg,xqg,qj->dij", mo_coeff_masked, depsilon_dnablarho * weight, dmu_dr, dmu_dr, mocc_masked)
            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum(
            #     "pi,xg,dqg,xpg,qj->dij", mo_coeff_masked, depsilon_dnablarho * weight, dmu_dr, dmu_dr, mocc_masked)
            # dFock_mo_occ[i_atom_of_grids, :, :, :] += 0.5 * cupy.einsum(
            #     "pi,g,dxpg,xqg,qj->dij", mo_coeff_masked, depsilon_dtau * weight, d2mu_dr2, dmu_dr, mocc_masked)
            # dFock_mo_occ[i_atom_of_grids, :, :, :] += 0.5 * cupy.einsum(
            #     "pi,g,dxqg,xpg,qj->dij", mo_coeff_masked, depsilon_dtau * weight, d2mu_dr2, dmu_dr, mocc_masked)
            dmudA_nu_ao_ao = dmudA_nu_ao + dmudA_nu_ao.transpose(0,2,1)
            del dmudA_nu_ao
            dFock_sparse_ao_occ[i_atom_of_grids, :, :, :] += contract("dpq,qj->dpj", dmudA_nu_ao_ao, mocc_masked)
            del dmudA_nu_ao_ao

            dFock_ao_occ[:, :, idx, :] += dFock_sparse_ao_occ
            del dFock_sparse_ao_occ

            g0 = g1
        assert g1 == ngrids

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f"xctype = {xctype} not supported")

    dFock_mo_occ = contract("Adpj,pi->Adij", dFock_ao_occ, mo_coeff_sorted)
    del dFock_ao_occ
    for i_atom in range(0, natm):
        ao_of_atom_i = cupy.where(i_atom_of_aos == i_atom)[0]
        if ao_of_atom_i.size > 0:
            dFock_ao_of_atom_i = dFock_orbital_response_dmudA_nu_term[:, ao_of_atom_i, :]

            dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i @ mocc_sorted
            dFock_mo_occ[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff_sorted[ao_of_atom_i], dFock_ao_of_atom_i_mocc)
            dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i.transpose(0,2,1) @ mocc_sorted[ao_of_atom_i]
            dFock_mo_occ[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff_sorted, dFock_ao_of_atom_i_mocc)
            del dFock_ao_of_atom_i_mocc, dFock_ao_of_atom_i
    del dFock_orbital_response_dmudA_nu_term

    return dFock_mo_occ

def get_dweight_dA(mol, grids, grid_range = None):
    ngrids = grids.coords.shape[0]
    assert grids.atm_idx.shape[0] == ngrids
    assert grids.quadrature_weights.shape[0] == ngrids
    atm_coords = cupy.asarray(mol.atom_coords(), order = "F")

    inv_atom_distance = 1 / cupy.linalg.norm(atm_coords[:, None, :] - atm_coords[None, :, :], axis = 2)
    cupy.fill_diagonal(inv_atom_distance, 0)

    from gpu4pyscf.dft import radi
    if grids.radii_adjust is None:
        # a_factor = cupy.zeros([mol.natm, mol.natm])
        a_factor_ptr = lib.c_null_ptr()
    else:
        assert grids.radii_adjust == radi.treutler_atomic_radii_adjust
        a_factor = radi.get_treutler_fac(mol, grids.atomic_radii) # Please make sure this is antisymmetric
        a_factor_ptr = ctypes.cast(a_factor.data.ptr, ctypes.c_void_p)

    from gpu4pyscf.dft.gen_grid import get_C_interface_scheme_id
    scheme_id = get_C_interface_scheme_id(grids.becke_scheme)

    grids_coords = cupy.asarray(grids.coords, order = "F")
    grids_quadrature_weights = cupy.asarray(grids.quadrature_weights)
    grids_atm_idx = cupy.asarray(grids.atm_idx)
    if grid_range is not None:
        assert numpy.asarray(grid_range).shape == (2,)
        assert grid_range[1] > grid_range[0]
        ngrids = grid_range[1] - grid_range[0]
        grids_coords = cupy.asfortranarray(grids_coords[grid_range[0] : grid_range[1]])
        # The next two arrays are 1D, so slicing without copy is fine.
        grids_quadrature_weights = grids_quadrature_weights[grid_range[0] : grid_range[1]]
        grids_atm_idx = grids_atm_idx[grid_range[0] : grid_range[1]]
        assert grids_coords.shape == (ngrids, 3)
        assert grids_quadrature_weights.shape == (ngrids,)
        assert grids_atm_idx.shape == (ngrids,)

    Ar_distance = cupy.linalg.norm(atm_coords[:, None, :] - grids_coords[None, :, :], axis = 2)
    assert Ar_distance.shape == (mol.natm, ngrids)

    P_B = cupy.zeros([mol.natm, ngrids], order = "C")
    libgdft.GDFTbecke_eval_PB(
        ctypes.cast(P_B.data.ptr, ctypes.c_void_p),
        a_factor_ptr,
        ctypes.cast(inv_atom_distance.data.ptr, ctypes.c_void_p),
        ctypes.cast(Ar_distance.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(mol.natm),
        ctypes.c_int(scheme_id),
    )
    sum_P_B = cupy.sum(P_B, axis = 0)
    inv_sum_P_B = cupy.zeros(ngrids)
    nonzero_sum_P_B_location = (sum_P_B > 1e-14)
    inv_sum_P_B[nonzero_sum_P_B_location] = 1.0 / sum_P_B[nonzero_sum_P_B_location]
    nonzero_sum_P_B_location = None
    sum_P_B = None

    dweight_dA = cupy.zeros([mol.natm, 3, ngrids], order = "C")
    libgdft.GDFTbecke_partition_weight_derivative(
        ctypes.cast(dweight_dA.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_quadrature_weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(atm_coords.data.ptr, ctypes.c_void_p),
        a_factor_ptr,
        ctypes.cast(inv_atom_distance.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_atm_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(Ar_distance.data.ptr, ctypes.c_void_p),
        ctypes.cast(P_B.data.ptr, ctypes.c_void_p),
        ctypes.cast(inv_sum_P_B.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(mol.natm),
        ctypes.c_int(scheme_id),
    )
    dweight_dA[grids_atm_idx, 0, cupy.arange(ngrids)] = -cupy.sum(dweight_dA[:, 0, :], axis=[0])
    dweight_dA[grids_atm_idx, 1, cupy.arange(ngrids)] = -cupy.sum(dweight_dA[:, 1, :], axis=[0])
    dweight_dA[grids_atm_idx, 2, cupy.arange(ngrids)] = -cupy.sum(dweight_dA[:, 2, :], axis=[0])

    return dweight_dA

def get_d2weight_dAdB(mol, grids, grid_range = None):
    ngrids = grids.coords.shape[0]
    assert grids.atm_idx.shape[0] == ngrids
    assert grids.quadrature_weights.shape[0] == ngrids
    atm_coords = cupy.asarray(mol.atom_coords(), order = "F")

    inv_atom_distance = 1 / cupy.linalg.norm(atm_coords[:, None, :] - atm_coords[None, :, :], axis = 2)
    cupy.fill_diagonal(inv_atom_distance, 0)

    from gpu4pyscf.dft import radi
    if grids.radii_adjust is None:
        # a_factor = cupy.zeros([mol.natm, mol.natm])
        a_factor_ptr = lib.c_null_ptr()
    else:
        assert grids.radii_adjust == radi.treutler_atomic_radii_adjust
        a_factor = radi.get_treutler_fac(mol, grids.atomic_radii) # Please make sure this is antisymmetric
        a_factor_ptr = ctypes.cast(a_factor.data.ptr, ctypes.c_void_p)

    from gpu4pyscf.dft.gen_grid import get_C_interface_scheme_id
    scheme_id = get_C_interface_scheme_id(grids.becke_scheme)

    grids_coords = cupy.asarray(grids.coords, order = "F")
    grids_quadrature_weights = cupy.asarray(grids.quadrature_weights)
    grids_atm_idx = cupy.asarray(grids.atm_idx)
    if grid_range is not None:
        assert numpy.asarray(grid_range).shape == (2,)
        assert grid_range[1] > grid_range[0]
        ngrids = grid_range[1] - grid_range[0]
        grids_coords = cupy.asfortranarray(grids_coords[grid_range[0] : grid_range[1]])
        # The next two arrays are 1D, so slicing without copy is fine.
        grids_quadrature_weights = grids_quadrature_weights[grid_range[0] : grid_range[1]]
        grids_atm_idx = grids_atm_idx[grid_range[0] : grid_range[1]]
        assert grids_coords.shape == (ngrids, 3)
        assert grids_quadrature_weights.shape == (ngrids,)
        assert grids_atm_idx.shape == (ngrids,)

    Ar_distance = cupy.linalg.norm(atm_coords[:, None, :] - grids_coords[None, :, :], axis = 2)
    assert Ar_distance.shape == (mol.natm, ngrids)

    P_B = cupy.zeros([mol.natm, ngrids], order = "C")
    libgdft.GDFTbecke_eval_PB(
        ctypes.cast(P_B.data.ptr, ctypes.c_void_p),
        a_factor_ptr,
        ctypes.cast(inv_atom_distance.data.ptr, ctypes.c_void_p),
        ctypes.cast(Ar_distance.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(mol.natm),
        ctypes.c_int(scheme_id),
    )
    sum_P_B = cupy.sum(P_B, axis = 0)
    inv_sum_P_B = cupy.zeros(ngrids)
    nonzero_sum_P_B_location = (sum_P_B > 1e-14)
    inv_sum_P_B[nonzero_sum_P_B_location] = 1.0 / sum_P_B[nonzero_sum_P_B_location]
    nonzero_sum_P_B_location = None
    sum_P_B = None

    d2weight_dAdB = cupy.zeros([mol.natm, mol.natm, 3, 3, ngrids], order = "C")
    assert d2weight_dAdB.size < numpy.iinfo(numpy.int32).max
    libgdft.GDFTbecke_partition_weight_second_derivative(
        ctypes.cast(d2weight_dAdB.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_quadrature_weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(atm_coords.data.ptr, ctypes.c_void_p),
        a_factor_ptr,
        ctypes.cast(inv_atom_distance.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_atm_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(Ar_distance.data.ptr, ctypes.c_void_p),
        ctypes.cast(P_B.data.ptr, ctypes.c_void_p),
        ctypes.cast(inv_sum_P_B.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(mol.natm),
        ctypes.c_int(scheme_id),
    )

    range_ngrids = cupy.arange(ngrids)
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            for j_xyz in range(3):
                d2weight_dAdB[i_atom, grids_atm_idx, i_xyz, j_xyz, range_ngrids] = -cupy.sum(d2weight_dAdB[i_atom, :, i_xyz, j_xyz, :], axis=[0])

    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            for j_xyz in range(3):
                d2weight_dAdB[grids_atm_idx, i_atom, i_xyz, j_xyz, range_ngrids] = -cupy.sum(d2weight_dAdB[:, i_atom, i_xyz, j_xyz, :], axis=[0])

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

    grids = mf.nlcgrids
    if not grid_response:
        if grids.coords is None:
            grids.build()

        ni = mf._numint
        opt = getattr(ni, 'gdftopt', None)
        if opt is None:
            ni.build(mol, grids.coords)
            opt = ni.gdftopt
    else:
        grids = grids.copy()
        grids.build(sort_grids_of_each_atom = True)

        ni = numint.NumInt()
        ni.gdftopt = None
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    natm = _sorted_mol.natm
    mol = None

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    if dm0.ndim == 3:
        assert dm0.shape[0] == 2
        dm0 = dm0[0] + dm0[1]
    dm0_sorted = opt.sort_orbitals(dm0, axis=[0,1])
    dm_mask_buf = cupy.empty(nao * nao)
    dm0 = None

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
    beta = 0.03125 * (3.0 / nlc_pars[0]**2)**0.75

    ngrids_full = grids.coords.shape[0]
    rho_drho = cupy.empty([4, ngrids_full])
    g1 = 0
    for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 1, strict_grid_order = True):
        g0, g1 = g1, g1 + weight.size
        dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)
        rho_drho[:, g0:g1] = numint.eval_rho(_sorted_mol, ao, dm0_masked, xctype = "NLC", hermi = 1)
    assert g1 == ngrids_full

    rho_i = rho_drho[0,:]

    rho_nonzero_mask = cupy.logical_and(
        rho_i >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD,
        cupy.abs(grids.weights) > 1e-14,
    )

    rho_i = rho_i[rho_nonzero_mask]
    grids_coords = cupy.ascontiguousarray(grids.coords[rho_nonzero_mask, :])
    grids_weights = grids.weights[rho_nonzero_mask]
    ngrids = grids_coords.shape[0]

    nabla_rho_i = rho_drho[1:4, rho_nonzero_mask]
    gamma_i = batched_vec_norm2(nabla_rho_i.T)

    stream = cupy.cuda.get_current_stream()

    omega_i               = cupy.empty(ngrids)
    domega_drho_i         = cupy.empty(ngrids)
    domega_dgamma_i       = cupy.empty(ngrids)
    d2omega_drho2_i       = cupy.empty(ngrids)
    d2omega_dgamma2_i     = cupy.empty(ngrids)
    d2omega_drho_dgamma_i = cupy.empty(ngrids)
    libgdft.VXC_vv10nlc_hess_eval_omega_derivative(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
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
    kappa_i         = kappa_prefactor * rho_i**(1.0/6.0)
    dkappa_drho_i   = (kappa_prefactor * (1.0/6.0)) * rho_i**(-5.0/6.0)
    d2kappa_drho2_i = (kappa_prefactor * (-5.0/36.0)) * rho_i**(-11.0/6.0)

    rho_weight_i = rho_i * grids_weights
    U_i = cupy.empty(ngrids)
    W_i = cupy.empty(ngrids)
    A_i = cupy.empty(ngrids)
    B_i = cupy.empty(ngrids)
    C_i = cupy.empty(ngrids)
    E_i = cupy.empty(ngrids)

    libgdft.VXC_vv10nlc_hess_eval_UWABCE(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(U_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(W_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(B_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(C_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(E_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_weight_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids)
    )
    del rho_weight_i

    f_rho_i = beta + E_i + rho_i * (dkappa_drho_i * U_i + domega_drho_i * W_i)
    f_gamma_i = rho_i * domega_dgamma_i * W_i
    fw_rho_i   =   f_rho_i * grids_weights
    fw_gamma_i = f_gamma_i * grids_weights

    ao_loc_sorted = _sorted_mol.ao_loc
    ao_expand = ao_loc_sorted[1:] - ao_loc_sorted[:-1]
    from pyscf.gto.mole import ATOM_OF
    i_atom_of_aos = numpy.repeat(_sorted_mol._bas[:,ATOM_OF], ao_expand)
    i_atom_of_aos = cupy.asarray(i_atom_of_aos, dtype = cupy.int32)

    drho_dA   = cupy.empty([natm, 3, ngrids], order = "C")
    dgamma_dA = cupy.empty([natm, 3, ngrids], order = "C")
    dnablarho_dA = cupy.empty([natm, 3, 3, ngrids], order = "C")

    g0_full = 0
    g0_nonzero = 0
    for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv = 2, strict_grid_order = True):
        g1_full = g0_full + weight.shape[0]

        ao = ao[:, :, rho_nonzero_mask[g0_full : g1_full]]

        if ao.size == 0:
            g0_full = g1_full
            continue

        g1_nonzero = g0_nonzero + ao.shape[-1]

        mu = ao[0, :, :]
        dmu_dr = ao[1:4, :, :]
        d2mu_dr2 = get_d2mu_dr2(ao)

        split_drho_dr = nabla_rho_i[:, g0_nonzero : g1_nonzero]

        dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)
        if grid_response:
            i_atom_of_grids = int(grids.atm_idx[g0_full])
            assert cupy.max(cupy.abs(grids.atm_idx[g0_full : g1_full] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)
        else:
            i_atom_of_grids = None
        masked_i_atom_of_aos = i_atom_of_aos[idx]

        drho_dA_orbital_response, drho_dA_grid_response = \
            get_drho_dA_sparse(dm0_masked, "GGA", natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr, d2mu_dr2, with_grid_response = grid_response)
        if not grid_response:
            drho_dA_full = drho_dA_orbital_response
        else:
            drho_dA_full = drho_dA_orbital_response + drho_dA_grid_response
        del drho_dA_orbital_response, drho_dA_grid_response
        drho_dA[:, :, g0_nonzero : g1_nonzero] = drho_dA_full[:, :, 0, :]
        dnablarho_dA[:, :, :, g0_nonzero : g1_nonzero] = drho_dA_full[:, :, 1:4, :]
        dgamma_dA[:, :, g0_nonzero : g1_nonzero] = 2 * contract("Adxg,xg->Adg", drho_dA_full[:, :, 1:4, :], split_drho_dr)

        del drho_dA_full
        del split_drho_dr
        del mu, dmu_dr, d2mu_dr2

        g0_nonzero = g1_nonzero
        g0_full = g1_full
    assert g1_full == ngrids_full
    assert g1_nonzero == ngrids

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
    del drho_dA, dgamma_dA

    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[mo_coeff.ndim-2])
    nmo = mo_coeff.shape[-1]
    if mo_coeff.ndim == 3:
        mocca = mo_coeff[0][:, mo_occ[0]>0]
        moccb = mo_coeff[1][:, mo_occ[1]>0]
        nocca = mocca.shape[1]
        noccb = moccb.shape[1]
        dFock_mo_occa = cupy.zeros([natm, 3, nmo, nocca])
        dFock_mo_occb = cupy.zeros([natm, 3, nmo, noccb])
    else:
        mocc = mo_coeff[:, mo_occ>0]
        nocc = mocc.shape[1]
        dFock_mo_occ = cupy.zeros([natm, 3, nmo, nocc])

    dFock_orbital_response_dmudA_nu_term = cupy.zeros((3, nao, nao))

    g0_full = 0
    g0_nonzero = 0
    for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv = 2, strict_grid_order = True):
        g1_full = g0_full + weight.shape[0]

        ao = ao[:, :, rho_nonzero_mask[g0_full : g1_full]]

        if ao.size == 0:
            g0_full = g1_full
            continue

        g1_nonzero = g0_nonzero + ao.shape[-1]

        mu = ao[0, :, :]
        dmu_dr = ao[1:4, :, :]
        d2mu_dr2 = get_d2mu_dr2(ao)

        split_drho_dr = nabla_rho_i[:, g0_nonzero : g1_nonzero]
        split_dnablarho_dA = dnablarho_dA[:, :, :, g0_nonzero : g1_nonzero]
        split_f_rho_A_i = f_rho_A_i[:, :, g0_nonzero : g1_nonzero]
        split_f_gamma_A_i = f_gamma_A_i[:, :, g0_nonzero : g1_nonzero]

        weight = weight[rho_nonzero_mask[g0_full : g1_full]]

        # w_i \phi_{\mu i} \phi_{\nu i} f_i^{\rho, A}
        # dFock_sparse_ao_occ = cupy.einsum('Adg,pg,qg,qj->Adpj', split_f_rho_A_i, mu, mu * weight, mocc_masked)

        # w_i 2 (\nabla\rho)_i \cdot (\nabla(\phi_\mu \phi_nu))_i f_i^{\gamma, A}
        # dFock_sparse_ao_occ += 2 * cupy.einsum('Adg,xpg,qg,xg,qj->Adpj', split_f_gamma_A_i, dmu_dr, mu, split_drho_dr * weight, mocc_masked)
        # dFock_sparse_ao_occ += 2 * cupy.einsum('Adg,xqg,pg,xg,qj->Adpj', split_f_gamma_A_i, dmu_dr, mu, split_drho_dr * weight, mocc_masked)

        # w_i 2 f_i^\gamma \nabla_A \nabla\rho \cdot \nabla(\phi_\mu \phi_nu)_i
        # dFock_sparse_ao_occ += 2 * cupy.einsum('Adxg,xpg,qg,qj->Adpj', split_dnablarho_dA, dmu_dr, mu * fw_gamma_i[g0_nonzero : g1_nonzero], mocc_masked)
        # dFock_sparse_ao_occ += 2 * cupy.einsum('Adxg,xqg,pg,qj->Adpj', split_dnablarho_dA, dmu_dr, mu * fw_gamma_i[g0_nonzero : g1_nonzero], mocc_masked)

        if mo_coeff.ndim == 3:
            mocca_masked = mocca[idx]
            moccb_masked = moccb[idx]

            dFock_sparse_ao_occa = cupy.zeros((natm, 3, mu.shape[0], nocca))
            dFock_sparse_ao_occb = cupy.zeros((natm, 3, mu.shape[0], noccb))

            dmudr_drhodr = contract("xpg,xg->pg", dmu_dr, split_drho_dr)
            w_mu_mocca = weight[:,None] * (mu.T @ mocca_masked)
            w_mu_moccb = weight[:,None] * (mu.T @ moccb_masked)
            w_dmudr_mocca = contract("dpg,pj->djg", dmu_dr, mocca_masked) * weight
            w_dmudr_moccb = contract("dpg,pj->djg", dmu_dr, moccb_masked) * weight
            drhodr_w_dmudr_mocca = contract("xg,xjg->gj", split_drho_dr, w_dmudr_mocca)
            drhodr_w_dmudr_moccb = contract("xg,xjg->gj", split_drho_dr, w_dmudr_moccb)
            fwgamma_mu_occa = f_gamma_i[g0_nonzero : g1_nonzero, None] * w_mu_mocca
            fwgamma_mu_occb = f_gamma_i[g0_nonzero : g1_nonzero, None] * w_mu_moccb
            fwgamma_dmudr_occa = w_dmudr_mocca * f_gamma_i[g0_nonzero : g1_nonzero]
            fwgamma_dmudr_occb = w_dmudr_moccb * f_gamma_i[g0_nonzero : g1_nonzero]
            del w_dmudr_mocca, w_dmudr_moccb
            for i_atom in range(natm):
                frhoA_w_mu_mocc = contract("dg,gj->dgj", split_f_rho_A_i[i_atom], w_mu_mocca)
                dFock_sparse_ao_occa[i_atom] += contract("pg,dgj->dpj", mu, frhoA_w_mu_mocc)
                frhoA_w_mu_mocc = contract("dg,gj->dgj", split_f_rho_A_i[i_atom], w_mu_moccb)
                dFock_sparse_ao_occb[i_atom] += contract("pg,dgj->dpj", mu, frhoA_w_mu_mocc)
                del frhoA_w_mu_mocc
                fgammaA_w_mu_mocc = contract("dg,gj->dgj", split_f_gamma_A_i[i_atom], w_mu_mocca)
                dFock_sparse_ao_occa[i_atom] += 2 * contract("pg,dgj->dpj", dmudr_drhodr, fgammaA_w_mu_mocc)
                fgammaA_w_mu_mocc = contract("dg,gj->dgj", split_f_gamma_A_i[i_atom], w_mu_moccb)
                dFock_sparse_ao_occb[i_atom] += 2 * contract("pg,dgj->dpj", dmudr_drhodr, fgammaA_w_mu_mocc)
                del fgammaA_w_mu_mocc
                fgammaA_mu = contract("dg,pg->dpg", split_f_gamma_A_i[i_atom], mu)
                dFock_sparse_ao_occa[i_atom] += 2 * contract("dpg,gj->dpj", fgammaA_mu, drhodr_w_dmudr_mocca)
                dFock_sparse_ao_occb[i_atom] += 2 * contract("dpg,gj->dpj", fgammaA_mu, drhodr_w_dmudr_moccb)
                del fgammaA_mu
                dnablarhodA_dmudr = contract("dxg,xpg->dpg", split_dnablarho_dA[i_atom], dmu_dr)
                dFock_sparse_ao_occa[i_atom] += 2 * contract("dpg,gj->dpj", dnablarhodA_dmudr, fwgamma_mu_occa)
                dFock_sparse_ao_occb[i_atom] += 2 * contract("dpg,gj->dpj", dnablarhodA_dmudr, fwgamma_mu_occb)
                del dnablarhodA_dmudr
                dnablarhodA_fwgamma_dmudr_occ = contract("dxg,xjg->djg", split_dnablarho_dA[i_atom], fwgamma_dmudr_occa)
                dFock_sparse_ao_occa[i_atom] += 2 * contract("pg,djg->dpj", mu, dnablarhodA_fwgamma_dmudr_occ)
                dnablarhodA_fwgamma_dmudr_occ = contract("dxg,xjg->djg", split_dnablarho_dA[i_atom], fwgamma_dmudr_occb)
                dFock_sparse_ao_occb[i_atom] += 2 * contract("pg,djg->dpj", mu, dnablarhodA_fwgamma_dmudr_occ)
                del dnablarhodA_fwgamma_dmudr_occ

            del dmudr_drhodr
            del w_mu_mocca, drhodr_w_dmudr_mocca, fwgamma_mu_occa, fwgamma_dmudr_occa
            del w_mu_moccb, drhodr_w_dmudr_moccb, fwgamma_mu_occb, fwgamma_dmudr_occb
        else:
            mocc_masked = mocc[idx]

            dFock_sparse_ao_occ = cupy.zeros((natm, 3, mu.shape[0], nocc))

            dmudr_drhodr = contract("xpg,xg->pg", dmu_dr, split_drho_dr)
            w_mu_mocc = weight[:,None] * (mu.T @ mocc_masked)
            w_dmudr_mocc = contract("dpg,pj->djg", dmu_dr, mocc_masked) * weight
            drhodr_w_dmudr_mocc = contract("xg,xjg->gj", split_drho_dr, w_dmudr_mocc)
            fwgamma_mu_occ = f_gamma_i[g0_nonzero : g1_nonzero, None] * w_mu_mocc
            fwgamma_dmudr_occ = w_dmudr_mocc * f_gamma_i[g0_nonzero : g1_nonzero]
            del w_dmudr_mocc
            for i_atom in range(natm):
                frhoA_w_mu_mocc = contract("dg,gj->dgj", split_f_rho_A_i[i_atom], w_mu_mocc)
                dFock_sparse_ao_occ[i_atom] += contract("pg,dgj->dpj", mu, frhoA_w_mu_mocc)
                del frhoA_w_mu_mocc
                fgammaA_w_mu_mocc = contract("dg,gj->dgj", split_f_gamma_A_i[i_atom], w_mu_mocc)
                dFock_sparse_ao_occ[i_atom] += 2 * contract("pg,dgj->dpj", dmudr_drhodr, fgammaA_w_mu_mocc)
                del fgammaA_w_mu_mocc
                fgammaA_mu = contract("dg,pg->dpg", split_f_gamma_A_i[i_atom], mu)
                dFock_sparse_ao_occ[i_atom] += 2 * contract("dpg,gj->dpj", fgammaA_mu, drhodr_w_dmudr_mocc)
                del fgammaA_mu
                dnablarhodA_dmudr = contract("dxg,xpg->dpg", split_dnablarho_dA[i_atom], dmu_dr)
                dFock_sparse_ao_occ[i_atom] += 2 * contract("dpg,gj->dpj", dnablarhodA_dmudr, fwgamma_mu_occ)
                del dnablarhodA_dmudr
                dnablarhodA_fwgamma_dmudr_occ = contract("dxg,xjg->djg", split_dnablarho_dA[i_atom], fwgamma_dmudr_occ)
                dFock_sparse_ao_occ[i_atom] += 2 * contract("pg,djg->dpj", mu, dnablarhodA_fwgamma_dmudr_occ)
                del dnablarhodA_fwgamma_dmudr_occ

            del w_mu_mocc, drhodr_w_dmudr_mocc, dmudr_drhodr, fwgamma_mu_occ, fwgamma_dmudr_occ

        # w_i f_i^\rho \nabla_A (\phi_\mu \phi_nu)_i
        # dmudA_nu_ao = cupy.einsum('dpg,qg->dpq', dmu_dr, mu * fw_rho_i[g0_nonzero : g1_nonzero])

        # w_i 2 f_i^\gamma \nabla\rho \cdot \nabla_A \nabla(\phi_\mu \phi_nu)_i
        # dmudA_nu_ao += 2 * cupy.einsum('dxpg,qg,xg->dpq', d2mu_dr2, mu, split_drho_dr * fw_gamma_i[g0_nonzero : g1_nonzero])
        # dmudA_nu_ao += 2 * cupy.einsum('dpg,xqg,xg->dpq', dmu_dr, dmu_dr, split_drho_dr * fw_gamma_i[g0_nonzero : g1_nonzero])
        mu_prefactor = dmu_dr * fw_rho_i[g0_nonzero : g1_nonzero]
        mu_prefactor += 2 * contract("dxpg,xg->dpg", d2mu_dr2, split_drho_dr * fw_gamma_i[g0_nonzero : g1_nonzero])
        dmudA_nu_ao = mu_prefactor @ mu.T
        dmudr_prefactor = contract("xpg,xg->pg", dmu_dr, split_drho_dr * fw_gamma_i[g0_nonzero : g1_nonzero])
        dmudA_nu_ao += 2 * dmu_dr @ dmudr_prefactor.T

        del mu, dmu_dr, d2mu_dr2

        dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += dmudA_nu_ao

        if grid_response:
            i_atom_of_grids = int(grids.atm_idx[g0_full])
            assert cupy.max(cupy.abs(grids.atm_idx[g0_full : g1_full] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)

            dmudA_nu_ao_ao = dmudA_nu_ao + dmudA_nu_ao.transpose(0,2,1)
            if mo_coeff.ndim == 3:
                dFock_sparse_ao_occa[i_atom_of_grids, :, :, :] += contract("dpq,qj->dpj", dmudA_nu_ao_ao, mocca_masked)
                dFock_sparse_ao_occb[i_atom_of_grids, :, :, :] += contract("dpq,qj->dpj", dmudA_nu_ao_ao, moccb_masked)
            else:
                dFock_sparse_ao_occ[i_atom_of_grids, :, :, :] += contract("dpq,qj->dpj", dmudA_nu_ao_ao, mocc_masked)
            del dmudA_nu_ao_ao
        del dmudA_nu_ao

        if mo_coeff.ndim == 3:
            dFock_mo_occa += cupy.einsum("pi,Adpj->Adij", mo_coeff[0, idx, :], dFock_sparse_ao_occa)
            dFock_mo_occb += cupy.einsum("pi,Adpj->Adij", mo_coeff[1, idx, :], dFock_sparse_ao_occb)
            del dFock_sparse_ao_occa, dFock_sparse_ao_occb
        else:
            dFock_mo_occ += cupy.einsum("pi,Adpj->Adij", mo_coeff[idx, :], dFock_sparse_ao_occ)
            del dFock_sparse_ao_occ

        g0_nonzero = g1_nonzero
        g0_full = g1_full
    assert g1_full == ngrids_full
    assert g1_nonzero == ngrids

    for i_atom in range(0, natm):
        ao_of_atom_i = cupy.where(i_atom_of_aos == i_atom)[0]
        if ao_of_atom_i.size > 0:
            dFock_ao_of_atom_i = dFock_orbital_response_dmudA_nu_term[:, ao_of_atom_i, :]

            if mo_coeff.ndim == 3:
                dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i @ mocca
                dFock_mo_occa[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff[0, ao_of_atom_i, :], dFock_ao_of_atom_i_mocc)
                dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i @ moccb
                dFock_mo_occb[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff[1, ao_of_atom_i, :], dFock_ao_of_atom_i_mocc)
                dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i.transpose(0,2,1) @ mocca[ao_of_atom_i]
                dFock_mo_occa[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff[0], dFock_ao_of_atom_i_mocc)
                dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i.transpose(0,2,1) @ moccb[ao_of_atom_i]
                dFock_mo_occb[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff[1], dFock_ao_of_atom_i_mocc)
            else:
                dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i @ mocc
                dFock_mo_occ[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff[ao_of_atom_i], dFock_ao_of_atom_i_mocc)
                dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i.transpose(0,2,1) @ mocc[ao_of_atom_i]
                dFock_mo_occ[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff, dFock_ao_of_atom_i_mocc)
            del dFock_ao_of_atom_i_mocc, dFock_ao_of_atom_i
    del dFock_orbital_response_dmudA_nu_term

    if grid_response:
        assert grids.atm_idx.shape[0] == grids.coords.shape[0]
        grid_to_atom_index_map = grids.atm_idx[rho_nonzero_mask]

        grid_offsets_of_atom = cupy.r_[0, cupy.flatnonzero(cupy.diff(grid_to_atom_index_map)) + 1]
        if grid_to_atom_index_map[-1] < 0:
            pass # There's padded grids whose index < 0, and the first index of padded grids is the number of valid grids
        else:
            grid_offsets_of_atom = cupy.append(grid_offsets_of_atom, grid_to_atom_index_map.shape[0])
        grid_offsets_of_atom = cupy.asarray(grid_offsets_of_atom, dtype = cupy.int32)

        assert grid_offsets_of_atom.shape == (natm + 1,)
        for i_atom in range(natm):
            assert cupy.all(grid_to_atom_index_map[grid_offsets_of_atom[i_atom] : grid_offsets_of_atom[i_atom + 1]] == i_atom)
        assert cupy.all(grid_to_atom_index_map[grid_offsets_of_atom[natm] : ] < 0)

        rho_weight_i = rho_i * grids_weights
        E_Bgr_i = cupy.empty([natm, 3, ngrids], order = "C")
        U_Bgr_i = cupy.empty([natm, 3, ngrids], order = "C")
        W_Bgr_i = cupy.empty([natm, 3, ngrids], order = "C")
        libgdft.VXC_vv10nlc_hess_eval_EUW_grid_response_offdiagonal(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(E_Bgr_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(U_Bgr_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(W_Bgr_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
            ctypes.cast(rho_weight_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grid_to_atom_index_map.data.ptr, ctypes.c_void_p),
            ctypes.cast(grid_offsets_of_atom.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(natm),
        )
        del rho_weight_i
        for i_atom in range(natm):
            g0, g1 = grid_offsets_of_atom[i_atom], grid_offsets_of_atom[i_atom + 1]

            E_Bgr_i[i_atom, :, g0:g1] = -cupy.sum(E_Bgr_i[:, :, g0:g1], axis = 0)
            U_Bgr_i[i_atom, :, g0:g1] = -cupy.sum(U_Bgr_i[:, :, g0:g1], axis = 0)
            W_Bgr_i[i_atom, :, g0:g1] = -cupy.sum(W_Bgr_i[:, :, g0:g1], axis = 0)

        grids_weights_1 = get_dweight_dA(_sorted_mol, grids)
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
        del E_Bw_i, U_Bw_i, W_Bw_i
        del E_Bgr_i, U_Bgr_i, W_Bgr_i

        g0_full = 0
        g0_nonzero = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv = 1, strict_grid_order = True):
            g1_full = g0_full + weight.shape[0]

            ao = ao[:, :, rho_nonzero_mask[g0_full : g1_full]]

            if ao.size == 0:
                g0_full = g1_full
                continue

            g1_nonzero = g0_nonzero + ao.shape[-1]

            mu = ao[0, :, :]
            dmu_dr = ao[1:4, :, :]

            split_drho_dr = nabla_rho_i[:, g0_nonzero : g1_nonzero]
            split_dwdA = grids_weights_1[:, :, g0_nonzero : g1_nonzero]
            split_f_rho_grid_response_i = f_rho_grid_response_i[:, :, g0_nonzero : g1_nonzero]
            split_f_gamma_grid_response_i = f_gamma_grid_response_i[:, :, g0_nonzero : g1_nonzero]

            weight = weight[rho_nonzero_mask[g0_full : g1_full]]

            if mo_coeff.ndim == 3:
                mocca_masked = mocca[idx]
                moccb_masked = moccb[idx]

                dFock_sparse_ao_occa = cupy.zeros((natm, 3, mu.shape[0], nocca))
                dFock_sparse_ao_occb = cupy.zeros((natm, 3, mu.shape[0], noccb))
            else:
                mocc_masked = mocc[idx]

                dFock_sparse_ao_occ = cupy.zeros((natm, 3, mu.shape[0], nocc))

            # # \nabla_A w_i term
            # dFock_sparse_ao_occ += cupy.einsum('Adg,pg,qg,qj->Adpj', split_dwdA, mu, mu * f_rho_i[g0_nonzero : g1_nonzero], mocc_masked)
            # dFock_sparse_ao_occ += 2 * cupy.einsum(
            #     'Adg,xpg,qg,xg,qj->Adpj', split_dwdA, dmu_dr, mu, split_drho_dr * f_gamma_i[g0_nonzero : g1_nonzero], mocc_masked)
            # dFock_sparse_ao_occ += 2 * cupy.einsum(
            #     'Adg,xqg,pg,xg,qj->Adpj', split_dwdA, dmu_dr, mu, split_drho_dr * f_gamma_i[g0_nonzero : g1_nonzero], mocc_masked)

            # # E_i^{Aw} and E_i^{Agr} terms combined
            # dFock_sparse_ao_occ += cupy.einsum('Adg,pg,qg,qj->Adpj', split_f_rho_grid_response_i, mu, mu * weight, mocc_masked)
            # dFock_sparse_ao_occ += 2 * cupy.einsum('Adg,xpg,qg,xg,qj->Adpj', split_f_gamma_grid_response_i, dmu_dr, mu, split_drho_dr * weight, mocc_masked)
            # dFock_sparse_ao_occ += 2 * cupy.einsum('Adg,xqg,pg,xg,qj->Adpj', split_f_gamma_grid_response_i, dmu_dr, mu, split_drho_dr * weight, mocc_masked)
            rho_prefactor = split_dwdA * f_rho_i[g0_nonzero : g1_nonzero] + split_f_rho_grid_response_i * weight
            gamma_prefactor = split_dwdA * f_gamma_i[g0_nonzero : g1_nonzero] + split_f_gamma_grid_response_i * weight

            if mo_coeff.ndim == 3:
                dmudr_drhodr = contract("xpg,xg->pg", dmu_dr, split_drho_dr)
                mu_mocca = mu.T @ mocca_masked
                mu_moccb = mu.T @ moccb_masked
                dmudr_mocca = contract("dpg,pj->djg", dmu_dr, mocca_masked)
                dmudr_moccb = contract("dpg,pj->djg", dmu_dr, moccb_masked)
                dmudr_drhodr_mocca = contract("xjg,xg->gj", dmudr_mocca, split_drho_dr)
                dmudr_drhodr_moccb = contract("xjg,xg->gj", dmudr_moccb, split_drho_dr)
                for i_atom in range(natm):
                    rho_prefactor_mu = contract("dg,pg->dpg", rho_prefactor[i_atom], mu)
                    dFock_sparse_ao_occa[i_atom] += contract("dpg,gj->dpj", rho_prefactor_mu, mu_mocca)
                    dFock_sparse_ao_occb[i_atom] += contract("dpg,gj->dpj", rho_prefactor_mu, mu_moccb)
                    del rho_prefactor_mu
                    gamma_prefactor_mu = contract("dg,pg->dpg", gamma_prefactor[i_atom], mu)
                    dFock_sparse_ao_occa[i_atom] += 2 * contract("dpg,gj->dpj", gamma_prefactor_mu, dmudr_drhodr_mocca)
                    dFock_sparse_ao_occb[i_atom] += 2 * contract("dpg,gj->dpj", gamma_prefactor_mu, dmudr_drhodr_moccb)
                    del gamma_prefactor_mu
                    gamma_prefactor_dmudr_drhodr = contract("dg,pg->dpg", gamma_prefactor[i_atom], dmudr_drhodr)
                    dFock_sparse_ao_occa[i_atom] += 2 * contract("dpg,gj->dpj", gamma_prefactor_dmudr_drhodr, mu_mocca)
                    dFock_sparse_ao_occb[i_atom] += 2 * contract("dpg,gj->dpj", gamma_prefactor_dmudr_drhodr, mu_moccb)
                    del gamma_prefactor_dmudr_drhodr

                del dmudr_drhodr
                del mu_mocca, dmudr_mocca, dmudr_drhodr_mocca
                del mu_moccb, dmudr_moccb, dmudr_drhodr_moccb
            else:
                dmudr_drhodr = contract("xpg,xg->pg", dmu_dr, split_drho_dr)
                mu_mocc = mu.T @ mocc_masked
                dmudr_mocc = contract("dpg,pj->djg", dmu_dr, mocc_masked)
                dmudr_drhodr_mocc = contract("xjg,xg->gj", dmudr_mocc, split_drho_dr)
                for i_atom in range(natm):
                    rho_prefactor_mu = contract("dg,pg->dpg", rho_prefactor[i_atom], mu)
                    dFock_sparse_ao_occ[i_atom] += contract("dpg,gj->dpj", rho_prefactor_mu, mu_mocc)
                    del rho_prefactor_mu
                    gamma_prefactor_mu = contract("dg,pg->dpg", gamma_prefactor[i_atom], mu)
                    dFock_sparse_ao_occ[i_atom] += 2 * contract("dpg,gj->dpj", gamma_prefactor_mu, dmudr_drhodr_mocc)
                    del gamma_prefactor_mu
                    gamma_prefactor_dmudr_drhodr = contract("dg,pg->dpg", gamma_prefactor[i_atom], dmudr_drhodr)
                    dFock_sparse_ao_occ[i_atom] += 2 * contract("dpg,gj->dpj", gamma_prefactor_dmudr_drhodr, mu_mocc)
                    del gamma_prefactor_dmudr_drhodr

                del mu_mocc, dmudr_mocc, dmudr_drhodr, dmudr_drhodr_mocc

            if mo_coeff.ndim == 3:
                dFock_mo_occa += cupy.einsum("pi,Adpj->Adij", mo_coeff[0, idx, :], dFock_sparse_ao_occa)
                dFock_mo_occb += cupy.einsum("pi,Adpj->Adij", mo_coeff[1, idx, :], dFock_sparse_ao_occb)
                del dFock_sparse_ao_occa, dFock_sparse_ao_occb
            else:
                dFock_mo_occ += cupy.einsum("pi,Adpj->Adij", mo_coeff[idx, :], dFock_sparse_ao_occ)
                del dFock_sparse_ao_occ

            g0_nonzero = g1_nonzero
            g0_full = g1_full
        assert g1_full == ngrids_full
        assert g1_nonzero == ngrids

    if mo_coeff.ndim == 3:
        return (dFock_mo_occa, dFock_mo_occb)
    else:
        return dFock_mo_occ

def get_drho_dA_full(dm0, xctype, natm, ngrids, aoslices = None, atom_to_grid_index_map = None,
                     mu = None, dmu_dr = None, d2mu_dr2 = None,
                     with_orbital_response = True, with_grid_response = True):
    if xctype == "LDA":
        with_nablarho = False
        with_tau = False
    elif xctype == "GGA":
        with_nablarho = True
        with_tau = False
    elif xctype == "MGGA":
        with_nablarho = True
        with_tau = True
    else:
        raise NotImplementedError(f"Unrecognized xctype = {xctype}")

    nao = dm0.shape[-1]
    assert mu is not None and mu.shape == (nao, ngrids)
    assert dmu_dr is not None and dmu_dr.shape == (3, nao, ngrids)
    if with_nablarho or with_tau:
        assert d2mu_dr2 is not None and d2mu_dr2.shape == (3, 3, nao, ngrids)

    if with_orbital_response:
        assert aoslices is not None and len(aoslices) == natm
    if with_grid_response:
        assert atom_to_grid_index_map is not None and len(atom_to_grid_index_map) == natm

    dm_dmT = dm0 + dm0.T
    dm_dot_mu_and_nu = dm_dmT @ mu
    if with_nablarho:
        dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr, dm_dmT)

    drho_dA_orbital_response = None
    dnablarho_dA_orbital_response = None
    dtau_dA_orbital_response = None

    if with_orbital_response:
        drho_dA_orbital_response = cupy.zeros([natm, 3, ngrids])
        if with_nablarho:
            dnablarho_dA_orbital_response = cupy.zeros([natm, 3, 3, ngrids]) # The last 3 is the nabla direction
        if with_tau:
            dtau_dA_orbital_response = cupy.zeros([natm, 3, ngrids])

        for i_atom in range(natm):
            p0, p1 = aoslices[i_atom][2:]
            dm_dot_mu_and_nu_i = dm_dot_mu_and_nu[p0:p1, :]
            drho_dA_orbital_response[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], dm_dot_mu_and_nu_i)

            if with_nablarho:
                dnablarho_dA_orbital_response[i_atom, :, :] += contract('dDig,ig->dDg', -d2mu_dr2[:, :, p0:p1, :], dm_dot_mu_and_nu_i)
                dm_dot_dmu_and_dnu_i = dm_dot_dmu_and_dnu[:, p0:p1, :]
                dnablarho_dA_orbital_response[i_atom, :, :] += contract('dig,Dig->dDg', -dmu_dr[:, p0:p1, :], dm_dot_dmu_and_dnu_i)
            dm_dot_mu_and_nu_i = None

            if with_tau:
                dtau_dA_orbital_response[i_atom, :, :] += 0.5 * contract('dDig,Dig->dg', -d2mu_dr2[:, :, p0:p1, :], dm_dot_dmu_and_dnu_i)
            dm_dot_dmu_and_dnu_i = None

    drho_dA_grid_response = None
    dnablarho_dA_grid_response = None
    dtau_dA_grid_response = None

    if with_grid_response:
        drho_dA_grid_response = cupy.zeros([natm, 3, ngrids])
        if with_nablarho:
            dnablarho_dA_grid_response = cupy.zeros([natm, 3, 3, ngrids]) # The last 3 is the nabla direction
        if with_tau:
            dtau_dA_grid_response = cupy.zeros([natm, 3, ngrids])

        for i_atom in range(natm):
            associated_grid_index = atom_to_grid_index_map[i_atom]
            if len(associated_grid_index) == 0:
                continue
            dmu_dr_grid_i = dmu_dr[:, :, associated_grid_index]

            dm_dot_mu_and_nu_i = dm_dot_mu_and_nu[:, associated_grid_index]
            rho_response = contract('dig,ig->dg', dmu_dr_grid_i, dm_dot_mu_and_nu_i)
            drho_dA_grid_response[i_atom][:, associated_grid_index] = rho_response
            rho_response = None

            if with_nablarho:
                d2mu_dr2_grid_i = d2mu_dr2[:, :, :, associated_grid_index]

                nablarho_response = contract('dDig,ig->dDg', d2mu_dr2_grid_i, dm_dot_mu_and_nu_i)
                dm_dot_dmu_and_dnu_i = dm_dot_dmu_and_dnu[:, :, associated_grid_index]
                nablarho_response += contract('dig,Dig->dDg', dmu_dr_grid_i, dm_dot_dmu_and_dnu_i)
                dnablarho_dA_grid_response[i_atom][:, :, associated_grid_index] = nablarho_response
                nablarho_response = None
            dm_dot_mu_and_nu_i = None
            dmu_dr_grid_i = None

            if with_tau:
                tau_reponse = 0.5 * contract('dDig,Dig->dg', d2mu_dr2_grid_i, dm_dot_dmu_and_dnu_i)
                dtau_dA_grid_response[i_atom][:, associated_grid_index] = tau_reponse
                tau_reponse = None
            dm_dot_dmu_and_dnu_i = None
            d2mu_dr2_grid_i = None

    if (not with_nablarho) and (not with_tau):
        return drho_dA_orbital_response, drho_dA_grid_response
    elif not with_tau:
        return drho_dA_orbital_response, dnablarho_dA_orbital_response, drho_dA_grid_response, dnablarho_dA_grid_response
    else:
        return drho_dA_orbital_response, dnablarho_dA_orbital_response, dtau_dA_orbital_response, \
               drho_dA_grid_response, dnablarho_dA_grid_response, dtau_dA_grid_response

def get_drho_dA_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos = None, i_atom_of_grids = None,
                       mu = None, dmu_dr = None, d2mu_dr2 = None,
                       with_orbital_response = True, with_grid_response = True):
    if xctype == "LDA":
        with_nablarho = False
        with_tau = False
        n_component = 1
    elif xctype == "GGA":
        with_nablarho = True
        with_tau = False
        n_component = 4
    elif xctype == "MGGA":
        with_nablarho = True
        with_tau = True
        n_component = 5
    else:
        raise NotImplementedError(f"Unrecognized xctype = {xctype}")

    nao = dm0_masked.shape[-1]
    assert mu is not None
    ngrids = mu.shape[1]
    assert mu.shape == (nao, ngrids)
    assert dmu_dr is not None and dmu_dr.shape == (3, nao, ngrids)
    if with_nablarho or with_tau:
        assert d2mu_dr2 is not None and d2mu_dr2.shape == (3, 3, nao, ngrids)

    if with_orbital_response:
        assert masked_i_atom_of_aos is not None
        masked_i_atom_of_aos = cupy.asarray(masked_i_atom_of_aos, dtype = numpy.int32)
        assert masked_i_atom_of_aos.shape == (nao,)
    if with_grid_response:
        assert i_atom_of_grids is not None
        i_atom_of_grids = int(i_atom_of_grids)
        assert 0 <= i_atom_of_grids and i_atom_of_grids < natm

    dm_dmT = dm0_masked + dm0_masked.T
    dm_dot_mu_and_nu = dm_dmT @ mu
    if with_nablarho:
        dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr, dm_dmT)

    assert with_orbital_response or with_grid_response, "Why are you calling this function?"

    drho_dA_orbital_response = None
    if with_orbital_response:
        drho_dA_orbital_response = cupy.zeros([natm, 3, n_component, ngrids])
    drho_dA_grid_response = None
    if with_grid_response:
        drho_dA_grid_response = cupy.zeros([natm, 3, n_component, ngrids])

    rho_response = dmu_dr * dm_dot_mu_and_nu[None, :, :]
    if with_orbital_response:
        cupy.add.at(drho_dA_orbital_response[:, :, 0, :], masked_i_atom_of_aos, rho_response.transpose(1,0,2))
    if with_grid_response:
        rho_response = cupy.einsum('dig->dg', rho_response)
        drho_dA_grid_response[i_atom_of_grids, :, 0, :] = rho_response
    del rho_response

    if with_nablarho:
        nablarho_response = d2mu_dr2 * dm_dot_mu_and_nu[None, None, :, :]
        nablarho_response += contract('dig,Dig->dDig', dmu_dr, dm_dot_dmu_and_dnu)
        if with_orbital_response:
            cupy.add.at(drho_dA_orbital_response[:, :, 1:4, :], masked_i_atom_of_aos, nablarho_response.transpose(2,0,1,3))
        if with_grid_response:
            nablarho_response = cupy.einsum('dDig->dDg', nablarho_response)
            drho_dA_grid_response[i_atom_of_grids, :, 1:4, :] = nablarho_response
        del nablarho_response

    if with_tau:
        tau_response = 0.5 * contract('dDig,Dig->dig', d2mu_dr2, dm_dot_dmu_and_dnu)
        if with_orbital_response:
            cupy.add.at(drho_dA_orbital_response[:, :, 4, :], masked_i_atom_of_aos, tau_response.transpose(1,0,2))
        if with_grid_response:
            tau_response = cupy.einsum('dig->dg', tau_response)
            drho_dA_grid_response[i_atom_of_grids, :, 4, :] = tau_response
        del tau_response

    drho_dA_orbital_response *= -1

    if xctype == "LDA":
        if drho_dA_orbital_response is not None:
            drho_dA_orbital_response = drho_dA_orbital_response[:, :, 0, :]
        if drho_dA_grid_response is not None:
            drho_dA_grid_response = drho_dA_grid_response[:, :, 0, :]
    return drho_dA_orbital_response, drho_dA_grid_response

def get_d2rho_dAdB_full(dm0, xctype, natm, ngrids, aoslices = None, atom_to_grid_index_map = None,
                        mu = None, dmu_dr = None, d2mu_dr2 = None, d3mu_dr3 = None,
                        with_orbital_response = True, with_grid_response = True):
    """
        This function should never be used in practice. It requires crazy amount of memory,
        and it's left for debug purpose only. Use the contract function instead.
    """
    if xctype == "LDA":
        with_nablarho = False
        with_tau = False
    elif xctype == "GGA":
        with_nablarho = True
        with_tau = False
    elif xctype == "MGGA":
        with_nablarho = True
        with_tau = True
    else:
        raise NotImplementedError(f"Unrecognized xctype = {xctype}")

    nao = dm0.shape[-1]
    assert mu is not None and mu.shape == (nao, ngrids)
    assert dmu_dr is not None and dmu_dr.shape == (3, nao, ngrids)
    assert d2mu_dr2 is not None and d2mu_dr2.shape == (3, 3, nao, ngrids)
    if with_nablarho or with_tau:
        assert d3mu_dr3 is not None and d3mu_dr3.shape == (3, 3, 3, nao, ngrids)

    if with_orbital_response or with_grid_response: # There are cross terms in grid response
        assert aoslices is not None and len(aoslices) == natm
    if with_grid_response:
        assert atom_to_grid_index_map is not None and len(atom_to_grid_index_map) == natm

    d2rho_dAdB_orbital_response = None
    d2rho_dAdB_grid_response = None

    if with_orbital_response:
        d2rho_dAdB_orbital_response = cupy.zeros([natm, natm, 3, 3, ngrids])
        for i_atom in range(natm):
            pi0, pi1 = aoslices[i_atom][2:]
            # d2mu/dr2 * nu, A orbital, B orbital
            d2rho_dAdB_orbital_response[i_atom, i_atom, :, :, :] += cupy.einsum('dDig,jg,ij->dDg',
                d2mu_dr2[:, :, pi0:pi1, :], mu, dm0[pi0:pi1, :] + dm0[:, pi0:pi1].T)

            for j_atom in range(natm):
                pj0, pj1 = aoslices[j_atom][2:]
                # dmu/dr * dnu/dr, A orbital, B orbital
                d2rho_dAdB_orbital_response[i_atom, j_atom, :, :, :] += cupy.einsum('dig,Djg,ij->dDg',
                    dmu_dr[:, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], dm0[pi0:pi1, pj0:pj1] + dm0[pj0:pj1, pi0:pi1].T)

    if with_grid_response:
        d2rho_dAdB_grid_response = cupy.zeros([natm, natm, 3, 3, ngrids])
        for i_atom in range(natm):
            g_i_with_response = atom_to_grid_index_map[i_atom]
            if len(g_i_with_response) > 0:
                # d2mu/dr2 * nu, A grid, B grid
                d2rho_dAdB_grid_response[i_atom, i_atom][:, :, g_i_with_response] += cupy.einsum('dDig,jg,ij->dDg',
                    d2mu_dr2[:, :, :, g_i_with_response], mu[:, g_i_with_response], dm0 + dm0.T)
                # dmu/dr * dnu/dr, A grid, B grid
                d2rho_dAdB_grid_response[i_atom, i_atom][:, :, g_i_with_response] += cupy.einsum('dig,Djg,ij->dDg',
                    dmu_dr[:, :, g_i_with_response], dmu_dr[:, :, g_i_with_response], dm0 + dm0.T)

            for j_atom in range(natm):
                pj0, pj1 = aoslices[j_atom][2:]
                if len(g_i_with_response) > 0:
                    # d2mu/dr2 * nu, A orbital, B grid
                    # Why is there a transpose for this equation? Because in the einsum, j index is written as i for simplicity. Same for many later cases.
                    d2rho_dAdB_grid_response[i_atom, j_atom][:, :, g_i_with_response] -= cupy.einsum('dDig,jg,ij->dDg',
                        d2mu_dr2[:, :, pj0:pj1, g_i_with_response], mu[:, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T).transpose(1,0,2)
                    d2rho_dAdB_grid_response[j_atom, i_atom][:, :, g_i_with_response] -= cupy.einsum('dDig,jg,ij->dDg',
                        d2mu_dr2[:, :, pj0:pj1, g_i_with_response], mu[:, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T)

                    # dmu/dr * dnu/dr, A orbital, B grid
                    d2rho_dAdB_grid_response[i_atom, j_atom][:, :, g_i_with_response] -= cupy.einsum('dig,Djg,ij->dDg',
                        dmu_dr[:, pj0:pj1, g_i_with_response], dmu_dr[:, :, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T).transpose(1,0,2)
                    d2rho_dAdB_grid_response[j_atom, i_atom][:, :, g_i_with_response] -= cupy.einsum('dig,Djg,ij->dDg',
                        dmu_dr[:, pj0:pj1, g_i_with_response], dmu_dr[:, :, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T)

    d2nablarho_dAdB_orbital_response = None
    d2nablarho_dAdB_grid_response = None

    if with_orbital_response and with_nablarho:
        d2nablarho_dAdB_orbital_response = cupy.zeros([natm, natm, 3, 3, 3, ngrids]) # The last 3 is the nabla direction
        for i_atom in range(natm):
            pi0, pi1 = aoslices[i_atom][2:]
            # d3mu/(dr dA dB) * nu, A orbital, B orbital
            d2nablarho_dAdB_orbital_response[i_atom, i_atom, :, :, :, :] += cupy.einsum('dDxig,jg,ij->dDxg',
                d3mu_dr3[:, :, :, pi0:pi1, :], mu, dm0[pi0:pi1, :] + dm0[:, pi0:pi1].T)
            # d2mu/(dA dB) * dnu/dr, A orbital, B orbital
            d2nablarho_dAdB_orbital_response[i_atom, i_atom, :, :, :, :] += cupy.einsum('dDig,xjg,ij->dDxg',
                d2mu_dr2[:, :, pi0:pi1, :], dmu_dr, dm0[pi0:pi1, :] + dm0[:, pi0:pi1].T)

            for j_atom in range(natm):
                pj0, pj1 = aoslices[j_atom][2:]
                # d2mu/(dr dA) * dnu/dB, A orbital, B orbital
                d2nablarho_dAdB_orbital_response[i_atom, j_atom, :, :, :, :] += cupy.einsum('dxig,Djg,ij->dDxg',
                    d2mu_dr2[:, :, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], dm0[pi0:pi1, pj0:pj1] + dm0[pj0:pj1, pi0:pi1].T)
                d2nablarho_dAdB_orbital_response[j_atom, i_atom, :, :, :, :] += cupy.einsum('dxig,Djg,ij->dDxg',
                    d2mu_dr2[:, :, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], dm0[pi0:pi1, pj0:pj1] + dm0[pj0:pj1, pi0:pi1].T).transpose(1,0,2,3)

    if with_grid_response and with_nablarho:
        d2nablarho_dAdB_grid_response = cupy.zeros([natm, natm, 3, 3, 3, ngrids]) # The last 3 is the nabla direction
        for i_atom in range(natm):
            g_i_with_response = atom_to_grid_index_map[i_atom]
            if len(g_i_with_response) > 0:
                # d3mu/(dr dA dB) * nu, A grid, B grid
                d2nablarho_dAdB_grid_response[i_atom, i_atom][:, :, :, g_i_with_response] += cupy.einsum('dDxig,jg,ij->dDxg',
                    d3mu_dr3[:, :, :, :, g_i_with_response], mu[:, g_i_with_response], dm0 + dm0.T)
                # d2mu/(dA dB) * dnu/dr, A grid, B grid
                d2nablarho_dAdB_grid_response[i_atom, i_atom][:, :, :, g_i_with_response] += cupy.einsum('dDig,xjg,ij->dDxg',
                    d2mu_dr2[:, :, :, g_i_with_response], dmu_dr[:, :, g_i_with_response], dm0 + dm0.T)

                # d2mu/(dr dA) * dnu/dB, A grid, B grid
                d2nablarho_dAdB_grid_response[i_atom, i_atom][:, :, :, g_i_with_response] += cupy.einsum('dxig,Djg,ij->dDxg',
                    d2mu_dr2[:, :, :, g_i_with_response], dmu_dr[:, :, g_i_with_response], dm0 + dm0.T)
                d2nablarho_dAdB_grid_response[i_atom, i_atom][:, :, :, g_i_with_response] += cupy.einsum('dxig,Djg,ij->dDxg',
                    d2mu_dr2[:, :, :, g_i_with_response], dmu_dr[:, :, g_i_with_response], dm0 + dm0.T).transpose(1,0,2,3)

            for j_atom in range(natm):
                pj0, pj1 = aoslices[j_atom][2:]
                if len(g_i_with_response) > 0:
                    # d3mu/(dr dA dB) * nu, A orbital, B grid
                    d2nablarho_dAdB_grid_response[i_atom, j_atom][:, :, :, g_i_with_response] -= cupy.einsum('dDxig,jg,ij->dDxg',
                        d3mu_dr3[:, :, :, pj0:pj1, g_i_with_response], mu[:, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T).transpose(1,0,2,3)
                    d2nablarho_dAdB_grid_response[j_atom, i_atom][:, :, :, g_i_with_response] -= cupy.einsum('dDxig,jg,ij->dDxg',
                        d3mu_dr3[:, :, :, pj0:pj1, g_i_with_response], mu[:, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T)

                    # d2mu/(dA dB) * dnu/dr, A orbital, B grid
                    d2nablarho_dAdB_grid_response[i_atom, j_atom][:, :, :, g_i_with_response] -= cupy.einsum('dDig,xjg,ij->dDxg',
                        d2mu_dr2[:, :, pj0:pj1, g_i_with_response], dmu_dr[:, :, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T).transpose(1,0,2,3)
                    d2nablarho_dAdB_grid_response[j_atom, i_atom][:, :, :, g_i_with_response] -= cupy.einsum('dDig,xjg,ij->dDxg',
                        d2mu_dr2[:, :, pj0:pj1, g_i_with_response], dmu_dr[:, :, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T)

                    # d2mu/(dr dA) * dnu/dB, A orbital, B grid
                    d2nablarho_dAdB_grid_response[i_atom, j_atom][:, :, :, g_i_with_response] -= cupy.einsum('dxig,Djg,ij->dDxg',
                        d2mu_dr2[:, :, pj0:pj1, g_i_with_response], dmu_dr[:, :, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T).transpose(1,0,2,3)
                    d2nablarho_dAdB_grid_response[j_atom, i_atom][:, :, :, g_i_with_response] -= cupy.einsum('dxig,Djg,ij->dDxg',
                        d2mu_dr2[:, :, pj0:pj1, g_i_with_response], dmu_dr[:, :, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T)
                    d2nablarho_dAdB_grid_response[i_atom, j_atom][:, :, :, g_i_with_response] -= cupy.einsum('dig,Dxjg,ij->dDxg',
                        dmu_dr[:, pj0:pj1, g_i_with_response], d2mu_dr2[:, :, :, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T).transpose(1,0,2,3)
                    d2nablarho_dAdB_grid_response[j_atom, i_atom][:, :, :, g_i_with_response] -= cupy.einsum('dig,Dxjg,ij->dDxg',
                        dmu_dr[:, pj0:pj1, g_i_with_response], d2mu_dr2[:, :, :, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T)

    d2tau_dAdB_orbital_response = None
    d2tau_dAdB_grid_response = None

    if with_orbital_response and with_tau:
        d2tau_dAdB_orbital_response = cupy.zeros([natm, natm, 3, 3, ngrids])
        for i_atom in range(natm):
            pi0, pi1 = aoslices[i_atom][2:]
            # d3mu/(dA dB dr) * dnu/dr, A orbital, B orbital
            d2tau_dAdB_orbital_response[i_atom, i_atom, :, :, :] += cupy.einsum('dDxig,xjg,ij->dDg',
                d3mu_dr3[:, :, :, pi0:pi1, :], dmu_dr, dm0[pi0:pi1, :] + dm0[:, pi0:pi1].T)

            for j_atom in range(natm):
                pj0, pj1 = aoslices[j_atom][2:]
                # d2mu/(dA dr) * d2nu/(dB dr), A orbital, B orbital
                d2tau_dAdB_orbital_response[i_atom, j_atom, :, :, :] += cupy.einsum('dxig,Dxjg,ij->dDg',
                    d2mu_dr2[:, :, pi0:pi1, :], d2mu_dr2[:, :, pj0:pj1, :], dm0[pi0:pi1, pj0:pj1] + dm0[pj0:pj1, pi0:pi1].T)

        d2tau_dAdB_orbital_response *= 0.5

    if with_grid_response and with_tau:
        d2tau_dAdB_grid_response = cupy.zeros([natm, natm, 3, 3, ngrids])
        for i_atom in range(natm):
            g_i_with_response = atom_to_grid_index_map[i_atom]
            if len(g_i_with_response) > 0:
                # d3mu/(dA dB dr) * dnu/dr, A grid, B grid
                d2tau_dAdB_grid_response[i_atom, i_atom][:, :, g_i_with_response] += cupy.einsum('dDxig,xjg,ij->dDg',
                    d3mu_dr3[:, :, :, :, g_i_with_response], dmu_dr[:, :, g_i_with_response], dm0 + dm0.T)
                # d2mu/(dA dr) * d2nu/(dB dr), A grid, B grid
                d2tau_dAdB_grid_response[i_atom, i_atom][:, :, g_i_with_response] += cupy.einsum('dxig,Dxjg,ij->dDg',
                    d2mu_dr2[:, :, :, g_i_with_response], d2mu_dr2[:, :, :, g_i_with_response], dm0 + dm0.T)

            for j_atom in range(natm):
                pj0, pj1 = aoslices[j_atom][2:]
                if len(g_i_with_response) > 0:
                    # d3mu/(dA dB dr) * dnu/dr, A orbital, B grid
                    d2tau_dAdB_grid_response[i_atom, j_atom][:, :, g_i_with_response] -= cupy.einsum('dDxig,xjg,ij->dDg',
                        d3mu_dr3[:, :, :, pj0:pj1, g_i_with_response], dmu_dr[:, :, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T).transpose(1,0,2)
                    d2tau_dAdB_grid_response[j_atom, i_atom][:, :, g_i_with_response] -= cupy.einsum('dDxig,xjg,ij->dDg',
                        d3mu_dr3[:, :, :, pj0:pj1, g_i_with_response], dmu_dr[:, :, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T)

                    # d2mu/(dA dr) * d2nu/(dB dr), A orbital, B grid
                    d2tau_dAdB_grid_response[i_atom, j_atom][:, :, g_i_with_response] -= cupy.einsum('dxig,Dxjg,ij->dDg',
                        d2mu_dr2[:, :, pj0:pj1, g_i_with_response], d2mu_dr2[:, :, :, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T).transpose(1,0,2)
                    d2tau_dAdB_grid_response[j_atom, i_atom][:, :, g_i_with_response] -= cupy.einsum('dxig,Dxjg,ij->dDg',
                        d2mu_dr2[:, :, pj0:pj1, g_i_with_response], d2mu_dr2[:, :, :, g_i_with_response], dm0[pj0:pj1, :] + dm0[:, pj0:pj1].T)

        d2tau_dAdB_grid_response *= 0.5

    if (not with_nablarho) and (not with_tau):
        return d2rho_dAdB_orbital_response, d2rho_dAdB_grid_response
    elif not with_tau:
        return d2rho_dAdB_orbital_response, d2nablarho_dAdB_orbital_response, d2rho_dAdB_grid_response, d2nablarho_dAdB_grid_response
    else:
        return d2rho_dAdB_orbital_response, d2nablarho_dAdB_orbital_response, d2tau_dAdB_orbital_response, \
               d2rho_dAdB_grid_response, d2nablarho_dAdB_grid_response, d2tau_dAdB_grid_response

def contract_d2rho_dAdB_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos = None, i_atom_of_grids = None,
                               mu = None, dmu_dr = None, d2mu_dr2 = None, d3mu_dr3 = None,
                               weight_depsilon_drho = None, weight_depsilon_dnablarho = None, weight_depsilon_dtau = None,
                               with_orbital_response = True, with_grid_response = True):
    r"""
        This function does the following in an optimized way:

        d2rho_dAdB_orbital_response, d2nablarho_dAdB_orbital_response, d2tau_dAdB_orbital_response, \
            d2rho_dAdB_grid_response, d2nablarho_dAdB_grid_response, d2tau_dAdB_grid_response = \
            get_d2rho_dAdB_full(
                dm0, xctype, natm, g1 - g0, aoslices, atom_to_grid_index_map, mu, dmu_dr, d2mu_dr2, d3mu_dr3,
            )

        d2E_dAdB_orbital_response += d2rho_dAdB_orbital_response @ weight_depsilon_drho[g0:g1]
        d2E_dAdB_orbital_response += cp.einsum("ABdDxg,xg->ABdD", d2nablarho_dAdB_orbital_response, weight_depsilon_dnablarho[:, g0:g1])
        d2E_dAdB_orbital_response += d2tau_dAdB_orbital_response @ weight_depsilon_dtau[g0:g1]

        If you think the logic in this function is nonsense, I agree. Refer to get_d2rho_dAdB_full() for a readable code.
        This function has went through the following optimizations:
        1. Split the chain contraction into binary contractions, and reorder the contractions
        2. Allow a sparse AO, and guarantees a block of grids belongs to the same atom
        3. Merge grid response diagonal terms into off-digonal term
        4. Share the intermediate variables among orbital and grid responses
    """
    if xctype == "LDA":
        with_nablarho = False
        with_tau = False
    elif xctype == "GGA":
        with_nablarho = True
        with_tau = False
    elif xctype == "MGGA":
        with_nablarho = True
        with_tau = True
    else:
        raise NotImplementedError(f"Unrecognized xctype = {xctype}")

    nao = dm0_masked.shape[-1]
    assert mu is not None
    ngrids = mu.shape[1]
    assert mu.shape == (nao, ngrids)
    assert dmu_dr is not None and dmu_dr.shape == (3, nao, ngrids)
    assert d2mu_dr2 is not None and d2mu_dr2.shape == (3, 3, nao, ngrids)
    if with_nablarho or with_tau:
        assert d3mu_dr3 is not None and d3mu_dr3.shape == (3, 3, 3, nao, ngrids)

    assert weight_depsilon_drho is not None and weight_depsilon_drho.shape == (ngrids,)
    if with_nablarho:
        assert weight_depsilon_dnablarho is not None and weight_depsilon_dnablarho.shape == (3, ngrids)
    if with_tau:
        assert weight_depsilon_dtau is not None and weight_depsilon_dtau.shape == (ngrids,)

    if with_orbital_response or with_grid_response: # There are cross terms in grid response
        assert masked_i_atom_of_aos is not None
        masked_i_atom_of_aos = cupy.asarray(masked_i_atom_of_aos, dtype = numpy.int32)
        assert masked_i_atom_of_aos.shape == (nao,)
    if with_grid_response:
        assert i_atom_of_grids is not None
        i_atom_of_grids = int(i_atom_of_grids)
        assert 0 <= i_atom_of_grids and i_atom_of_grids < natm

    dm_dmT = dm0_masked + dm0_masked.T
    dm_dot_mu = dm_dmT @ mu
    dm_dot_dmudr = contract("ij,djg->dig", dm_dmT, dmu_dr)

    d2e_ii_AA_ao_uncontracted = 0
    d2V = 0
    d2e_ji_AB_ao_uncontracted = 0

    # d2mu/dAdB * nu
    d2rhodAdG = contract("dDig,ig->dDi", d2mu_dr2, dm_dot_mu * weight_depsilon_drho)
    if with_orbital_response:
        d2e_ii_AA_ao_uncontracted += d2rhodAdG
    if with_grid_response:
        d2e_ji_AB_ao_uncontracted += d2rhodAdG
    del d2rhodAdG

    # dmu/dA * dnu/dB
    dmudr_wv = dmu_dr * weight_depsilon_drho
    if with_orbital_response:
        d2rhodAdB = contract("dig,Djg->dDij", dmu_dr, dmudr_wv)
        d2V += d2rhodAdB
        del d2rhodAdB
    if with_grid_response:
        d2rhodAdG = contract("dig,Dig->dDi", dmudr_wv, dm_dot_dmudr)
        d2e_ji_AB_ao_uncontracted += d2rhodAdG
        del d2rhodAdG
    del dmudr_wv

    if with_nablarho:
        # d3mu/(dr dA dB) * nu
        d3mudr3_wv = contract("dDxig,xg->dDig", d3mu_dr3, weight_depsilon_dnablarho)
        d2nablarhodAdG = contract("dDig,ig->dDi", d3mudr3_wv, dm_dot_mu)
        del d3mudr3_wv
        # d2mu/(dA dB) * dnu/dr
        dm_dmudr_wv = contract("xig,xg->ig", dm_dot_dmudr, weight_depsilon_dnablarho)
        d2nablarhodAdG += contract("dDig,ig->dDi", d2mu_dr2, dm_dmudr_wv)
        del dm_dmudr_wv
        d2mudr2_wv = contract("dxig,xg->dig", d2mu_dr2, weight_depsilon_dnablarho)
        if with_orbital_response:
            d2e_ii_AA_ao_uncontracted += d2nablarhodAdG
            # d2mu/(dr dA) * dnu/dB, A orbital, B orbital
            d2mudr2_dnudr_wv = contract("dig,Djg->dDij", d2mudr2_wv, dmu_dr)
            d2V += d2mudr2_dnudr_wv + d2mudr2_dnudr_wv.transpose(1,0,3,2)
            del d2mudr2_dnudr_wv
        if with_grid_response:
            # d2mu/(dr dA) * dnu/dB, A orbital, B grid
            d2nablarhodAdG += contract("dig,Dig->dDi", d2mudr2_wv, dm_dot_dmudr)
            # d2mu/(dr dA) * dnu/dB, A grid, B orbital
            dm_d2mudr2_wv = contract("ij,djg->dig", dm_dmT, d2mudr2_wv)
            d2nablarhodAdG += contract("dig,Dig->dDi", dmu_dr, dm_d2mudr2_wv)
            del dm_d2mudr2_wv
            d2e_ji_AB_ao_uncontracted += d2nablarhodAdG
        del d2mudr2_wv
        del d2nablarhodAdG

    if with_tau:
        # d3mu/(dA dB dr) * dnu/dr
        d2taudAdG = 0.5 * contract("dDxig,xig->dDi", d3mu_dr3, dm_dot_dmudr * weight_depsilon_dtau)
        # d2mu/(dA dr) * d2nu/(dB dr)
        d2mudAdr_d2nudBdr_wv = 0.5 * contract("dxig,Dxjg->dDij", d2mu_dr2, d2mu_dr2 * weight_depsilon_dtau)
        if with_orbital_response:
            d2e_ii_AA_ao_uncontracted += d2taudAdG
            d2V += d2mudAdr_d2nudBdr_wv
        if with_grid_response:
            d2taudAdG += contract("dDij,ij->dDi", d2mudAdr_d2nudBdr_wv, dm_dmT)
            d2e_ji_AB_ao_uncontracted += d2taudAdG
        del d2mudAdr_d2nudBdr_wv
        del d2taudAdG

    d2e = cupy.zeros([natm, natm, 3, 3])

    if with_orbital_response:
        d2e_ii_AA = cupy.zeros((natm, 3, 3))
        d2e_ii_AA_ao_uncontracted = d2e_ii_AA_ao_uncontracted.transpose(2,0,1)
        cupy.add.at(d2e_ii_AA, masked_i_atom_of_aos, d2e_ii_AA_ao_uncontracted)
        del d2e_ii_AA_ao_uncontracted
        for i_atom in range(natm):
            d2e[i_atom, i_atom, :, :] += d2e_ii_AA[i_atom, :, :]
        del d2e_ii_AA

        d2V = d2V * dm_dmT[None, None, :, :]
        d2V = d2V.transpose(2,3,0,1)
        cupy.add.at(d2e, (masked_i_atom_of_aos[:, None], masked_i_atom_of_aos[None, :], slice(None), slice(None)), d2V)

    if with_grid_response:
        d2e[i_atom_of_grids, i_atom_of_grids, :, :] += cupy.einsum("dDi->dD", d2e_ji_AB_ao_uncontracted)

        d2e_ji_AB = cupy.zeros((natm, 3, 3))
        d2e_ji_AB_ao_uncontracted = d2e_ji_AB_ao_uncontracted.transpose(2,0,1)
        cupy.add.at(d2e_ji_AB, masked_i_atom_of_aos, d2e_ji_AB_ao_uncontracted)
        del d2e_ji_AB_ao_uncontracted
        # Why is there a transpose for this equation? Because we're using j index at where it supposes to be i.
        d2e[i_atom_of_grids, :, :, :] -= d2e_ji_AB.transpose(0,2,1)
        d2e[:, i_atom_of_grids, :, :] -= d2e_ji_AB
        del d2e_ji_AB

    return d2e

def _get_exc_deriv2_grid_response(hessobj, mo_coeff, mo_occ, max_memory):
    """
        xc energy 2nd derivative grid response contribution
    """

    mol = hessobj.mol
    mf = hessobj.base
    ni = numint.NumInt()
    xctype = ni._xc_type(mf.xc)

    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    grids = grids.copy()
    grids.build(sort_grids_of_each_atom = True)
    ngrids = grids.coords.shape[0]

    ni.gdftopt = None
    ni.build(mol, grids.coords)
    opt = ni.gdftopt

    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    natm = mol.natm
    mol = None

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    assert dm0.ndim == 2
    dm0_sorted = opt.sort_orbitals(dm0, axis=[0,1])
    dm_mask_buf = cupy.empty(nao * nao)
    dm0 = None

    nonzero_weight_mask = cupy.abs(grids.weights) > 1e-20 # There are d2rho/dAdB terms with very low weight but very big contribution

    ao_loc_sorted = _sorted_mol.ao_loc
    ao_expand = ao_loc_sorted[1:] - ao_loc_sorted[:-1]
    from pyscf.gto.mole import ATOM_OF
    i_atom_of_aos = numpy.repeat(_sorted_mol._bas[:,ATOM_OF], ao_expand)
    i_atom_of_aos = cupy.asarray(i_atom_of_aos, dtype = cupy.int32)

    d2e = cupy.zeros([natm, natm, 3, 3])

    if xctype == 'LDA':
        g0 = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 0, strict_grid_order = True):
            g1 = g0 + weight.shape[0]

            if ao.size == 0:
                g0 = g1
                continue

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)
            rho = numint.eval_rho(_sorted_mol, ao, dm0_masked, xctype = xctype, hermi = 1)
            exc = ni.eval_xc_eff(mf.xc, rho, deriv = 0, xctype=xctype, spin=0)[0]

            epsilon = exc * rho
            del rho, exc

            d2w_dAdB = get_d2weight_dAdB(_sorted_mol, grids, (g0,g1))
            d2e += contract("ABdDg,g->ABdD", d2w_dAdB, epsilon)
            del d2w_dAdB, epsilon
            g0 = g1
        assert g1 == ngrids

        g0 = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 2, strict_grid_order = True):
            g1 = g0 + weight.shape[0]

            ao = ao[:, :, nonzero_weight_mask[g0:g1]]

            if ao.size == 0:
                g0 = g1
                continue

            mu = ao[0]
            dmu_dr = ao[1:4]
            d2mu_dr2 = get_d2mu_dr2(ao)

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)

            rho = numint.eval_rho(_sorted_mol, ao[0], dm0_masked, xctype = xctype, hermi = 1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype = xctype, spin=0)[1:3]
            del rho

            depsilon_drho = vxc[0]

            i_atom_of_grids = int(grids.atm_idx[g0])
            assert cupy.max(cupy.abs(grids.atm_idx[g0:g1] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)

            masked_i_atom_of_aos = i_atom_of_aos[idx]

            drho_dA_orbital_response, drho_dA_grid_response = \
                get_drho_dA_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr)
            drho_dA_full_response = drho_dA_orbital_response + drho_dA_grid_response
            del drho_dA_orbital_response, drho_dA_grid_response

            dw_dA = get_dweight_dA(_sorted_mol, grids, (g0,g1))
            dw_dA = dw_dA[:, :, nonzero_weight_mask[g0:g1]]

            # d2e += cupy.einsum("Adg,g,BDg->ABdD", dw_dA, depsilon_drho, drho_dA_full_response)
            # d2e += cupy.einsum("Adg,g,BDg->BADd", dw_dA, depsilon_drho, drho_dA_full_response)
            d2e_dwdA_term = contract("Adg,BDg->ABdD", dw_dA, drho_dA_full_response * depsilon_drho)
            del dw_dA
            d2e += d2e_dwdA_term + d2e_dwdA_term.transpose(1,0,3,2)
            del d2e_dwdA_term

            weight = weight[nonzero_weight_mask[g0:g1]]
            fwxc = fxc[0,0] * weight
            del fxc
            d2e += contract("Adg,BDg->ABdD", drho_dA_full_response, drho_dA_full_response * fwxc)
            del fwxc, drho_dA_full_response

            d2e += contract_d2rho_dAdB_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                                              mu, dmu_dr, d2mu_dr2, None,
                                              depsilon_drho * weight)

            g0 = g1
        assert g1 == ngrids

    elif xctype == 'GGA':
        g0 = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 1, strict_grid_order = True):
            g1 = g0 + weight.shape[0]

            if ao.size == 0:
                g0 = g1
                continue

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)
            rho = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked, xctype = xctype, hermi = 1)
            exc = ni.eval_xc_eff(mf.xc, rho, deriv = 0, xctype=xctype, spin=0)[0]

            epsilon = exc * rho[0, :]
            del rho, exc

            d2w_dAdB = get_d2weight_dAdB(_sorted_mol, grids, (g0,g1))
            d2e += contract("ABdDg,g->ABdD", d2w_dAdB, epsilon)
            del d2w_dAdB, epsilon
            g0 = g1
        assert g1 == ngrids

        g0 = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 3, strict_grid_order = True):
            g1 = g0 + weight.shape[0]

            ao = ao[:, :, nonzero_weight_mask[g0:g1]]

            if ao.size == 0:
                g0 = g1
                continue

            mu = ao[0]
            dmu_dr = ao[1:4]
            d2mu_dr2 = get_d2mu_dr2(ao)
            d3mu_dr3 = get_d3mu_dr3(ao)

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)

            rho = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked, xctype = xctype, hermi = 1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype = xctype, spin=0)[1:3]
            del rho

            depsilon_drho = vxc[0]
            depsilon_dnablarho = vxc[1:4]

            i_atom_of_grids = int(grids.atm_idx[g0])
            assert cupy.max(cupy.abs(grids.atm_idx[g0:g1] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)

            masked_i_atom_of_aos = i_atom_of_aos[idx]

            drho_dA_orbital_response, drho_dA_grid_response = \
                get_drho_dA_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr, d2mu_dr2)
            drho_dA_full_response = drho_dA_orbital_response + drho_dA_grid_response
            del drho_dA_orbital_response, drho_dA_grid_response

            dw_dA = get_dweight_dA(_sorted_mol, grids, (g0,g1))
            dw_dA = dw_dA[:, :, nonzero_weight_mask[g0:g1]]

            # d2e += cupy.einsum("Adg,g,BDg->ABdD", dw_dA, depsilon_drho, drho_dA_full_response[:,:,0,:])
            # d2e += cupy.einsum("Adg,g,BDg->BADd", dw_dA, depsilon_drho, drho_dA_full_response[:,:,0,:])
            # d2e += cupy.einsum("Adg,xg,BDxg->ABdD", dw_dA, depsilon_dnablarho, drho_dA_full_response[:,:,1:4,:])
            # d2e += cupy.einsum("Adg,xg,BDxg->BADd", dw_dA, depsilon_dnablarho, drho_dA_full_response[:,:,1:4,:])
            depsilondnablarho_dnablarhodA = contract("xg,Adxg->Adg", depsilon_dnablarho, drho_dA_full_response[:,:,1:4,:])
            d2e_dwdA_term = contract("Adg,BDg->ABdD", dw_dA, drho_dA_full_response[:,:,0,:] * depsilon_drho + depsilondnablarho_dnablarhodA)
            del depsilondnablarho_dnablarhodA
            del dw_dA
            d2e += d2e_dwdA_term + d2e_dwdA_term.transpose(1,0,3,2)
            del d2e_dwdA_term

            weight = weight[nonzero_weight_mask[g0:g1]]
            fwxc = fxc * weight
            del fxc
            drhodA_fwxc = contract("xyg,Adyg->Adxg", fwxc, drho_dA_full_response)
            del fwxc
            d2e += contract("Adxg,BDxg->ABdD", drho_dA_full_response, drhodA_fwxc)
            del drhodA_fwxc, drho_dA_full_response

            d2e += contract_d2rho_dAdB_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                                              mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                                              depsilon_drho * weight, depsilon_dnablarho * weight)

            g0 = g1
        assert g1 == ngrids

    elif xctype == 'MGGA':
        g0 = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 1, strict_grid_order = True):
            g1 = g0 + weight.shape[0]

            if ao.size == 0:
                g0 = g1
                continue

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)
            rho = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked, xctype = xctype, hermi = 1)
            exc = ni.eval_xc_eff(mf.xc, rho, deriv = 0, xctype=xctype, spin=0)[0]

            epsilon = exc * rho[0, :]
            del rho, exc

            d2w_dAdB = get_d2weight_dAdB(_sorted_mol, grids, (g0,g1))
            d2e += contract("ABdDg,g->ABdD", d2w_dAdB, epsilon)
            del d2w_dAdB, epsilon
            g0 = g1
        assert g1 == ngrids

        g0 = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 3, strict_grid_order = True):
            g1 = g0 + weight.shape[0]

            ao = ao[:, :, nonzero_weight_mask[g0:g1]]

            if ao.size == 0:
                g0 = g1
                continue

            mu = ao[0]
            dmu_dr = ao[1:4]
            d2mu_dr2 = get_d2mu_dr2(ao)
            d3mu_dr3 = get_d3mu_dr3(ao)

            dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)

            rho = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked, xctype = xctype, hermi = 1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype = xctype, spin=0)[1:3]
            del rho

            depsilon_drho = vxc[0]
            depsilon_dnablarho = vxc[1:4]
            depsilon_dtau = vxc[4]

            i_atom_of_grids = int(grids.atm_idx[g0])
            assert cupy.max(cupy.abs(grids.atm_idx[g0:g1] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)

            masked_i_atom_of_aos = i_atom_of_aos[idx]

            drho_dA_orbital_response, drho_dA_grid_response = \
                get_drho_dA_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr, d2mu_dr2)
            drho_dA_full_response = drho_dA_orbital_response + drho_dA_grid_response
            del drho_dA_orbital_response, drho_dA_grid_response

            dw_dA = get_dweight_dA(_sorted_mol, grids, (g0,g1))
            dw_dA = dw_dA[:, :, nonzero_weight_mask[g0:g1]]

            # d2e += cupy.einsum("Adg,g,BDg->ABdD", dw_dA, depsilon_drho, drho_dA_full_response[:,:,0,:])
            # d2e += cupy.einsum("Adg,g,BDg->BADd", dw_dA, depsilon_drho, drho_dA_full_response[:,:,0,:])
            # d2e += cupy.einsum("Adg,xg,BDxg->ABdD", dw_dA, depsilon_dnablarho, drho_dA_full_response[:,:,1:4,:])
            # d2e += cupy.einsum("Adg,xg,BDxg->BADd", dw_dA, depsilon_dnablarho, drho_dA_full_response[:,:,1:4,:])
            # d2e += cupy.einsum("Adg,g,BDg->ABdD", dw_dA, depsilon_dtau, drho_dA_full_response[:,:,4,:])
            # d2e += cupy.einsum("Adg,g,BDg->BADd", dw_dA, depsilon_dtau, drho_dA_full_response[:,:,4,:])
            depsilondnablarho_dnablarhodA = contract("xg,Adxg->Adg", depsilon_dnablarho, drho_dA_full_response[:,:,1:4,:])
            d2e_dwdA_term = contract("Adg,BDg->ABdD", dw_dA,
                drho_dA_full_response[:,:,0,:] * depsilon_drho + depsilondnablarho_dnablarhodA + drho_dA_full_response[:,:,4,:] * depsilon_dtau)
            del depsilondnablarho_dnablarhodA
            del dw_dA
            d2e += d2e_dwdA_term + d2e_dwdA_term.transpose(1,0,3,2)
            del d2e_dwdA_term

            weight = weight[nonzero_weight_mask[g0:g1]]
            fwxc = fxc * weight
            del fxc
            drhodA_fwxc = contract("xyg,Adyg->Adxg", fwxc, drho_dA_full_response)
            del fwxc
            d2e += contract("Adxg,BDxg->ABdD", drho_dA_full_response, drhodA_fwxc)
            del drhodA_fwxc, drho_dA_full_response

            d2e += contract_d2rho_dAdB_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos, i_atom_of_grids,
                                              mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                                              depsilon_drho * weight, depsilon_dnablarho * weight, depsilon_dtau * weight)

            g0 = g1
        assert g1 == ngrids

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f"xctype = {xctype} not supported")

    return d2e

def _nr_rks_fxc_mo_task(ni, mol, grids, xc_code, fxc, mo_coeff, mo1, mocc,
                        verbose=None, hermi=1, device_id=0):
    with cupy.cuda.Device(device_id):
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
        #### Setup memory buffer
        if xctype == 'LDA':
            ncomp = 1
        elif xctype == 'GGA':
            ncomp = 4
        else:
            ncomp = 5
        fxc_w_buf = cupy.empty(ncomp*ncomp*MIN_BLK_SIZE)
        buf = cupy.empty(MIN_BLK_SIZE * nao)
        vtmp_buf = cupy.empty(nao*nao)
        for ao, mask, weights, coords in ni.block_loop(_sorted_mol, grids, nao, ao_deriv,
                                                       max_memory=None, blksize=None,
                                                       grid_range=(grid_start, grid_end)):
            blk_size = len(weights)
            nao_sub = len(mask)
            vtmp = cupy.ndarray((nao_sub, nao_sub), memptr=vtmp_buf.data)

            p0, p1 = p1, p1+len(weights)
            occ_coeff_mask = mocc[mask]
            rho1 = numint.eval_rho4(_sorted_mol, ao, 2.0*occ_coeff_mask, mo1[:,mask],
                                    xctype=xctype, hermi=hermi)
            t1 = log.timer_debug2('eval rho', *t1)
            if xctype == 'HF':
                continue
            # precompute fxc_w
            fxc_w = cupy.ndarray((ncomp, ncomp, blk_size), memptr=fxc_w_buf.data)
            fxc_w = cupy.multiply(fxc[:,:,p0:p1], weights, out=fxc_w)
            wv = contract('axg,xyg->ayg', rho1, fxc_w, out=rho1)

            for i in range(nset):
                if xctype == 'LDA':
                    aow = numint._scale_ao(ao, wv[i][0], out=buf)
                    add_sparse(vmat[i], ao.dot(aow.T, out=vtmp), mask)
                    # vmat_tmp = ao.dot(numint._scale_ao(ao, wv[i][0]).T)
                elif xctype == 'GGA':
                    wv[i,0] *= .5
                    aow = numint._scale_ao(ao, wv[i], out=buf)
                    add_sparse(vmat[i], ao[0].dot(aow.T, out=vtmp), mask)
                elif xctype == 'NLC':
                    raise NotImplementedError('NLC')
                else:
                    wv[i,0] *= .5
                    wv[i,4] *= .5
                    vtmp = numint._tau_dot(ao, ao, wv[i,4], buf=buf, out=vtmp)
                    aow = numint._scale_ao(ao[:4], wv[i,:4], out=buf)
                    vtmp = contract('ig, jg->ij', ao[0], aow, beta=1, out=vtmp) # ao[0].dot(aow.T, out=vtmp)
                    add_sparse(vmat[i], vtmp, mask)

            t1 = log.timer_debug2('integration', *t1)
            ao = rho1 = None
        t0 = log.timer_debug1(f'vxc on Device {device_id} ', *t0)
        if xctype != 'LDA':
            transpose_sum(vmat)
        vmat = _ao2mo(vmat, mocc, mo_coeff)
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

    TODO: check the effect of different grid, using mf.nlcgrids right now

    Args:
        mo_coeff: array of shape (2, nao, nmo) or (nao, nmo)
            0-th order RKS or UKS MO coefficients
            mo_occ: array of shape (2, nmo) or (nmo,)
            0-th order RKS or UKS MO occupancies
        dm1s: array of shape (*, nao, nao)
            Spin-traced first order density matrices
        return_in_mo:
            Whether to return NLC matrices in MO representations. When UKS
            orbitals are supplied, a two-element tuple of matrices
            (spin-up, spin-down) are evaluated and returned.
    """
    nao = mol.nao
    output_in_2d = False
    if dm1s.ndim == 2:
        assert dm1s.shape == (nao, nao)
        dm1s = dm1s.reshape((1, nao, nao))
        output_in_2d = True
    else:
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
    mol = None

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

    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[mo_coeff.ndim-2])
    ngrids_full = grids.coords.shape[0]
    rho_drho = cupy.empty([4, ngrids_full])
    g1 = 0
    for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv = 1, strict_grid_order = True):
        g0, g1 = g1, g1 + weight.size
        if mo_coeff.ndim == 2:
            rho_drho[:, g0:g1] = numint.eval_rho2(_sorted_mol, ao, mo_coeff[idx, :], mo_occ, None, 'GGA')
        else:
            rho_drho[:, g0:g1]  = numint.eval_rho2(_sorted_mol, ao, mo_coeff[0, idx, :], mo_occ[0], None, 'GGA')
            rho_drho[:, g0:g1] += numint.eval_rho2(_sorted_mol, ao, mo_coeff[1, idx, :], mo_occ[1], None, 'GGA')
    assert g1 == ngrids_full

    rho_i = rho_drho[0,:]

    rho_nonzero_mask = cupy.logical_and(
        rho_i >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD,
        cupy.abs(grids.weights) > 1e-14,
    )

    rho_i = rho_i[rho_nonzero_mask]
    nabla_rho_i = rho_drho[1:4, rho_nonzero_mask]
    grids_coords = cupy.ascontiguousarray(grids.coords[rho_nonzero_mask, :])
    grids_weights = grids.weights[rho_nonzero_mask]
    ngrids = grids_coords.shape[0]

    gamma_i = batched_vec_norm2(nabla_rho_i.T)

    stream = cupy.cuda.get_current_stream()

    omega_i               = cupy.empty(ngrids)
    domega_drho_i         = cupy.empty(ngrids)
    domega_dgamma_i       = cupy.empty(ngrids)
    d2omega_drho2_i       = cupy.empty(ngrids)
    d2omega_dgamma2_i     = cupy.empty(ngrids)
    d2omega_drho_dgamma_i = cupy.empty(ngrids)
    libgdft.VXC_vv10nlc_hess_eval_omega_derivative(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
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
    kappa_i         = kappa_prefactor * rho_i**(1.0/6.0)
    dkappa_drho_i   = (kappa_prefactor * (1.0/6.0)) * rho_i**(-5.0/6.0)
    d2kappa_drho2_i = (kappa_prefactor * (-5.0/36.0)) * rho_i**(-11.0/6.0)

    rho_weight_i = rho_i * grids_weights
    U_i = cupy.empty(ngrids)
    W_i = cupy.empty(ngrids)
    A_i = cupy.empty(ngrids)
    B_i = cupy.empty(ngrids)
    C_i = cupy.empty(ngrids)
    E_i = cupy.empty(ngrids) # Not used
    libgdft.VXC_vv10nlc_hess_eval_UWABCE(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(U_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(W_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(B_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(C_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(E_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_weight_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids)
    )
    del rho_weight_i
    del E_i

    f_gamma_i = rho_i * domega_dgamma_i * W_i

    dm1s_sorted = opt.sort_orbitals(dm1s, axis=[1,2])
    dm_mask_buf = cupy.empty(nao * nao)
    dm1s = None

    if return_in_mo:
        if mo_coeff.ndim == 3:
            mocca = mo_coeff[0][:, mo_occ[0]>0]
            moccb = mo_coeff[1][:, mo_occ[1]>0]
            vmata = cupy.zeros([n_dm1, mo_coeff.shape[2], mocca.shape[1]])
            vmatb = cupy.zeros([n_dm1, mo_coeff.shape[2], moccb.shape[1]])
        else:
            mocc = mo_coeff[:, mo_occ>0]
            vmat = cupy.zeros([n_dm1, mo_coeff.shape[1], mocc.shape[1]])
    else:
        vmat = cupy.zeros([n_dm1, nao, nao])

    available_gpu_memory = get_avail_mem()
    available_gpu_memory = int(available_gpu_memory * 0.25) # Don't use too much gpu memory
    fxc_nbytes_per_dm1 = ((1*6 + 3*2) * ngrids + (1*2 + 3*2) * ngrids_full) * 8
    ndm1_per_batch = int(available_gpu_memory / fxc_nbytes_per_dm1)
    if ndm1_per_batch < 6:
        raise MemoryError(f"Out of GPU memory for NLC response (orbital hessian), available gpu memory = {get_avail_mem()}"
                          f" bytes, nao = {nao}, natm = {_sorted_mol.natm}, ngrids (nonzero rho) = {ngrids}")
    ndm1_per_batch = (ndm1_per_batch + 6 - 1) // 6 * 6

    for i_dm1_batch in range(0, n_dm1, ndm1_per_batch):
        n_dm1_batch = min(ndm1_per_batch, n_dm1 - i_dm1_batch)

        rho_drho_t = cupy.empty([n_dm1_batch, 4, ngrids_full])
        g1 = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv = 1, strict_grid_order = True):
            g0, g1 = g1, g1 + weight.size
            for i_dm in range(n_dm1_batch):
                dm1_sorted = dm1s_sorted[i_dm + i_dm1_batch, :, :]
                dm1_masked = take_last2d(dm1_sorted, idx, out = dm_mask_buf)
                rho_drho_t[i_dm, :, g0:g1] = numint.eval_rho(_sorted_mol, ao, dm1_masked, xctype = "NLC", hermi = 0)
                dm1_sorted = None
                dm1_masked = None
        assert g1 == ngrids_full

        rho_drho_t = rho_drho_t[:, :, rho_nonzero_mask]

        rho_t_i = rho_drho_t[:, 0, :]
        nabla_rho_t_i = rho_drho_t[:, 1:4, :]
        gamma_t_i = nabla_rho_i[0, :] * nabla_rho_t_i[:, 0, :] \
                    + nabla_rho_i[1, :] * nabla_rho_t_i[:, 1, :] \
                    + nabla_rho_i[2, :] * nabla_rho_t_i[:, 2, :]
        gamma_t_i *= 2 # Account for the factor of 2 before gamma_j^t term in equation (22)
        del rho_drho_t

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
        del rho_t_i, gamma_t_i

        fxc_rho = f_rho_t_i * grids_weights
        del f_rho_t_i
        fxc_gamma  = contract("dg,tg->tdg", nabla_rho_i, f_gamma_t_i)
        del f_gamma_t_i
        fxc_gamma += nabla_rho_t_i * f_gamma_i
        del nabla_rho_t_i
        fxc_gamma = 2 * fxc_gamma * grids_weights

        g0_full = 0
        g0_nonzero = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv = 1, strict_grid_order = True):
            g1_full = g0_full + weight.shape[0]

            ao = ao[:, :, rho_nonzero_mask[g0_full : g1_full]]

            if ao.size == 0:
                g0_full = g1_full
                continue

            g1_nonzero = g0_nonzero + ao.shape[-1]

            split_fxc_rho = fxc_rho[:, g0_nonzero : g1_nonzero]
            split_fxc_gamma = fxc_gamma[:, :, g0_nonzero : g1_nonzero]

            for i_dm in range(n_dm1_batch):
                # \mu \nu
                V_munu = contract("ig,jg->ij", ao[0], ao[0] * split_fxc_rho[i_dm, :])

                # \mu \nabla\nu + \nabla\mu \nu
                nabla_fxc_dot_nabla_ao = contract("dg,dig->ig", split_fxc_gamma[i_dm, :, :], ao[1:4])
                V_munu_gamma = contract("ig,jg->ij", ao[0], nabla_fxc_dot_nabla_ao)
                del nabla_fxc_dot_nabla_ao
                V_munu += V_munu_gamma
                V_munu += V_munu_gamma.T
                del V_munu_gamma

                if return_in_mo:
                    if mo_coeff.ndim == 3:
                        vmata[i_dm + i_dm1_batch, :, :] += mo_coeff[0, idx, :].T @ V_munu @ mocca[idx, :]
                        vmatb[i_dm + i_dm1_batch, :, :] += mo_coeff[1, idx, :].T @ V_munu @ moccb[idx, :]
                    else:
                        vmat[i_dm + i_dm1_batch, :, :] += mo_coeff[idx, :].T @ V_munu @ mocc[idx, :]
                else:
                    add_sparse(vmat[i_dm + i_dm1_batch], V_munu, idx)

            g0_nonzero = g1_nonzero
            g0_full = g1_full
        assert g1_full == ngrids_full
        assert g1_nonzero == ngrids

    if return_in_mo and mo_coeff.ndim == 3:
        if output_in_2d:
            vmata = vmata[0]
            vmatb = vmatb[0]
        return (vmata, vmatb)

    if not return_in_mo:
        vmat = opt.unsort_orbitals(vmat, axis=[1,2])

    if output_in_2d:
        vmat = vmat[0]
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
