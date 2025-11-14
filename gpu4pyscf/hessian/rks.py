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
from gpu4pyscf.lib.cupy_helper import (contract, add_sparse, get_avail_mem,
                                       reduce_to_device, transpose_sum, take_last2d)
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

    t1 = log.timer_debug1('hessian of 2e part', *t1)
    de2 += _get_exc_deriv2(hessobj, mo_coeff, mo_occ, dm0, max_memory, atmlst, log)
    if mf.do_nlc():
        de2 += _get_enlc_deriv2(hessobj, mo_coeff, mo_occ, max_memory, log)

    log.timer('RKS partial hessian', *time0)
    return de2

def _get_exc_deriv2(hessobj, mo_coeff, mo_occ, dm0, max_memory, atmlst = None, log = None):
    if log is None:
        log = logger.new_logger(hessobj)
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
    if hessobj.grid_response:
        log.info("Calculating grid response for DFT Hessian")
        de2 += _get_exc_deriv2_grid_response(hessobj, mo_coeff, mo_occ, max_memory)

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
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1][0]
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
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
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
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
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
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
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
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
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
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
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

def get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    natm = len(aoslices)
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert dm0.shape == (nao, nao)

    dm_dmT = dm0 + dm0.T
    dm_dot_mu_and_nu = dm_dmT @ mu
    dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr, dm_dmT)
    dm_dmT = None

    d2rho_dAdr = cupy.zeros([natm, 3, 3, ngrids])
    for i_atom in range(natm):
        p0, p1 = aoslices[i_atom][2:]
        # d2rho_dAdr[i_atom, :, :, :] += cupy.einsum('dDig,jg,ij->dDg', -d2mu_dr2[:, :, p0:p1, :], mu, dm0[p0:p1, :])
        # d2rho_dAdr[i_atom, :, :, :] += cupy.einsum('dDig,jg,ij->dDg', -d2mu_dr2[:, :, p0:p1, :], mu, dm0[:, p0:p1].T)
        # d2rho_dAdr[i_atom, :, :, :] += cupy.einsum('dig,Djg,ij->dDg', -dmu_dr[:, p0:p1, :], dmu_dr, dm0[p0:p1, :])
        # d2rho_dAdr[i_atom, :, :, :] += cupy.einsum('dig,Djg,ij->dDg', -dmu_dr[:, p0:p1, :], dmu_dr, dm0[:, p0:p1].T)
        dm_dot_mu_and_nu_i = dm_dot_mu_and_nu[p0:p1, :]
        d2rho_dAdr[i_atom, :, :, :] += contract('dDig,ig->dDg', -d2mu_dr2[:, :, p0:p1, :], dm_dot_mu_and_nu_i)
        dm_dot_mu_and_nu_i = None
        dm_dot_dmu_and_dnu_i = dm_dot_dmu_and_dnu[:, p0:p1, :]
        d2rho_dAdr[i_atom, :, :, :] += contract('dig,Dig->dDg', -dmu_dr[:, p0:p1, :], dm_dot_dmu_and_dnu_i)
        dm_dot_dmu_and_dnu_i = None
    return d2rho_dAdr

def get_d2rho_dAdr_grid_response(d2mu_dr2, dmu_dr, mu, dm0, atom_to_grid_index_map = None, i_atom = None):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert dm0.shape == (nao, nao)

    dm_dmT = dm0 + dm0.T

    if i_atom is None:
        assert atom_to_grid_index_map is not None
        natm = len(atom_to_grid_index_map)

        dm_dot_mu_and_nu = dm_dmT @ mu
        dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr, dm_dmT)
        dm_dmT = None

        d2rho_dAdr_grid_response = cupy.zeros([natm, 3, 3, ngrids])
        for i_atom in range(natm):
            associated_grid_index = atom_to_grid_index_map[i_atom]
            if len(associated_grid_index) == 0:
                continue
            # d2rho_dAdr_response  = cupy.einsum('dDig,jg,ij->dDg', d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index], dm0)
            # d2rho_dAdr_response += cupy.einsum('dDig,jg,ij->dDg', d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index], dm0.T)
            # d2rho_dAdr_response += cupy.einsum('dig,Djg,ij->dDg', dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], dm0)
            # d2rho_dAdr_response += cupy.einsum('dig,Djg,ij->dDg', dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], dm0.T)
            dm_dot_mu_and_nu_i = dm_dot_mu_and_nu[:, associated_grid_index]
            d2rho_dAdr_response  = contract('dDig,ig->dDg', d2mu_dr2[:, :, :, associated_grid_index], dm_dot_mu_and_nu_i)
            dm_dot_mu_and_nu_i = None
            dm_dot_dmu_and_dnu_i = dm_dot_dmu_and_dnu[:, :, associated_grid_index]
            d2rho_dAdr_response += contract('dig,Dig->dDg', dmu_dr[:, :, associated_grid_index], dm_dot_dmu_and_dnu_i)
            dm_dot_dmu_and_dnu_i = None

            d2rho_dAdr_grid_response[i_atom][:, :, associated_grid_index] = d2rho_dAdr_response
    else:
        assert atom_to_grid_index_map is None

        # Here we assume all grids belong to atom i
        dm_dot_mu_and_nu = dm_dmT @ mu
        d2rho_dAdr_grid_response  = contract('dDig,ig->dDg', d2mu_dr2, dm_dot_mu_and_nu)
        dm_dot_mu_and_nu = None
        dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr, dm_dmT)
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

    dm_dmT = dm0 + dm0.T
    dm_dot_mu_and_nu = dm_dmT @ mu
    drhodr_dot_dmudr = contract('Djg,Dg->jg', dmu_dr, drho_dr)
    drhodr_dot_dmu_dnu_dot_dm = dm_dmT @ drhodr_dot_dmudr
    dm_dmT = None
    drhodr_dot_dmudr = None

    drho_dA = cupy.zeros([natm, 3, ngrids])
    dgamma_dA = cupy.zeros([natm, 3, ngrids])
    for i_atom in range(natm):
        p0, p1 = aoslices[i_atom][2:]

        # drho_dA[i_atom, :, :] += cupy.einsum('dig,jg,ij->dg', -dmu_dr[:, p0:p1, :], mu, dm0[p0:p1, :])
        # drho_dA[i_atom, :, :] += cupy.einsum('dig,jg,ij->dg', -dmu_dr[:, p0:p1, :], mu, dm0[:, p0:p1].T)
        dm_dot_mu_and_nu_i = dm_dot_mu_and_nu[p0:p1, :]
        drho_dA[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], dm_dot_mu_and_nu_i)

        # dgamma_dA[i_atom, :, :] += cupy.einsum('dDig,jg,Dg,ij->dg', -d2mu_dr2[:, :, p0:p1, :], mu, drho_dr, dm0[p0:p1, :])
        # dgamma_dA[i_atom, :, :] += cupy.einsum('dDig,jg,Dg,ij->dg', -d2mu_dr2[:, :, p0:p1, :], mu, drho_dr, dm0[:, p0:p1].T)
        # dgamma_dA[i_atom, :, :] += cupy.einsum('dig,Djg,Dg,ij->dg', -dmu_dr[:, p0:p1, :], dmu_dr, drho_dr, dm0[p0:p1, :])
        # dgamma_dA[i_atom, :, :] += cupy.einsum('dig,Djg,Dg,ij->dg', -dmu_dr[:, p0:p1, :], dmu_dr, drho_dr, dm0[:, p0:p1].T)
        d2mudAdr_dot_drhodr = contract('dDig,Dg->dig', -d2mu_dr2[:, :, p0:p1, :], drho_dr)
        dgamma_dA[i_atom, :, :] += contract('dig,ig->dg', d2mudAdr_dot_drhodr, dm_dot_mu_and_nu_i)
        d2mudAdr_dot_drhodr = None
        dm_dot_mu_and_nu_i = None
        drhodr_dot_dmu_dnu_dot_dm_i = drhodr_dot_dmu_dnu_dot_dm[p0:p1, :]
        dgamma_dA[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], drhodr_dot_dmu_dnu_dot_dm_i)
        drhodr_dot_dmu_dnu_dot_dm_i = None
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

    dm_dmT = dm0 + dm0.T

    if i_atom is None:
        assert atom_to_grid_index_map is not None

        natm = len(atom_to_grid_index_map)
        drho_dA_grid_response   = cupy.zeros([natm, 3, ngrids])
        dgamma_dA_grid_response = cupy.zeros([natm, 3, ngrids])

        dm_dot_mu_and_nu = dm_dmT @ mu
        dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr, dm_dmT)
        dm_dmT = None

        for i_atom in range(natm):
            associated_grid_index = atom_to_grid_index_map[i_atom]
            if len(associated_grid_index) == 0:
                continue
            # rho_response  = cupy.einsum('dig,jg,ij->dg', dmu_dr[:, :, associated_grid_index], mu[:, associated_grid_index], dm0)
            # rho_response += cupy.einsum('dig,jg,ij->dg', dmu_dr[:, :, associated_grid_index], mu[:, associated_grid_index], dm0.T)
            dmu_dr_grid_i = dmu_dr[:, :, associated_grid_index]
            dm_dot_mu_and_nu_i = dm_dot_mu_and_nu[:, associated_grid_index]
            rho_response = contract('dig,ig->dg', dmu_dr_grid_i, dm_dot_mu_and_nu_i)
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
            gamma_response  = contract('dig,ig->dg', d2mudr2_dot_drhodr, dm_dot_mu_and_nu_i)
            d2mudr2_dot_drhodr = None
            dm_dot_mu_and_nu_i = None
            dm_dot_dmu_and_dnu_i = dm_dot_dmu_and_dnu[:, :, associated_grid_index]
            dmudr_dot_drhodr = contract('dig,dg->ig', dmu_dr_grid_i, drho_dr[:, associated_grid_index])
            dmu_dr_grid_i = None
            gamma_response += contract('dig,ig->dg', dm_dot_dmu_and_dnu_i, dmudr_dot_drhodr)
            dmudr_dot_drhodr = None
            dm_dot_dmu_and_dnu_i = None
            dgamma_dA_grid_response[i_atom][:, associated_grid_index] = gamma_response
            gamma_response = None
        dm_dot_mu_and_nu = None
        dm_dot_dmu_and_dnu = None

    else:
        assert atom_to_grid_index_map is None

        # Here we assume all grids belong to atom i
        dm_dot_mu_and_nu = dm_dmT @ mu
        drho_dA_grid_response = contract('dig,ig->dg', dmu_dr, dm_dot_mu_and_nu)

        d2mudr2_dot_drhodr = contract('dDig,Dg->dig', d2mu_dr2, drho_dr)
        dgamma_dA_grid_response = contract('dig,ig->dg', d2mudr2_dot_drhodr, dm_dot_mu_and_nu)
        d2mudr2_dot_drhodr = None
        dm_dot_mu_and_nu = None
        dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr, dm_dmT)
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
        d2rho_dAdB[i_atom, i_atom, :, :, :] += cupy.einsum('dDig,jg,ij->dDg',
            d2mu_dr2[:, :, pi0:pi1, :], mu, dm0[pi0:pi1, :] + dm0[:, pi0:pi1].T)
        d2gamma_dAdB[i_atom, i_atom, :, :, :] += cupy.einsum('dDPig,jg,Pg,ij->dDg',
            d3mu_dr3[:, :, :, pi0:pi1, :], mu, drho, dm0[pi0:pi1, :] + dm0[:, pi0:pi1].T)
        d2gamma_dAdB[i_atom, i_atom, :, :, :] += cupy.einsum('dDig,Pjg,Pg,ij->dDg',
            d2mu_dr2[:, :, pi0:pi1, :], dmu_dr, drho, dm0[pi0:pi1, :] + dm0[:, pi0:pi1].T)
        for j_atom in range(natm):
            pj0, pj1 = aoslices[j_atom][2:]
            d2rho_dAdB[i_atom, j_atom, :, :, :] += cupy.einsum('dig,Djg,ij->dDg',
                dmu_dr[:, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], dm0[pi0:pi1, pj0:pj1] + dm0[pj0:pj1, pi0:pi1].T)
            d2gamma_dAdB[i_atom, j_atom, :, :, :] += cupy.einsum('dPig,Djg,Pg,ij->dDg',
                d2mu_dr2[:, :, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], drho, dm0[pi0:pi1, pj0:pj1] + dm0[pj0:pj1, pi0:pi1].T)
            d2gamma_dAdB[i_atom, j_atom, :, :, :] += cupy.einsum('dig,DPjg,Pg,ij->dDg',
                dmu_dr[:, pi0:pi1, :], d2mu_dr2[:, :, pj0:pj1, :], drho, dm0[pi0:pi1, pj0:pj1] + dm0[pj0:pj1, pi0:pi1].T)

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

    dm_dmT = dm0 + dm0.T
    mu_dot_dm = dm_dmT @ mu
    drhodr_dot_dmudr = contract('djg,dg->jg', dmu_dr, drho_dr)
    drhodr_dot_dmudr_dot_dm = dm_dmT @ drhodr_dot_dmudr
    drhodr_dot_dmudr = None

    d2e_rho_dAdB = cupy.zeros([natm, natm, 3, 3])
    d2e_gamma_dAdB = cupy.zeros([natm, natm, 3, 3])
    for i_atom in range(natm):
        pi0, pi1 = aoslices[i_atom][2:]

        mu_dot_dm_i = mu_dot_dm[pi0:pi1, :]
        d2rho_dA2  = contract('dDig,ig->dDg', d2mu_dr2[:, :, pi0:pi1, :], mu_dot_dm_i)
        d2e_rho_dAdB[i_atom, i_atom, :, :] += contract('dDg,g->dD', d2rho_dA2, fw_rho)
        d2rho_dA2 = None

        d3mudA2dr_dot_drhodr = contract('dDPig,Pg->dDig', d3mu_dr3[:, :, :, pi0:pi1, :], drho_dr)
        d2gamma_dA2 = contract('dDig,ig->dDg', d3mudA2dr_dot_drhodr, mu_dot_dm_i)
        d3mudA2dr_dot_drhodr = None
        mu_dot_dm_i = None
        drhodr_dot_dmudr_dot_dm_i = drhodr_dot_dmudr_dot_dm[pi0:pi1, :]
        d2gamma_dA2 += contract('dDig,ig->dDg', d2mu_dr2[:, :, pi0:pi1, :], drhodr_dot_dmudr_dot_dm_i)
        drhodr_dot_dmudr_dot_dm_i = None
        d2e_gamma_dAdB[i_atom, i_atom, :, :] += contract('dDg,g->dD', d2gamma_dA2, fw_gamma)
        d2gamma_dA2 = None

        for j_atom in range(natm):
            pj0, pj1 = aoslices[j_atom][2:]
            dmudr_dot_dm_ij = contract('djg,ij->dig', dmu_dr[:, pj0:pj1, :], dm_dmT[pi0:pi1, pj0:pj1])
            d2rho_dAdB = contract('dig,Dig->dDg', dmu_dr[:, pi0:pi1, :], dmudr_dot_dm_ij)
            d2e_rho_dAdB[i_atom, j_atom, :, :] += contract('dDg,g->dD', d2rho_dAdB, fw_rho)
            d2rho_dAdB = None

            drhodr_dot_d2mudAdr = contract('dDig,Dg->dig', d2mu_dr2[:, :, pi0:pi1, :], drho_dr)
            d2gamma_dAdB = contract('dig,Dig->dDg', drhodr_dot_d2mudAdr, dmudr_dot_dm_ij)
            dmudr_dot_dm_ij = None
            drhodr_dot_d2mudAdr = None
            d2gamma_dAdB = contract('dDg,g->dD', d2gamma_dAdB, fw_gamma)
            d2e_gamma_dAdB[i_atom, j_atom, :, :] += d2gamma_dAdB
            d2e_gamma_dAdB[j_atom, i_atom, :, :] += d2gamma_dAdB.T
            d2gamma_dAdB = None

    d2rho_dAdr = get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices)
    d2e_gamma_dAdB += contract('AdPg,BDPg->ABdD', d2rho_dAdr, d2rho_dAdr * fw_gamma)

    return d2e_rho_dAdB + 2 * d2e_gamma_dAdB

def get_d2rhodAdB_d2gammadAdB_grid_response(mol, grids_coords, dm0, atom_to_grid_index_map):
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

    _, _, d2rho_dAdB_grid_response, d2nablarho_dAdB_grid_response = \
        get_d2rho_dAdB_full(dm0, "GGA", natm, ngrids, aoslices, atom_to_grid_index_map, mu, dmu_dr, d2mu_dr2, d3mu_dr3, with_orbital_response = False)

    d2gamma_dAdB_grid_response = cupy.einsum("ABdDxg,xg->ABdDg", d2nablarho_dAdB_grid_response, drho)

    d2rho_dAdr_orbital_response = get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices)
    d2rho_dAdr_grid_response = get_d2rho_dAdr_grid_response(d2mu_dr2, dmu_dr, mu, dm0, atom_to_grid_index_map)
    d2gamma_dAdB_grid_response += cupy.einsum("Adxg,BDxg->ABdDg", d2rho_dAdr_orbital_response, d2rho_dAdr_grid_response)
    d2gamma_dAdB_grid_response += cupy.einsum("Adxg,BDxg->ABdDg", d2rho_dAdr_grid_response, d2rho_dAdr_orbital_response)
    d2gamma_dAdB_grid_response += cupy.einsum("Adxg,BDxg->ABdDg", d2rho_dAdr_grid_response, d2rho_dAdr_grid_response)

    d2gamma_dAdB_grid_response *= 2
    return d2rho_dAdB_grid_response, d2gamma_dAdB_grid_response

def contract_d2rhodAdB_d2gammadAdB_grid_response(d3mu_dr3, d2mu_dr2, dmu_dr, mu, drho_dr, dm0, aoslices, atom_to_grid_index_map, fw_rho, fw_gamma):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    natm = len(aoslices)
    assert d3mu_dr3.shape == (3, 3, 3, nao, ngrids)
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert drho_dr.shape == (3, ngrids)
    assert dm0.shape == (nao, nao)

    # Factor of 2 coming from the derivative of gamma = |nabla rho|^2
    drhodr_weight_depsilondgamma = 2 * drho_dr * fw_gamma

    d2e_rho_gamma = contract_d2rho_dAdB_full(dm0, "GGA", natm, ngrids, aoslices, atom_to_grid_index_map,
                                             mu, dmu_dr, d2mu_dr2, d3mu_dr3, fw_rho, drhodr_weight_depsilondgamma, None,
                                             with_orbital_response = False)

    d2rho_dAdr_orbital_response = get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices)
    d2rho_dAdr_grid_response = get_d2rho_dAdr_grid_response(d2mu_dr2, dmu_dr, mu, dm0, atom_to_grid_index_map)
    d2rhodAdr_gridresponse_weight_depsilondgamma = d2rho_dAdr_grid_response * fw_gamma
    d2edgamma2_drhodA_cross_term = cupy.einsum("Adxg,BDxg->ABdD", d2rho_dAdr_orbital_response, d2rhodAdr_gridresponse_weight_depsilondgamma)
    d2edgamma2_drhodA_cross_term += d2edgamma2_drhodA_cross_term.transpose(1,0,3,2)
    d2edgamma2_drhodA_cross_term += cupy.einsum("Adxg,BDxg->ABdD", d2rho_dAdr_grid_response, d2rhodAdr_gridresponse_weight_depsilondgamma)
    d2e_rho_gamma += 2 * d2edgamma2_drhodA_cross_term

    return d2e_rho_gamma

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
    weight_f_rho_i   =   f_rho_i * grids_weights
    weight_f_gamma_i = f_gamma_i * grids_weights

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
    #                                      weight_f_rho_i, weight_f_gamma_i)

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

        split_fw_rho   = weight_f_rho_i  [g0:g1]
        split_fw_gamma = weight_f_gamma_i[g0:g1]
        d2e += contract_d2rhodAdB_d2gammadAdB(d3mu_dr3, d2mu_dr2, dmu_dr, mu, split_drho_dr, dm0, aoslices, split_fw_rho, split_fw_gamma)

        split_ao = None
        mu = None
        dmu_dr = None
        d2mu_dr2 = None
        d3mu_dr3 = None
        split_drho_dA = None
        split_dgamma_dA = None

    weight_f_rho_i = None
    weight_f_gamma_i = None

    if not hessobj.grid_response:
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

    if hessobj.grid_response:
        # The code above includes only the orbital response piece of E_{G,G}^{AB} in Eq 37.
        log.info("Calculating grid response for VV10 Hessian")

        # # First half of E_{w,w}^{AB} in Eq 32
        # d2w_dAdB = get_d2weight_dAdB(mol, grids)
        # d2w_dAdB = d2w_dAdB[:, :, :, :, rho_nonzero_mask]
        # d2e += contract("ABdDg,g->ABdD", d2w_dAdB, rho_i * (beta + E_i))
        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
        ao_nbytes_per_grid = ((9 * 2) * mol.natm * mol.natm + 2) * 8
        ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
        if ngrids_per_batch < 16:
            raise MemoryError(f"Out of GPU memory for NLC energy second derivative, available gpu memory = {get_avail_mem()}"
                              f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids = {ngrids_full}")
        ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
        ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

        g0_nonzero = 0
        for g0_full in range(0, ngrids_full, ngrids_per_batch):
            g1_full = min(g0_full + ngrids_per_batch, ngrids_full)
            d2w_dAdB = get_d2weight_dAdB(mol, grids, (g0_full, g1_full))
            d2w_dAdB = d2w_dAdB[:, :, :, :, rho_nonzero_mask[g0_full : g1_full]]
            g1_nonzero = g0_nonzero + d2w_dAdB.shape[4]
            d2e += contract("ABdDg,g->ABdD", d2w_dAdB, rho_i[g0_nonzero : g1_nonzero] * (beta + E_i[g0_nonzero : g1_nonzero]))
            g0_nonzero = g1_nonzero
        assert g0_nonzero == ngrids

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

        # Second half of E_{w,w}^{AB} in Eq 32
        d2e += contract("Adg,BDg->ABdD", grids_weights_1, E_Bw_i * rho_i)

        grid_to_atom_index_map = grids.atm_idx[rho_nonzero_mask]
        atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]

        drho_dA_full_response   = cupy.empty([natm, 3, ngrids], order = "C")
        dgamma_dA_full_response = cupy.empty([natm, 3, ngrids], order = "C")

        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
        ao_nbytes_per_grid = ((10 + 1*4 + 3*4 + 9) * mol.nao + (3*4) * mol.natm) * 8
        ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
        if ngrids_per_batch < 16:
            raise MemoryError(f"Out of GPU memory for NLC energy second derivative, available gpu memory = {get_avail_mem()}"
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
            split_grid_to_atom_index_map = grid_to_atom_index_map[g0:g1]
            split_atom_to_grid_index_map = [cupy.where(split_grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]

            split_drho_dA_orbital_response, split_dgamma_dA_orbital_response = \
                get_drhodA_dgammadA_orbital_response(d2mu_dr2, dmu_dr, mu, split_drho_dr, dm0, aoslices)
            split_drho_dA_grid_response,    split_dgamma_dA_grid_response = \
                get_drhodA_dgammadA_grid_response(d2mu_dr2, dmu_dr, mu, split_drho_dr, dm0, split_atom_to_grid_index_map)

            drho_dA_full_response  [:, :, g0:g1] =   split_drho_dA_orbital_response +   split_drho_dA_grid_response
            dgamma_dA_full_response[:, :, g0:g1] = split_dgamma_dA_orbital_response + split_dgamma_dA_grid_response
            split_ao = None
            mu = None
            dmu_dr = None
            d2mu_dr2 = None
            split_drho_dA_orbital_response   = None
            split_dgamma_dA_orbital_response = None
            split_drho_dA_grid_response   = None
            split_dgamma_dA_grid_response = None

        # First term of E_{G,w}^{AB} in Eq 34, and its transpose
        E_Gw_AB_term_1_right = drho_dA_full_response * f_rho_i + dgamma_dA_full_response * f_gamma_i
        E_Gw_AB_term_1 = contract("Adg,BDg->ABdD", grids_weights_1, E_Gw_AB_term_1_right)
        E_Gw_AB_term_1_right = None
        d2e += E_Gw_AB_term_1 + E_Gw_AB_term_1.transpose(1,0,3,2)
        E_Gw_AB_term_1 = None
        # Second term of E_{G,w}^{AB} in Eq 34, and its transpose
        E_Gw_AB_term_2_right = (E_Bw_i + (U_Bw_i * dkappa_drho_i + W_Bw_i * domega_drho_i) * rho_i) * grids_weights
        E_Gw_AB_term_2 = contract("Adg,BDg->ABdD", drho_dA_full_response, E_Gw_AB_term_2_right)
        E_Gw_AB_term_2_right = None
        d2e += E_Gw_AB_term_2 + E_Gw_AB_term_2.transpose(1,0,3,2)
        E_Gw_AB_term_2 = None
        # Third term of E_{G,w}^{AB} in Eq 34, and its transpose
        E_Gw_AB_term_3_right = W_Bw_i * domega_dgamma_i * rho_i * grids_weights
        E_Gw_AB_term_3 = contract("Adg,BDg->ABdD", dgamma_dA_full_response, E_Gw_AB_term_3_right)
        E_Gw_AB_term_3_right = None
        d2e += E_Gw_AB_term_3 + E_Gw_AB_term_3.transpose(1,0,3,2)
        E_Gw_AB_term_3 = None

        E_Bw_i = None
        U_Bw_i = None
        W_Bw_i = None

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

        # E_{w,gr}^{AB} in Eq 33, and its transpose
        E_wgr_AB_term = contract("Adg,BDg->ABdD", grids_weights_1, E_Bgr_i * rho_i)
        d2e += E_wgr_AB_term
        d2e += E_wgr_AB_term.transpose(1,0,3,2)
        grids_weights_1 = None
        E_wgr_AB_term = None

        # First term in E_{G,gr}^{AB} in Eq 35, and its transpose
        E_Ggr_AB_term_1_right = (E_Bgr_i + (U_Bgr_i * dkappa_drho_i + W_Bgr_i * domega_drho_i) * rho_i) * grids_weights
        E_Ggr_AB_term_1 = contract("Adg,BDg->ABdD", drho_dA_full_response, E_Ggr_AB_term_1_right)
        E_Ggr_AB_term_1_right = None
        d2e += E_Ggr_AB_term_1 + E_Ggr_AB_term_1.transpose(1,0,3,2)
        E_Ggr_AB_term_1 = None
        # Second term in E_{G,gr}^{AB} in Eq 35, and its transpose
        E_Ggr_AB_term_2_right = W_Bgr_i * domega_dgamma_i * rho_i * grids_weights
        E_Ggr_AB_term_2 = contract("Adg,BDg->ABdD", dgamma_dA_full_response, E_Ggr_AB_term_2_right)
        E_Ggr_AB_term_2_right = None
        d2e += E_Ggr_AB_term_2 + E_Ggr_AB_term_2.transpose(1,0,3,2)
        E_Ggr_AB_term_2 = None

        E_Bgr_i = None
        U_Bgr_i = None
        W_Bgr_i = None

        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
        ao_nbytes_per_grid = ((20 + 9 + 27 + 4*4 + 12*4 + 4*2) * mol.nao + (9*2) * mol.natm + 4 + 18*4 + 27*2*4 + 3) * 8
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
            split_grid_to_atom_index_map = grid_to_atom_index_map[g0:g1]
            split_atom_to_grid_index_map = [cupy.where(split_grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]

            d2e += contract_d2rhodAdB_d2gammadAdB_grid_response(d3mu_dr3, d2mu_dr2, dmu_dr, mu, split_drho_dr, dm0, aoslices, split_atom_to_grid_index_map,
                                                                f_rho_i[g0:g1] * grids_weights[g0:g1], f_gamma_i[g0:g1] * grids_weights[g0:g1])

            split_ao = None
            mu = None
            dmu_dr = None
            d2mu_dr2 = None
            d3mu_dr3 = None

        # Last two terms in E_{G,G}^{AB} in Eq 37, orbital + grid response contribution
        drho_dA_full_response   = cupy.ascontiguousarray(drho_dA_full_response)
        dgamma_dA_full_response = cupy.ascontiguousarray(dgamma_dA_full_response)
        f_rho_A_i_full_response   = cupy.empty([mol.natm, 3, ngrids], order = "C")
        f_gamma_A_i_full_response = cupy.empty([mol.natm, 3, ngrids], order = "C")

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
            ctypes.c_int(3 * mol.natm),
        )

        d2e += contract("Adg,BDg->ABdD",   drho_dA_full_response,   f_rho_A_i_full_response * grids_weights)
        d2e += contract("Adg,BDg->ABdD", dgamma_dA_full_response, f_gamma_A_i_full_response * grids_weights)

        f_rho_A_i_full_response   = None
        f_gamma_A_i_full_response = None
        drho_dA_full_response   = None
        dgamma_dA_full_response = None

        # E_{gr,gr}^{AB} in Eq 36
        D_B_i = cupy.empty([mol.natm, 3, 3, ngrids], order = "C")
        libgdft.VXC_vv10nlc_hess_eval_D_B_in_double_grid_response(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(D_B_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_weights.data.ptr, ctypes.c_void_p),
            ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grid_to_atom_index_map.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(natm),
        )

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
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
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
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
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
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
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

        if hessobj.grid_response:
            t2 = log.init_timer()

            if xctype == 'LDA':
                # If you wonder why not using ni.block_loop(), because I need the exact grid index range (g0, g1).
                available_gpu_memory = get_avail_mem()
                available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
                ao_nbytes_per_grid = ((4 + 3 + 2 + 3*2 + 1) * mol.nao + (3*2) * mol.natm + 16 + 4) * 8
                ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
                if ngrids_per_batch < 16:
                    raise MemoryError(f"Out of GPU memory for LDA Fock first derivative, available gpu memory = {get_avail_mem()}"
                                      f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (one GPU) = {grid_end - grid_start}, device_id = {device_id}")
                ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
                ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

                for g0 in range(grid_start, grid_end, ngrids_per_batch):
                    g1 = min(g0 + ngrids_per_batch, grid_end)
                    split_grids_coords = cupy.asarray(grids.coords)[g0:g1, :]
                    split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 1, gdftopt = None, transpose = False)

                    mu = split_ao[0]
                    dmu_dr = split_ao[1:4]

                    rho = numint.eval_rho2(mol, mu, mo_coeff, mo_occ, xctype=xctype)
                    vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype=xctype)[1:3]
                    rho = None

                    depsilon_drho = vxc[0] # Just of shape (ngrids,)
                    d2epsilon_drho2 = fxc[0,0] # Just of shape (ngrids,)

                    dw_dA = get_dweight_dA(mol, grids, (g0,g1))
                    # # Negative here to cancel the overall negative sign before return
                    # vmat -= cupy.einsum("Adg,g,pg,qg,qj->Adpj", dw_dA, depsilon_drho, mu, mu, mocc)
                    dwdA_depsilondrho = dw_dA * depsilon_drho
                    dw_dA = None
                    mu_occ = mu.T @ mocc
                    for i_atom in range(natm):
                        dwdA_depsilondrho_mu = contract("dg,pg->dpg", dwdA_depsilondrho[i_atom, :, :], mu)
                        vmat[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_depsilondrho_mu, mu_occ)
                        dwdA_depsilondrho_mu = None
                    dwdA_depsilondrho = None

                    grid_to_atom_index_map = cupy.asarray(grids.atm_idx)[g0:g1]
                    atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]
                    grid_to_atom_index_map = None

                    _, drho_dA_grid_response = \
                        get_drho_dA_full(dm0, xctype, natm, g1 - g0, None, atom_to_grid_index_map, mu, dmu_dr, with_orbital_response = False)

                    weight = cupy.asarray(grids.weights)[g0:g1]
                    # # Negative here to cancel the overall negative sign before return
                    # vmat -= cupy.einsum("g,g,Adg,pg,qg,qj->Adpj", weight, d2epsilon_drho2, drho_dA_grid_response, mu, mu, mocc)
                    weight_d2epsilondrho2_drhodA_grid_response = drho_dA_grid_response * (weight * d2epsilon_drho2)
                    drho_dA_grid_response = None
                    for i_atom in range(natm):
                        weight_d2epsilondrho2_drhodA_grid_response_mu = contract("dg,pg->dpg", weight_d2epsilondrho2_drhodA_grid_response[i_atom, :, :], mu)
                        vmat[i_atom, :, :, :] -= contract("dpg,gj->dpj", weight_d2epsilondrho2_drhodA_grid_response_mu, mu_occ)
                        weight_d2epsilondrho2_drhodA_grid_response_mu = None
                    mu_occ = None

                    for i_atom in range(natm):
                        associated_grid_index = atom_to_grid_index_map[i_atom]
                        if len(associated_grid_index) == 0:
                            continue

                        # # Negative here to cancel the overall negative sign before return
                        # vmat[i_atom, :, :, :] -= cupy.einsum("g,g,dpg,qg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_drho[associated_grid_index],
                        #     dmu_dr[:, :, associated_grid_index], mu[:, associated_grid_index], mocc)
                        # vmat[i_atom, :, :, :] -= cupy.einsum("g,g,dqg,pg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_drho[associated_grid_index],
                        #     dmu_dr[:, :, associated_grid_index], mu[:, associated_grid_index], mocc)
                        mu_grid_i = mu[:, associated_grid_index]
                        weight_depsilondrho_dmudr_grid_i = dmu_dr[:, :, associated_grid_index] * \
                                                           (weight[associated_grid_index] * depsilon_drho[associated_grid_index])
                        mu_occ_grid_i = mu_grid_i.T @ mocc
                        vmat[i_atom, :, :, :] -= weight_depsilondrho_dmudr_grid_i @ mu_occ_grid_i
                        mu_occ_grid_i = None
                        weight_depsilondrho_dmudr_occ_grid_i = contract("dqg,qj->djg", weight_depsilondrho_dmudr_grid_i, mocc)
                        weight_depsilondrho_dmudr_grid_i = None
                        vmat[i_atom, :, :, :] -= contract("pg,djg->dpj", mu_grid_i, weight_depsilondrho_dmudr_occ_grid_i)
                        weight_depsilondrho_dmudr_occ_grid_i = None
                        mu_grid_i = None

            elif xctype == 'GGA':
                # If you wonder why not using ni.block_loop(), because I need the exact grid index range (g0, g1).
                available_gpu_memory = get_avail_mem()
                available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
                ao_nbytes_per_grid = ((10 + 9 + 2 + 3*2 + 2 + 3*2 + 3*3 + 9 + 4*3) * mol.nao + (3 + 3 + 9 + 3*4*2) * mol.natm + 4*2 + 16*2 + 2 + 2) * 8
                ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
                if ngrids_per_batch < 16:
                    raise MemoryError(f"Out of GPU memory for GGA Fock first derivative, available gpu memory = {get_avail_mem()}"
                                      f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (one GPU) = {grid_end - grid_start}, device_id = {device_id}")
                ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
                ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

                for g0 in range(grid_start, grid_end, ngrids_per_batch):
                    g1 = min(g0 + ngrids_per_batch, grid_end)
                    split_grids_coords = cupy.asarray(grids.coords)[g0:g1, :]
                    split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2, gdftopt = None, transpose = False)

                    mu = split_ao[0]
                    dmu_dr = split_ao[1:4]
                    d2mu_dr2 = get_d2mu_dr2(split_ao)

                    rho_drho = numint.eval_rho2(mol, split_ao[:4], mo_coeff, mo_occ, xctype=xctype)
                    vxc, fxc = ni.eval_xc_eff(mf.xc, rho_drho, deriv = 2, xctype=xctype)[1:3]

                    # rho = rho_drho[0]
                    # drho_dr = rho_drho[1:4]
                    rho_drho_tau = None

                    depsilon_drho = vxc[0]
                    depsilon_dnablarho = vxc[1:4]
                    # d2epsilon_drho2 = fxc[0,0]
                    # d2epsilon_drho_dnablarho = fxc[0,1:4]
                    # d2epsilon_dnablarho2 = fxc[1:4,1:4]

                    dw_dA = get_dweight_dA(mol, grids, (g0,g1))
                    # # Negative here to cancel the overall negative sign before return
                    # vmat -= cupy.einsum("Adg,g,pg,qg,qj->Adpj", dw_dA, depsilon_drho, mu, mu, mocc)
                    # vmat -= cupy.einsum("Adg,xg,xpg,qg,qj->Adpj", dw_dA, depsilon_dnablarho, dmu_dr, mu, mocc)
                    # vmat -= cupy.einsum("Adg,xg,xqg,pg,qj->Adpj", dw_dA, depsilon_dnablarho, dmu_dr, mu, mocc)
                    depsilondnablarho_dmudr = contract("xg,xpg->pg", depsilon_dnablarho, dmu_dr)
                    depsilondrho_mu = mu * depsilon_drho
                    mu_occ = mu.T @ mocc
                    for i_atom in range(natm):
                        dwdA_depsilondrho_mu = contract("dg,pg->dpg", dw_dA[i_atom, :, :], depsilondrho_mu + depsilondnablarho_dmudr)
                        vmat[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_depsilondrho_mu, mu_occ)
                        dwdA_depsilondrho_mu = None
                    depsilondrho_mu = None
                    mu_occ = None
                    depsilondnablarho_dmudr_occ = depsilondnablarho_dmudr.T @ mocc
                    depsilondnablarho_dmudr = None
                    for i_atom in range(natm):
                        dwdA_mu = contract("dg,pg->dpg", dw_dA[i_atom, :, :], mu)
                        vmat[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_mu, depsilondnablarho_dmudr_occ)
                        dwdA_mu = None
                    depsilondnablarho_dmudr_occ = None
                    dw_dA = None

                    grid_to_atom_index_map = cupy.asarray(grids.atm_idx)[g0:g1]
                    atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]
                    grid_to_atom_index_map = None

                    _, _, drho_dA_grid_response, dnablarho_dA_grid_response = \
                        get_drho_dA_full(dm0, xctype, natm, g1 - g0, None, atom_to_grid_index_map, mu, dmu_dr, d2mu_dr2, with_orbital_response = False)

                    weight = cupy.asarray(grids.weights)[g0:g1]
                    # # Negative here to cancel the overall negative sign before return
                    # # d2epsilon/drho2 * drho/dR * mu * nu
                    # vmat -= cupy.einsum("g,g,Adg,pg,qg,qj->Adpj", weight, d2epsilon_drho2, drho_dA_grid_response, mu, mu, mocc)
                    # # d2epsilon/(drho d_nabla_rho) * d_nabla_rho/dR * mu * nu
                    # vmat -= cupy.einsum("g,xg,Adxg,pg,qg,qj->Adpj", weight, d2epsilon_drho_dnablarho, dnablarho_dA_grid_response, mu, mu, mocc)
                    # # d2epsilon/(d_nabla_rho drho) * drho/dR * nabla(mu * nu)
                    # vmat -= cupy.einsum("g,xg,Adg,xpg,qg,qj->Adpj", weight, d2epsilon_drho_dnablarho, drho_dA_grid_response, dmu_dr, mu, mocc)
                    # vmat -= cupy.einsum("g,xg,Adg,xqg,pg,qj->Adpj", weight, d2epsilon_drho_dnablarho, drho_dA_grid_response, dmu_dr, mu, mocc)
                    # # d2epsilon/(d_nabla_rho d_nabla_rho) * d_nabla_rho/dR * nabla(mu * nu)
                    # vmat -= cupy.einsum("g,xyg,Adxg,ypg,qg,qj->Adpj", weight, d2epsilon_dnablarho2, dnablarho_dA_grid_response, dmu_dr, mu, mocc)
                    # vmat -= cupy.einsum("g,xyg,Adxg,yqg,pg,qj->Adpj", weight, d2epsilon_dnablarho2, dnablarho_dA_grid_response, dmu_dr, mu, mocc)
                    combined_d_dA_grid_response = cupy.concatenate((drho_dA_grid_response[:, :, None, :], dnablarho_dA_grid_response), axis = 2)
                    drho_dA_grid_response = None
                    dnablarho_dA_grid_response = None

                    fwxc = fxc * weight
                    fxc = None
                    drhodA_grid_response_fwxc = contract("xyg,Adyg->Adxg", fwxc, combined_d_dA_grid_response)
                    combined_d_dA_grid_response = None
                    fwxc = None

                    mu_occ = mu.T @ mocc
                    dmudr_occ = contract("dqg,qj->dgj", dmu_dr, mocc)
                    for i_atom in range(natm):
                        drhodA_grid_response_fwxc_rho_term_mu = contract("dg,pg->dpg", drhodA_grid_response_fwxc[i_atom, :, 0, :], mu)
                        vmat[i_atom, :, :, :] -= contract("dpg,gj->dpj", drhodA_grid_response_fwxc_rho_term_mu, mu_occ)
                        drhodA_grid_response_fwxc_rho_term_mu = None
                        drhodA_grid_response_fwxc_nablarho_term_dmudr_occ = contract("dxg,xgj->dgj", drhodA_grid_response_fwxc[i_atom, :, 1:4, :], dmudr_occ)
                        vmat[i_atom, :, :, :] -= contract("dgj,pg->dpj", drhodA_grid_response_fwxc_nablarho_term_dmudr_occ, mu)
                        drhodA_grid_response_fwxc_nablarho_term_dmudr_occ = None
                        drhodA_grid_response_fwxc_nablarho_term_dmudr = contract("dxg,xpg->dpg", drhodA_grid_response_fwxc[i_atom, :, 1:4, :], dmu_dr)
                        vmat[i_atom, :, :, :] -= contract("dpg,gj->dpj", drhodA_grid_response_fwxc_nablarho_term_dmudr, mu_occ)
                        drhodA_grid_response_fwxc_nablarho_term_dmudr = None
                    drhodA_grid_response_fwxc = None
                    mu_occ = None
                    dmudr_occ = None

                    for i_atom in range(natm):
                        associated_grid_index = atom_to_grid_index_map[i_atom]
                        if len(associated_grid_index) == 0:
                            continue
                        # # Negative here to cancel the overall negative sign before return
                        # vmat[i_atom, :, :, :] -= cupy.einsum("g,g,dpg,qg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_drho[associated_grid_index],
                        #     dmu_dr[:, :, associated_grid_index], mu[:, associated_grid_index], mocc)
                        # vmat[i_atom, :, :, :] -= cupy.einsum("g,g,dqg,pg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_drho[associated_grid_index],
                        #     dmu_dr[:, :, associated_grid_index], mu[:, associated_grid_index], mocc)
                        # vmat[i_atom, :, :, :] -= cupy.einsum("g,Dg,dDpg,qg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_dnablarho[:, associated_grid_index],
                        #     d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index], mocc)
                        # vmat[i_atom, :, :, :] -= cupy.einsum("g,Dg,dDqg,pg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_dnablarho[:, associated_grid_index],
                        #     d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index], mocc)
                        # vmat[i_atom, :, :, :] -= cupy.einsum("g,Dg,dpg,Dqg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_dnablarho[:, associated_grid_index],
                        #     dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], mocc)
                        # vmat[i_atom, :, :, :] -= cupy.einsum("g,Dg,dqg,Dpg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_dnablarho[:, associated_grid_index],
                        #     dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], mocc)
                        mu_grid_i = mu[:, associated_grid_index]
                        dmu_dr_grid_i = dmu_dr[:, :, associated_grid_index]

                        mu_occ_grid_i = mu_grid_i.T @ mocc
                        dmudr_occ_grid_i = contract("dqg,qj->dgj", dmu_dr_grid_i, mocc)

                        weight_depsilondrho_grid_i = weight[associated_grid_index] * depsilon_drho[associated_grid_index]
                        vmat[i_atom, :, :, :] -= (dmu_dr_grid_i * weight_depsilondrho_grid_i) @ mu_occ_grid_i
                        vmat[i_atom, :, :, :] -= contract("pg,dgj->dpj", mu_grid_i * weight_depsilondrho_grid_i, dmudr_occ_grid_i)
                        weight_depsilondrho_grid_i = None

                        d2mu_dr2_grid_i = d2mu_dr2[:, :, :, associated_grid_index]

                        weight_depsilondnablarho_grid_i = weight[associated_grid_index] * depsilon_dnablarho[:, associated_grid_index]
                        weight_depsilondnablarho_d2mudr2 = contract("Dg,dDpg->dpg", weight_depsilondnablarho_grid_i, d2mu_dr2_grid_i)
                        d2mu_dr2_grid_i = None
                        vmat[i_atom, :, :, :] -= contract("dpg,gj->dpj", weight_depsilondnablarho_d2mudr2, mu_occ_grid_i)
                        mu_occ_grid_i = None
                        weight_depsilondnablarho_d2mudr2_occ = contract("dpg,pj->dgj", weight_depsilondnablarho_d2mudr2, mocc)
                        weight_depsilondnablarho_d2mudr2 = None
                        vmat[i_atom, :, :, :] -= contract("pg,dgj->dpj", mu_grid_i, weight_depsilondnablarho_d2mudr2_occ)
                        mu_grid_i = None
                        weight_depsilondnablarho_d2mudr2_occ = None
                        weight_depsilondnablarho_dmudr = contract("Dg,Dpg->pg", weight_depsilondnablarho_grid_i, dmu_dr_grid_i)
                        vmat[i_atom, :, :, :] -= contract("pg,dgj->dpj", weight_depsilondnablarho_dmudr, dmudr_occ_grid_i)
                        dmudr_occ_grid_i = None
                        weight_depsilondnablarho_dmudr_occ = weight_depsilondnablarho_dmudr.T @ mocc
                        weight_depsilondnablarho_dmudr = None
                        vmat[i_atom, :, :, :] -= contract("dpg,gj->dpj", dmu_dr_grid_i, weight_depsilondnablarho_dmudr_occ)
                        weight_depsilondnablarho_dmudr_occ = None
                        dmu_dr_grid_i = None
                        weight_depsilondnablarho_grid_i = None

            elif xctype == 'MGGA':
                # If you wonder why not using ni.block_loop(), because I need the exact grid index range (g0, g1).
                available_gpu_memory = get_avail_mem()
                available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
                ao_nbytes_per_grid = ((10 + 9 + 24 + 4*2 + 24 + 3 + 24) * mol.nao + (3 + 15*3) * mol.natm + 5*2 + 25*2 + 2 + 2) * 8
                ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
                if ngrids_per_batch < 16:
                    raise MemoryError(f"Out of GPU memory for mGGA Fock first derivative, available gpu memory = {get_avail_mem()}"
                                      f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (one GPU) = {grid_end - grid_start}, device_id = {device_id}")
                ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
                ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

                for g0 in range(grid_start, grid_end, ngrids_per_batch):
                    g1 = min(g0 + ngrids_per_batch, grid_end)
                    split_grids_coords = cupy.asarray(grids.coords)[g0:g1, :]
                    split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2, gdftopt = None, transpose = False)

                    mu = split_ao[0]
                    dmu_dr = split_ao[1:4]
                    d2mu_dr2 = get_d2mu_dr2(split_ao)

                    rho_drho_tau = numint.eval_rho2(mol, split_ao[:4], mo_coeff, mo_occ, xctype=xctype)
                    vxc, fxc = ni.eval_xc_eff(mf.xc, rho_drho_tau, deriv = 2, xctype=xctype)[1:3]

                    # rho = rho_drho_tau[0]
                    # drho_dr = rho_drho_tau[1:4]
                    # tau = rho_drho_tau[4]
                    rho_drho_tau = None

                    depsilon_drho = vxc[0]
                    depsilon_dnablarho = vxc[1:4]
                    depsilon_dtau = vxc[4]
                    # d2epsilon_drho2 = fxc[0,0]
                    # d2epsilon_drho_dnablarho = fxc[0,1:4]
                    # d2epsilon_drho_dtau = fxc[0,4]
                    # d2epsilon_dnablarho2 = fxc[1:4,1:4]
                    # d2epsilon_dnablarho_dtau = fxc[1:4,4]
                    # d2epsilon_dtau2 = fxc[4,4]

                    dw_dA = get_dweight_dA(mol, grids, (g0,g1))
                    # # Negative here to cancel the overall negative sign before return
                    # vmat -= cupy.einsum("Adg,g,pg,qg,qj->Adpj", dw_dA, depsilon_drho, mu, mu, mocc)
                    # vmat -= cupy.einsum("Adg,xg,xpg,qg,qj->Adpj", dw_dA, depsilon_dnablarho, dmu_dr, mu, mocc)
                    # vmat -= cupy.einsum("Adg,xg,xqg,pg,qj->Adpj", dw_dA, depsilon_dnablarho, dmu_dr, mu, mocc)
                    # vmat -= 0.5 * cupy.einsum("Adg,g,xpg,xqg,qj->Adpj", dw_dA, depsilon_dtau, dmu_dr, dmu_dr, mocc)
                    depsilondnablarho_dmudr = contract("xg,xpg->pg", depsilon_dnablarho, dmu_dr)
                    depsilondrho_mu = mu * depsilon_drho
                    mu_occ = mu.T @ mocc
                    for i_atom in range(natm):
                        dwdA_depsilondrho_mu = contract("dg,pg->dpg", dw_dA[i_atom, :, :], depsilondrho_mu + depsilondnablarho_dmudr)
                        vmat[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_depsilondrho_mu, mu_occ)
                        dwdA_depsilondrho_mu = None
                    depsilondrho_mu = None
                    mu_occ = None
                    depsilondnablarho_dmudr_occ = depsilondnablarho_dmudr.T @ mocc
                    depsilondnablarho_dmudr = None
                    for i_atom in range(natm):
                        dwdA_mu = contract("dg,pg->dpg", dw_dA[i_atom, :, :], mu)
                        vmat[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_mu, depsilondnablarho_dmudr_occ)
                        dwdA_mu = None
                    depsilondnablarho_dmudr_occ = None
                    depsilondtau_dmudr_occ = contract("dqg,qj->dgj", dmu_dr * depsilon_dtau, mocc)
                    for i_atom in range(natm):
                        dwdA_dmudr = contract("dg,xpg->dpxg", dw_dA[i_atom, :, :], dmu_dr)
                        vmat[i_atom, :, :, :] -= 0.5 * contract("dpxg,xgj->dpj", dwdA_dmudr, depsilondtau_dmudr_occ)
                        dwdA_dmudr = None
                    depsilondtau_dmudr_occ = None
                    dw_dA = None

                    grid_to_atom_index_map = cupy.asarray(grids.atm_idx)[g0:g1]
                    atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]
                    grid_to_atom_index_map = None

                    _, _, _, drho_dA_grid_response, dnablarho_dA_grid_response, dtau_dA_grid_response = \
                        get_drho_dA_full(dm0, xctype, natm, g1 - g0, None, atom_to_grid_index_map, mu, dmu_dr, d2mu_dr2, with_orbital_response = False)

                    weight = cupy.asarray(grids.weights)[g0:g1]
                    # # d2epsilon/drho2 * drho/dR * mu * nu
                    # vmat -= cupy.einsum("g,g,Adg,pg,qg,qj->Adpj", weight, d2epsilon_drho2, drho_dA_grid_response, mu, mu, mocc)
                    # # d2epsilon/(drho d_nabla_rho) * d_nabla_rho/dR * mu * nu
                    # vmat -= cupy.einsum("g,xg,Adxg,pg,qg,qj->Adpj", weight, d2epsilon_drho_dnablarho, dnablarho_dA_grid_response, mu, mu, mocc)
                    # # d2epsilon/(d_nabla_rho drho) * drho/dR * nabla(mu * nu)
                    # vmat -= cupy.einsum("g,xg,Adg,xpg,qg,qj->Adpj", weight, d2epsilon_drho_dnablarho, drho_dA_grid_response, dmu_dr, mu, mocc)
                    # vmat -= cupy.einsum("g,xg,Adg,xqg,pg,qj->Adpj", weight, d2epsilon_drho_dnablarho, drho_dA_grid_response, dmu_dr, mu, mocc)
                    # # d2epsilon/(d_nabla_rho d_nabla_rho) * d_nabla_rho/dR * nabla(mu * nu)
                    # vmat -= cupy.einsum("g,xyg,Adxg,ypg,qg,qj->Adpj", weight, d2epsilon_dnablarho2, dnablarho_dA_grid_response, dmu_dr, mu, mocc)
                    # vmat -= cupy.einsum("g,xyg,Adxg,yqg,pg,qj->Adpj", weight, d2epsilon_dnablarho2, dnablarho_dA_grid_response, dmu_dr, mu, mocc)
                    # # d2epsilon/(drho dtau) * dtau/dR * mu * nu
                    # vmat -= cupy.einsum("g,g,Adg,pg,qg,qj->Adpj", weight, d2epsilon_drho_dtau, dtau_dA_grid_response, mu, mu, mocc)
                    # # d2epsilon/(d_nabla_rho dtau) * dtau/dR * nabla(mu * nu)
                    # vmat -= cupy.einsum("g,xg,Adg,xpg,qg,qj->Adpj", weight, d2epsilon_dnablarho_dtau, dtau_dA_grid_response, dmu_dr, mu, mocc)
                    # vmat -= cupy.einsum("g,xg,Adg,xqg,pg,qj->Adpj", weight, d2epsilon_dnablarho_dtau, dtau_dA_grid_response, dmu_dr, mu, mocc)
                    # # d2epsilon/(dtau drho) * drho/dR * nabla_mu * nabla_nu
                    # vmat -= 0.5 * cupy.einsum("g,g,Adg,xpg,xqg,qj->Adpj", weight, d2epsilon_drho_dtau, drho_dA_grid_response, dmu_dr, dmu_dr, mocc)
                    # # d2epsilon/(dtau d_nabla_rho) * d_nabla_rho/dR * nabla_mu * nabla_nu
                    # vmat -= 0.5 * cupy.einsum("g,xg,Adxg,ypg,yqg,qj->Adpj",
                    #     weight, d2epsilon_dnablarho_dtau, dnablarho_dA_grid_response, dmu_dr, dmu_dr, mocc)
                    # # d2epsilon/dtau2 * dtau/dR * nabla_mu * nabla_nu
                    # vmat -= 0.5 * cupy.einsum("g,g,Adg,xpg,xqg,qj->Adpj", weight, d2epsilon_dtau2, dtau_dA_grid_response, dmu_dr, dmu_dr, mocc)
                    combined_d_dA_grid_response = cupy.concatenate(
                        (drho_dA_grid_response[:, :, None, :], dnablarho_dA_grid_response, dtau_dA_grid_response[:, :, None, :]),
                        axis = 2
                    )
                    drho_dA_grid_response = None
                    dnablarho_dA_grid_response = None
                    dtau_dA_grid_response = None

                    fwxc = fxc * weight
                    fxc = None
                    drhodA_grid_response_fwxc = contract("xyg,Adyg->Adxg", fwxc, combined_d_dA_grid_response)
                    combined_d_dA_grid_response = None
                    fwxc = None

                    mu_occ = mu.T @ mocc
                    dmudr_occ = contract("dqg,qj->dgj", dmu_dr, mocc)
                    for i_atom in range(natm):
                        drhodA_grid_response_fwxc_rho_term_mu = contract("dg,pg->dpg", drhodA_grid_response_fwxc[i_atom, :, 0, :], mu)
                        vmat[i_atom, :, :, :] -= contract("dpg,gj->dpj", drhodA_grid_response_fwxc_rho_term_mu, mu_occ)
                        drhodA_grid_response_fwxc_rho_term_mu = None
                        drhodA_grid_response_fwxc_nablarho_term_dmudr_occ = contract("dxg,xgj->dgj", drhodA_grid_response_fwxc[i_atom, :, 1:4, :], dmudr_occ)
                        vmat[i_atom, :, :, :] -= contract("dgj,pg->dpj", drhodA_grid_response_fwxc_nablarho_term_dmudr_occ, mu)
                        drhodA_grid_response_fwxc_nablarho_term_dmudr_occ = None
                        drhodA_grid_response_fwxc_nablarho_term_dmudr = contract("dxg,xpg->dpg", drhodA_grid_response_fwxc[i_atom, :, 1:4, :], dmu_dr)
                        vmat[i_atom, :, :, :] -= contract("dpg,gj->dpj", drhodA_grid_response_fwxc_nablarho_term_dmudr, mu_occ)
                        drhodA_grid_response_fwxc_nablarho_term_dmudr = None
                        drhodA_grid_response_fwxc_tau_term_dmudr = contract("dg,xpg->dpxg", drhodA_grid_response_fwxc[i_atom, :, 4, :], dmu_dr)
                        vmat[i_atom, :, :, :] -= 0.5 * contract("dpxg,xgj->dpj", drhodA_grid_response_fwxc_tau_term_dmudr, dmudr_occ)
                        drhodA_grid_response_fwxc_tau_term_dmudr = None
                    drhodA_grid_response_fwxc = None
                    mu_occ = None
                    dmudr_occ = None

                    for i_atom in range(natm):
                        associated_grid_index = atom_to_grid_index_map[i_atom]
                        if len(associated_grid_index) == 0:
                            continue
                        # # Negative here to cancel the overall negative sign before return
                        # vmat[i_atom, :, :, :] -= cupy.einsum("g,g,dpg,qg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_drho[associated_grid_index],
                        #     dmu_dr[:, :, associated_grid_index], mu[:, associated_grid_index], mocc)
                        # vmat[i_atom, :, :, :] -= cupy.einsum("g,g,dqg,pg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_drho[associated_grid_index],
                        #     dmu_dr[:, :, associated_grid_index], mu[:, associated_grid_index], mocc)
                        # vmat[i_atom, :, :, :] -= cupy.einsum("g,Dg,dDpg,qg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_dnablarho[:, associated_grid_index],
                        #     d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index], mocc)
                        # vmat[i_atom, :, :, :] -= cupy.einsum("g,Dg,dDqg,pg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_dnablarho[:, associated_grid_index],
                        #     d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index], mocc)
                        # vmat[i_atom, :, :, :] -= cupy.einsum("g,Dg,dpg,Dqg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_dnablarho[:, associated_grid_index],
                        #     dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], mocc)
                        # vmat[i_atom, :, :, :] -= cupy.einsum("g,Dg,dqg,Dpg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_dnablarho[:, associated_grid_index],
                        #     dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], mocc)
                        # vmat[i_atom, :, :, :] -= 0.5 * cupy.einsum("g,g,dDpg,Dqg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_dtau[associated_grid_index],
                        #     d2mu_dr2[:, :, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], mocc)
                        # vmat[i_atom, :, :, :] -= 0.5 * cupy.einsum("g,g,dDqg,Dpg,qj->dpj",
                        #     weight[associated_grid_index], depsilon_dtau[associated_grid_index],
                        #     d2mu_dr2[:, :, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], mocc)
                        mu_grid_i = mu[:, associated_grid_index]
                        dmu_dr_grid_i = dmu_dr[:, :, associated_grid_index]

                        mu_occ_grid_i = mu_grid_i.T @ mocc
                        dmudr_occ_grid_i = contract("dqg,qj->dgj", dmu_dr_grid_i, mocc)

                        weight_depsilondrho_grid_i = weight[associated_grid_index] * depsilon_drho[associated_grid_index]
                        vmat[i_atom, :, :, :] -= (dmu_dr_grid_i * weight_depsilondrho_grid_i) @ mu_occ_grid_i
                        vmat[i_atom, :, :, :] -= contract("pg,dgj->dpj", mu_grid_i * weight_depsilondrho_grid_i, dmudr_occ_grid_i)
                        weight_depsilondrho_grid_i = None

                        d2mu_dr2_grid_i = d2mu_dr2[:, :, :, associated_grid_index]

                        weight_depsilondnablarho_grid_i = weight[associated_grid_index] * depsilon_dnablarho[:, associated_grid_index]
                        weight_depsilondnablarho_d2mudr2 = contract("Dg,dDpg->dpg", weight_depsilondnablarho_grid_i, d2mu_dr2_grid_i)
                        vmat[i_atom, :, :, :] -= contract("dpg,gj->dpj", weight_depsilondnablarho_d2mudr2, mu_occ_grid_i)
                        mu_occ_grid_i = None
                        weight_depsilondnablarho_d2mudr2_occ = contract("dpg,pj->dgj", weight_depsilondnablarho_d2mudr2, mocc)
                        weight_depsilondnablarho_d2mudr2 = None
                        vmat[i_atom, :, :, :] -= contract("pg,dgj->dpj", mu_grid_i, weight_depsilondnablarho_d2mudr2_occ)
                        mu_grid_i = None
                        weight_depsilondnablarho_d2mudr2_occ = None
                        weight_depsilondnablarho_dmudr = contract("Dg,Dpg->pg", weight_depsilondnablarho_grid_i, dmu_dr_grid_i)
                        vmat[i_atom, :, :, :] -= contract("pg,dgj->dpj", weight_depsilondnablarho_dmudr, dmudr_occ_grid_i)
                        weight_depsilondnablarho_dmudr_occ = weight_depsilondnablarho_dmudr.T @ mocc
                        weight_depsilondnablarho_dmudr = None
                        vmat[i_atom, :, :, :] -= contract("dpg,gj->dpj", dmu_dr_grid_i, weight_depsilondnablarho_dmudr_occ)
                        weight_depsilondnablarho_dmudr_occ = None
                        weight_depsilondnablarho_grid_i = None

                        weight_depsilondrho_grid_i = weight[associated_grid_index] * depsilon_dtau[associated_grid_index]
                        vmat[i_atom, :, :, :] -= 0.5 * cupy.einsum("dDpg,Dgj->dpj", d2mu_dr2_grid_i * weight_depsilondrho_grid_i, dmudr_occ_grid_i)
                        dmudr_occ_grid_i = None
                        d2mudr2_occ_grid_i = contract("dDqg,qj->dDgj", d2mu_dr2_grid_i, mocc)
                        d2mu_dr2_grid_i = None
                        vmat[i_atom, :, :, :] -= 0.5 * contract("Dpg,dDgj->dpj", dmu_dr_grid_i * weight_depsilondrho_grid_i, d2mudr2_occ_grid_i)
                        dmu_dr_grid_i = None
                        d2mudr2_occ_grid_i = None

            elif xctype == 'HF':
                pass
            else:
                raise NotImplementedError(f"xctype = {xctype} not supported")
            t2 = log.timer_debug2('grid response', *t2)

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

def get_dweight_dA(mol, grids, grid_range = None):
    ngrids = grids.coords.shape[0]
    assert grids.atm_idx.shape[0] == ngrids
    assert grids.quadrature_weights.shape[0] == ngrids
    atm_coords = cupy.asarray(mol.atom_coords(), order = "C")

    from gpu4pyscf.dft import radi
    a_factor = radi.get_treutler_fac(mol, grids.atomic_radii)

    grids_coords = cupy.asarray(grids.coords)
    grids_quadrature_weights = cupy.asarray(grids.quadrature_weights)
    grids_atm_idx = cupy.asarray(grids.atm_idx)
    if grid_range is not None:
        assert numpy.asarray(grid_range).shape == (2,)
        assert grid_range[1] > grid_range[0]
        ngrids = grid_range[1] - grid_range[0]
        grids_coords = grids_coords[grid_range[0] : grid_range[1]]
        grids_quadrature_weights = grids_quadrature_weights[grid_range[0] : grid_range[1]]
        grids_atm_idx = grids_atm_idx[grid_range[0] : grid_range[1]]

    dweight_dA = cupy.zeros([mol.natm, 3, ngrids], order = "C")
    libgdft.GDFTbecke_partition_weight_derivative(
        ctypes.cast(dweight_dA.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_quadrature_weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(atm_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(a_factor.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_atm_idx.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(mol.natm),
    )
    dweight_dA[grids_atm_idx, 0, cupy.arange(ngrids)] = -cupy.sum(dweight_dA[:, 0, :], axis=[0])
    dweight_dA[grids_atm_idx, 1, cupy.arange(ngrids)] = -cupy.sum(dweight_dA[:, 1, :], axis=[0])
    dweight_dA[grids_atm_idx, 2, cupy.arange(ngrids)] = -cupy.sum(dweight_dA[:, 2, :], axis=[0])

    return dweight_dA

def get_d2weight_dAdB(mol, grids, grid_range = None):
    ngrids = grids.coords.shape[0]
    assert grids.atm_idx.shape[0] == ngrids
    assert grids.quadrature_weights.shape[0] == ngrids
    atm_coords = cupy.asarray(mol.atom_coords(), order = "C")

    from gpu4pyscf.dft import radi
    a_factor = radi.get_treutler_fac(mol, grids.atomic_radii)

    grids_coords = cupy.asarray(grids.coords)
    grids_quadrature_weights = cupy.asarray(grids.quadrature_weights)
    grids_atm_idx = cupy.asarray(grids.atm_idx)
    if grid_range is not None:
        assert numpy.asarray(grid_range).shape == (2,)
        assert grid_range[1] > grid_range[0]
        ngrids = grid_range[1] - grid_range[0]
        grids_coords = grids_coords[grid_range[0] : grid_range[1]]
        grids_quadrature_weights = grids_quadrature_weights[grid_range[0] : grid_range[1]]
        grids_atm_idx = grids_atm_idx[grid_range[0] : grid_range[1]]

    P_B = cupy.zeros([mol.natm, ngrids], order = "C")
    libgdft.GDFTbecke_eval_PB(
        ctypes.cast(P_B.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(atm_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(a_factor.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(mol.natm),
    )
    sum_P_B = cupy.sum(P_B, axis = 0)
    inv_sum_P_B = cupy.zeros(ngrids)
    nonzero_sum_P_B_location = (sum_P_B > 1e-14)
    inv_sum_P_B[nonzero_sum_P_B_location] = 1.0 / sum_P_B[nonzero_sum_P_B_location]
    nonzero_sum_P_B_location = None
    sum_P_B = None

    d2weight_dAdB = cupy.zeros([mol.natm, mol.natm, 3, 3, ngrids], order = "C")
    libgdft.GDFTbecke_partition_weight_second_derivative(
        ctypes.cast(d2weight_dAdB.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_quadrature_weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(atm_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(a_factor.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_atm_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(P_B.data.ptr, ctypes.c_void_p),
        ctypes.cast(inv_sum_P_B.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(mol.natm),
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
            if len(associated_grid_index) == 0:
                continue
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
            if len(associated_grid_index) == 0:
                continue
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

            vmat_mo[i_atom, :, :, :] += _ao2mo(dF, mocc, mo_coeff)
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
            if len(associated_grid_index) == 0:
                continue
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

                vmat_mo[i_atom, :, :, :] += _ao2mo(dF_ao, mocc, mo_coeff)
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

                vmat_mo[i_atom, :, :, :] += _ao2mo(dF_ao, mocc, mo_coeff)
                dF_ao = None

    return vmat_mo

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

def contract_d2rho_dAdB_full(dm0, xctype, natm, ngrids, aoslices = None, atom_to_grid_index_map = None,
                             mu = None, dmu_dr = None, d2mu_dr2 = None, d3mu_dr3 = None,
                             weight_depsilon_drho = None, weight_depsilon_dnablarho = None, weight_depsilon_dtau = None,
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
    assert d2mu_dr2 is not None and d2mu_dr2.shape == (3, 3, nao, ngrids)
    if with_nablarho or with_tau:
        assert d3mu_dr3 is not None and d3mu_dr3.shape == (3, 3, 3, nao, ngrids)

    assert weight_depsilon_drho is not None and weight_depsilon_drho.shape == (ngrids,)
    if with_nablarho:
        assert weight_depsilon_dnablarho is not None and weight_depsilon_dnablarho.shape == (3, ngrids)
    if with_tau:
        assert weight_depsilon_dtau is not None and weight_depsilon_dtau.shape == (ngrids,)

    if with_orbital_response or with_grid_response: # There are cross terms in grid response
        assert aoslices is not None and len(aoslices) == natm
    if with_grid_response:
        assert atom_to_grid_index_map is not None and len(atom_to_grid_index_map) == natm

    dm_dmT = dm0 + dm0.T
    dm_dot_mu = dm_dmT @ mu
    dm_dot_dmudr = contract("djg,ij->dig", dmu_dr, dm_dmT)
    if with_grid_response and (with_nablarho or with_tau):
        dm_dot_d2mudr2 = contract("dDjg,ij->dDig", d2mu_dr2, dm_dmT)

    d2e = cupy.zeros([natm, natm, 3, 3])

    if with_orbital_response:
        for i_atom in range(natm):
            pi0, pi1 = aoslices[i_atom][2:]
            # d2mu/dr2 * nu, A orbital, B orbital
            dm_dot_mu_i = dm_dot_mu[pi0:pi1, :]
            d2mudAdA_nu = contract("dDig,ig->dDg", d2mu_dr2[:, :, pi0:pi1, :], dm_dot_mu_i)
            d2e[i_atom, i_atom, :, :] += d2mudAdA_nu @ weight_depsilon_drho
            d2mudAdA_nu = None

            if with_nablarho:
                # d3mu/(dr dA dB) * nu, A orbital, B orbital
                d3mudAdAdr_nu = contract("dDxig,ig->dDxg", d3mu_dr3[:, :, :, pi0:pi1, :], dm_dot_mu_i)
                d2e[i_atom, i_atom, :, :] += contract("dDxg,xg->dD", d3mudAdAdr_nu, weight_depsilon_dnablarho)
                d3mudAdAdr_nu = None
                # d2mu/(dA dB) * dnu/dr, A orbital, B orbital
                dm_dot_dmudr_i = dm_dot_dmudr[:, pi0:pi1, :]
                d2mudAdA_dnudr = contract("dDig,xig->dDxg", d2mu_dr2[:, :, pi0:pi1, :], dm_dot_dmudr_i)
                d2e[i_atom, i_atom, :, :] += contract("dDxg,xg->dD", d2mudAdA_dnudr, weight_depsilon_dnablarho)
                d2mudAdA_dnudr = None
            dm_dot_mu_i = None

            if with_tau:
                # d3mu/(dA dB dr) * dnu/dr, A orbital, B orbital
                d3mudAdAdr_dnudr = contract("dDxig,xig->dDg", d3mu_dr3[:, :, :, pi0:pi1, :], dm_dot_dmudr_i)
                d2e[i_atom, i_atom, :, :] += 0.5 * d3mudAdAdr_dnudr @ weight_depsilon_dtau
                d3mudAdAdr_dnudr = None
            dm_dot_dmudr_i = None

            for j_atom in range(natm):
                pj0, pj1 = aoslices[j_atom][2:]
                # dmu/dr * dnu/dr, A orbital, B orbital
                dm_dot_dmudr_ij = contract("djg,ij->dig", dmu_dr[:, pj0:pj1, :], dm_dmT[pi0:pi1, pj0:pj1])
                dmudA_dnudB = contract("dig,Dig->dDg", dmu_dr[:, pi0:pi1, :], dm_dot_dmudr_ij)
                d2e[i_atom, j_atom, :, :] += dmudA_dnudB @ weight_depsilon_drho

                if with_nablarho:
                    # d2mu/(dr dA) * dnu/dB, A orbital, B orbital
                    d2mudAdr_dnudB = contract("dxig,Dig->dDxg", d2mu_dr2[:, :, pi0:pi1, :], dm_dot_dmudr_ij)
                    d2e_ij_AB = contract("dDxg,xg->dD", d2mudAdr_dnudB, weight_depsilon_dnablarho)
                    d2mudAdr_dnudB = None
                    d2e[i_atom, j_atom, :, :] += d2e_ij_AB
                    d2e[j_atom, i_atom, :, :] += d2e_ij_AB.T
                    d2e_ij_AB = None
                dm_dot_dmudr_ij = None

                if with_tau:
                    pj0, pj1 = aoslices[j_atom][2:]
                    # d2mu/(dA dr) * d2nu/(dB dr), A orbital, B orbital
                    dm_dot_d2mudr2_ij = contract("dDjg,ij->dDig", d2mu_dr2[:, :, pj0:pj1, :], dm_dmT[pi0:pi1, pj0:pj1])
                    d2mudAdr_d2nudBdr = contract("dxig,Dxig->dDg", d2mu_dr2[:, :, pi0:pi1, :], dm_dot_d2mudr2_ij)
                    dm_dot_d2mudr2_ij = None
                    d2e[i_atom, j_atom, :, :] += 0.5 * d2mudAdr_d2nudBdr @ weight_depsilon_dtau
                    d2mudAdr_d2nudBdr = None

    if with_grid_response:
        buf_27_ngrids = cupy.empty(ngrids * 27)

        for i_atom in range(natm):
            g_i_with_response = atom_to_grid_index_map[i_atom]
            if len(g_i_with_response) == 0:
                continue

            weight_depsilon_drho_grid_i = weight_depsilon_drho[g_i_with_response]
            if with_nablarho:
                weight_depsilon_dnablarho_grid_i = weight_depsilon_dnablarho[:, g_i_with_response]
            if with_tau:
                weight_depsilon_dtau_grid_i = weight_depsilon_dtau[g_i_with_response]
            dmu_dr_grid_i = dmu_dr[:, :, g_i_with_response]
            d2mu_dr2_grid_i = d2mu_dr2[:, :, :, g_i_with_response]
            if with_nablarho or with_tau:
                d3mu_dr3_grid_i = d3mu_dr3[:, :, :, :, g_i_with_response]
            ngrids_i = len(g_i_with_response)

            d2rhodGdG = cupy.ndarray([3, 3, ngrids_i], memptr = buf_27_ngrids.data)
            # d2mu/dr2 * nu, A grid, B grid
            dm_dot_mu_i = dm_dot_mu[:, g_i_with_response]
            contract("dDig,ig->dDg", d2mu_dr2_grid_i, dm_dot_mu_i, beta = 0.0, out = d2rhodGdG)
            # dmu/dr * dnu/dr, A grid, B grid
            dm_dot_dmudr_i = dm_dot_dmudr[:, :, g_i_with_response]
            contract("dig,Dig->dDg", dmu_dr_grid_i, dm_dot_dmudr_i, beta = 1.0, out = d2rhodGdG)
            d2e[i_atom, i_atom, :, :] += d2rhodGdG @ weight_depsilon_drho_grid_i
            d2rhodGdG = None

            if with_nablarho:
                d2nablarhodGdG = cupy.ndarray([3, 3, 3, ngrids_i], memptr = buf_27_ngrids.data)

                # d3mu/(dr dA dB) * nu, A grid, B grid
                contract("dDxig,ig->dDxg", d3mu_dr3_grid_i, dm_dot_mu_i, beta = 0.0, out = d2nablarhodGdG)
                # d2mu/(dA dB) * dnu/dr, A grid, B grid
                contract("dDig,xig->dDxg", d2mu_dr2_grid_i, dm_dot_dmudr_i, beta = 1.0, out = d2nablarhodGdG)
                d2e[i_atom, i_atom, :, :] += contract("dDxg,xg->dD", d2nablarhodGdG, weight_depsilon_dnablarho_grid_i)

                # d2mu/(dr dA) * dnu/dB, A grid, B grid
                contract("dxig,Dig->dDxg", d2mu_dr2_grid_i, dm_dot_dmudr_i, beta = 0.0, out = d2nablarhodGdG)
                d2e_ii_AA = contract("dDxg,xg->dD", d2nablarhodGdG, weight_depsilon_dnablarho_grid_i)
                d2nablarhodGdG = None
                d2e[i_atom, i_atom, :, :] += d2e_ii_AA + d2e_ii_AA.T
                d2e_ii_AA = None

                dm_dot_d2mudr2_i = dm_dot_d2mudr2[:, :, :, g_i_with_response]

            if with_tau:
                d2taudGdG = cupy.ndarray([3, 3, ngrids_i], memptr = buf_27_ngrids.data)
                # d3mu/(dA dB dr) * dnu/dr, A grid, B grid
                contract("dDxig,xig->dDg", d3mu_dr3_grid_i, dm_dot_dmudr_i, beta = 0.0, out = d2taudGdG)
                # d2mu/(dA dr) * d2nu/(dB dr), A grid, B grid
                contract("dxig,Dxig->dDg", d2mu_dr2_grid_i, dm_dot_d2mudr2_i, beta = 1.0, out = d2taudGdG)
                d2e[i_atom, i_atom, :, :] += 0.5 * d2taudGdG @ weight_depsilon_dtau_grid_i
                d2taudGdG = None

            for j_atom in range(natm):
                pj0, pj1 = aoslices[j_atom][2:]
                d2rhodAdG = cupy.ndarray([3, 3, ngrids_i], memptr = buf_27_ngrids.data)
                # d2mu/dr2 * nu, A orbital, B grid
                dm_dot_mu_ji = dm_dot_mu_i[pj0:pj1, :]
                contract("dDig,ig->dDg", d2mu_dr2_grid_i[:, :, pj0:pj1, :], dm_dot_mu_ji, beta = 0.0, out = d2rhodAdG)
                # dmu/dr * dnu/dr, A orbital, B grid
                dm_dot_dmudr_ji = dm_dot_dmudr_i[:, pj0:pj1, :]
                contract("dig,Dig->dDg", dmu_dr_grid_i[:, pj0:pj1, :], dm_dot_dmudr_ji, beta = 1.0, out = d2rhodAdG)
                d2e_ji_AB = d2rhodAdG @ weight_depsilon_drho_grid_i
                d2rhodAdG = None

                if with_nablarho:
                    d2nablarhodAdG = cupy.ndarray([3, 3, 3, ngrids_i], memptr = buf_27_ngrids.data)
                    # d2mu/(dA dB) * dnu/dr, A orbital, B grid
                    contract("dDig,xig->dDxg", d2mu_dr2_grid_i[:, :, pj0:pj1, :], dm_dot_dmudr_ji, beta = 0.0, out = d2nablarhodAdG)
                    # d2mu/(dr dA) * dnu/dB, A orbital, B grid
                    d2nablarhodAdG += d2nablarhodAdG.transpose(0,2,1,3)
                    # d3mu/(dr dA dB) * nu, A orbital, B grid
                    contract("dDxig,ig->dDxg", d3mu_dr3_grid_i[:, :, :, pj0:pj1, :], dm_dot_mu_ji, beta = 1.0, out = d2nablarhodAdG)
                    # d2mu/(dr dA) * dnu/dB, A grid, B orbital
                    dm_dot_d2mudr2_ij = dm_dot_d2mudr2_i[:, :, pj0:pj1, :]
                    contract("dig,Dxig->dDxg", dmu_dr_grid_i[:, pj0:pj1, :], dm_dot_d2mudr2_ij, beta = 1.0, out = d2nablarhodAdG)

                    d2e_ji_AB += contract("dDxg,xg->dD", d2nablarhodAdG, weight_depsilon_dnablarho_grid_i)
                    d2nablarhodAdG = None
                dm_dot_mu_ji = None

                if with_tau:
                    d2taudAdG = cupy.ndarray([3, 3, ngrids_i], memptr = buf_27_ngrids.data)
                    # d3mu/(dA dB dr) * dnu/dr, A orbital, B grid
                    contract("dDxig,xig->dDg", d3mu_dr3_grid_i[:, :, :, pj0:pj1, :], dm_dot_dmudr_ji, beta = 0.0, out = d2taudAdG)
                    # d2mu/(dA dr) * d2nu/(dB dr), A orbital, B grid
                    contract("dxig,Dxig->dDg", d2mu_dr2_grid_i[:, :, pj0:pj1, :], dm_dot_d2mudr2_ij, beta = 1.0, out = d2taudAdG)
                    d2e_ji_AB += 0.5 * d2taudAdG @ weight_depsilon_dtau_grid_i
                    d2taudAdG = None

                dm_dot_dmudr_ji = None
                dm_dot_d2mudr2_ij = None

                # Why is there a transpose for this equation? Because we're using j index at where it supposes to be i.
                d2e[i_atom, j_atom, :, :] -= d2e_ji_AB.T
                d2e[j_atom, i_atom, :, :] -= d2e_ji_AB
                d2e_ji_AB = None

            weight_depsilon_drho_grid_i = None
            if with_nablarho:
                weight_depsilon_dnablarho_grid_i = None
            if with_tau:
                weight_depsilon_dtau_grid_i = None
            dm_dot_mu_i = None
            dm_dot_dmudr_i = None
            dm_dot_d2mudr2_i = None
            dmu_dr_grid_i = None
            d2mu_dr2_grid_i = None
            if with_nablarho or with_tau:
                d3mu_dr3_grid_i = None

    return d2e

def _get_exc_deriv2_grid_response(hessobj, mo_coeff, mo_occ, max_memory):
    """
        xc energy 2nd derivative grid response contribution
    """

    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)

    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build()
    ngrids = grids.coords.shape[0]

    natm = mol.natm
    aoslices = mol.aoslice_by_atom()

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    d2e = cupy.zeros([mol.natm, mol.natm, 3, 3])

    if xctype == 'LDA':
        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
        ao_nbytes_per_grid = ((1) * mol.nao + (9) * mol.natm * mol.natm + 2) * 8
        ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
        if ngrids_per_batch < 16:
            raise MemoryError(f"Out of GPU memory for LDA energy second derivative, available gpu memory = {get_avail_mem()}"
                              f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids = {ngrids}")
        ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
        ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

        for g0 in range(0, ngrids, ngrids_per_batch):
            g1 = min(g0 + ngrids_per_batch, ngrids)
            split_grids_coords = grids.coords[g0:g1, :]
            split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 0, gdftopt = None, transpose = False)
            rho = numint.eval_rho2(mol, split_ao, mo_coeff, mo_occ, xctype=xctype)
            exc = ni.eval_xc_eff(mf.xc, rho, deriv = 0, xctype=xctype)[0]

            epsilon = exc[:, 0] * rho
            rho = None
            exc = None

            d2w_dAdB = get_d2weight_dAdB(mol, grids, (g0,g1))
            d2e += contract("ABdDg,g->ABdD", d2w_dAdB, epsilon)
            d2w_dAdB = None
            epsilon = None

        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
        ao_nbytes_per_grid = ((10 + 9 + 2 + 4*4) * mol.nao + (3*3 + 3) * mol.natm + 3 + 1 + 18*4) * 8
        ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
        if ngrids_per_batch < 16:
            raise MemoryError(f"Out of GPU memory for LDA energy second derivative, available gpu memory = {get_avail_mem()}"
                                f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids = {ngrids}")
        ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
        ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

        for g0 in range(0, ngrids, ngrids_per_batch):
            g1 = min(g0 + ngrids_per_batch, ngrids)
            split_grids_coords = grids.coords[g0:g1, :]
            split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2, gdftopt = None, transpose = False)

            mu = split_ao[0]
            dmu_dr = split_ao[1:4]
            d2mu_dr2 = get_d2mu_dr2(split_ao)

            rho = numint.eval_rho2(mol, mu, mo_coeff, mo_occ, xctype=xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype=xctype)[1:3]
            rho = None

            depsilon_drho = vxc[0] # Just of shape (ngrids,)
            d2epsilon_drho2 = fxc[0,0] # Just of shape (ngrids,)

            grid_to_atom_index_map = grids.atm_idx[g0:g1]
            atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]
            grid_to_atom_index_map = None

            drho_dA_orbital_response, drho_dA_grid_response = \
                get_drho_dA_full(dm0, xctype, natm, g1 - g0, aoslices, atom_to_grid_index_map, mu, dmu_dr)

            drho_dA_full = drho_dA_orbital_response + drho_dA_grid_response

            dw_dA = get_dweight_dA(mol, grids, (g0,g1))
            # d2e += cupy.einsum("Adg,g,BDg->ABdD", dw_dA, depsilon_drho, drho_dA_full)
            # d2e += cupy.einsum("Adg,g,BDg->BADd", dw_dA, depsilon_drho, drho_dA_full)
            d2e_dwdA_term = contract("Adg,BDg->ABdD", dw_dA, drho_dA_full * depsilon_drho)
            dw_dA = None
            drho_dA_full = None
            d2e += d2e_dwdA_term + d2e_dwdA_term.transpose(1,0,3,2)
            d2e_dwdA_term = None

            weight = grids.weights[g0:g1]
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_drho2, drho_dA_orbital_response, drho_dA_grid_response)
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_drho2, drho_dA_grid_response, drho_dA_orbital_response)
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_drho2, drho_dA_grid_response, drho_dA_grid_response)
            drhodA_grid_response_weight_d2epsilondrho2 = drho_dA_grid_response * (weight * d2epsilon_drho2)
            d2e_drhodA_cross_term = contract("Adg,BDg->ABdD", drho_dA_orbital_response, drhodA_grid_response_weight_d2epsilondrho2)
            drho_dA_orbital_response = None
            d2e_drhodA_cross_term += d2e_drhodA_cross_term.transpose(1,0,3,2)
            d2e_drhodA_cross_term += contract("Adg,BDg->ABdD", drho_dA_grid_response, drhodA_grid_response_weight_d2epsilondrho2)
            drho_dA_grid_response = None
            drhodA_grid_response_weight_d2epsilondrho2 = None
            d2e += d2e_drhodA_cross_term
            d2e_drhodA_cross_term = None

            d2e += contract_d2rho_dAdB_full(dm0, xctype, natm, g1 - g0, aoslices, atom_to_grid_index_map,
                                            mu, dmu_dr, d2mu_dr2, None,
                                            depsilon_drho * weight,
                                            with_orbital_response = False)

    elif xctype == 'GGA':
        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
        ao_nbytes_per_grid = ((4) * mol.nao + (9) * mol.natm * mol.natm + 4 + 1*2) * 8
        ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
        if ngrids_per_batch < 16:
            raise MemoryError(f"Out of GPU memory for GGA energy second derivative, available gpu memory = {get_avail_mem()}"
                                f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids = {ngrids}")
        ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
        ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

        for g0 in range(0, ngrids, ngrids_per_batch):
            g1 = min(g0 + ngrids_per_batch, ngrids)
            split_grids_coords = grids.coords[g0:g1, :]
            split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 1, gdftopt = None, transpose = False)

            rho_drho = numint.eval_rho2(mol, split_ao, mo_coeff, mo_occ, xctype=xctype)
            exc = ni.eval_xc_eff(mf.xc, rho_drho, deriv = 0, xctype=xctype)[0]

            rho = rho_drho[0]
            rho_drho = None

            epsilon = exc[:, 0] * rho
            rho = None
            exc = None

            d2w_dAdB = get_d2weight_dAdB(mol, grids, (g0,g1))
            d2e += contract("ABdDg,g->ABdD", d2w_dAdB, epsilon)
            d2w_dAdB = None
            epsilon = None

        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
        ao_nbytes_per_grid = ((20 + 9 + 27 + 2 + 3*2 + 4*4 + 12*4) * mol.nao
                              + (3*3 + 9*3 + 3 + 2*3 + 4*2*3) * mol.natm + 4*2 + 16*2 + 18*4 + 27*2*4) * 8
        ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
        if ngrids_per_batch < 16:
            raise MemoryError(f"Out of GPU memory for GGA energy second derivative, available gpu memory = {get_avail_mem()}"
                                f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids = {ngrids}")
        ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
        ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

        for g0 in range(0, ngrids, ngrids_per_batch):
            g1 = min(g0 + ngrids_per_batch, ngrids)
            split_grids_coords = grids.coords[g0:g1, :]
            split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 3, gdftopt = None, transpose = False)

            mu = split_ao[0]
            dmu_dr = split_ao[1:4]
            d2mu_dr2 = get_d2mu_dr2(split_ao)
            d3mu_dr3 = get_d3mu_dr3(split_ao)

            rho_drho = numint.eval_rho2(mol, split_ao[:4], mo_coeff, mo_occ, xctype=xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho_drho, deriv = 2, xctype=xctype)[1:3]
            rho_drho = None

            depsilon_drho = vxc[0]
            depsilon_dnablarho = vxc[1:4]
            # d2epsilon_drho2 = fxc[0,0]
            # d2epsilon_drho_dnablarho = fxc[0,1:4]
            # d2epsilon_dnablarho2 = fxc[1:4,1:4]

            grid_to_atom_index_map = grids.atm_idx[g0:g1]
            atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]
            grid_to_atom_index_map = None

            drho_dA_orbital_response, dnablarho_dA_orbital_response, drho_dA_grid_response, dnablarho_dA_grid_response = \
                get_drho_dA_full(dm0, xctype, natm, g1 - g0, aoslices, atom_to_grid_index_map, mu, dmu_dr, d2mu_dr2)

            drho_dA_full = drho_dA_orbital_response + drho_dA_grid_response
            dnablarho_dA_full = dnablarho_dA_orbital_response + dnablarho_dA_grid_response

            dw_dA = get_dweight_dA(mol, grids, (g0,g1))
            # d2e += cupy.einsum("Adg,g,BDg->ABdD", dw_dA, depsilon_drho, drho_dA_full)
            # d2e += cupy.einsum("Adg,g,BDg->BADd", dw_dA, depsilon_drho, drho_dA_full)
            # d2e += cupy.einsum("Adg,xg,BDxg->ABdD", dw_dA, depsilon_dnablarho, dnablarho_dA_full)
            # d2e += cupy.einsum("Adg,xg,BDxg->BADd", dw_dA, depsilon_dnablarho, dnablarho_dA_full)
            depsilondnablarho_dnablarhodA = contract("xg,Adxg->Adg", depsilon_dnablarho, dnablarho_dA_full)
            dnablarho_dA_full = None
            d2e_dwdA_term = contract("Adg,BDg->ABdD", dw_dA, drho_dA_full * depsilon_drho + depsilondnablarho_dnablarhodA)
            drho_dA_full = None
            depsilondnablarho_dnablarhodA = None
            dw_dA = None
            d2e += d2e_dwdA_term + d2e_dwdA_term.transpose(1,0,3,2)
            d2e_dwdA_term = None

            weight = grids.weights[g0:g1]
            # # d2epsilon/drho2 * drho/dA * drho/dB
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_drho2, drho_dA_orbital_response, drho_dA_grid_response)
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_drho2, drho_dA_grid_response, drho_dA_orbital_response)
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_drho2, drho_dA_grid_response, drho_dA_grid_response)
            # # d2epsilon/(drho d_nabla_rho) * d_nabla_rho/dA * drho/dB
            # d2e += cupy.einsum("g,xg,Adg,BDxg->ABdD", weight, d2epsilon_drho_dnablarho, drho_dA_orbital_response, dnablarho_dA_grid_response)
            # d2e += cupy.einsum("g,xg,Adg,BDxg->ABdD", weight, d2epsilon_drho_dnablarho, drho_dA_grid_response, dnablarho_dA_orbital_response)
            # d2e += cupy.einsum("g,xg,Adg,BDxg->ABdD", weight, d2epsilon_drho_dnablarho, drho_dA_grid_response, dnablarho_dA_grid_response)
            # # d2epsilon/(drho d_nabla_rho) * drho/dA * d_nabla_rho/dB
            # d2e += cupy.einsum("g,xg,Adg,BDxg->BADd", weight, d2epsilon_drho_dnablarho, drho_dA_orbital_response, dnablarho_dA_grid_response)
            # d2e += cupy.einsum("g,xg,Adg,BDxg->BADd", weight, d2epsilon_drho_dnablarho, drho_dA_grid_response, dnablarho_dA_orbital_response)
            # d2e += cupy.einsum("g,xg,Adg,BDxg->BADd", weight, d2epsilon_drho_dnablarho, drho_dA_grid_response, dnablarho_dA_grid_response)
            # # d2epsilon/(d_nabla_rho d_nabla_rho) * d_nabla_rho/dA * d_nabla_rho/dB
            # d2e += cupy.einsum("g,xyg,Adxg,BDyg->ABdD", weight, d2epsilon_dnablarho2, dnablarho_dA_orbital_response, dnablarho_dA_grid_response)
            # d2e += cupy.einsum("g,xyg,Adxg,BDyg->ABdD", weight, d2epsilon_dnablarho2, dnablarho_dA_grid_response, dnablarho_dA_orbital_response)
            # d2e += cupy.einsum("g,xyg,Adxg,BDyg->ABdD", weight, d2epsilon_dnablarho2, dnablarho_dA_grid_response, dnablarho_dA_grid_response)
            combined_d_dA_orbital_response = cupy.concatenate((drho_dA_orbital_response[:, :, None, :], dnablarho_dA_orbital_response), axis = 2)
            combined_d_dA_grid_response    = cupy.concatenate((   drho_dA_grid_response[:, :, None, :],    dnablarho_dA_grid_response), axis = 2)
            drho_dA_orbital_response = None
            dnablarho_dA_orbital_response = None
            drho_dA_grid_response = None
            dnablarho_dA_grid_response = None
            fwxc = fxc * weight
            fxc = None

            drhodA_grid_response_fwxc = contract("xyg,Adyg->Adxg", fwxc, combined_d_dA_grid_response)
            fwxc = None
            d2e_drhodA_cross_term = contract("Adxg,BDxg->ABdD", combined_d_dA_orbital_response, drhodA_grid_response_fwxc)
            combined_d_dA_orbital_response = None
            d2e_drhodA_cross_term += d2e_drhodA_cross_term.transpose(1,0,3,2)
            d2e_drhodA_cross_term += contract("Adxg,BDxg->ABdD", combined_d_dA_grid_response, drhodA_grid_response_fwxc)
            combined_d_dA_grid_response = None
            drhodA_grid_response_fwxc = None
            d2e += d2e_drhodA_cross_term
            d2e_drhodA_cross_term = None

            d2e += contract_d2rho_dAdB_full(dm0, xctype, natm, g1 - g0, aoslices, atom_to_grid_index_map,
                                            mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                                            depsilon_drho * weight, depsilon_dnablarho * weight,
                                            with_orbital_response = False)

    elif xctype == 'MGGA':
        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
        ao_nbytes_per_grid = ((4) * mol.nao + (9) * mol.natm * mol.natm + 5 + 1*2) * 8
        ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
        if ngrids_per_batch < 16:
            raise MemoryError(f"Out of GPU memory for mGGA energy second derivative, available gpu memory = {get_avail_mem()}"
                                f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids = {ngrids}")
        ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
        ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

        for g0 in range(0, ngrids, ngrids_per_batch):
            g1 = min(g0 + ngrids_per_batch, ngrids)
            split_grids_coords = grids.coords[g0:g1, :]
            split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 1, gdftopt = None, transpose = False)

            rho_drho_tau = numint.eval_rho2(mol, split_ao, mo_coeff, mo_occ, xctype=xctype)
            exc = ni.eval_xc_eff(mf.xc, rho_drho_tau, deriv = 0, xctype=xctype)[0]

            rho = rho_drho_tau[0]
            rho_drho_tau = None

            epsilon = exc[:, 0] * rho
            rho = None
            exc = None

            d2w_dAdB = get_d2weight_dAdB(mol, grids, (g0,g1))
            d2e += contract("ABdDg,g->ABdD", d2w_dAdB, epsilon)
            d2w_dAdB = None
            epsilon = None

        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
        ao_nbytes_per_grid = ((20 + 9 + 27 + 2 + 3*2 + 4*4 + 12*4 + 9*4) * mol.nao
                              + (3*3 + 9*3 + 3*3 + 3 + 3*3 + 5*2*3) * mol.natm + 5*2 + 25*2 + 18*4 + 27*2*4 + 18*4) * 8
        ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
        if ngrids_per_batch < 16:
            raise MemoryError(f"Out of GPU memory for mGGA energy second derivative, available gpu memory = {get_avail_mem()}"
                                f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids = {ngrids}")
        ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
        ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

        for g0 in range(0, ngrids, ngrids_per_batch):
            g1 = min(g0 + ngrids_per_batch, ngrids)
            split_grids_coords = grids.coords[g0:g1, :]
            split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 3, gdftopt = None, transpose = False)

            mu = split_ao[0]
            dmu_dr = split_ao[1:4]
            d2mu_dr2 = get_d2mu_dr2(split_ao)
            d3mu_dr3 = get_d3mu_dr3(split_ao)

            rho_drho_tau = numint.eval_rho2(mol, split_ao[:4], mo_coeff, mo_occ, xctype=xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho_drho_tau, deriv = 2, xctype=xctype)[1:3]
            rho_drho_tau = None

            depsilon_drho = vxc[0]
            depsilon_dnablarho = vxc[1:4]
            depsilon_dtau = vxc[4]
            # d2epsilon_drho2 = fxc[0,0]
            # d2epsilon_drho_dnablarho = fxc[0,1:4]
            # d2epsilon_drho_dtau = fxc[0,4]
            # d2epsilon_dnablarho2 = fxc[1:4,1:4]
            # d2epsilon_dnablarho_dtau = fxc[1:4,4]
            # d2epsilon_dtau2 = fxc[4,4]

            grid_to_atom_index_map = grids.atm_idx[g0:g1]
            atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]
            grid_to_atom_index_map = None

            drho_dA_orbital_response, dnablarho_dA_orbital_response, dtau_dA_orbital_response, \
                drho_dA_grid_response, dnablarho_dA_grid_response, dtau_dA_grid_response = \
                    get_drho_dA_full(dm0, xctype, natm, g1 - g0, aoslices, atom_to_grid_index_map, mu, dmu_dr, d2mu_dr2)

            drho_dA_full = drho_dA_orbital_response + drho_dA_grid_response
            dnablarho_dA_full = dnablarho_dA_orbital_response + dnablarho_dA_grid_response
            dtau_dA_full = dtau_dA_orbital_response + dtau_dA_grid_response

            dw_dA = get_dweight_dA(mol, grids, (g0,g1))
            # d2e += cupy.einsum("Adg,g,BDg->ABdD", dw_dA, depsilon_drho, drho_dA_full)
            # d2e += cupy.einsum("Adg,g,BDg->BADd", dw_dA, depsilon_drho, drho_dA_full)
            # d2e += cupy.einsum("Adg,xg,BDxg->ABdD", dw_dA, depsilon_dnablarho, dnablarho_dA_full)
            # d2e += cupy.einsum("Adg,xg,BDxg->BADd", dw_dA, depsilon_dnablarho, dnablarho_dA_full)
            # d2e += cupy.einsum("Adg,g,BDg->ABdD", dw_dA, depsilon_dtau, dtau_dA_full)
            # d2e += cupy.einsum("Adg,g,BDg->BADd", dw_dA, depsilon_dtau, dtau_dA_full)
            depsilondnablarho_dnablarhodA = contract("xg,Adxg->Adg", depsilon_dnablarho, dnablarho_dA_full)
            dnablarho_dA_full = None
            d2e_dwdA_term = contract("Adg,BDg->ABdD", dw_dA, drho_dA_full * depsilon_drho + depsilondnablarho_dnablarhodA + dtau_dA_full * depsilon_dtau)
            drho_dA_full = None
            dtau_dA_full = None
            depsilondnablarho_dnablarhodA = None
            dw_dA = None
            d2e += d2e_dwdA_term + d2e_dwdA_term.transpose(1,0,3,2)
            d2e_dwdA_term = None

            weight = grids.weights[g0:g1]
            # # d2epsilon/drho2 * drho/dA * drho/dB
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_drho2, drho_dA_orbital_response, drho_dA_grid_response)
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_drho2, drho_dA_grid_response, drho_dA_orbital_response)
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_drho2, drho_dA_grid_response, drho_dA_grid_response)
            # # d2epsilon/(drho d_nabla_rho) * d_nabla_rho/dA * drho/dB
            # d2e += cupy.einsum("g,xg,Adg,BDxg->ABdD", weight, d2epsilon_drho_dnablarho, drho_dA_orbital_response, dnablarho_dA_grid_response)
            # d2e += cupy.einsum("g,xg,Adg,BDxg->ABdD", weight, d2epsilon_drho_dnablarho, drho_dA_grid_response, dnablarho_dA_orbital_response)
            # d2e += cupy.einsum("g,xg,Adg,BDxg->ABdD", weight, d2epsilon_drho_dnablarho, drho_dA_grid_response, dnablarho_dA_grid_response)
            # # d2epsilon/(drho d_nabla_rho) * drho/dA * d_nabla_rho/dB
            # d2e += cupy.einsum("g,xg,Adg,BDxg->BADd", weight, d2epsilon_drho_dnablarho, drho_dA_orbital_response, dnablarho_dA_grid_response)
            # d2e += cupy.einsum("g,xg,Adg,BDxg->BADd", weight, d2epsilon_drho_dnablarho, drho_dA_grid_response, dnablarho_dA_orbital_response)
            # d2e += cupy.einsum("g,xg,Adg,BDxg->BADd", weight, d2epsilon_drho_dnablarho, drho_dA_grid_response, dnablarho_dA_grid_response)
            # # d2epsilon/(d_nabla_rho d_nabla_rho) * d_nabla_rho/dA * d_nabla_rho/dB
            # d2e += cupy.einsum("g,xyg,Adxg,BDyg->ABdD", weight, d2epsilon_dnablarho2, dnablarho_dA_orbital_response, dnablarho_dA_grid_response)
            # d2e += cupy.einsum("g,xyg,Adxg,BDyg->ABdD", weight, d2epsilon_dnablarho2, dnablarho_dA_grid_response, dnablarho_dA_orbital_response)
            # d2e += cupy.einsum("g,xyg,Adxg,BDyg->ABdD", weight, d2epsilon_dnablarho2, dnablarho_dA_grid_response, dnablarho_dA_grid_response)
            # # d2epsilon/(drho dtau) * dtau/dA * drho/dB
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_drho_dtau, drho_dA_orbital_response, dtau_dA_grid_response)
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_drho_dtau, drho_dA_grid_response, dtau_dA_orbital_response)
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_drho_dtau, drho_dA_grid_response, dtau_dA_grid_response)
            # # d2epsilon/(drho dtau) * drho/dA * dtau/dB
            # d2e += cupy.einsum("g,g,Adg,BDg->BADd", weight, d2epsilon_drho_dtau, drho_dA_orbital_response, dtau_dA_grid_response)
            # d2e += cupy.einsum("g,g,Adg,BDg->BADd", weight, d2epsilon_drho_dtau, drho_dA_grid_response, dtau_dA_orbital_response)
            # d2e += cupy.einsum("g,g,Adg,BDg->BADd", weight, d2epsilon_drho_dtau, drho_dA_grid_response, dtau_dA_grid_response)
            # # d2epsilon/(d_nabla_rho dtau) * dtau/dA * d_nabla_rho/dB
            # d2e += cupy.einsum("g,xg,Adxg,BDg->ABdD", weight, d2epsilon_dnablarho_dtau, dnablarho_dA_orbital_response, dtau_dA_grid_response)
            # d2e += cupy.einsum("g,xg,Adxg,BDg->ABdD", weight, d2epsilon_dnablarho_dtau, dnablarho_dA_grid_response, dtau_dA_orbital_response)
            # d2e += cupy.einsum("g,xg,Adxg,BDg->ABdD", weight, d2epsilon_dnablarho_dtau, dnablarho_dA_grid_response, dtau_dA_grid_response)
            # # d2epsilon/(d_nabla_rho dtau) * d_nabla_rho/dA * dtau/dB
            # d2e += cupy.einsum("g,xg,Adxg,BDg->BADd", weight, d2epsilon_dnablarho_dtau, dnablarho_dA_orbital_response, dtau_dA_grid_response)
            # d2e += cupy.einsum("g,xg,Adxg,BDg->BADd", weight, d2epsilon_dnablarho_dtau, dnablarho_dA_grid_response, dtau_dA_orbital_response)
            # d2e += cupy.einsum("g,xg,Adxg,BDg->BADd", weight, d2epsilon_dnablarho_dtau, dnablarho_dA_grid_response, dtau_dA_grid_response)
            # # d2epsilon/dtau2 * dtau/dA * dtau/dB
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_dtau2, dtau_dA_orbital_response, dtau_dA_grid_response)
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_dtau2, dtau_dA_grid_response, dtau_dA_orbital_response)
            # d2e += cupy.einsum("g,g,Adg,BDg->ABdD", weight, d2epsilon_dtau2, dtau_dA_grid_response, dtau_dA_grid_response)
            combined_d_dA_orbital_response = cupy.concatenate(
                (drho_dA_orbital_response[:, :, None, :], dnablarho_dA_orbital_response, dtau_dA_orbital_response[:, :, None, :]),
                axis = 2
            )
            combined_d_dA_grid_response    = cupy.concatenate(
                (   drho_dA_grid_response[:, :, None, :],    dnablarho_dA_grid_response,    dtau_dA_grid_response[:, :, None, :]),
                axis = 2
            )
            drho_dA_orbital_response = None
            dnablarho_dA_orbital_response = None
            dtau_dA_orbital_response = None
            drho_dA_grid_response = None
            dnablarho_dA_grid_response = None
            dtau_dA_grid_response = None
            fwxc = fxc * weight
            fxc = None

            drhodA_grid_response_fwxc = contract("xyg,Adyg->Adxg", fwxc, combined_d_dA_grid_response)
            fwxc = None
            d2e_drhodA_cross_term = contract("Adxg,BDxg->ABdD", combined_d_dA_orbital_response, drhodA_grid_response_fwxc)
            combined_d_dA_orbital_response = None
            d2e_drhodA_cross_term += d2e_drhodA_cross_term.transpose(1,0,3,2)
            d2e_drhodA_cross_term += contract("Adxg,BDxg->ABdD", combined_d_dA_grid_response, drhodA_grid_response_fwxc)
            combined_d_dA_grid_response = None
            drhodA_grid_response_fwxc = None
            d2e += d2e_drhodA_cross_term
            d2e_drhodA_cross_term = None

            d2e += contract_d2rho_dAdB_full(dm0, xctype, natm, g1 - g0, aoslices, atom_to_grid_index_map,
                                            mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                                            depsilon_drho * weight, depsilon_dnablarho * weight, depsilon_dtau * weight,
                                            with_orbital_response = False)

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
