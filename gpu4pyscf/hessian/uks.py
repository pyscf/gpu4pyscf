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

'''
Non-relativistic UKS analytical Hessian
'''


import cupy
import numpy
import cupy as cp
from pyscf import lib
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.hessian import uhf as uhf_hess
from gpu4pyscf.hessian.rhf import _ao2mo
from gpu4pyscf.hessian.rks import get_dweight_dA, get_d2weight_dAdB, get_d2mu_dr2, get_d3mu_dr3
from gpu4pyscf.hessian.rks import _get_enlc_deriv2, _get_vnlc_deriv1, nr_rks_fnlc_mo
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import (contract, add_sparse, get_avail_mem, take_last2d)
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import min_grid_blksize

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

    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = mocca.dot(mocca.T)
    dm0b = moccb.dot(moccb.T)
    dm0 = cp.asarray((dm0a, dm0b))

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
    de2, ejk = uhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
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

    t1 = log.timer_debug1('hessian of JK part', *t1)

    de2 += _get_exc_deriv2(hessobj, mo_coeff, mo_occ, (dm0a, dm0b), max_memory, atmlst, log)
    if mf.do_nlc():
        de2 += _get_enlc_deriv2(hessobj, mo_coeff, mo_occ, max_memory, log)

    t1 = log.timer_debug1('hessian of XC part', *t1)
    log.timer('UKS partial hessian', *time0)
    return de2

def _get_exc_deriv2(hessobj, mo_coeff, mo_occ, dm0, max_memory, atmlst = None, log = None):
    if log is None:
        log = logger.new_logger(hessobj)

    if hessobj.grid_response:
        log.info("Calculating grid response for unrestricted DFT Hessian")
        return _get_exc_deriv2_grid_response(hessobj, mo_coeff, mo_occ, max_memory)

    mol = hessobj.mol
    mf = hessobj.base
    dm0a = dm0[0]
    dm0b = dm0[1]

    de2 = cupy.zeros([mol.natm, mol.natm, 3, 3])

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    veffa_diag, veffb_diag = _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)

    aoslices = mol.aoslice_by_atom()
    vxca_dm, vxcb_dm = _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)
    if atmlst is None:
        atmlst = range(mol.natm)
    for i0, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        veffa_dm = vxca_dm[ia]
        veffb_dm = vxcb_dm[ia]
        de2[i0,i0] += contract('xypq,pq->xy', veffa_diag[:,:,p0:p1], dm0a[p0:p1])*2
        de2[i0,i0] += contract('xypq,pq->xy', veffb_diag[:,:,p0:p1], dm0b[p0:p1])*2
        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            de2[i0,j0] += 2.0 * veffa_dm[:,:,q0:q1].sum(axis=2)
            de2[i0,j0] += 2.0 * veffb_dm[:,:,q0:q1].sum(axis=2)
        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    return de2

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    mf = hessobj.base
    natm = mol.natm
    assert atmlst is None or atmlst == range(natm)
    mo_a, mo_b = mo_coeff
    mocca = mo_a[:,mo_occ[0]>0]
    moccb = mo_b[:,mo_occ[1]>0]
    dm0a = mocca.dot(mocca.T)
    dm0b = moccb.dot(moccb.T)
    avail_mem = get_avail_mem()
    max_memory = avail_mem * .8e-6

    h1moa, h1mob = _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    if mf.do_nlc():
        h1moa_nlc, h1mob_nlc = _get_vnlc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
        h1moa += h1moa_nlc
        h1mob += h1mob_nlc

    grad_obj = hessobj.base.Gradients()
    h1moa += rhf_grad.get_grad_hcore(grad_obj, mo_a, mo_occ[0])
    h1mob += rhf_grad.get_grad_hcore(grad_obj, mo_b, mo_occ[1])

    mf = hessobj.base
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)

    # Estimate the size of intermediate variables
    # dm, vj, and vk in [natm,3,nao_cart,nao_cart]
    nao_cart = mol.nao_cart()
    avail_mem -= 8 * (h1moa.size + h1mob.size)
    slice_size = int(avail_mem*0.5) // (8*3*nao_cart*nao_cart*6)
    for atoms_slice in lib.prange(0, natm, slice_size):
        vja, vka = rhf_hess._get_jk_ip1(mol, dm0a, with_k=with_k, atoms_slice=atoms_slice, verbose=verbose)
        vjb, vkb = rhf_hess._get_jk_ip1(mol, dm0b, with_k=with_k, atoms_slice=atoms_slice, verbose=verbose)
        vj = vja + vjb
        if with_k:
            #:veffa = vja + vjb - hyb * vka
            #:veffb = vja + vjb - hyb * vkb
            veffa = vka
            veffb = vkb
            veffa *= -hyb
            veffb *= -hyb
            veffa += vj
            veffb += vj
        else:
            veffa = vj
            veffb = vj.copy()
        vj = vja = vjb = vka = vkb = None
        if abs(omega) > 1e-10 and abs(alpha-hyb) > 1e-10:
            with mol.with_range_coulomb(omega):
                vka_lr = rhf_hess._get_jk_ip1(mol, dm0a, with_j=False, atoms_slice=atoms_slice, verbose=verbose)[1]
                vkb_lr = rhf_hess._get_jk_ip1(mol, dm0b, with_j=False, atoms_slice=atoms_slice, verbose=verbose)[1]
                vka_lr *= (alpha-hyb)
                vkb_lr *= (alpha-hyb)
                veffa -= vka_lr
                veffb -= vkb_lr
                vka_lr = vkb_lr = None

        atom0, atom1 = atoms_slice
        for i, ia in enumerate(range(atom0, atom1)):
            for ix in range(3):
                h1moa[ia,ix] += mo_a.T.dot(veffa[i,ix].dot(mocca))
                h1mob[ia,ix] += mo_b.T.dot(veffb[i,ix].dot(moccb))
        veffa = veffb = None
    return h1moa, h1mob

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

    nao_sph, nmo = mo_coeff[0].shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol

    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])
    nao = mo_coeff.shape[1]
    # TODO: check mol in opt?
    vmata = cupy.zeros((6,nao,nao))
    vmatb = cupy.zeros((6,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            mo_coeff_mask = mo_coeff[:,mask,:]
            rhoa = numint.eval_rho2(_sorted_mol, ao[0], mo_coeff_mask[0], mo_occ[0], mask, xctype)
            rhob = numint.eval_rho2(_sorted_mol, ao[0], mo_coeff_mask[1], mo_occ[1], mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 1, xctype=xctype, spin=1)[1]
            wv = weight * vxc[:,0]
            aowa = numint._scale_ao(ao[0], wv[0])
            aowb = numint._scale_ao(ao[0], wv[1])
            for i in range(6):
                vmat_tmp = numint._dot_ao_ao(mol, ao[i+4], aowa, mask, shls_slice, ao_loc)
                add_sparse(vmata[i], vmat_tmp, mask)
                vmat_tmp = numint._dot_ao_ao(mol, ao[i+4], aowb, mask, shls_slice, ao_loc)
                add_sparse(vmatb[i], vmat_tmp, mask)
            aowa = aowb = None

    elif xctype == 'GGA':
        def contract_(ao, aoidx, wv, mask):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1])
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2])
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3])
            return numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            mo_coeff_mask = mo_coeff[:,mask,:]
            rhoa = numint.eval_rho2(_sorted_mol, ao[:4], mo_coeff_mask[0], mo_occ[0], mask, xctype)
            rhob = numint.eval_rho2(_sorted_mol, ao[:4], mo_coeff_mask[1], mo_occ[1], mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 1, xctype=xctype, spin=1)[1]
            wv = weight * vxc
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
            aowa = numint._scale_ao(ao[:4], wv[0,:4])
            aowb = numint._scale_ao(ao[:4], wv[1,:4])
            vmata_tmp = [0]*6
            vmatb_tmp = [0]*6
            for i in range(6):
                vmata_tmp[i] = numint._dot_ao_ao(mol, ao[i+4], aowa, mask, shls_slice, ao_loc)
                vmatb_tmp[i] = numint._dot_ao_ao(mol, ao[i+4], aowb, mask, shls_slice, ao_loc)
            vmata_tmp[0] += contract_(ao, [XXX,XXY,XXZ], wv[0], mask)
            vmata_tmp[1] += contract_(ao, [XXY,XYY,XYZ], wv[0], mask)
            vmata_tmp[2] += contract_(ao, [XXZ,XYZ,XZZ], wv[0], mask)
            vmata_tmp[3] += contract_(ao, [XYY,YYY,YYZ], wv[0], mask)
            vmata_tmp[4] += contract_(ao, [XYZ,YYZ,YZZ], wv[0], mask)
            vmata_tmp[5] += contract_(ao, [XZZ,YZZ,ZZZ], wv[0], mask)
            vmatb_tmp[0] += contract_(ao, [XXX,XXY,XXZ], wv[1], mask)
            vmatb_tmp[1] += contract_(ao, [XXY,XYY,XYZ], wv[1], mask)
            vmatb_tmp[2] += contract_(ao, [XXZ,XYZ,XZZ], wv[1], mask)
            vmatb_tmp[3] += contract_(ao, [XYY,YYY,YYZ], wv[1], mask)
            vmatb_tmp[4] += contract_(ao, [XYZ,YYZ,YZZ], wv[1], mask)
            vmatb_tmp[5] += contract_(ao, [XZZ,YZZ,ZZZ], wv[1], mask)
            for i in range(6):
                add_sparse(vmata[i], vmata_tmp[i], mask)
                add_sparse(vmatb[i], vmatb_tmp[i], mask)
            rhoa = rhob = vxc = wv = aowa = aowb = None
    elif xctype == 'MGGA':
        def contract_(ao, aoidx, wv, mask):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1])
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2])
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3])
            return numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            mo_coeff_mask = mo_coeff[:,mask,:]
            rhoa = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff_mask[0], mo_occ[0], mask, xctype)
            rhob = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff_mask[1], mo_occ[1], mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 1, xctype=xctype, spin=1)[1]
            wv = weight * vxc
            wv[:,4] *= .5  # for the factor 1/2 in tau
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
            vmata_tmp = [0]*6
            vmatb_tmp = [0]*6
            aowa = numint._scale_ao(ao[:4], wv[0,:4])
            aowb = numint._scale_ao(ao[:4], wv[1,:4])
            for i in range(6):
                vmata_tmp[i] = numint._dot_ao_ao(mol, ao[i+4], aowa, mask, shls_slice, ao_loc)
                vmatb_tmp[i] = numint._dot_ao_ao(mol, ao[i+4], aowb, mask, shls_slice, ao_loc)
            vmata_tmp[0] += contract_(ao, [XXX,XXY,XXZ], wv[0], mask)
            vmata_tmp[1] += contract_(ao, [XXY,XYY,XYZ], wv[0], mask)
            vmata_tmp[2] += contract_(ao, [XXZ,XYZ,XZZ], wv[0], mask)
            vmata_tmp[3] += contract_(ao, [XYY,YYY,YYZ], wv[0], mask)
            vmata_tmp[4] += contract_(ao, [XYZ,YYZ,YZZ], wv[0], mask)
            vmata_tmp[5] += contract_(ao, [XZZ,YZZ,ZZZ], wv[0], mask)
            vmatb_tmp[0] += contract_(ao, [XXX,XXY,XXZ], wv[1], mask)
            vmatb_tmp[1] += contract_(ao, [XXY,XYY,XYZ], wv[1], mask)
            vmatb_tmp[2] += contract_(ao, [XXZ,XYZ,XZZ], wv[1], mask)
            vmatb_tmp[3] += contract_(ao, [XYY,YYY,YYZ], wv[1], mask)
            vmatb_tmp[4] += contract_(ao, [XYZ,YYZ,YZZ], wv[1], mask)
            vmatb_tmp[5] += contract_(ao, [XZZ,YZZ,ZZZ], wv[1], mask)
            aowa = [numint._scale_ao(ao[i], wv[0,4]) for i in range(1, 4)]
            aowb = [numint._scale_ao(ao[i], wv[1,4]) for i in range(1, 4)]
            for i, j in enumerate([XXX, XXY, XXZ, XYY, XYZ, XZZ]):
                vmata_tmp[i] += numint._dot_ao_ao(mol, ao[j], aowa[0], mask, shls_slice, ao_loc)
                vmatb_tmp[i] += numint._dot_ao_ao(mol, ao[j], aowb[0], mask, shls_slice, ao_loc)

            for i, j in enumerate([XXY, XYY, XYZ, YYY, YYZ, YZZ]):
                vmata_tmp[i] += numint._dot_ao_ao(mol, ao[j], aowa[1], mask, shls_slice, ao_loc)
                vmatb_tmp[i] += numint._dot_ao_ao(mol, ao[j], aowb[1], mask, shls_slice, ao_loc)

            for i, j in enumerate([XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ]):
                vmata_tmp[i] += numint._dot_ao_ao(mol, ao[j], aowa[2], mask, shls_slice, ao_loc)
                vmatb_tmp[i] += numint._dot_ao_ao(mol, ao[j], aowb[2], mask, shls_slice, ao_loc)
            for i in range(6):
                add_sparse(vmata[i], vmata_tmp[i], mask)
                add_sparse(vmatb[i], vmatb_tmp[i], mask)
    vmata = vmata[[0,1,2,
                 1,3,4,
                 2,4,5]]
    vmatb = vmatb[[0,1,2,
                 1,3,4,
                 2,4,5]]
    vmata = opt.unsort_orbitals(vmata, axis=[1,2])
    vmata = vmata.reshape(3,3,nao_sph,nao_sph)
    vmatb = opt.unsort_orbitals(vmatb, axis=[1,2])
    vmatb = vmatb.reshape(3,3,nao_sph,nao_sph)
    return vmata, vmatb

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
    shls_slice = (0, mol.nbas)
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
    '''Partially contracted vxc*dm'''
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

    nao, nmo = mo_coeff[0].shape
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
    dm0a, dm0b = mf.make_rdm1(mo_coeff, mo_occ)
    dm0a_sorted = opt.sort_orbitals(dm0a, axis=[0,1])
    dm0b_sorted = opt.sort_orbitals(dm0b, axis=[0,1])
    vmata_dm = cupy.zeros((mol.natm,3,3,nao))
    vmatb_dm = cupy.zeros((mol.natm,3,3,nao))
    ipipa = cupy.zeros((3,3,nao,nao))
    ipipb = cupy.zeros((3,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        t1 = log.init_timer()
        for ao_mask, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            nao_non0 = len(mask)
            ao = contract('nip,ij->njp', ao_mask, coeff[mask])
            rhoa = numint.eval_rho2(_sorted_mol, ao[0], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = numint.eval_rho2(_sorted_mol, ao[0], mo_coeff[1], mo_occ[1], mask, xctype)
            t1 = log.timer_debug2('eval rho', *t1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 2, xctype=xctype, spin=1)[1:3]
            t1 = log.timer_debug2('eval vxc', *t1)
            wv = weight * vxc[:,0]
            aowa = [numint._scale_ao(ao[i], wv[0]) for i in range(1, 4)]
            _d1d2_dot_(ipipa, mol, aowa, ao[1:4], mask, ao_loc, False)
            aowb = [numint._scale_ao(ao[i], wv[1]) for i in range(1, 4)]
            _d1d2_dot_(ipipb, mol, aowb, ao[1:4], mask, ao_loc, False)
            dm0a_mask = dm0a_sorted[mask[:,None], mask]
            dm0b_mask = dm0b_sorted[mask[:,None], mask]

            ao_dma_mask = contract('nig,ij->njg', ao_mask[:4], dm0a_mask)
            ao_dmb_mask = contract('nig,ij->njg', ao_mask[:4], dm0b_mask)

            ao_dm0a = numint._dot_ao_dm(mol, ao[0], dm0a, mask, shls_slice, ao_loc)
            ao_dm0b = numint._dot_ao_dm(mol, ao[0], dm0b, mask, shls_slice, ao_loc)
            wf = weight * fxc[:,0,:,0]
            for ia in range(_sorted_mol.natm):
                p0, p1 = aoslices[ia][2:]
                # *2 for \nabla|ket> in rho1
                rho1a = contract('xig,ig->xg', ao[1:,p0:p1,:], ao_dm0a[p0:p1,:]) * 2
                rho1b = contract('xig,ig->xg', ao[1:,p0:p1,:], ao_dm0b[p0:p1,:]) * 2
                # aow ~ rho1 ~ d/dR1
                wv = wf[0,:,None] * rho1a
                wv+= wf[1,:,None] * rho1b
                aow = cupy.empty_like(ao_dma_mask[1:4])
                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dma_mask[0], wv[0,i])
                vmata_dm[ia][:,:,mask] += contract('yjg,xjg->xyj', ao_mask[1:4], aow)
                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dmb_mask[0], wv[1,i])
                vmatb_dm[ia][:,:,mask] += contract('yjg,xjg->xyj', ao_mask[1:4], aow)
            ao_dm0a = ao_dm0b = aow = None
            t1 = log.timer_debug2('integration', *t1)
        vmata_dm = opt.unsort_orbitals(vmata_dm, axis=[3])
        vmatb_dm = opt.unsort_orbitals(vmatb_dm, axis=[3])
        for ia in range(_sorted_mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmata_dm[ia] += contract('xypq,pq->xyp', ipipa[:,:,:,p0:p1], dm0a[:,p0:p1])
            vmatb_dm[ia] += contract('xypq,pq->xyp', ipipb[:,:,:,p0:p1], dm0b[:,p0:p1])
    elif xctype == 'GGA':
        ao_deriv = 2
        t1 = log.init_timer()
        for ao_mask, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            nao_non0 = len(mask)
            ao = contract('nip,ij->njp', ao_mask, coeff[mask])
            rhoa = numint.eval_rho2(_sorted_mol, ao[:4], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = numint.eval_rho2(_sorted_mol, ao[:4], mo_coeff[1], mo_occ[1], mask, xctype)
            t1 = log.timer_debug2('eval rho', *t1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 2, xctype=xctype, spin=1)[1:3]
            t1 = log.timer_debug2('eval vxc', *t1)
            wv = weight * vxc
            wv[:,0] *= .5
            aow = rks_grad._make_dR_dao_w(ao, wv[0])
            _d1d2_dot_(ipipa, mol, aow, ao[1:4], mask, ao_loc, False)
            aow = rks_grad._make_dR_dao_w(ao, wv[1])
            _d1d2_dot_(ipipb, mol, aow, ao[1:4], mask, ao_loc, False)
            ao_dm0a = [numint._dot_ao_dm(mol, ao[i], dm0a, mask, shls_slice, ao_loc) for i in range(4)]
            ao_dm0b = [numint._dot_ao_dm(mol, ao[i], dm0b, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            dm0a_mask = dm0a_sorted[mask[:,None], mask]
            dm0b_mask = dm0b_sorted[mask[:,None], mask]
            ao_dma_mask = contract('nig,ij->njg', ao_mask[:4], dm0a_mask)
            ao_dmb_mask = contract('nig,ij->njg', ao_mask[:4], dm0b_mask)
            vmata_dm_tmp = cupy.empty([3,3,nao_non0])
            vmatb_dm_tmp = cupy.empty([3,3,nao_non0])
            for ia in range(_sorted_mol.natm):
                dR_rho1a = _make_dR_rho1(ao, ao_dm0a, ia, aoslices, xctype)
                dR_rho1b = _make_dR_rho1(ao, ao_dm0b, ia, aoslices, xctype)
                wv = contract('xbyg,sxg->bsyg', wf[0], dR_rho1a)
                wv+= contract('xbyg,sxg->bsyg', wf[1], dR_rho1b)
                wv[:,:,0] *= .5
                wva, wvb = wv
                for i in range(3):
                    aow = rks_grad._make_dR_dao_w(ao_mask, wva[i])
                    vmata_dm_tmp[i] = contract('xjg,jg->xj', aow, ao_dma_mask[0])
                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dma_mask[:4], wva[i,:4])
                vmata_dm_tmp += contract('yjg,xjg->xyj', ao_mask[1:4], aow)
                vmata_dm[ia][:,:,mask] += vmata_dm_tmp

                for i in range(3):
                    aow = rks_grad._make_dR_dao_w(ao_mask, wvb[i])
                    vmatb_dm_tmp[i] = contract('xjg,jg->xj', aow, ao_dmb_mask[0])
                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dmb_mask[:4], wvb[i,:4])
                vmatb_dm_tmp += contract('yjg,xjg->xyj', ao_mask[1:4], aow)
                vmatb_dm[ia][:,:,mask] += vmatb_dm_tmp
            ao_dm0a = ao_dm0b = aow = None
            t1 = log.timer_debug2('integration', *t1)
        vmata_dm = opt.unsort_orbitals(vmata_dm, axis=[3])
        vmatb_dm = opt.unsort_orbitals(vmatb_dm, axis=[3])
        for ia in range(_sorted_mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmata_dm[ia] += contract('xypq,pq->xyp', ipipa[:,:,:,p0:p1], dm0a[:,p0:p1])
            vmata_dm[ia] += contract('yxqp,pq->xyp', ipipa[:,:,p0:p1], dm0a[:,p0:p1])
            vmatb_dm[ia] += contract('xypq,pq->xyp', ipipb[:,:,:,p0:p1], dm0b[:,p0:p1])
            vmatb_dm[ia] += contract('yxqp,pq->xyp', ipipb[:,:,p0:p1], dm0b[:,p0:p1])
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
            rhoa = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff[1], mo_occ[1], mask, xctype)
            t1 = log.timer_debug2('eval rho', *t1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa, rhob)), 2, xctype=xctype, spin=1)[1:3]
            t1 = log.timer_debug2('eval vxc', *t1)
            wv = weight * vxc
            wv[:,0] *= .5
            wv[:,4] *= .25
            aow = rks_grad._make_dR_dao_w(ao, wv[0])
            _d1d2_dot_(ipipa, mol, aow, ao[1:4], mask, ao_loc, False)
            aow = rks_grad._make_dR_dao_w(ao, wv[1])
            _d1d2_dot_(ipipb, mol, aow, ao[1:4], mask, ao_loc, False)

            aow = [numint._scale_ao(ao[i], wv[0,4]) for i in range(4, 10)]
            _d1d2_dot_(ipipa, mol, [aow[0], aow[1], aow[2]], [ao[XX], ao[XY], ao[XZ]], mask, ao_loc, False)
            _d1d2_dot_(ipipa, mol, [aow[1], aow[3], aow[4]], [ao[YX], ao[YY], ao[YZ]], mask, ao_loc, False)
            _d1d2_dot_(ipipa, mol, [aow[2], aow[4], aow[5]], [ao[ZX], ao[ZY], ao[ZZ]], mask, ao_loc, False)
            aow = [numint._scale_ao(ao[i], wv[1,4]) for i in range(4, 10)]
            _d1d2_dot_(ipipb, mol, [aow[0], aow[1], aow[2]], [ao[XX], ao[XY], ao[XZ]], mask, ao_loc, False)
            _d1d2_dot_(ipipb, mol, [aow[1], aow[3], aow[4]], [ao[YX], ao[YY], ao[YZ]], mask, ao_loc, False)
            _d1d2_dot_(ipipb, mol, [aow[2], aow[4], aow[5]], [ao[ZX], ao[ZY], ao[ZZ]], mask, ao_loc, False)

            dm0a_mask = dm0a_sorted[mask[:,None], mask]
            dm0b_mask = dm0b_sorted[mask[:,None], mask]
            ao_dm0a = [numint._dot_ao_dm(mol, ao[i], dm0a, mask, shls_slice, ao_loc) for i in range(4)]
            ao_dm0b = [numint._dot_ao_dm(mol, ao[i], dm0b, mask, shls_slice, ao_loc) for i in range(4)]
            ao_dma_mask = contract('nig,ij->njg', ao_mask[:4], dm0a_mask)
            ao_dmb_mask = contract('nig,ij->njg', ao_mask[:4], dm0b_mask)
            wf = weight * fxc
            for ia in range(_sorted_mol.natm):
                dR_rho1a = _make_dR_rho1(ao, ao_dm0a, ia, aoslices, xctype)
                dR_rho1b = _make_dR_rho1(ao, ao_dm0b, ia, aoslices, xctype)
                wv = contract('xbyg,sxg->bsyg', wf[0], dR_rho1a)
                wv+= contract('xbyg,sxg->bsyg', wf[1], dR_rho1b)
                wv[:,:,0] *= .5
                wv[:,:,4] *= .5  # for the factor 1/2 in tau
                wva, wvb = wv

                vmata_dm_tmp = cupy.empty([3,3,nao_non0])
                vmatb_dm_tmp = cupy.empty([3,3,nao_non0])
                for i in range(3):
                    aow = rks_grad._make_dR_dao_w(ao_mask, wva[i])
                    vmata_dm_tmp[i] = contract('xjg,jg->xj', aow, ao_dma_mask[0])

                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dma_mask[:4], wva[i,:4])
                vmata_dm_tmp += contract('yjg,xjg->xyj', ao_mask[1:4], aow)

                for i in range(3):
                    aow = rks_grad._make_dR_dao_w(ao_mask, wvb[i])
                    vmatb_dm_tmp[i] = contract('xjg,jg->xj', aow, ao_dmb_mask[0])

                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dmb_mask[:4], wvb[i,:4])
                vmatb_dm_tmp += contract('yjg,xjg->xyj', ao_mask[1:4], aow)

                # alpha
                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dma_mask[1], wva[i,4])
                vmata_dm_tmp[:,0] += contract('jg,xjg->xj', ao_mask[XX], aow)
                vmata_dm_tmp[:,1] += contract('jg,xjg->xj', ao_mask[XY], aow)
                vmata_dm_tmp[:,2] += contract('jg,xjg->xj', ao_mask[XZ], aow)

                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dma_mask[2], wva[i,4])
                vmata_dm_tmp[:,0] += contract('jg,xjg->xj', ao_mask[YX], aow)
                vmata_dm_tmp[:,1] += contract('jg,xjg->xj', ao_mask[YY], aow)
                vmata_dm_tmp[:,2] += contract('jg,xjg->xj', ao_mask[YZ], aow)

                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dma_mask[3], wva[i,4])
                vmata_dm_tmp[:,0] += contract('jg,xjg->xj', ao_mask[ZX], aow)
                vmata_dm_tmp[:,1] += contract('jg,xjg->xj', ao_mask[ZY], aow)
                vmata_dm_tmp[:,2] += contract('jg,xjg->xj', ao_mask[ZZ], aow)

                # beta
                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dmb_mask[1], wvb[i,4])
                vmatb_dm_tmp[:,0] += contract('jg,xjg->xj', ao_mask[XX], aow)
                vmatb_dm_tmp[:,1] += contract('jg,xjg->xj', ao_mask[XY], aow)
                vmatb_dm_tmp[:,2] += contract('jg,xjg->xj', ao_mask[XZ], aow)

                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dmb_mask[2], wvb[i,4])
                vmatb_dm_tmp[:,0] += contract('jg,xjg->xj', ao_mask[YX], aow)
                vmatb_dm_tmp[:,1] += contract('jg,xjg->xj', ao_mask[YY], aow)
                vmatb_dm_tmp[:,2] += contract('jg,xjg->xj', ao_mask[YZ], aow)

                for i in range(3):
                    aow[i] = numint._scale_ao(ao_dmb_mask[3], wvb[i,4])
                vmatb_dm_tmp[:,0] += contract('jg,xjg->xj', ao_mask[ZX], aow)
                vmatb_dm_tmp[:,1] += contract('jg,xjg->xj', ao_mask[ZY], aow)
                vmatb_dm_tmp[:,2] += contract('jg,xjg->xj', ao_mask[ZZ], aow)

                vmata_dm[ia][:,:,mask] += vmata_dm_tmp
                vmatb_dm[ia][:,:,mask] += vmatb_dm_tmp
            t1 = log.timer_debug2('integration', *t1)
        vmata_dm = opt.unsort_orbitals(vmata_dm, axis=[3])
        vmatb_dm = opt.unsort_orbitals(vmatb_dm, axis=[3])
        for ia in range(_sorted_mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmata_dm[ia] += contract('xypq,pq->xyp', ipipa[:,:,:,p0:p1], dm0a[:,p0:p1])
            vmata_dm[ia] += contract('yxqp,pq->xyp', ipipa[:,:,p0:p1], dm0a[:,p0:p1])
            vmatb_dm[ia] += contract('xypq,pq->xyp', ipipb[:,:,:,p0:p1], dm0b[:,p0:p1])
            vmatb_dm[ia] += contract('yxqp,pq->xyp', ipipb[:,:,p0:p1], dm0b[:,p0:p1])
    return vmata_dm, vmatb_dm

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
    assert dm0_masked.shape == (2, nao, nao)
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

    dm_dmT = dm0_masked + dm0_masked.transpose(0,2,1)
    dm_dot_mu_and_nu = dm_dmT @ mu
    if with_nablarho:
        dm_dot_dmu_and_dnu = contract('djg,uij->udig', dmu_dr, dm_dmT)

    assert with_orbital_response or with_grid_response, "Why are you calling this function?"

    drho_dA_orbital_response = None
    if with_orbital_response:
        drho_dA_orbital_response = cupy.zeros([natm, 3, 2, n_component, ngrids])
    drho_dA_grid_response = None
    if with_grid_response:
        drho_dA_grid_response = cupy.zeros([natm, 3, 2, n_component, ngrids])

    rho_response = contract("dig,uig->duig", dmu_dr, dm_dot_mu_and_nu)
    if with_orbital_response:
        cupy.add.at(drho_dA_orbital_response[:, :, :, 0, :], masked_i_atom_of_aos, rho_response.transpose(2,0,1,3))
    if with_grid_response:
        rho_response = cupy.einsum('duig->dug', rho_response)
        drho_dA_grid_response[i_atom_of_grids, :, :, 0, :] = rho_response
    del rho_response

    if with_nablarho:
        nablarho_response = contract("dDig,uig->duDig", d2mu_dr2, dm_dot_mu_and_nu)
        nablarho_response += contract('dig,uDig->duDig', dmu_dr, dm_dot_dmu_and_dnu)
        if with_orbital_response:
            cupy.add.at(drho_dA_orbital_response[:, :, :, 1:4, :], masked_i_atom_of_aos, nablarho_response.transpose(3,0,1,2,4))
        if with_grid_response:
            nablarho_response = cupy.einsum('duDig->duDg', nablarho_response)
            drho_dA_grid_response[i_atom_of_grids, :, :, 1:4, :] = nablarho_response
        del nablarho_response

    if with_tau:
        tau_response = 0.5 * contract('dDig,uDig->duig', d2mu_dr2, dm_dot_dmu_and_dnu)
        if with_orbital_response:
            cupy.add.at(drho_dA_orbital_response[:, :, :, 4, :], masked_i_atom_of_aos, tau_response.transpose(2,0,1,3))
        if with_grid_response:
            tau_response = cupy.einsum('duig->dug', tau_response)
            drho_dA_grid_response[i_atom_of_grids, :, :, 4, :] = tau_response
        del tau_response

    drho_dA_orbital_response *= -1

    if xctype == "LDA":
        if drho_dA_orbital_response is not None:
            drho_dA_orbital_response = drho_dA_orbital_response[:, :, :, 0, :]
        if drho_dA_grid_response is not None:
            drho_dA_grid_response = drho_dA_grid_response[:, :, :, 0, :]
    return drho_dA_orbital_response, drho_dA_grid_response

def contract_d2rho_dAdB_sparse(dm0_masked, xctype, natm, masked_i_atom_of_aos = None, i_atom_of_grids = None,
                               mu = None, dmu_dr = None, d2mu_dr2 = None, d3mu_dr3 = None,
                               weight_depsilon_drho = None, weight_depsilon_dnablarho = None, weight_depsilon_dtau = None,
                               with_orbital_response = True, with_grid_response = True):
    """
        This function is the unrestricted version of gpu4pyscf.hessian.rks.contract_d2rho_dAdB_sparse().
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
    assert dm0_masked.shape == (2, nao, nao)
    assert mu is not None
    ngrids = mu.shape[1]
    assert mu.shape == (nao, ngrids)
    assert dmu_dr is not None and dmu_dr.shape == (3, nao, ngrids)
    assert d2mu_dr2 is not None and d2mu_dr2.shape == (3, 3, nao, ngrids)
    if with_nablarho or with_tau:
        assert d3mu_dr3 is not None and d3mu_dr3.shape == (3, 3, 3, nao, ngrids)

    assert weight_depsilon_drho is not None and weight_depsilon_drho.shape == (2, ngrids)
    if with_nablarho:
        assert weight_depsilon_dnablarho is not None and weight_depsilon_dnablarho.shape == (2, 3, ngrids)
    if with_tau:
        assert weight_depsilon_dtau is not None and weight_depsilon_dtau.shape == (2, ngrids)

    if with_orbital_response or with_grid_response: # There are cross terms in grid response
        assert masked_i_atom_of_aos is not None
        masked_i_atom_of_aos = cupy.asarray(masked_i_atom_of_aos, dtype = numpy.int32)
        assert masked_i_atom_of_aos.shape == (nao,)
    if with_grid_response:
        assert i_atom_of_grids is not None
        i_atom_of_grids = int(i_atom_of_grids)
        assert 0 <= i_atom_of_grids and i_atom_of_grids < natm

    dm_dmT = dm0_masked + dm0_masked.transpose(0,2,1)
    dm_dot_mu = dm_dmT @ mu
    dm_dot_dmudr = contract("uij,djg->udig", dm_dmT, dmu_dr)

    d2e_ii_AA_ao_uncontracted = 0
    d2V = 0
    d2e_ji_AB_ao_uncontracted = 0

    # d2mu/dAdB * nu
    dm_mu_wv = contract("uig,ug->ig", dm_dot_mu, weight_depsilon_drho)
    d2rhodAdG = contract("dDig,ig->dDi", d2mu_dr2, dm_mu_wv)
    del dm_mu_wv
    if with_orbital_response:
        d2e_ii_AA_ao_uncontracted += d2rhodAdG
    if with_grid_response:
        d2e_ji_AB_ao_uncontracted += d2rhodAdG
    del d2rhodAdG

    # dmu/dA * dnu/dB
    dmudr_wv = contract("dig,ug->udig", dmu_dr, weight_depsilon_drho)
    if with_orbital_response:
        d2rhodAdB = contract("dig,uDjg->udDij", dmu_dr, dmudr_wv)
        d2V += d2rhodAdB
        del d2rhodAdB
    if with_grid_response:
        d2rhodAdG = contract("udig,uDig->dDi", dmudr_wv, dm_dot_dmudr)
        d2e_ji_AB_ao_uncontracted += d2rhodAdG
        del d2rhodAdG
    del dmudr_wv

    if with_nablarho:
        # d3mu/(dr dA dB) * nu
        d3mudr3_wv = contract("dDxig,uxg->dDuig", d3mu_dr3, weight_depsilon_dnablarho)
        d2nablarhodAdG = contract("dDuig,uig->dDi", d3mudr3_wv, dm_dot_mu)
        del d3mudr3_wv
        # d2mu/(dA dB) * dnu/dr
        dm_dmudr_wv = contract("uxig,uxg->ig", dm_dot_dmudr, weight_depsilon_dnablarho)
        d2nablarhodAdG += contract("dDig,ig->dDi", d2mu_dr2, dm_dmudr_wv)
        del dm_dmudr_wv
        d2mudr2_wv = contract("dxig,uxg->udig", d2mu_dr2, weight_depsilon_dnablarho)
        if with_orbital_response:
            d2e_ii_AA_ao_uncontracted += d2nablarhodAdG
            # d2mu/(dr dA) * dnu/dB, A orbital, B orbital
            d2mudr2_dnudr_wv = contract("udig,Djg->udDij", d2mudr2_wv, dmu_dr)
            d2V += d2mudr2_dnudr_wv + d2mudr2_dnudr_wv.transpose(0,2,1,4,3)
            del d2mudr2_dnudr_wv
        if with_grid_response:
            # d2mu/(dr dA) * dnu/dB, A orbital, B grid
            d2nablarhodAdG += contract("udig,uDig->dDi", d2mudr2_wv, dm_dot_dmudr)
            # d2mu/(dr dA) * dnu/dB, A grid, B orbital
            dm_d2mudr2_wv = contract("uij,udjg->dig", dm_dmT, d2mudr2_wv)
            d2nablarhodAdG += contract("dig,Dig->dDi", dmu_dr, dm_d2mudr2_wv)
            del dm_d2mudr2_wv
            d2e_ji_AB_ao_uncontracted += d2nablarhodAdG
        del d2mudr2_wv
        del d2nablarhodAdG

    if with_tau:
        # d3mu/(dA dB dr) * dnu/dr
        dm_dmudr_wv = contract("udig,ug->dig", dm_dot_dmudr, weight_depsilon_dtau)
        d2taudAdG = 0.5 * contract("dDxig,xig->dDi", d3mu_dr3, dm_dmudr_wv)
        del dm_dmudr_wv
        # d2mu/(dA dr) * d2nu/(dB dr)
        d2mudr2_wv = contract("dxig,ug->udxig", d2mu_dr2, weight_depsilon_dtau)
        d2mudAdr_d2nudBdr_wv = 0.5 * contract("dxig,uDxjg->udDij", d2mu_dr2, d2mudr2_wv)
        del d2mudr2_wv
        if with_orbital_response:
            d2e_ii_AA_ao_uncontracted += d2taudAdG
            d2V += d2mudAdr_d2nudBdr_wv
        if with_grid_response:
            d2taudAdG += contract("udDij,uij->dDi", d2mudAdr_d2nudBdr_wv, dm_dmT)
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

        d2V = contract("udDij,uij->dDij", d2V, dm_dmT)
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
    assert dm0.ndim == 3 and dm0.shape[0] == 2
    dm0_sorted = opt.sort_orbitals(dm0, axis=[1,2])
    dm_mask_buf = cupy.empty(2 * nao * nao)
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
            rhoa = numint.eval_rho(_sorted_mol, ao, dm0_masked[0], xctype = xctype, hermi = 1)
            rhob = numint.eval_rho(_sorted_mol, ao, dm0_masked[1], xctype = xctype, hermi = 1)
            rho = cupy.asarray((rhoa, rhob))
            exc = ni.eval_xc_eff(mf.xc, rho, deriv = 0, xctype=xctype, spin=1)[0]
            del rho

            epsilon = exc * (rhoa + rhob)
            del rhoa, rhob, exc

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

            rhoa = numint.eval_rho(_sorted_mol, ao[0], dm0_masked[0], xctype = xctype, hermi = 1)
            rhob = numint.eval_rho(_sorted_mol, ao[0], dm0_masked[1], xctype = xctype, hermi = 1)
            rho = cupy.asarray((rhoa, rhob))
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype = xctype, spin=1)[1:3]
            del rhoa, rhob, rho

            depsilon_drho = vxc[:,0,:] # Just of shape (2,ngrids)

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
            drhodA_v = contract("Adug,ug->Adg", drho_dA_full_response, depsilon_drho)
            d2e_dwdA_term = contract("Adg,BDg->ABdD", dw_dA, drhodA_v)
            del dw_dA, drhodA_v
            d2e += d2e_dwdA_term + d2e_dwdA_term.transpose(1,0,3,2)
            del d2e_dwdA_term

            weight = weight[nonzero_weight_mask[g0:g1]]
            fwxc = fxc[:,0,:,0,:] * weight # Just of shape (2,2,ngrids)
            del fxc
            drhodA_fw = contract("Adug,uvg->Advg", drho_dA_full_response, fwxc)
            del fwxc
            d2e += contract("Adug,BDug->ABdD", drho_dA_full_response, drhodA_fw)
            del drhodA_fw, drho_dA_full_response

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
            rhoa = numint.eval_rho(_sorted_mol, ao, dm0_masked[0], xctype = xctype, hermi = 1)
            rhob = numint.eval_rho(_sorted_mol, ao, dm0_masked[1], xctype = xctype, hermi = 1)
            rho = cupy.asarray((rhoa, rhob))
            exc = ni.eval_xc_eff(mf.xc, rho, deriv = 0, xctype=xctype, spin=1)[0]
            del rho

            epsilon = exc * (rhoa[0, :] + rhob[0, :])
            del rhoa, rhob, exc

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

            rhoa = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked[0], xctype = xctype, hermi = 1)
            rhob = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked[1], xctype = xctype, hermi = 1)
            rho = cupy.asarray((rhoa, rhob))
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype = xctype, spin=1)[1:3]
            del rhoa, rhob, rho

            depsilon_drho = vxc[:, 0, :]
            depsilon_dnablarho = vxc[:, 1:4, :]

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
            depsilondrho_drhodA = contract("ug,Adug->Adg", depsilon_drho, drho_dA_full_response[:,:,:,0,:])
            depsilondnablarho_dnablarhodA = contract("uxg,Aduxg->Adg", depsilon_dnablarho, drho_dA_full_response[:,:,:,1:4,:])
            d2e_dwdA_term = contract("Adg,BDg->ABdD", dw_dA, depsilondrho_drhodA + depsilondnablarho_dnablarhodA)
            del depsilondrho_drhodA, depsilondnablarho_dnablarhodA
            del dw_dA
            d2e += d2e_dwdA_term + d2e_dwdA_term.transpose(1,0,3,2)
            del d2e_dwdA_term

            weight = weight[nonzero_weight_mask[g0:g1]]
            fwxc = fxc * weight
            del fxc
            drhodA_fwxc = contract("uxvyg,Advyg->Aduxg", fwxc, drho_dA_full_response)
            del fwxc
            d2e += contract("Aduxg,BDuxg->ABdD", drho_dA_full_response, drhodA_fwxc)
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
            rhoa = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked[0], xctype = xctype, hermi = 1)
            rhob = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked[1], xctype = xctype, hermi = 1)
            rho = cupy.asarray((rhoa, rhob))
            exc = ni.eval_xc_eff(mf.xc, rho, deriv = 0, xctype=xctype, spin=1)[0]
            del rho

            epsilon = exc * (rhoa[0, :] + rhob[0, :])
            del rhoa, rhob, exc

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

            rhoa = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked[0], xctype = xctype, hermi = 1)
            rhob = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked[1], xctype = xctype, hermi = 1)
            rho = cupy.asarray((rhoa, rhob))
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype = xctype, spin=1)[1:3]
            del rhoa, rhob, rho

            depsilon_drho = vxc[:, 0, :]
            depsilon_dnablarho = vxc[:, 1:4, :]
            depsilon_dtau = vxc[:, 4, :]

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
            depsilondrho_drhodA = contract("ug,Adug->Adg", depsilon_drho, drho_dA_full_response[:,:,:,0,:])
            depsilondnablarho_dnablarhodA = contract("uxg,Aduxg->Adg", depsilon_dnablarho, drho_dA_full_response[:,:,:,1:4,:])
            depsilondtau_dtaudA = contract("ug,Adug->Adg", depsilon_dtau, drho_dA_full_response[:,:,:,4,:])
            d2e_dwdA_term = contract("Adg,BDg->ABdD", dw_dA,
                depsilondrho_drhodA + depsilondnablarho_dnablarhodA + depsilondtau_dtaudA)
            del depsilondrho_drhodA, depsilondnablarho_dnablarhodA, depsilondtau_dtaudA
            del dw_dA
            d2e += d2e_dwdA_term + d2e_dwdA_term.transpose(1,0,3,2)
            del d2e_dwdA_term

            weight = weight[nonzero_weight_mask[g0:g1]]
            fwxc = fxc * weight
            del fxc
            drhodA_fwxc = contract("uxvyg,Advyg->Aduxg", fwxc, drho_dA_full_response)
            del fwxc
            d2e += contract("Aduxg,BDuxg->ABdD", drho_dA_full_response, drhodA_fwxc)
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

def _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
    if hessobj.grid_response:
        return _get_vxc_deriv1_grid_response(hessobj, mo_coeff, mo_occ, max_memory)

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
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]

    nao, nmo = mo_coeff[0].shape
    natm = mol.natm
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
    dm0a, dm0b = mf.make_rdm1(mo_coeff, mo_occ)

    vipa = cupy.zeros((3,nao,nao))
    vipb = cupy.zeros((3,nao,nao))
    vmata = cupy.zeros((natm,3,nao,nocca))
    vmatb = cupy.zeros((natm,3,nao,noccb))
    max_memory = None
    if xctype == 'LDA':
        ao_deriv = 1
        t1 = t0 = log.init_timer()
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            ao = contract('nip,ij->njp', ao, coeff[mask])
            rhoa = numint.eval_rho2(_sorted_mol, ao[0], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = numint.eval_rho2(_sorted_mol, ao[0], mo_coeff[1], mo_occ[1], mask, xctype)
            t1 = log.timer_debug2('eval rho', *t1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 2, xctype=xctype, spin=1)[1:3]
            t1 = log.timer_debug2('eval vxc', *t1)
            wv = weight * vxc[:,0]
            aow = numint._scale_ao(ao[0], wv[0])
            vipa += rks_grad._d1_dot_(ao[1:4], aow.T)
            aow = numint._scale_ao(ao[0], wv[1])
            vipb += rks_grad._d1_dot_(ao[1:4], aow.T)
            moa = contract('xig,ip->xpg', ao, mocca)
            mob = contract('xig,ip->xpg', ao, moccb)
            ao_dm0a = numint._dot_ao_dm(mol, ao[0], dm0a, mask, shls_slice, ao_loc)
            ao_dm0b = numint._dot_ao_dm(mol, ao[0], dm0b, mask, shls_slice, ao_loc)
            wf = weight * fxc[:,0,:,0]
            for ia in range(natm):
                p0, p1 = aoslices[ia][2:]
# First order density = rho1 * 2.  *2 is not applied because + c.c. in the end
                rho1a = contract('xig,ig->xg', ao[1:,p0:p1,:], ao_dm0a[p0:p1,:])
                rho1b = contract('xig,ig->xg', ao[1:,p0:p1,:], ao_dm0b[p0:p1,:])
                wv = wf[0,:,None] * rho1a
                wv+= wf[1,:,None] * rho1b
                aow = [numint._scale_ao(ao[0], wv[0,i]) for i in range(3)]
                mow = [numint._scale_ao(moa[0], wv[0,i]) for i in range(3)]
                vmata[ia] += rks_grad._d1_dot_(aow, moa[0].T)
                vmata[ia] += rks_grad._d1_dot_(mow, ao[0].T).transpose([0,2,1])

                aow = [numint._scale_ao(ao[0], wv[1,i]) for i in range(3)]
                mow = [numint._scale_ao(mob[0], wv[1,i]) for i in range(3)]
                vmatb[ia] += rks_grad._d1_dot_(aow, mob[0].T)
                vmatb[ia] += rks_grad._d1_dot_(mow, ao[0].T).transpose([0,2,1])
            ao_dm0a = ao_dm0b = aow = None
            t1 = log.timer_debug2('integration', *t1)
    elif xctype == 'GGA':
        ao_deriv = 2
        t1 = t0 = log.init_timer()
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            ao = contract('nip,ij->njp', ao, coeff[mask])
            rhoa = numint.eval_rho2(_sorted_mol, ao[:4], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = numint.eval_rho2(_sorted_mol, ao[:4], mo_coeff[1], mo_occ[1], mask, xctype)
            t1 = log.timer_debug2('eval rho', *t1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 2, xctype=xctype, spin=1)[1:3]
            t1 = log.timer_debug2('eval vxc', *t1)
            wv = weight * vxc
            wv[:,0] *= .5
            vipa += rks_grad._gga_grad_sum_(ao, wv[0])
            vipb += rks_grad._gga_grad_sum_(ao, wv[1])
            moa = contract('xig,ip->xpg', ao, mocca)
            mob = contract('xig,ip->xpg', ao, moccb)
            ao_dm0a = [numint._dot_ao_dm(mol, ao[i], dm0a, mask, shls_slice, ao_loc)
                      for i in range(4)]
            ao_dm0b = [numint._dot_ao_dm(mol, ao[i], dm0b, mask, shls_slice, ao_loc)
                      for i in range(4)]
            wf = weight * fxc
            for ia in range(natm):
                dR_rho1a = _make_dR_rho1(ao, ao_dm0a, ia, aoslices, xctype)
                dR_rho1b = _make_dR_rho1(ao, ao_dm0b, ia, aoslices, xctype)
                wv = contract('xbyg,sxg->bsyg', wf[0], dR_rho1a)
                wv+= contract('xbyg,sxg->bsyg', wf[1], dR_rho1b)
                wv[:,:,0] *= .5
                wva,wvb = wv
                aow = [numint._scale_ao(ao[:4], wva[i,:4]) for i in range(3)]
                mow = [numint._scale_ao(moa[:4], wva[i,:4]) for i in range(3)]
                vmata[ia] += rks_grad._d1_dot_(aow, moa[0].T)
                vmata[ia] += rks_grad._d1_dot_(mow, ao[0].T).transpose([0,2,1])

                aow = [numint._scale_ao(ao[:4], wvb[i,:4]) for i in range(3)]
                mow = [numint._scale_ao(mob[:4], wvb[i,:4]) for i in range(3)]
                vmatb[ia] += rks_grad._d1_dot_(aow, mob[0].T)
                vmatb[ia] += rks_grad._d1_dot_(mow, ao[0].T).transpose([0,2,1])
            t1 = log.timer_debug2('integration', *t1)
            ao_dm0a = ao_dm0b = aow = None
    elif xctype == 'MGGA':
        if grids.level < 5:
            log.warn('MGGA Hessian is sensitive to dft grids.')
        ao_deriv = 2
        t1 = t0 = log.init_timer()
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            ao = contract('nip,ij->njp', ao, coeff[mask])
            rhoa = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff[1], mo_occ[1], mask, xctype)
            t1 = log.timer_debug2('eval rho', *t1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 2, xctype=xctype, spin=1)[1:3]
            t1 = log.timer_debug2('eval vxc', *t0)
            wv = weight * vxc
            wv[:,0] *= .5
            wv[:,4] *= .5  # for the factor 1/2 in tau
            vipa += rks_grad._gga_grad_sum_(ao, wv[0])
            vipa += rks_grad._tau_grad_dot_(ao, wv[0,4])
            vipb += rks_grad._gga_grad_sum_(ao, wv[1])
            vipb += rks_grad._tau_grad_dot_(ao, wv[1,4])

            moa = contract('xig,ip->xpg', ao, mocca)
            mob = contract('xig,ip->xpg', ao, moccb)
            ao_dm0a = [numint._dot_ao_dm(mol, ao[i], dm0a, mask, shls_slice, ao_loc) for i in range(4)]
            ao_dm0b = [numint._dot_ao_dm(mol, ao[i], dm0b, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            for ia in range(natm):
                dR_rho1a = _make_dR_rho1(ao, ao_dm0a, ia, aoslices, xctype)
                dR_rho1b = _make_dR_rho1(ao, ao_dm0b, ia, aoslices, xctype)
                wv = contract('xbyg,sxg->bsyg', wf[0], dR_rho1a)
                wv+= contract('xbyg,sxg->bsyg', wf[1], dR_rho1b)
                wv[:,:,0] *= .5
                wv[:,:,4] *= .25
                wva,wvb = wv
                aow = [numint._scale_ao(ao[:4], wva[i,:4]) for i in range(3)]
                mow = [numint._scale_ao(moa[:4], wva[i,:4]) for i in range(3)]
                vmata[ia] += rks_grad._d1_dot_(aow, moa[0].T)
                vmata[ia] += rks_grad._d1_dot_(mow, ao[0].T).transpose([0,2,1])

                aow = [numint._scale_ao(ao[:4], wvb[i,:4]) for i in range(3)]
                mow = [numint._scale_ao(mob[:4], wvb[i,:4]) for i in range(3)]
                vmatb[ia] += rks_grad._d1_dot_(aow, mob[0].T)
                vmatb[ia] += rks_grad._d1_dot_(mow, ao[0].T).transpose([0,2,1])
                for j in range(1, 4):
                    aow = [numint._scale_ao(ao[j], wva[i,4]) for i in range(3)]
                    mow = [numint._scale_ao(moa[j], wva[i,4]) for i in range(3)]
                    vmata[ia] += rks_grad._d1_dot_(aow, moa[j].T)
                    vmata[ia] += rks_grad._d1_dot_(mow, ao[j].T).transpose([0,2,1])

                    aow = [numint._scale_ao(ao[j], wvb[i,4]) for i in range(3)]
                    mow = [numint._scale_ao(mob[j], wvb[i,4]) for i in range(3)]
                    vmatb[ia] += rks_grad._d1_dot_(aow, mob[j].T)
                    vmatb[ia] += rks_grad._d1_dot_(mow, ao[j].T).transpose([0,2,1])
            ao_dm0a = ao_dm0b = aow = None
            t1 = log.timer_debug2('integration', *t1)

    va_mo = cupy.ndarray((natm,3,nmo,nocca), dtype=vmata.dtype, memptr=vmata.data)
    vb_mo = cupy.ndarray((natm,3,nmo,noccb), dtype=vmatb.dtype, memptr=vmatb.data)
    vmat_tmp = cupy.empty([3,nao,nao])
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        vmat_tmp[:] = 0.
        vmat_tmp[:,p0:p1] += vipa[:,p0:p1]
        vmat_tmp[:,:,p0:p1] += vipa[:,p0:p1].transpose(0,2,1)
        tmp = contract('xij,jq->xiq', vmat_tmp, mocca)
        tmp += vmata[ia]
        contract('xiq,ip->xpq', tmp, mo_coeff[0], alpha=-1., out=va_mo[ia])

        vmat_tmp[:] = 0.
        vmat_tmp[:,p0:p1] += vipb[:,p0:p1]
        vmat_tmp[:,:,p0:p1] += vipb[:,p0:p1].transpose(0,2,1)
        tmp = contract('xij,jq->xiq', vmat_tmp, moccb)
        tmp += vmatb[ia]
        contract('xiq,ip->xpq', tmp, mo_coeff[1], alpha=-1., out=vb_mo[ia])
    return va_mo, vb_mo

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
    assert mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2
    mo_coeff_sorted = opt.sort_orbitals(mo_coeff, axis=[1])
    mocca_sorted = mo_coeff_sorted[0][:, mo_occ[0] > 0]
    moccb_sorted = mo_coeff_sorted[1][:, mo_occ[1] > 0]
    nao = _sorted_mol.nao
    # nmo = mo_coeff_sorted.shape[2]
    nocca = mocca_sorted.shape[1]
    noccb = moccb_sorted.shape[1]

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    assert dm0.ndim == 3 and dm0.shape[0] == 2
    dm0_sorted = opt.sort_orbitals(dm0, axis=[1,2])
    dm_mask_buf = cupy.empty(2 * nao * nao)
    dm0 = None
    mo_coeff = None

    nonzero_weight_mask = cupy.abs(grids.weights) > 1e-20 # There are d2rho/dAdB terms with very low weight but very big contribution

    ao_loc_sorted = _sorted_mol.ao_loc
    ao_expand = ao_loc_sorted[1:] - ao_loc_sorted[:-1]
    from pyscf.gto.mole import ATOM_OF
    i_atom_of_aos = numpy.repeat(_sorted_mol._bas[:,ATOM_OF], ao_expand)
    i_atom_of_aos = cupy.asarray(i_atom_of_aos, dtype = cupy.int32)

    dFock_orbital_response_dmudA_nu_term = cupy.zeros((2, 3, nao, nao))
    dFock_ao_occa = cupy.zeros((natm, 3, nao, nocca))
    dFock_ao_occb = cupy.zeros((natm, 3, nao, noccb))

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
            mocca_masked = mocca_sorted[idx, :]
            moccb_masked = moccb_sorted[idx, :]

            rhoa = numint.eval_rho(_sorted_mol, ao[0], dm0_masked[0], xctype = xctype, hermi = 1)
            rhob = numint.eval_rho(_sorted_mol, ao[0], dm0_masked[1], xctype = xctype, hermi = 1)
            rho = cupy.asarray((rhoa, rhob))
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype=xctype, spin=1)[1:3]
            del rhoa, rhob, rho

            depsilon_drho = vxc[:, 0, :] # Just of shape (2, ngrids)
            d2epsilon_drho2 = fxc[:, 0, :, 0, :] # Just of shape (2, 2, ngrids)
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
            mu_nu_term_prefactor = contract("Advg,uvg->Adug", drho_dA_full_response, d2epsilon_drho2 * weight)
            mu_nu_term_prefactor += contract("Adg,ug->Adug", dw_dA, depsilon_drho)
            del drho_dA_full_response, d2epsilon_drho2, dw_dA

            dFock_sparse_ao_occa = cupy.zeros((natm, 3, mu.shape[0], nocca))
            dFock_sparse_ao_occb = cupy.zeros((natm, 3, mu.shape[0], noccb))

            mu_mocca = mu.T @ mocca_masked
            mu_moccb = mu.T @ moccb_masked
            for i_atom in range(natm):
                prefactor_with_nu_mocc = contract("dg,gj->dgj", mu_nu_term_prefactor[i_atom, :, 0, :], mu_mocca)
                dFock_sparse_ao_occa[i_atom, :, :, :] += contract("pg,dgj->dpj", mu, prefactor_with_nu_mocc)
                del prefactor_with_nu_mocc
                prefactor_with_nu_mocc = contract("dg,gj->dgj", mu_nu_term_prefactor[i_atom, :, 1, :], mu_moccb)
                dFock_sparse_ao_occb[i_atom, :, :, :] += contract("pg,dgj->dpj", mu, prefactor_with_nu_mocc)
                del prefactor_with_nu_mocc
            del mu_mocca, mu_moccb
            del mu_nu_term_prefactor

            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += cupy.einsum("g,dpg,qg->dpq", depsilon_drho * weight, dmu_dr, mu)
            mu_vw = contract("pg,ug->upg", mu, depsilon_drho * weight)
            dmudA_nu_ao = contract("dpg,uqg->udpq", dmu_dr, mu_vw)
            del mu_vw
            dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(2), range(3), idx, idx)] += dmudA_nu_ao

            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum("pi,g,dpg,qg,qj->dij", mo_coeff_masked, depsilon_drho * weight, dmu_dr, mu, mocc_masked)
            # dFock_mo_occ[i_atom_of_grids, :, :, :] += cupy.einsum("pi,g,dqg,pg,qj->dij", mo_coeff_masked, depsilon_drho * weight, dmu_dr, mu, mocc_masked)
            dmudA_nu_ao_ao = dmudA_nu_ao + dmudA_nu_ao.transpose(0,1,3,2)
            del dmudA_nu_ao
            dFock_sparse_ao_occa[i_atom_of_grids, :, :, :] += contract("dpq,qj->dpj", dmudA_nu_ao_ao[0], mocca_masked)
            dFock_sparse_ao_occb[i_atom_of_grids, :, :, :] += contract("dpq,qj->dpj", dmudA_nu_ao_ao[1], moccb_masked)
            del dmudA_nu_ao_ao

            dFock_ao_occa[:, :, idx, :] += dFock_sparse_ao_occa
            dFock_ao_occb[:, :, idx, :] += dFock_sparse_ao_occb
            del dFock_sparse_ao_occa, dFock_sparse_ao_occb

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
            mocca_masked = mocca_sorted[idx, :]
            moccb_masked = moccb_sorted[idx, :]

            rhoa = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked[0], xctype = xctype, hermi = 1)
            rhob = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked[1], xctype = xctype, hermi = 1)
            rho = cupy.asarray((rhoa, rhob))
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype=xctype, spin=1)[1:3]
            del rhoa, rhob, rho

            dw_dA = get_dweight_dA(_sorted_mol, grids, (g0,g1))
            dw_dA = dw_dA[:, :, nonzero_weight_mask[g0:g1]]

            # dFock_mo_occ += cupy.einsum("pi,Adg,g,pg,qg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_drho, mu, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adg,xg,xpg,qg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_dnablarho, dmu_dr, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adg,xg,xqg,pg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_dnablarho, dmu_dr, mu, mocc_masked)
            dwdA_vxc = contract("Adg,uxg->Aduxg", dw_dA, vxc)
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
            drhodA_fwxc = contract("uxvyg,Advyg->Aduxg", fwxc, drho_dA_full_response)
            del drho_dA_full_response
            del fwxc

            # dFock_mo_occ += cupy.einsum("pi,Adg,pg,qg,qj->Adij", mo_coeff_masked, drhodA_fwxc[:, :, 0, :], mu, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adxg,xpg,qg,qj->Adij", mo_coeff_masked, drhodA_fwxc[:, :, 1:4, :], dmu_dr, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adxg,xqg,pg,qj->Adij", mo_coeff_masked, drhodA_fwxc[:, :, 1:4, :], dmu_dr, mu, mocc_masked)
            mu_nu_term_prefactor = drhodA_fwxc + dwdA_vxc
            del drhodA_fwxc, dwdA_vxc

            dFock_sparse_ao_occa = cupy.zeros((natm, 3, mu.shape[0], nocca))
            dFock_sparse_ao_occb = cupy.zeros((natm, 3, mu.shape[0], noccb))

            mu_mocca = mu.T @ mocca_masked
            mu_moccb = mu.T @ moccb_masked
            dmudr_mocca = contract("dqg,qj->djg", dmu_dr, mocca_masked)
            dmudr_moccb = contract("dqg,qj->djg", dmu_dr, moccb_masked)
            for i_atom in range(natm):
                mu_on_right_term = contract("xpg,duxg->udpg", ao[0:4], mu_nu_term_prefactor[i_atom, :, :, 0:4, :])
                dFock_sparse_ao_occa[i_atom, :, :, :] += contract("dpg,gj->dpj", mu_on_right_term[0], mu_mocca)
                dFock_sparse_ao_occb[i_atom, :, :, :] += contract("dpg,gj->dpj", mu_on_right_term[1], mu_moccb)
                del mu_on_right_term
                nablarho_dmudr_on_right_term = contract("dxg,xjg->djg", mu_nu_term_prefactor[i_atom, :, 0, 1:4, :], dmudr_mocca)
                dFock_sparse_ao_occa[i_atom, :, :, :] += contract("pg,djg->dpj", mu, nablarho_dmudr_on_right_term)
                nablarho_dmudr_on_right_term = contract("dxg,xjg->djg", mu_nu_term_prefactor[i_atom, :, 1, 1:4, :], dmudr_moccb)
                dFock_sparse_ao_occb[i_atom, :, :, :] += contract("pg,djg->dpj", mu, nablarho_dmudr_on_right_term)
                del nablarho_dmudr_on_right_term
            del mu_mocca, mu_moccb, dmudr_mocca, dmudr_moccb
            del mu_nu_term_prefactor

            wv = vxc * weight
            del vxc
            weight_depsilon_drho = wv[:, 0, :]
            weight_depsilon_dnablarho = wv[:, 1:4, :]
            del wv

            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += cupy.einsum("g,dpg,qg->dpq", depsilon_drho * weight, dmu_dr, mu)
            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += cupy.einsum("xg,dxpg,qg->dpq", depsilon_dnablarho * weight, d2mu_dr2, mu)
            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += cupy.einsum("xg,dpg,xqg->dpq", depsilon_dnablarho * weight, dmu_dr, dmu_dr)
            mu_weight_depsilondrho = contract("pg,ug->upg", mu, weight_depsilon_drho)
            dmudr_weight_depsilondnablarho = contract("uxg,xpg->upg", weight_depsilon_dnablarho, dmu_dr)
            dmudA_nu_ao = contract("dpg,uqg->udpq", dmu_dr, mu_weight_depsilondrho + dmudr_weight_depsilondnablarho)
            del mu_weight_depsilondrho, dmudr_weight_depsilondnablarho
            d2mudr2_weight_depsilondnablarho = contract("dxpg,uxg->udpg", d2mu_dr2, weight_depsilon_dnablarho)
            dmudA_nu_ao += contract("udpg,qg->udpq", d2mudr2_weight_depsilondnablarho, mu)
            del d2mudr2_weight_depsilondnablarho

            dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(2), range(3), idx, idx)] += dmudA_nu_ao

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
            dmudA_nu_ao_ao = dmudA_nu_ao + dmudA_nu_ao.transpose(0,1,3,2)
            del dmudA_nu_ao
            dFock_sparse_ao_occa[i_atom_of_grids, :, :, :] += contract("dpq,qj->dpj", dmudA_nu_ao_ao[0], mocca_masked)
            dFock_sparse_ao_occb[i_atom_of_grids, :, :, :] += contract("dpq,qj->dpj", dmudA_nu_ao_ao[1], moccb_masked)
            del dmudA_nu_ao_ao

            dFock_ao_occa[:, :, idx, :] += dFock_sparse_ao_occa
            dFock_ao_occb[:, :, idx, :] += dFock_sparse_ao_occb
            del dFock_sparse_ao_occa, dFock_sparse_ao_occb

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
            mocca_masked = mocca_sorted[idx, :]
            moccb_masked = moccb_sorted[idx, :]

            rhoa = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked[0], xctype = xctype, hermi = 1)
            rhob = numint.eval_rho(_sorted_mol, ao[:4], dm0_masked[1], xctype = xctype, hermi = 1)
            rho = cupy.asarray((rhoa, rhob))
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype=xctype, spin=1)[1:3]
            del rhoa, rhob, rho

            dw_dA = get_dweight_dA(_sorted_mol, grids, (g0,g1))
            dw_dA = dw_dA[:, :, nonzero_weight_mask[g0:g1]]

            # dFock_mo_occ += cupy.einsum("pi,Adg,g,pg,qg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_drho, mu, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adg,xg,xpg,qg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_dnablarho, dmu_dr, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adg,xg,xqg,pg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_dnablarho, dmu_dr, mu, mocc_masked)
            # dFock_mo_occ += 0.5 * cupy.einsum("pi,Adg,g,xpg,xqg,qj->Adij", mo_coeff_masked, dw_dA, depsilon_dtau, dmu_dr, dmu_dr, mocc_masked)
            dwdA_vxc = contract("Adg,uxg->Aduxg", dw_dA, vxc)
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
            drhodA_fwxc = contract("uxvyg,Advyg->Aduxg", fwxc, drho_dA_full_response)
            del drho_dA_full_response
            del fwxc

            # dFock_mo_occ += cupy.einsum("pi,Adg,pg,qg,qj->Adij", mo_coeff_masked, drhodA_fwxc[:, :, 0, :], mu, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adxg,xpg,qg,qj->Adij", mo_coeff_masked, drhodA_fwxc[:, :, 1:4, :], dmu_dr, mu, mocc_masked)
            # dFock_mo_occ += cupy.einsum("pi,Adxg,xqg,pg,qj->Adij", mo_coeff_masked, drhodA_fwxc[:, :, 1:4, :], dmu_dr, mu, mocc_masked)
            # dFock_mo_occ += 0.5 * cupy.einsum("pi,Adg,xpg,xqg,qj->Adij", mo_coeff_masked, drhodA_fwxc[:, :, 4, :], dmu_dr, dmu_dr, mocc_masked)
            mu_nu_term_prefactor = drhodA_fwxc + dwdA_vxc
            del drhodA_fwxc, dwdA_vxc

            dFock_sparse_ao_occa = cupy.zeros((natm, 3, mu.shape[0], nocca))
            dFock_sparse_ao_occb = cupy.zeros((natm, 3, mu.shape[0], noccb))

            mu_mocca = mu.T @ mocca_masked
            mu_moccb = mu.T @ moccb_masked
            dmudr_mocca = contract("dqg,qj->djg", dmu_dr, mocca_masked)
            dmudr_moccb = contract("dqg,qj->djg", dmu_dr, moccb_masked)
            for i_atom in range(natm):
                mu_on_right_term = contract("xpg,duxg->udpg", ao[0:4], mu_nu_term_prefactor[i_atom, :, :, 0:4, :])
                dFock_sparse_ao_occa[i_atom, :, :, :] += contract("dpg,gj->dpj", mu_on_right_term[0], mu_mocca)
                dFock_sparse_ao_occb[i_atom, :, :, :] += contract("dpg,gj->dpj", mu_on_right_term[1], mu_moccb)
                del mu_on_right_term
                nablarho_dmudr_on_right_term = contract("dxg,xjg->djg", mu_nu_term_prefactor[i_atom, :, 0, 1:4, :], dmudr_mocca)
                dFock_sparse_ao_occa[i_atom, :, :, :] += contract("pg,djg->dpj", mu, nablarho_dmudr_on_right_term)
                nablarho_dmudr_on_right_term = contract("dxg,xjg->djg", mu_nu_term_prefactor[i_atom, :, 1, 1:4, :], dmudr_moccb)
                dFock_sparse_ao_occb[i_atom, :, :, :] += contract("pg,djg->dpj", mu, nablarho_dmudr_on_right_term)
                del nablarho_dmudr_on_right_term
                tau_term = contract("dg,xjg->dxjg", mu_nu_term_prefactor[i_atom, :, 0, 4, :], dmudr_mocca)
                dFock_sparse_ao_occa[i_atom, :, :, :] += 0.5 * contract("xpg,dxjg->dpj", dmu_dr, tau_term)
                tau_term = contract("dg,xjg->dxjg", mu_nu_term_prefactor[i_atom, :, 1, 4, :], dmudr_moccb)
                dFock_sparse_ao_occb[i_atom, :, :, :] += 0.5 * contract("xpg,dxjg->dpj", dmu_dr, tau_term)
                del tau_term
            del mu_mocca, mu_moccb, dmudr_mocca, dmudr_moccb
            del mu_nu_term_prefactor

            wv = vxc * weight
            del vxc
            weight_depsilon_drho = wv[:, 0, :]
            weight_depsilon_dnablarho = wv[:, 1:4, :]
            weight_depsilon_dtau = wv[:, 4, :]
            del wv

            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += cupy.einsum(
            #     "g,dpg,qg->dpq", depsilon_drho * weight, dmu_dr, mu)
            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += cupy.einsum(
            #     "xg,dxpg,qg->dpq", depsilon_dnablarho * weight, d2mu_dr2, mu)
            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += cupy.einsum(
            #     "xg,dpg,xqg->dpq", depsilon_dnablarho * weight, dmu_dr, dmu_dr)
            # dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += 0.5 * cupy.einsum(
            #     "g,dxpg,xqg->dpq", depsilon_dtau * weight, d2mu_dr2, dmu_dr)
            mu_weight_depsilondrho = contract("pg,ug->upg", mu, weight_depsilon_drho)
            dmudr_weight_depsilondnablarho = contract("uxg,xpg->upg", weight_depsilon_dnablarho, dmu_dr)
            dmudA_nu_ao = contract("dpg,uqg->udpq", dmu_dr, mu_weight_depsilondrho + dmudr_weight_depsilondnablarho)
            del mu_weight_depsilondrho, dmudr_weight_depsilondnablarho
            dmudr_weight_depsilondtau = 0.5 * contract("dpg,ug->udpg", dmu_dr, weight_depsilon_dtau)
            mu_weight_depsilondnablarho = contract("uxg,qg->uxqg", weight_depsilon_dnablarho, mu)
            dmudA_nu_ao += contract("dxpg,uxqg->udpq", d2mu_dr2, dmudr_weight_depsilondtau + mu_weight_depsilondnablarho)
            del mu_weight_depsilondnablarho, dmudr_weight_depsilondtau

            dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(2), range(3), idx, idx)] += dmudA_nu_ao

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
            dmudA_nu_ao_ao = dmudA_nu_ao + dmudA_nu_ao.transpose(0,1,3,2)
            del dmudA_nu_ao
            dFock_sparse_ao_occa[i_atom_of_grids, :, :, :] += contract("dpq,qj->dpj", dmudA_nu_ao_ao[0], mocca_masked)
            dFock_sparse_ao_occb[i_atom_of_grids, :, :, :] += contract("dpq,qj->dpj", dmudA_nu_ao_ao[1], moccb_masked)
            del dmudA_nu_ao_ao

            dFock_ao_occa[:, :, idx, :] += dFock_sparse_ao_occa
            dFock_ao_occb[:, :, idx, :] += dFock_sparse_ao_occb
            del dFock_sparse_ao_occa, dFock_sparse_ao_occb

            g0 = g1
        assert g1 == ngrids

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f"xctype = {xctype} not supported")

    dFock_mo_occa = contract("Adpj,pi->Adij", dFock_ao_occa, mo_coeff_sorted[0])
    dFock_mo_occb = contract("Adpj,pi->Adij", dFock_ao_occb, mo_coeff_sorted[1])
    del dFock_ao_occa, dFock_ao_occb
    for i_atom in range(0, natm):
        ao_of_atom_i = cupy.where(i_atom_of_aos == i_atom)[0]
        if ao_of_atom_i.size > 0:
            dFock_ao_of_atom_i = dFock_orbital_response_dmudA_nu_term[:, :, ao_of_atom_i, :]

            dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i[0] @ mocca_sorted
            dFock_mo_occa[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff_sorted[0][ao_of_atom_i, :], dFock_ao_of_atom_i_mocc)
            dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i[0].transpose(0,2,1) @ mocca_sorted[ao_of_atom_i]
            dFock_mo_occa[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff_sorted[0], dFock_ao_of_atom_i_mocc)

            dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i[1] @ moccb_sorted
            dFock_mo_occb[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff_sorted[1][ao_of_atom_i, :], dFock_ao_of_atom_i_mocc)
            dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i[1].transpose(0,2,1) @ moccb_sorted[ao_of_atom_i]
            dFock_mo_occb[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff_sorted[1], dFock_ao_of_atom_i_mocc)
            del dFock_ao_of_atom_i_mocc, dFock_ao_of_atom_i
    del dFock_orbital_response_dmudA_nu_term

    return (dFock_mo_occa, dFock_mo_occb)

def get_veff_resp_mo(hessobj, mol, dms, mo_coeff, mo_occ, hermi=1):
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

    nao, nmoa = mo_coeff[0].shape
    nao, nmob = mo_coeff[1].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]

    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    hermi = 1

    rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, mf.xc,
                                        mo_coeff, mo_occ, 1)
    v1 = ni.nr_uks_fxc(mol, grids, mf.xc, None, dms, 0, hermi,
                        rho0, vxc, fxc, max_memory=None)
    nset = dms.shape[1]
    v1vo = cupy.empty([nset, nmoa*nocca+nmob*noccb])
    v1vo[:,:nmoa*nocca] = _ao2mo(v1[0], mocca, mo_coeff[0]).reshape(-1,nmoa*nocca)
    v1vo[:,nmoa*nocca:] = _ao2mo(v1[1], moccb, mo_coeff[1]).reshape(-1,nmob*noccb)

    if mf.do_nlc():
        vnlca, vnlcb = nr_uks_fnlc_mo(mf, mol, mo_coeff, mo_occ, dms)
        v1vo[:,:nmoa*nocca] += vnlca.reshape(-1, nmoa*nocca)
        v1vo[:,nmoa*nocca:] += vnlcb.reshape(-1, nmob*noccb)

    if hybrid:
        vj, vk = hessobj.get_jk_mo(mol, dms, mo_coeff, mo_occ, hermi=1)
        vk *= hyb
        if omega > 1e-10:
            _, vk_lr = hessobj.get_jk_mo(mol, dms, mo_coeff, mo_occ,
                                         hermi, with_j=False, omega=omega)
            vk_lr *= (alpha-hyb)
            vk += vk_lr
        v1vo += vj - vk
    else:
        v1vo += hessobj.get_jk_mo(mol, dms, mo_coeff, mo_occ,
                                  hermi=1, with_k=False)[0]
    return v1vo

def nr_uks_fnlc_mo(mf, mol, mo_coeff, mo_occ, dm1s, return_in_mo=True):
    """
    The UKS version of nr_rks_fnlc_mo

    Args:
        mo_coeff: array of shape (2, nao, nmo)
            0-th order UKS MO coefficients
        mo_occ: array of shape (2, nmo)
            0-th order UKS MO occupancies
        dm1s: array of shape (2, *, nao, nao)
            First order UKS density matrices
        return_in_mo:
            Whether to return NLC matrices in MO representations. When UKS
            orbitals are supplied, a two-element tuple of matrices
            (spin-up, spin-down) are evaluated and returned.
    """
    assert mo_coeff.ndim == 3
    return nr_rks_fnlc_mo(mf, mol, mo_coeff, mo_occ, dm1s[0]+dm1s[1], return_in_mo)

class Hessian(rhf_hess.HessianBase):
    '''Non-relativistic UKS hessian'''

    def __init__(self, mf):
        rhf_hess.Hessian.__init__(self, mf)
        self.grids = None
        self.grid_response = False

    hess_elec = uhf_hess.hess_elec
    solve_mo1 = uhf_hess.Hessian.solve_mo1
    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1
    gen_vind = uhf_hess.gen_vind
    get_jk_mo = uhf_hess._get_jk_mo
    get_veff_resp_mo = get_veff_resp_mo

from gpu4pyscf import dft
dft.uks.UKS.Hessian = lib.class_as_method(Hessian)
