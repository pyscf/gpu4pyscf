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
import cupy as cp
from pyscf import lib
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.hessian import uhf as uhf_hess
from gpu4pyscf.hessian.rhf import _ao2mo
from gpu4pyscf.hessian.rks import get_dweight_dA, get_d2weight_dAdB, get_d2mu_dr2, get_d3mu_dr3, get_drho_dA_full, contract_d2rho_dAdB_full
from gpu4pyscf.hessian.rks import _get_enlc_deriv2, _get_vnlc_deriv1, nr_rks_fnlc_mo
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import (contract, add_sparse, get_avail_mem)
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
    if hessobj.grid_response:
        log.info("Calculating grid response for unrestricted DFT Hessian")
        de2 += _get_exc_deriv2_grid_response(hessobj, mo_coeff, mo_occ, max_memory)

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
            vxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 1, xctype=xctype)[1]
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
            vxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 1, xctype=xctype)[1]
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
            vxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 1, xctype=xctype)[1]
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
            vxc, fxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 2, xctype=xctype)[1:3]
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
            vxc, fxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 2, xctype=xctype)[1:3]
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
            vxc, fxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa, rhob)), 2, xctype=xctype)[1:3]
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
    assert dm0.ndim == 3 and dm0.shape[0] == 2

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
            rhoa = numint.eval_rho2(mol, split_ao, mo_coeff[0], mo_occ[0], xctype=xctype)
            rhob = numint.eval_rho2(mol, split_ao, mo_coeff[1], mo_occ[1], xctype=xctype)
            rho = cupy.asarray((rhoa, rhob))
            exc = ni.eval_xc_eff(mf.xc, rho, deriv = 0, xctype=xctype)[0]
            rho = None

            epsilon = exc[:, 0] * (rhoa + rhob)
            rhoa = None
            rhob = None
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

            rhoa = numint.eval_rho2(mol, mu, mo_coeff[0], mo_occ[0], xctype=xctype)
            rhob = numint.eval_rho2(mol, mu, mo_coeff[1], mo_occ[1], xctype=xctype)
            rho = cupy.asarray((rhoa, rhob))
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype=xctype)[1:3]
            rho = None
            rhoa = None
            rhob = None

            depsilon_drho = vxc[:,0,:] # Just of shape (2,ngrids)
            d2epsilon_drho2 = fxc[:,0,:,0,:] # Just of shape (2,2,ngrids)

            grid_to_atom_index_map = grids.atm_idx[g0:g1]
            atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]
            grid_to_atom_index_map = None

            drho_dA_orbital_response = cp.zeros((2, natm, 3, g1-g0))
            drho_dA_grid_response = cp.zeros((2, natm, 3, g1-g0))
            for i_dm in range(2):
                drho_dA_orbital_response_i, drho_dA_grid_response_i = \
                    get_drho_dA_full(dm0[i_dm], xctype, natm, g1 - g0, aoslices, atom_to_grid_index_map, mu, dmu_dr)
                drho_dA_orbital_response[i_dm] = drho_dA_orbital_response_i
                drho_dA_grid_response[i_dm] = drho_dA_grid_response_i
                drho_dA_orbital_response_i = None
                drho_dA_grid_response_i = None

            drho_dA_full = drho_dA_orbital_response + drho_dA_grid_response

            dw_dA = get_dweight_dA(mol, grids, (g0,g1))
            depsilondrho_drhodA = contract("uAdg,ug->Adg", drho_dA_full, depsilon_drho)
            d2e_dwdA_term = contract("Adg,BDg->ABdD", dw_dA, depsilondrho_drhodA)
            depsilondrho_drhodA = None
            drho_dA_full = None
            d2e += d2e_dwdA_term + d2e_dwdA_term.transpose(1,0,3,2)
            d2e_dwdA_term = None
            dw_dA = None

            weight = grids.weights[g0:g1]
            drhodA_grid_response_weight_d2epsilondrho2 = contract("uAdg,uvg->vAdg", drho_dA_grid_response, d2epsilon_drho2 * weight)
            d2e_drhodA_cross_term = contract("uAdg,uBDg->ABdD", drho_dA_orbital_response, drhodA_grid_response_weight_d2epsilondrho2)
            drho_dA_orbital_response = None
            d2e_drhodA_cross_term += d2e_drhodA_cross_term.transpose(1,0,3,2)
            d2e_drhodA_cross_term += contract("uAdg,uBDg->ABdD", drho_dA_grid_response, drhodA_grid_response_weight_d2epsilondrho2)
            drho_dA_grid_response = None
            drhodA_grid_response_weight_d2epsilondrho2 = None
            d2e += d2e_drhodA_cross_term
            d2e_drhodA_cross_term = None

            for i_dm in range(2):
                d2e += contract_d2rho_dAdB_full(dm0[i_dm], xctype, natm, g1 - g0, aoslices, atom_to_grid_index_map,
                                                mu, dmu_dr, d2mu_dr2, None,
                                                depsilon_drho[i_dm] * weight,
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

            rho_drhoa = numint.eval_rho2(mol, split_ao, mo_coeff[0], mo_occ[0], xctype=xctype)
            rho_drhob = numint.eval_rho2(mol, split_ao, mo_coeff[1], mo_occ[1], xctype=xctype)
            rho_drho = cupy.asarray((rho_drhoa, rho_drhob))
            rho_drhoa = None
            rho_drhob = None
            exc = ni.eval_xc_eff(mf.xc, rho_drho, deriv = 0, xctype=xctype)[0]

            rho = rho_drho[:,0]
            rho_drho = None

            epsilon = exc[:, 0] * (rho[0] + rho[1])
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

            rho_drhoa = numint.eval_rho2(mol, split_ao, mo_coeff[0], mo_occ[0], xctype=xctype)
            rho_drhob = numint.eval_rho2(mol, split_ao, mo_coeff[1], mo_occ[1], xctype=xctype)
            rho_drho = cupy.asarray((rho_drhoa, rho_drhob))
            rho_drhoa = None
            rho_drhob = None
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho_drho, deriv = 2, xctype=xctype)[1:3]
            rho_drho = None

            depsilon_drho = vxc[:, 0, :]
            depsilon_dnablarho = vxc[:, 1:4, :]

            grid_to_atom_index_map = grids.atm_idx[g0:g1]
            atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]
            grid_to_atom_index_map = None

            drho_dA_orbital_response = cp.zeros((2, natm, 3, g1-g0))
            drho_dA_grid_response = cp.zeros((2, natm, 3, g1-g0))
            dnablarho_dA_orbital_response = cp.zeros((2, natm, 3, 3, g1-g0))
            dnablarho_dA_grid_response = cp.zeros((2, natm, 3, 3, g1-g0))
            for i_dm in range(2):
                drho_dA_orbital_response_i, dnablarho_dA_orbital_response_i, drho_dA_grid_response_i, dnablarho_dA_grid_response_i = \
                    get_drho_dA_full(dm0[i_dm], xctype, natm, g1 - g0, aoslices, atom_to_grid_index_map, mu, dmu_dr, d2mu_dr2)
                drho_dA_orbital_response[i_dm] = drho_dA_orbital_response_i
                drho_dA_grid_response[i_dm] = drho_dA_grid_response_i
                dnablarho_dA_orbital_response[i_dm] = dnablarho_dA_orbital_response_i
                dnablarho_dA_grid_response[i_dm] = dnablarho_dA_grid_response_i
                drho_dA_orbital_response_i = None
                drho_dA_grid_response_i = None
                dnablarho_dA_orbital_response_i = None
                dnablarho_dA_grid_response_i = None

            drho_dA_full = drho_dA_orbital_response + drho_dA_grid_response
            dnablarho_dA_full = dnablarho_dA_orbital_response + dnablarho_dA_grid_response

            dw_dA = get_dweight_dA(mol, grids, (g0,g1))
            depsilondnablarho_dnablarhodA = contract("uxg,uAdxg->Adg", depsilon_dnablarho, dnablarho_dA_full)
            dnablarho_dA_full = None
            depsilondrho_drhodA = contract("uAdg,ug->Adg", drho_dA_full, depsilon_drho)
            d2e_dwdA_term = contract("Adg,BDg->ABdD", dw_dA, depsilondrho_drhodA + depsilondnablarho_dnablarhodA)
            depsilondrho_drhodA = None
            drho_dA_full = None
            depsilondnablarho_dnablarhodA = None
            dw_dA = None
            d2e += d2e_dwdA_term + d2e_dwdA_term.transpose(1,0,3,2)
            d2e_dwdA_term = None

            weight = grids.weights[g0:g1]
            combined_d_dA_orbital_response = cupy.concatenate((drho_dA_orbital_response[:, :, :, None, :], dnablarho_dA_orbital_response), axis = 3)
            combined_d_dA_grid_response    = cupy.concatenate((   drho_dA_grid_response[:, :, :, None, :],    dnablarho_dA_grid_response), axis = 3)
            drho_dA_orbital_response = None
            dnablarho_dA_orbital_response = None
            drho_dA_grid_response = None
            dnablarho_dA_grid_response = None
            fwxc = fxc * weight
            fxc = None

            drhodA_grid_response_fwxc = contract("uxvyg,vAdyg->uAdxg", fwxc, combined_d_dA_grid_response)
            fwxc = None
            d2e_drhodA_cross_term = contract("uAdxg,uBDxg->ABdD", combined_d_dA_orbital_response, drhodA_grid_response_fwxc)
            combined_d_dA_orbital_response = None
            d2e_drhodA_cross_term += d2e_drhodA_cross_term.transpose(1,0,3,2)
            d2e_drhodA_cross_term += contract("uAdxg,uBDxg->ABdD", combined_d_dA_grid_response, drhodA_grid_response_fwxc)
            combined_d_dA_grid_response = None
            drhodA_grid_response_fwxc = None
            d2e += d2e_drhodA_cross_term
            d2e_drhodA_cross_term = None

            for i_dm in range(2):
                d2e += contract_d2rho_dAdB_full(dm0[i_dm], xctype, natm, g1 - g0, aoslices, atom_to_grid_index_map,
                                                mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                                                depsilon_drho[i_dm] * weight, depsilon_dnablarho[i_dm] * weight,
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

            rho_drho_taua = numint.eval_rho2(mol, split_ao, mo_coeff[0], mo_occ[0], xctype=xctype)
            rho_drho_taub = numint.eval_rho2(mol, split_ao, mo_coeff[1], mo_occ[1], xctype=xctype)
            rho_drho_tau = cupy.asarray((rho_drho_taua, rho_drho_taub))
            rho_drho_taua = None
            rho_drho_taub = None
            exc = ni.eval_xc_eff(mf.xc, rho_drho_tau, deriv = 0, xctype=xctype)[0]

            rho = rho_drho_tau[:, 0]
            rho_drho_tau = None

            epsilon = exc[:, 0] * (rho[0] + rho[1])
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

            rho_drho_taua = numint.eval_rho2(mol, split_ao, mo_coeff[0], mo_occ[0], xctype=xctype)
            rho_drho_taub = numint.eval_rho2(mol, split_ao, mo_coeff[1], mo_occ[1], xctype=xctype)
            rho_drho_tau = cupy.asarray((rho_drho_taua, rho_drho_taub))
            rho_drho_taua = None
            rho_drho_taub = None
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho_drho_tau, deriv = 2, xctype=xctype)[1:3]
            rho_drho_tau = None

            depsilon_drho = vxc[:, 0, :]
            depsilon_dnablarho = vxc[:, 1:4, :]
            depsilon_dtau = vxc[:, 4, :]

            grid_to_atom_index_map = grids.atm_idx[g0:g1]
            atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]
            grid_to_atom_index_map = None

            drho_dA_orbital_response = cp.zeros((2, natm, 3, g1-g0))
            drho_dA_grid_response = cp.zeros((2, natm, 3, g1-g0))
            dnablarho_dA_orbital_response = cp.zeros((2, natm, 3, 3, g1-g0))
            dnablarho_dA_grid_response = cp.zeros((2, natm, 3, 3, g1-g0))
            dtau_dA_orbital_response = cp.zeros((2, natm, 3, g1-g0))
            dtau_dA_grid_response = cp.zeros((2, natm, 3, g1-g0))
            for i_dm in range(2):
                drho_dA_orbital_response_i, dnablarho_dA_orbital_response_i, dtau_dA_orbital_response_i, \
                    drho_dA_grid_response_i, dnablarho_dA_grid_response_i, dtau_dA_grid_response_i = \
                    get_drho_dA_full(dm0[i_dm], xctype, natm, g1 - g0, aoslices, atom_to_grid_index_map, mu, dmu_dr, d2mu_dr2)
                drho_dA_orbital_response[i_dm] = drho_dA_orbital_response_i
                drho_dA_grid_response[i_dm] = drho_dA_grid_response_i
                dnablarho_dA_orbital_response[i_dm] = dnablarho_dA_orbital_response_i
                dnablarho_dA_grid_response[i_dm] = dnablarho_dA_grid_response_i
                dtau_dA_orbital_response[i_dm] = dtau_dA_orbital_response_i
                dtau_dA_grid_response[i_dm] = dtau_dA_grid_response_i
                drho_dA_orbital_response_i = None
                drho_dA_grid_response_i = None
                dnablarho_dA_orbital_response_i = None
                dnablarho_dA_grid_response_i = None
                dtau_dA_orbital_response_i = None
                dtau_dA_grid_response_i = None

            drho_dA_full = drho_dA_orbital_response + drho_dA_grid_response
            dnablarho_dA_full = dnablarho_dA_orbital_response + dnablarho_dA_grid_response
            dtau_dA_full = dtau_dA_orbital_response + dtau_dA_grid_response

            dw_dA = get_dweight_dA(mol, grids, (g0,g1))
            depsilondnablarho_dnablarhodA = contract("uxg,uAdxg->Adg", depsilon_dnablarho, dnablarho_dA_full)
            dnablarho_dA_full = None
            depsilondrho_drhodA = contract("uAdg,ug->Adg", drho_dA_full, depsilon_drho)
            depsilondtau_drhodA = contract("uAdg,ug->Adg", dtau_dA_full, depsilon_dtau)
            d2e_dwdA_term = contract("Adg,BDg->ABdD", dw_dA, depsilondrho_drhodA + depsilondnablarho_dnablarhodA + depsilondtau_drhodA)
            depsilondrho_drhodA = None
            depsilondtau_drhodA = None
            drho_dA_full = None
            dtau_dA_full = None
            depsilondnablarho_dnablarhodA = None
            dw_dA = None
            d2e += d2e_dwdA_term + d2e_dwdA_term.transpose(1,0,3,2)
            d2e_dwdA_term = None

            weight = grids.weights[g0:g1]
            combined_d_dA_orbital_response = cupy.concatenate(
                (drho_dA_orbital_response[:, :, :, None, :], dnablarho_dA_orbital_response, dtau_dA_orbital_response[:, :, :, None, :]),
                axis = 3
            )
            combined_d_dA_grid_response    = cupy.concatenate(
                (   drho_dA_grid_response[:, :, :, None, :],    dnablarho_dA_grid_response,    dtau_dA_grid_response[:, :, :, None, :]),
                axis = 3
            )
            drho_dA_orbital_response = None
            dnablarho_dA_orbital_response = None
            dtau_dA_orbital_response = None
            drho_dA_grid_response = None
            dnablarho_dA_grid_response = None
            dtau_dA_grid_response = None
            fwxc = fxc * weight
            fxc = None

            drhodA_grid_response_fwxc = contract("uxvyg,vAdyg->uAdxg", fwxc, combined_d_dA_grid_response)
            fwxc = None
            d2e_drhodA_cross_term = contract("uAdxg,uBDxg->ABdD", combined_d_dA_orbital_response, drhodA_grid_response_fwxc)
            combined_d_dA_orbital_response = None
            d2e_drhodA_cross_term += d2e_drhodA_cross_term.transpose(1,0,3,2)
            d2e_drhodA_cross_term += contract("uAdxg,uBDxg->ABdD", combined_d_dA_grid_response, drhodA_grid_response_fwxc)
            combined_d_dA_grid_response = None
            drhodA_grid_response_fwxc = None
            d2e += d2e_drhodA_cross_term
            d2e_drhodA_cross_term = None

            for i_dm in range(2):
                d2e += contract_d2rho_dAdB_full(dm0[i_dm], xctype, natm, g1 - g0, aoslices, atom_to_grid_index_map,
                                                mu, dmu_dr, d2mu_dr2, d3mu_dr3,
                                                depsilon_drho[i_dm] * weight, depsilon_dnablarho[i_dm] * weight, depsilon_dtau[i_dm] * weight,
                                                with_orbital_response = False)

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f"xctype = {xctype} not supported")

    return d2e

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
            vxc, fxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 2, xctype=xctype)[1:3]
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
            vxc, fxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 2, xctype=xctype)[1:3]
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
            vxc, fxc = ni.eval_xc_eff(mf.xc, cupy.asarray((rhoa,rhob)), 2, xctype=xctype)[1:3]
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

    if hessobj.grid_response:
        t2 = log.init_timer()
        grid_start, grid_end = 0, grids.coords.shape[0]
        dm0 = (dm0a, dm0b)

        if xctype == 'LDA':
            # If you wonder why not using ni.block_loop(), because I need the exact grid index range (g0, g1).
            available_gpu_memory = get_avail_mem()
            available_gpu_memory = int(available_gpu_memory * 0.5) # Don't use too much gpu memory
            ao_nbytes_per_grid = ((4 + 3 + 2 + 3*2 + 1) * mol.nao + (3*2) * mol.natm + 16 + 4) * 8
            ngrids_per_batch = int(available_gpu_memory / ao_nbytes_per_grid)
            if ngrids_per_batch < 16:
                raise MemoryError(f"Out of GPU memory for LDA Fock first derivative, available gpu memory = {get_avail_mem()}"
                                  f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (one GPU) = {grid_end - grid_start}")
            ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
            ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

            for g0 in range(grid_start, grid_end, ngrids_per_batch):
                g1 = min(g0 + ngrids_per_batch, grid_end)
                split_grids_coords = cupy.asarray(grids.coords)[g0:g1, :]
                split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 1, gdftopt = None, transpose = False)

                mu = split_ao[0]
                dmu_dr = split_ao[1:4]

                rhoa = numint.eval_rho2(mol, mu, mo_coeff[0], mo_occ[0], xctype=xctype)
                rhob = numint.eval_rho2(mol, mu, mo_coeff[1], mo_occ[1], xctype=xctype)
                rho = cupy.asarray((rhoa, rhob))
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, deriv = 2, xctype=xctype)[1:3]
                rho = None

                depsilon_drho = vxc[:,0,:] # Just of shape (2,ngrids)
                d2epsilon_drho2 = fxc[:,0,:,0,:] # Just of shape (2,2,ngrids)

                dw_dA = get_dweight_dA(mol, grids, (g0,g1))
                # # Negative here to cancel the overall negative sign before return
                dwdA_depsilondrho = contract("Adg,ug->uAdg", dw_dA, depsilon_drho)
                dw_dA = None
                mu_occ = (mu.T @ mocca, mu.T @ moccb)
                for i_atom in range(natm):
                    dwdA_depsilondrho_mu = contract("udg,pg->udpg", dwdA_depsilondrho[:, i_atom, :, :], mu)
                    vmata[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_depsilondrho_mu[0], mu_occ[0])
                    vmatb[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_depsilondrho_mu[1], mu_occ[1])
                    dwdA_depsilondrho_mu = None
                dwdA_depsilondrho = None

                grid_to_atom_index_map = cupy.asarray(grids.atm_idx)[g0:g1]
                atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]
                grid_to_atom_index_map = None

                drho_dA_grid_response = cp.zeros((2, natm, 3, g1-g0))
                for i_dm in range(2):
                    _, drho_dA_grid_response_i = \
                        get_drho_dA_full(dm0[i_dm], xctype, natm, g1 - g0, None, atom_to_grid_index_map, mu, dmu_dr, with_orbital_response = False)
                    drho_dA_grid_response[i_dm] = drho_dA_grid_response_i
                    drho_dA_grid_response_i = None

                weight = cupy.asarray(grids.weights)[g0:g1]
                # # Negative here to cancel the overall negative sign before return
                weight_d2epsilondrho2_drhodA_grid_response = contract("uAdg,uvg->vAdg", drho_dA_grid_response, d2epsilon_drho2 * weight)
                drho_dA_grid_response = None
                for i_atom in range(natm):
                    weight_d2epsilondrho2_drhodA_grid_response_mu = contract("vdg,pg->vdpg", weight_d2epsilondrho2_drhodA_grid_response[:, i_atom, :, :], mu)
                    vmata[i_atom, :, :, :] -= contract("dpg,gj->dpj", weight_d2epsilondrho2_drhodA_grid_response_mu[0], mu_occ[0])
                    vmatb[i_atom, :, :, :] -= contract("dpg,gj->dpj", weight_d2epsilondrho2_drhodA_grid_response_mu[1], mu_occ[1])
                    weight_d2epsilondrho2_drhodA_grid_response_mu = None
                mu_occ = None

                for i_atom in range(natm):
                    associated_grid_index = atom_to_grid_index_map[i_atom]
                    if len(associated_grid_index) == 0:
                        continue

                    mu_grid_i = mu[:, associated_grid_index]
                    weight_depsilondrho_dmudr_grid_i = contract("dpg,ug->udpg",
                                                                dmu_dr[:, :, associated_grid_index],
                                                                depsilon_drho[:, associated_grid_index] * weight[associated_grid_index])
                    mu_occ_grid_i = (mu_grid_i.T @ mocca, mu_grid_i.T @ moccb)
                    vmata[i_atom, :, :, :] -= weight_depsilondrho_dmudr_grid_i[0] @ mu_occ_grid_i[0]
                    vmatb[i_atom, :, :, :] -= weight_depsilondrho_dmudr_grid_i[1] @ mu_occ_grid_i[1]
                    mu_occ_grid_i = None

                    weight_depsilondrho_dmudr_occ_grid_i = contract("dqg,qj->djg", weight_depsilondrho_dmudr_grid_i[0], mocca)
                    vmata[i_atom, :, :, :] -= contract("pg,djg->dpj", mu_grid_i, weight_depsilondrho_dmudr_occ_grid_i)
                    weight_depsilondrho_dmudr_occ_grid_i = contract("dqg,qj->djg", weight_depsilondrho_dmudr_grid_i[1], moccb)
                    vmatb[i_atom, :, :, :] -= contract("pg,djg->dpj", mu_grid_i, weight_depsilondrho_dmudr_occ_grid_i)
                    weight_depsilondrho_dmudr_grid_i = None
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
                                  f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (one GPU) = {grid_end - grid_start}")
            ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
            ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

            for g0 in range(grid_start, grid_end, ngrids_per_batch):
                g1 = min(g0 + ngrids_per_batch, grid_end)
                split_grids_coords = cupy.asarray(grids.coords)[g0:g1, :]
                split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2, gdftopt = None, transpose = False)

                mu = split_ao[0]
                dmu_dr = split_ao[1:4]
                d2mu_dr2 = get_d2mu_dr2(split_ao)

                rho_drhoa = numint.eval_rho2(mol, split_ao[:4], mo_coeff[0], mo_occ[0], xctype=xctype)
                rho_drhob = numint.eval_rho2(mol, split_ao[:4], mo_coeff[1], mo_occ[1], xctype=xctype)
                rho_drho = cupy.asarray((rho_drhoa, rho_drhob))
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho_drho, deriv = 2, xctype=xctype)[1:3]
                rho_drho = None

                depsilon_drho = vxc[:, 0, :]
                depsilon_dnablarho = vxc[:, 1:4, :]

                dw_dA = get_dweight_dA(mol, grids, (g0,g1))
                depsilondnablarho_dmudr = contract("uxg,xpg->upg", depsilon_dnablarho, dmu_dr)
                depsilondrho_mu = contract("pg,ug->upg", mu, depsilon_drho)
                mu_occ = (mu.T @ mocca, mu.T @ moccb)
                for i_atom in range(natm):
                    dwdA_depsilondrho_mu = contract("dg,upg->udpg", dw_dA[i_atom, :, :], depsilondrho_mu + depsilondnablarho_dmudr)
                    vmata[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_depsilondrho_mu[0], mu_occ[0])
                    vmatb[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_depsilondrho_mu[1], mu_occ[1])
                    dwdA_depsilondrho_mu = None
                depsilondrho_mu = None
                mu_occ = None
                depsilondnablarho_dmudr_occ = (depsilondnablarho_dmudr[0].T @ mocca, depsilondnablarho_dmudr[1].T @ moccb)
                depsilondnablarho_dmudr = None
                for i_atom in range(natm):
                    dwdA_mu = contract("dg,pg->dpg", dw_dA[i_atom, :, :], mu)
                    vmata[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_mu, depsilondnablarho_dmudr_occ[0])
                    vmatb[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_mu, depsilondnablarho_dmudr_occ[1])
                    dwdA_mu = None
                depsilondnablarho_dmudr_occ = None
                dw_dA = None

                grid_to_atom_index_map = cupy.asarray(grids.atm_idx)[g0:g1]
                atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]
                grid_to_atom_index_map = None

                drho_dA_grid_response = cp.zeros((2, natm, 3, g1-g0))
                dnablarho_dA_grid_response = cp.zeros((2, natm, 3, 3, g1-g0))
                for i_dm in range(2):
                    _, _, drho_dA_grid_response_i, dnablarho_dA_grid_response_i = \
                        get_drho_dA_full(dm0[i_dm], xctype, natm, g1 - g0, None, atom_to_grid_index_map, mu, dmu_dr, d2mu_dr2, with_orbital_response = False)
                    drho_dA_grid_response[i_dm] = drho_dA_grid_response_i
                    dnablarho_dA_grid_response[i_dm] = dnablarho_dA_grid_response_i
                    drho_dA_grid_response_i = None
                    dnablarho_dA_grid_response_i = None

                weight = cupy.asarray(grids.weights)[g0:g1]
                # # Negative here to cancel the overall negative sign before return
                combined_d_dA_grid_response = cupy.concatenate((drho_dA_grid_response[:, :, :, None, :], dnablarho_dA_grid_response), axis = 3)
                drho_dA_grid_response = None
                dnablarho_dA_grid_response = None

                fwxc = fxc * weight
                fxc = None
                drhodA_grid_response_fwxc = contract("uxvyg,vAdyg->uAdxg", fwxc, combined_d_dA_grid_response)
                combined_d_dA_grid_response = None
                fwxc = None

                mu_occ = (mu.T @ mocca, mu.T @ moccb)
                dmudr_occ = (
                    contract("dqg,qj->dgj", dmu_dr, mocca),
                    contract("dqg,qj->dgj", dmu_dr, moccb),
                )
                for i_atom in range(natm):
                    drhodA_grid_response_fwxc_rho_term_mu = contract("udg,pg->udpg", drhodA_grid_response_fwxc[:, i_atom, :, 0, :], mu)
                    vmata[i_atom, :, :, :] -= contract("dpg,gj->dpj", drhodA_grid_response_fwxc_rho_term_mu[0], mu_occ[0])
                    vmatb[i_atom, :, :, :] -= contract("dpg,gj->dpj", drhodA_grid_response_fwxc_rho_term_mu[1], mu_occ[1])
                    drhodA_grid_response_fwxc_rho_term_mu = None
                    drhodA_grid_response_fwxc_nablarho_term_dmudr_occ = contract("dxg,xgj->dgj", drhodA_grid_response_fwxc[0, i_atom, :, 1:4, :], dmudr_occ[0])
                    vmata[i_atom, :, :, :] -= contract("dgj,pg->dpj", drhodA_grid_response_fwxc_nablarho_term_dmudr_occ, mu)
                    drhodA_grid_response_fwxc_nablarho_term_dmudr_occ = contract("dxg,xgj->dgj", drhodA_grid_response_fwxc[1, i_atom, :, 1:4, :], dmudr_occ[1])
                    vmatb[i_atom, :, :, :] -= contract("dgj,pg->dpj", drhodA_grid_response_fwxc_nablarho_term_dmudr_occ, mu)
                    drhodA_grid_response_fwxc_nablarho_term_dmudr_occ = None
                    drhodA_grid_response_fwxc_nablarho_term_dmudr = contract("udxg,xpg->udpg", drhodA_grid_response_fwxc[:, i_atom, :, 1:4, :], dmu_dr)
                    vmata[i_atom, :, :, :] -= contract("dpg,gj->dpj", drhodA_grid_response_fwxc_nablarho_term_dmudr[0], mu_occ[0])
                    vmatb[i_atom, :, :, :] -= contract("dpg,gj->dpj", drhodA_grid_response_fwxc_nablarho_term_dmudr[1], mu_occ[1])
                    drhodA_grid_response_fwxc_nablarho_term_dmudr = None
                drhodA_grid_response_fwxc = None
                mu_occ = None
                dmudr_occ = None

                for i_atom in range(natm):
                    associated_grid_index = atom_to_grid_index_map[i_atom]
                    if len(associated_grid_index) == 0:
                        continue
                    # # Negative here to cancel the overall negative sign before return
                    mu_grid_i = mu[:, associated_grid_index]
                    dmu_dr_grid_i = dmu_dr[:, :, associated_grid_index]

                    mu_occ_grid_i = (mu_grid_i.T @ mocca, mu_grid_i.T @ moccb)
                    dmudr_occ_grid_i = (
                        contract("dqg,qj->dgj", dmu_dr_grid_i, mocca),
                        contract("dqg,qj->dgj", dmu_dr_grid_i, moccb),
                    )

                    weight_depsilondrho_grid_i = depsilon_drho[:, associated_grid_index] * weight[associated_grid_index]
                    vmata[i_atom, :, :, :] -= (dmu_dr_grid_i * weight_depsilondrho_grid_i[0]) @ mu_occ_grid_i[0]
                    vmatb[i_atom, :, :, :] -= (dmu_dr_grid_i * weight_depsilondrho_grid_i[1]) @ mu_occ_grid_i[1]
                    vmata[i_atom, :, :, :] -= contract("pg,dgj->dpj", mu_grid_i * weight_depsilondrho_grid_i[0], dmudr_occ_grid_i[0])
                    vmatb[i_atom, :, :, :] -= contract("pg,dgj->dpj", mu_grid_i * weight_depsilondrho_grid_i[1], dmudr_occ_grid_i[1])
                    weight_depsilondrho_grid_i = None

                    d2mu_dr2_grid_i = d2mu_dr2[:, :, :, associated_grid_index]

                    weight_depsilondnablarho_grid_i = depsilon_dnablarho[:, :, associated_grid_index] * weight[associated_grid_index]
                    weight_depsilondnablarho_d2mudr2 = contract("uDg,dDpg->udpg", weight_depsilondnablarho_grid_i, d2mu_dr2_grid_i)
                    d2mu_dr2_grid_i = None
                    vmata[i_atom, :, :, :] -= contract("dpg,gj->dpj", weight_depsilondnablarho_d2mudr2[0], mu_occ_grid_i[0])
                    vmatb[i_atom, :, :, :] -= contract("dpg,gj->dpj", weight_depsilondnablarho_d2mudr2[1], mu_occ_grid_i[1])
                    mu_occ_grid_i = None
                    weight_depsilondnablarho_d2mudr2_occ = contract("dpg,pj->dgj", weight_depsilondnablarho_d2mudr2[0], mocca)
                    vmata[i_atom, :, :, :] -= contract("pg,dgj->dpj", mu_grid_i, weight_depsilondnablarho_d2mudr2_occ)
                    weight_depsilondnablarho_d2mudr2_occ = contract("dpg,pj->dgj", weight_depsilondnablarho_d2mudr2[1], moccb)
                    vmatb[i_atom, :, :, :] -= contract("pg,dgj->dpj", mu_grid_i, weight_depsilondnablarho_d2mudr2_occ)
                    weight_depsilondnablarho_d2mudr2 = None
                    mu_grid_i = None
                    weight_depsilondnablarho_d2mudr2_occ = None
                    weight_depsilondnablarho_dmudr = contract("uDg,Dpg->upg", weight_depsilondnablarho_grid_i, dmu_dr_grid_i)
                    vmata[i_atom, :, :, :] -= contract("pg,dgj->dpj", weight_depsilondnablarho_dmudr[0], dmudr_occ_grid_i[0])
                    vmatb[i_atom, :, :, :] -= contract("pg,dgj->dpj", weight_depsilondnablarho_dmudr[1], dmudr_occ_grid_i[1])
                    dmudr_occ_grid_i = None
                    weight_depsilondnablarho_dmudr_occ = (weight_depsilondnablarho_dmudr[0].T @ mocca, weight_depsilondnablarho_dmudr[1].T @ moccb)
                    weight_depsilondnablarho_dmudr = None
                    vmata[i_atom, :, :, :] -= contract("dpg,gj->dpj", dmu_dr_grid_i, weight_depsilondnablarho_dmudr_occ[0])
                    vmatb[i_atom, :, :, :] -= contract("dpg,gj->dpj", dmu_dr_grid_i, weight_depsilondnablarho_dmudr_occ[1])
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
                                  f" bytes, nao = {mol.nao}, natm = {mol.natm}, ngrids (one GPU) = {grid_end - grid_start}")
            ngrids_per_batch = (ngrids_per_batch + 16 - 1) // 16 * 16
            ngrids_per_batch = min(ngrids_per_batch, min_grid_blksize)

            for g0 in range(grid_start, grid_end, ngrids_per_batch):
                g1 = min(g0 + ngrids_per_batch, grid_end)
                split_grids_coords = cupy.asarray(grids.coords)[g0:g1, :]
                split_ao = numint.eval_ao(mol, split_grids_coords, deriv = 2, gdftopt = None, transpose = False)

                mu = split_ao[0]
                dmu_dr = split_ao[1:4]
                d2mu_dr2 = get_d2mu_dr2(split_ao)

                rho_drho_taua = numint.eval_rho2(mol, split_ao[:4], mo_coeff[0], mo_occ[0], xctype=xctype)
                rho_drho_taub = numint.eval_rho2(mol, split_ao[:4], mo_coeff[1], mo_occ[1], xctype=xctype)
                rho_drho_tau = cupy.asarray((rho_drho_taua, rho_drho_taub))
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho_drho_tau, deriv = 2, xctype=xctype)[1:3]
                rho_drho_tau = None

                depsilon_drho = vxc[:, 0, :]
                depsilon_dnablarho = vxc[:, 1:4, :]
                depsilon_dtau = vxc[:, 4, :]

                dw_dA = get_dweight_dA(mol, grids, (g0,g1))
                depsilondnablarho_dmudr = contract("uxg,xpg->upg", depsilon_dnablarho, dmu_dr)
                depsilondrho_mu = contract("pg,ug->upg", mu, depsilon_drho)
                mu_occ = (mu.T @ mocca, mu.T @ moccb)
                for i_atom in range(natm):
                    dwdA_depsilondrho_mu = contract("dg,upg->udpg", dw_dA[i_atom, :, :], depsilondrho_mu + depsilondnablarho_dmudr)
                    vmata[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_depsilondrho_mu[0], mu_occ[0])
                    vmatb[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_depsilondrho_mu[1], mu_occ[1])
                    dwdA_depsilondrho_mu = None
                depsilondrho_mu = None
                mu_occ = None
                depsilondnablarho_dmudr_occ = (depsilondnablarho_dmudr[0].T @ mocca, depsilondnablarho_dmudr[1].T @ moccb)
                depsilondnablarho_dmudr = None
                for i_atom in range(natm):
                    dwdA_mu = contract("dg,pg->dpg", dw_dA[i_atom, :, :], mu)
                    vmata[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_mu, depsilondnablarho_dmudr_occ[0])
                    vmatb[i_atom, :, :, :] -= contract("dpg,gj->dpj", dwdA_mu, depsilondnablarho_dmudr_occ[1])
                    dwdA_mu = None
                depsilondnablarho_dmudr_occ = None
                depsilondtau_dmudr = contract("dpg,ug->udpg", dmu_dr, depsilon_dtau)
                depsilondtau_dmudr_occ = (
                    depsilondtau_dmudr[0].transpose(0,2,1) @ mocca,
                    depsilondtau_dmudr[1].transpose(0,2,1) @ moccb,
                )
                depsilondtau_dmudr = None
                for i_atom in range(natm):
                    dwdA_dmudr = contract("dg,xpg->dpxg", dw_dA[i_atom, :, :], dmu_dr)
                    vmata[i_atom, :, :, :] -= 0.5 * contract("dpxg,xgj->dpj", dwdA_dmudr, depsilondtau_dmudr_occ[0])
                    vmatb[i_atom, :, :, :] -= 0.5 * contract("dpxg,xgj->dpj", dwdA_dmudr, depsilondtau_dmudr_occ[1])
                    dwdA_dmudr = None
                depsilondtau_dmudr_occ = None
                dw_dA = None

                grid_to_atom_index_map = cupy.asarray(grids.atm_idx)[g0:g1]
                atom_to_grid_index_map = [cupy.where(grid_to_atom_index_map == i_atom)[0] for i_atom in range(natm)]
                grid_to_atom_index_map = None

                drho_dA_grid_response = cp.zeros((2, natm, 3, g1-g0))
                dnablarho_dA_grid_response = cp.zeros((2, natm, 3, 3, g1-g0))
                dtau_dA_grid_response = cp.zeros((2, natm, 3, g1-g0))
                for i_dm in range(2):
                    _, _, _, drho_dA_grid_response_i, dnablarho_dA_grid_response_i, dtau_dA_grid_response_i = \
                        get_drho_dA_full(dm0[i_dm], xctype, natm, g1 - g0, None, atom_to_grid_index_map, mu, dmu_dr, d2mu_dr2, with_orbital_response = False)
                    drho_dA_grid_response[i_dm] = drho_dA_grid_response_i
                    dnablarho_dA_grid_response[i_dm] = dnablarho_dA_grid_response_i
                    dtau_dA_grid_response[i_dm] = dtau_dA_grid_response_i
                    drho_dA_grid_response_i = None
                    dnablarho_dA_grid_response_i = None
                    dtau_dA_grid_response_i = None

                weight = cupy.asarray(grids.weights)[g0:g1]
                combined_d_dA_grid_response = cupy.concatenate(
                    (drho_dA_grid_response[:, :, :, None, :], dnablarho_dA_grid_response, dtau_dA_grid_response[:, :, :, None, :]),
                    axis = 3
                )
                drho_dA_grid_response = None
                dnablarho_dA_grid_response = None
                dtau_dA_grid_response = None

                fwxc = fxc * weight
                fxc = None
                drhodA_grid_response_fwxc = contract("uxvyg,vAdyg->uAdxg", fwxc, combined_d_dA_grid_response)
                combined_d_dA_grid_response = None
                fwxc = None

                mu_occ = (mu.T @ mocca, mu.T @ moccb)
                dmudr_occ = (
                    contract("dqg,qj->dgj", dmu_dr, mocca),
                    contract("dqg,qj->dgj", dmu_dr, moccb),
                )
                for i_atom in range(natm):
                    drhodA_grid_response_fwxc_rho_term_mu = contract("udg,pg->udpg", drhodA_grid_response_fwxc[:, i_atom, :, 0, :], mu)
                    vmata[i_atom, :, :, :] -= contract("dpg,gj->dpj", drhodA_grid_response_fwxc_rho_term_mu[0], mu_occ[0])
                    vmatb[i_atom, :, :, :] -= contract("dpg,gj->dpj", drhodA_grid_response_fwxc_rho_term_mu[1], mu_occ[1])
                    drhodA_grid_response_fwxc_rho_term_mu = None
                    drhodA_grid_response_fwxc_nablarho_term_dmudr_occ = contract("dxg,xgj->dgj", drhodA_grid_response_fwxc[0, i_atom, :, 1:4, :], dmudr_occ[0])
                    vmata[i_atom, :, :, :] -= contract("dgj,pg->dpj", drhodA_grid_response_fwxc_nablarho_term_dmudr_occ, mu)
                    drhodA_grid_response_fwxc_nablarho_term_dmudr_occ = contract("dxg,xgj->dgj", drhodA_grid_response_fwxc[1, i_atom, :, 1:4, :], dmudr_occ[1])
                    vmatb[i_atom, :, :, :] -= contract("dgj,pg->dpj", drhodA_grid_response_fwxc_nablarho_term_dmudr_occ, mu)
                    drhodA_grid_response_fwxc_nablarho_term_dmudr_occ = None
                    drhodA_grid_response_fwxc_nablarho_term_dmudr = contract("udxg,xpg->udpg", drhodA_grid_response_fwxc[:, i_atom, :, 1:4, :], dmu_dr)
                    vmata[i_atom, :, :, :] -= contract("dpg,gj->dpj", drhodA_grid_response_fwxc_nablarho_term_dmudr[0], mu_occ[0])
                    vmatb[i_atom, :, :, :] -= contract("dpg,gj->dpj", drhodA_grid_response_fwxc_nablarho_term_dmudr[1], mu_occ[1])
                    drhodA_grid_response_fwxc_nablarho_term_dmudr = None
                    drhodA_grid_response_fwxc_tau_term_dmudr = contract("udg,xpg->udpxg", drhodA_grid_response_fwxc[:, i_atom, :, 4, :], dmu_dr)
                    vmata[i_atom, :, :, :] -= 0.5 * contract("dpxg,xgj->dpj", drhodA_grid_response_fwxc_tau_term_dmudr[0], dmudr_occ[0])
                    vmatb[i_atom, :, :, :] -= 0.5 * contract("dpxg,xgj->dpj", drhodA_grid_response_fwxc_tau_term_dmudr[1], dmudr_occ[1])
                    drhodA_grid_response_fwxc_tau_term_dmudr = None
                drhodA_grid_response_fwxc = None
                mu_occ = None
                dmudr_occ = None

                for i_atom in range(natm):
                    associated_grid_index = atom_to_grid_index_map[i_atom]
                    if len(associated_grid_index) == 0:
                        continue
                    # # Negative here to cancel the overall negative sign before return
                    mu_grid_i = mu[:, associated_grid_index]
                    dmu_dr_grid_i = dmu_dr[:, :, associated_grid_index]

                    mu_occ_grid_i = (mu_grid_i.T @ mocca, mu_grid_i.T @ moccb)
                    dmudr_occ_grid_i = (
                        contract("dqg,qj->dgj", dmu_dr_grid_i, mocca),
                        contract("dqg,qj->dgj", dmu_dr_grid_i, moccb),
                    )

                    weight_depsilondrho_grid_i = depsilon_drho[:, associated_grid_index] * weight[associated_grid_index]
                    vmata[i_atom, :, :, :] -= (dmu_dr_grid_i * weight_depsilondrho_grid_i[0]) @ mu_occ_grid_i[0]
                    vmatb[i_atom, :, :, :] -= (dmu_dr_grid_i * weight_depsilondrho_grid_i[1]) @ mu_occ_grid_i[1]
                    vmata[i_atom, :, :, :] -= contract("pg,dgj->dpj", mu_grid_i * weight_depsilondrho_grid_i[0], dmudr_occ_grid_i[0])
                    vmatb[i_atom, :, :, :] -= contract("pg,dgj->dpj", mu_grid_i * weight_depsilondrho_grid_i[1], dmudr_occ_grid_i[1])
                    weight_depsilondrho_grid_i = None

                    d2mu_dr2_grid_i = d2mu_dr2[:, :, :, associated_grid_index]

                    weight_depsilondnablarho_grid_i = depsilon_dnablarho[:, :, associated_grid_index] * weight[associated_grid_index]
                    weight_depsilondnablarho_d2mudr2 = contract("uDg,dDpg->udpg", weight_depsilondnablarho_grid_i, d2mu_dr2_grid_i)
                    vmata[i_atom, :, :, :] -= contract("dpg,gj->dpj", weight_depsilondnablarho_d2mudr2[0], mu_occ_grid_i[0])
                    vmatb[i_atom, :, :, :] -= contract("dpg,gj->dpj", weight_depsilondnablarho_d2mudr2[1], mu_occ_grid_i[1])
                    mu_occ_grid_i = None
                    weight_depsilondnablarho_d2mudr2_occ = contract("dpg,pj->dgj", weight_depsilondnablarho_d2mudr2[0], mocca)
                    vmata[i_atom, :, :, :] -= contract("pg,dgj->dpj", mu_grid_i, weight_depsilondnablarho_d2mudr2_occ)
                    weight_depsilondnablarho_d2mudr2_occ = contract("dpg,pj->dgj", weight_depsilondnablarho_d2mudr2[1], moccb)
                    vmatb[i_atom, :, :, :] -= contract("pg,dgj->dpj", mu_grid_i, weight_depsilondnablarho_d2mudr2_occ)
                    weight_depsilondnablarho_d2mudr2 = None
                    mu_grid_i = None
                    weight_depsilondnablarho_d2mudr2_occ = None
                    weight_depsilondnablarho_dmudr = contract("uDg,Dpg->upg", weight_depsilondnablarho_grid_i, dmu_dr_grid_i)
                    vmata[i_atom, :, :, :] -= contract("pg,dgj->dpj", weight_depsilondnablarho_dmudr[0], dmudr_occ_grid_i[0])
                    vmatb[i_atom, :, :, :] -= contract("pg,dgj->dpj", weight_depsilondnablarho_dmudr[1], dmudr_occ_grid_i[1])
                    weight_depsilondnablarho_dmudr_occ = (weight_depsilondnablarho_dmudr[0].T @ mocca, weight_depsilondnablarho_dmudr[1].T @ moccb)
                    weight_depsilondnablarho_dmudr = None
                    vmata[i_atom, :, :, :] -= contract("dpg,gj->dpj", dmu_dr_grid_i, weight_depsilondnablarho_dmudr_occ[0])
                    vmatb[i_atom, :, :, :] -= contract("dpg,gj->dpj", dmu_dr_grid_i, weight_depsilondnablarho_dmudr_occ[1])
                    weight_depsilondnablarho_dmudr_occ = None
                    weight_depsilondnablarho_grid_i = None

                    weight_depsilondrho_grid_i = depsilon_dtau[:, associated_grid_index] * weight[associated_grid_index]
                    vmata[i_atom, :, :, :] -= 0.5 * cupy.einsum("dDpg,Dgj->dpj", d2mu_dr2_grid_i * weight_depsilondrho_grid_i[0], dmudr_occ_grid_i[0])
                    vmatb[i_atom, :, :, :] -= 0.5 * cupy.einsum("dDpg,Dgj->dpj", d2mu_dr2_grid_i * weight_depsilondrho_grid_i[1], dmudr_occ_grid_i[1])
                    dmudr_occ_grid_i = None
                    d2mudr2_occ_grid_i = (
                        contract("dDqg,qj->dDgj", d2mu_dr2_grid_i, mocca),
                        contract("dDqg,qj->dDgj", d2mu_dr2_grid_i, moccb),
                    )
                    d2mu_dr2_grid_i = None
                    vmata[i_atom, :, :, :] -= 0.5 * contract("Dpg,dDgj->dpj", dmu_dr_grid_i * weight_depsilondrho_grid_i[0], d2mudr2_occ_grid_i[0])
                    vmatb[i_atom, :, :, :] -= 0.5 * contract("Dpg,dDgj->dpj", dmu_dr_grid_i * weight_depsilondrho_grid_i[1], d2mudr2_occ_grid_i[1])
                    dmu_dr_grid_i = None
                    d2mudr2_occ_grid_i = None

        elif xctype == 'HF':
            pass
        else:
            raise NotImplementedError(f"xctype = {xctype} not supported")
        t2 = log.timer_debug2('grid response', *t2)

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
        vnlca, vnlcb = nr_rks_fnlc_mo(mf, mol, mo_coeff, mo_occ, dms)
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
