# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Non-relativistic UKS analytical Hessian
'''


import numpy
import cupy
from pyscf import lib
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.hessian import uhf as uhf_hess
from gpu4pyscf.hessian import uks as uks_hess
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import contract, add_sparse, take_last2d
from gpu4pyscf.lib import logger

# import pyscf.grad.rks to activate nuc_grad_method method
from gpu4pyscf.grad import rks  # noqa
from gpu4pyscf.hessian import rhf as rhf_hess_gpu

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

    if mf.nlc != '':
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

    coeff = cupy.asarray(opt.coeff)
    mo_coeff = contract('nij,pi->npj', mo_coeff, coeff)
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
    vmata = contract('npq,qj->npj', vmata, coeff)
    vmata = contract('pi,npj->nij', coeff, vmata).reshape(3,3,nao_sph,nao_sph)
    vmatb = contract('npq,qj->npj', vmatb, coeff)
    vmatb = contract('pi,npj->nij', coeff, vmatb).reshape(3,3,nao_sph,nao_sph)
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
    dm0a_sorted = take_last2d(dm0a, opt.ao_idx)
    dm0b_sorted = take_last2d(dm0b, opt.ao_idx)
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
            dm0a_mask = dm0a_sorted[numpy.ix_(mask, mask)]
            dm0b_mask = dm0b_sorted[numpy.ix_(mask, mask)]

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
        for ia in range(_sorted_mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmata_dm[ia] = vmata_dm[ia][:,:,opt.rev_ao_idx]
            vmatb_dm[ia] = vmatb_dm[ia][:,:,opt.rev_ao_idx]
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
            dm0a_mask = dm0a_sorted[numpy.ix_(mask, mask)]
            dm0b_mask = dm0b_sorted[numpy.ix_(mask, mask)]
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
        for ia in range(_sorted_mol.natm):
            vmata_dm[ia] = vmata_dm[ia][:,:,opt.rev_ao_idx]
            vmatb_dm[ia] = vmatb_dm[ia][:,:,opt.rev_ao_idx]
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

            dm0a_mask = dm0a_sorted[numpy.ix_(mask, mask)]
            dm0b_mask = dm0b_sorted[numpy.ix_(mask, mask)]
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
        for ia in range(_sorted_mol.natm):
            vmata_dm[ia] = vmata_dm[ia][:,:,opt.rev_ao_idx]
            vmatb_dm[ia] = vmatb_dm[ia][:,:,opt.rev_ao_idx]
            p0, p1 = aoslices[ia][2:]
            vmata_dm[ia] += contract('xypq,pq->xyp', ipipa[:,:,:,p0:p1], dm0a[:,p0:p1])
            vmata_dm[ia] += contract('yxqp,pq->xyp', ipipa[:,:,p0:p1], dm0a[:,p0:p1])
            vmatb_dm[ia] += contract('xypq,pq->xyp', ipipb[:,:,:,p0:p1], dm0b[:,p0:p1])
            vmatb_dm[ia] += contract('yxqp,pq->xyp', ipipb[:,:,p0:p1], dm0b[:,p0:p1])
    return vmata_dm, vmatb_dm

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
    vmata = cupy.zeros((_sorted_mol.natm,3,nao,nocca))
    vmatb = cupy.zeros((_sorted_mol.natm,3,nao,noccb))
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
            for ia in range(_sorted_mol.natm):
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
            for ia in range(_sorted_mol.natm):
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
            for ia in range(_sorted_mol.natm):
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

    vmata = -contract("kxiq,ip->kxpq", vmata, mo_coeff[0])
    vmatb = -contract("kxiq,ip->kxpq", vmatb, mo_coeff[1])

    for ia in range(_sorted_mol.natm):
        p0, p1 = aoslices[ia][2:]
        vmat_tmp = cupy.zeros([3,nao,nao])
        vmat_tmp[:,p0:p1] += vipa[:,p0:p1]
        vmat_tmp[:,:,p0:p1] += vipa[:,p0:p1].transpose(0,2,1)

        vmat_tmp = contract('xij,jq->xiq', vmat_tmp, mocca)
        vmat_tmp = contract('xiq,ip->xpq', vmat_tmp, mo_coeff[0])
        vmata[ia] -= vmat_tmp

        vmat_tmp = cupy.zeros([3,nao,nao])
        vmat_tmp[:,p0:p1] += vipb[:,p0:p1]
        vmat_tmp[:,:,p0:p1] += vipb[:,p0:p1].transpose(0,2,1)

        vmat_tmp = contract('xij,jq->xiq', vmat_tmp, moccb)
        vmat_tmp = contract('xiq,ip->xpq', vmat_tmp, mo_coeff[1])
        vmatb[ia] -= vmat_tmp
    return vmata, vmatb


class Hessian(rhf_hess.HessianBase):
    '''Non-relativistic UKS hessian'''
    from gpu4pyscf.lib.utils import to_gpu, device

    def __init__(self, mf):
        rhf_hess.Hessian.__init__(self, mf)
        self.grids = None
        self.grid_response = False

    hess_elec = uhf_hess.hess_elec
    solve_mo1 = uhf_hess.Hessian.solve_mo1
    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1
    kernel = NotImplemented
    hess = NotImplemented

from gpu4pyscf import dft
dft.uks.UKS.Hessian = lib.class_as_method(Hessian)
