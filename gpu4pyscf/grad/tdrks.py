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


from functools import reduce
import cupy as cp
from pyscf import lib
from pyscf.lib import logger
from gpu4pyscf import lib as lib_gpu
from gpu4pyscf.lib.cupy_helper import  contract, add_sparse
from gpu4pyscf.df import int3c2e
from gpu4pyscf.dft import numint
from gpu4pyscf.scf import cphf
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.grad import tdrhf


#
# Given Y = 0, TDDFT gradients (XAX+XBY+YBX+YAY)^1 turn to TDA gradients (XAX)^1
#
def grad_elec(td_grad, x_y, singlet=True, atmlst=None, verbose=logger.INFO):
    '''
    Electronic part of TDA, TDDFT nuclear gradients

    Args:
        td_grad : grad.tdrhf.Gradients or grad.tdrks.Gradients object.

        x_y : a two-element list of cp arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ>0).sum())
    nvir = nmo - nocc
    x, y = x_y
    x = cp.asarray(x)
    y = cp.asarray(y)
    xpy = (x+y).reshape(nocc,nvir).T
    xmy = (x-y).reshape(nocc,nvir).T
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]
    dvv = cp.einsum('ai,bi->ab', xpy, xpy) + cp.einsum('ai,bi->ab', xmy, xmy) # 2 T_{ab}
    doo =-cp.einsum('ai,aj->ij', xpy, xpy) - cp.einsum('ai,aj->ij', xmy, xmy) # 2 T_{ij}
    dmxpy = reduce(cp.dot, (orbv, xpy, orbo.T)) # (X+Y) in ao basis
    dmxmy = reduce(cp.dot, (orbv, xmy, orbo.T)) # (X-Y) in ao basis
    dmzoo = reduce(cp.dot, (orbo, doo, orbo.T)) # T_{ij}*2 in ao basis
    dmzoo+= reduce(cp.dot, (orbv, dvv, orbv.T)) # T_{ij}*2 + T_{ab}*2 in ao basis

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    f1vo, f1oo, vxc1, k1ao = \
            _contract_xc_kernel(td_grad, mf.xc, dmxpy,
                                dmzoo, True, True, singlet)

    if ni.libxc.is_hybrid_xc(mf.xc):
        dm = (dmzoo, dmxpy+dmxpy.T, dmxmy-dmxmy.T)
        vj, vk = mf.get_jk(mol, dm, hermi=0)
        if not isinstance(vj, cp.ndarray): vj = cp.asarray(vj)
        if not isinstance(vk, cp.ndarray): vk = cp.asarray(vk)
        vk *= hyb
        if omega != 0:
            vk += cp.asarray(mf.get_k(mol, dm, hermi=0, omega=omega) * (alpha-hyb))
        veff0doo = vj[0] * 2 - vk[0] + f1oo[0] + k1ao[0] * 2
        wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
        if singlet:
            veff = vj[1] * 2 - vk[1] + f1vo[0] * 2
        else:
            veff = f1vo[0] - vk[1]
        veff0mop = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= cp.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy) * 2
        wvo += cp.einsum('ac,ai->ci', veff0mop[nocc:,nocc:], xpy) * 2
        veff = -vk[2]
        veff0mom = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= cp.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy) * 2
        wvo += cp.einsum('ac,ai->ci', veff0mom[nocc:,nocc:], xmy) * 2
    else:
        vj = mf.get_j(mol, (dmzoo, dmxpy+dmxpy.T), hermi=1)
        if not isinstance(vj, cp.ndarray): vj = cp.asarray(vj)
        veff0doo = vj[0] * 2 + f1oo[0] + k1ao[0] * 2
        wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
        if singlet:
            veff = vj[1] * 2 + f1vo[0] * 2
        else:
            veff = f1vo[0]
        veff0mop = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= cp.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy) * 2
        wvo += cp.einsum('ac,ai->ci', veff0mop[nocc:,nocc:], xpy) * 2
        veff0mom = cp.zeros((nmo,nmo))

    # set singlet=None, generate function for CPHF type response kernel
    vresp = mf.gen_response(singlet=None, hermi=1)
    def fvind(x):
        dm = reduce(cp.dot, (orbv, x.reshape(nvir,nocc)*2, orbo.T))
        v1ao = vresp(dm+dm.T)
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()
    z1 = cphf.solve(fvind, mo_energy, mo_occ, wvo,
                    max_cycle=td_grad.cphf_max_cycle,
                    tol=td_grad.cphf_conv_tol)[0]
    z1 = z1.reshape(nvir,nocc)
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    z1ao = reduce(cp.dot, (orbv, z1, orbo.T))
    veff = vresp(z1ao+z1ao.T)

    im0 = cp.zeros((nmo,nmo))
    im0[:nocc,:nocc] = reduce(cp.dot, (orbo.T, veff0doo+veff, orbo))
    im0[:nocc,:nocc]+= cp.einsum('ak,ai->ki', veff0mop[nocc:,:nocc], xpy)
    im0[:nocc,:nocc]+= cp.einsum('ak,ai->ki', veff0mom[nocc:,:nocc], xmy)
    im0[nocc:,nocc:] = cp.einsum('ci,ai->ac', veff0mop[nocc:,:nocc], xpy)
    im0[nocc:,nocc:]+= cp.einsum('ci,ai->ac', veff0mom[nocc:,:nocc], xmy)
    im0[nocc:,:nocc] = cp.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy)*2
    im0[nocc:,:nocc]+= cp.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy)*2

    zeta = lib_gpu.cupy_helper.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[nocc:]
    dm1 = cp.zeros((nmo,nmo))
    dm1[:nocc,:nocc] = doo
    dm1[nocc:,nocc:] = dvv
    dm1[nocc:,:nocc] = z1
    dm1[:nocc,:nocc] += cp.eye(nocc)*2 # for ground state
    im0 = reduce(cp.dot, (mo_coeff, im0+zeta*dm1, mo_coeff.T))

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td_grad.base._scf.nuc_grad_method()
    s1 = mf_grad.get_ovlp(mol)

    dmz1doo = z1ao + dmzoo
    oo0 = reduce(cp.dot, (orbo, orbo.T))

    if atmlst is None:
        atmlst = range(mol.natm)
    h1 = cp.asarray(mf_grad.get_hcore(mol)) # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_ground = contract('xij,ij->xi', h1, oo0*2)
    dh_td = contract('xij,ij->xi', h1, (dmz1doo+dmz1doo.T)*0.5)
    ds = contract('xij,ij->xi', s1, (im0+im0.T)*0.5)

    dh1e_ground = int3c2e.get_dh1e(mol, oo0*2) # 1/r like terms
    if mol.has_ecp():
        dh1e_ground += rhf_grad.get_dh1e_ecp(mol, oo0*2) # 1/r like terms
    dh1e_td = int3c2e.get_dh1e(mol, (dmz1doo+dmz1doo.T)*0.5) # 1/r like terms
    if mol.has_ecp():
        dh1e_td += rhf_grad.get_dh1e_ecp(mol, (dmz1doo+dmz1doo.T)*0.5) # 1/r like terms

    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    vhfopt = mf._opt_gpu.get(None, None)
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
    dvhf_DD_DP = rhf_grad._jk_energy_per_atom(mol, (dmz1doo+dmz1doo.T)*0.5 + oo0*2, vhfopt, j_factor, k_factor, verbose=verbose)
    dvhf_DD_DP-= rhf_grad._jk_energy_per_atom(mol, (dmz1doo+dmz1doo.T)*0.5, vhfopt, j_factor, k_factor, verbose=verbose)
    dvhf_xpy = rhf_grad._jk_energy_per_atom(mol, dmxpy+dmxpy.T, vhfopt, j_factor, k_factor, verbose=verbose)*2
    dvhf_xmy = rhf_grad._jk_energy_per_atom(mol, dmxmy-dmxmy.T, vhfopt, j_factor=0.0, k_factor=k_factor)*2
    
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
            dvhf_DD_DP += rhf_grad._jk_energy_per_atom(mol, (dmz1doo+dmz1doo.T)*0.5 + oo0*2, vhfopt, j_factor, k_factor, verbose=verbose)
            dvhf_DD_DP -= rhf_grad._jk_energy_per_atom(mol, (dmz1doo+dmz1doo.T)*0.5, vhfopt, j_factor, k_factor, verbose=verbose)
            dvhf_xpy += rhf_grad._jk_energy_per_atom(mol, dmxpy+dmxpy.T, vhfopt, j_factor, k_factor, verbose=verbose)*2
            dvhf_xmy += rhf_grad._jk_energy_per_atom(mol, dmxmy-dmxmy.T, vhfopt, j_factor=0.0, k_factor=k_factor)*2

    fxcz1 = _contract_xc_kernel(td_grad, mf.xc, z1ao, None,
                                False, False, True)[0]

    veff1_0 = vxc1[1:]
    veff1_1 =(f1oo[1:] + fxcz1[1:] + k1ao[1:]*2)*2 # *2 for dmz1doo+dmz1oo.T
    if singlet:
        veff1_2 = f1vo[1:] * 2
    else:
        veff1_2 = f1vo[1:]
    time1 = log.timer('2e AO integral derivatives', *time1)
    extra_force = cp.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        extra_force[k] += mf_grad.extra_force(ia, locals())

    delec = 2.0*(dh_ground + dh_td - ds)
    aoslices = mol.aoslice_by_atom()
    delec= cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:,2:]])
    de = 2.0 * (dvhf_DD_DP + dvhf_xpy + dvhf_xmy) + dh1e_ground + dh1e_td + delec + extra_force

    offsetdic = mol.offset_nr_by_atom()
    tmp = cp.zeros((3, nao, nao)) # TODO: not the same with UKS, can be changed!
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        tmp[:,p0:p1]   += veff1_0[:,p0:p1]
        tmp[:,:,p0:p1] += veff1_0[:,p0:p1].transpose(0,2,1)
        de[k] += cp.einsum('xpq,pq->x', tmp, oo0) * 2

        de[k] += cp.einsum('xpq,pq->x', tmp, dmz1doo)
        tmp *= 0.0

        de[k] += cp.einsum('xij,ij->x', veff1_1[:,p0:p1], oo0[p0:p1])
        de[k] += cp.einsum('xij,ij->x', veff1_2[:,p0:p1], dmxpy[p0:p1,:]) * 2
        de[k] += cp.einsum('xji,ij->x', veff1_2[:,p0:p1], dmxpy[:,p0:p1]) * 2

    log.timer('TDRKS nuclear gradients', *time0)
    return de.get()


# dmvo, dmoo in AO-representation
# Note spin-trace is applied for fxc, kxc
def _contract_xc_kernel(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True,
                        with_kxc=True, singlet=True):
    mol = td_grad.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])

    # dmvo ~ reduce(cp.dot, (orbv, Xai, orbo.T))
    dmvo = (dmvo + dmvo.T) * .5 # because K_{ia,jb} == K_{ia,bj}
    dmvo = opt.sort_orbitals(dmvo, axis=[0,1])

    f1vo = cp.zeros((4,nao,nao))  # 0th-order, d/dx, d/dy, d/dz
    deriv = 2
    if dmoo is not None:
        f1oo = cp.zeros((4,nao,nao))
        dmoo = opt.sort_orbitals(dmoo, axis=[0,1])
    else:
        f1oo = None
    if with_vxc:
        v1ao = cp.zeros((4,nao,nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = cp.zeros((4,nao,nao))
        deriv = 3
    else:
        k1ao = None

    if xctype == 'HF':
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == 'LDA':
        fmat_, ao_deriv = _lda_eval_mat_, 1
    elif xctype == 'GGA':
        fmat_, ao_deriv = _gga_eval_mat_, 2
    elif xctype == 'MGGA':
        fmat_, ao_deriv = _mgga_eval_mat_, 2
        logger.warn(td_grad, 'TDRKS-MGGA Gradients may be inaccurate due to grids response')
    else:
        raise NotImplementedError(f'td-rks for functional {xc_code}')

    if singlet:
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
            if xctype == 'LDA':
                ao0 = ao[0]
            else:
                ao0 = ao
            mo_coeff_mask = mo_coeff[mask,:]
            rho = ni.eval_rho2(_sorted_mol, ao0, mo_coeff_mask, mo_occ, mask, xctype, with_lapl=False)
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]
            dmvo_mask = dmvo[mask[:,None],mask]
            rho1 = ni.eval_rho(_sorted_mol, ao0, dmvo_mask, mask, xctype, hermi=1,
                               with_lapl=False) * 2  # *2 for alpha + beta
            if xctype == 'LDA':
                rho1 = rho1[cp.newaxis]
            wv = cp.einsum('yg,xyg,g->xg', rho1, fxc, weight)
            fmat_(_sorted_mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if dmoo is not None:
                dmoo_mask = dmoo[mask[:,None],mask]
                rho2 = ni.eval_rho(_sorted_mol, ao0, dmoo_mask, mask, xctype, hermi=1, with_lapl=False) * 2
                if xctype == 'LDA':
                    rho2 = rho2[cp.newaxis]
                wv = cp.einsum('yg,xyg,g->xg', rho2, fxc, weight)
                fmat_(_sorted_mol, f1oo, ao, wv, mask, shls_slice, ao_loc)
            if with_vxc:
                fmat_(_sorted_mol, v1ao, ao, vxc * weight, mask, shls_slice, ao_loc)
            if with_kxc:
                wv = cp.einsum('yg,zg,xyzg,g->xg', rho1, rho1, kxc, weight)
                fmat_(_sorted_mol, k1ao, ao, wv, mask, shls_slice, ao_loc)
    else:
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
            if xctype == 'LDA':
                ao0 = ao[0]
            else:
                ao0 = ao
            mo_coeff_mask = mo_coeff[mask,:]
            rho = ni.eval_rho2(_sorted_mol, ao0, mo_coeff_mask, mo_occ, mask, xctype, with_lapl=False)
            rho *= .5
            rho = cp.repeat(rho[cp.newaxis], 2, axis=0)
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]
            # fxc_t couples triplet excitation amplitudes
            # 1/2 int (tia - tIA) fxc (tjb - tJB) = tia fxc_t tjb
            fxc_t = fxc[:,:,0] - fxc[:,:,1]
            fxc_t = fxc_t[0] - fxc_t[1]
            dmvo_mask = dmvo[mask[:,None],mask]
            rho1 = ni.eval_rho(_sorted_mol, ao0, dmvo_mask, mask, xctype, hermi=1, with_lapl=False)
            if xctype == 'LDA':
                rho1 = rho1[cp.newaxis]
            wv = cp.einsum('yg,xyg,g->xg', rho1, fxc_t, weight)
            fmat_(_sorted_mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if dmoo is not None:
                # fxc_s == 2 * fxc of spin restricted xc kernel
                # provides f1oo to couple the interaction between first order MO
                # and density response of tddft amplitudes, which is described by dmoo
                fxc_s = fxc[0,:,0] + fxc[0,:,1]
                dmoo_mask = dmoo[mask[:,None],mask]
                rho2 = ni.eval_rho(_sorted_mol, ao0, dmoo_mask, mask, xctype, hermi=1, with_lapl=False)
                if xctype == 'LDA':
                    rho2 = rho2[cp.newaxis]
                wv = cp.einsum('yg,xyg,g->xg', rho2, fxc_s, weight)
                fmat_(_sorted_mol, f1oo, ao, wv, mask, shls_slice, ao_loc)
            if with_vxc:
                vxc = vxc[0]
                fmat_(_sorted_mol, v1ao, ao, vxc * weight, mask, shls_slice, ao_loc)
            if with_kxc:
                # kxc in terms of the triplet coupling
                # 1/2 int (tia - tIA) kxc (tjb - tJB) = tia kxc_t tjb
                kxc = kxc[0,:,0] - kxc[0,:,1]
                kxc = kxc[:,:,0] - kxc[:,:,1]
                wv = cp.einsum('yg,zg,xyzg,g->xg', rho1, rho1, kxc, weight)
                fmat_(_sorted_mol, k1ao, ao, wv, mask, shls_slice, ao_loc)

    f1vo[1:] *= -1
    f1vo = opt.unsort_orbitals(f1vo, axis=[1,2])
    if f1oo is not None: 
        f1oo[1:] *= -1
        f1oo = opt.unsort_orbitals(f1oo, axis=[1,2])
    if v1ao is not None: 
        v1ao[1:] *= -1
        v1ao = opt.unsort_orbitals(v1ao, axis=[1,2])
    if k1ao is not None: 
        k1ao[1:] *= -1
        k1ao = opt.unsort_orbitals(k1ao, axis=[1,2])

    return f1vo, f1oo, v1ao, k1ao

def _lda_eval_mat_(mol, vmat, ao, wv, mask, shls_slice, ao_loc):
    aow = numint._scale_ao(ao[0], wv[0])
    for k in range(4):
        vtmp = numint._dot_ao_ao(mol, ao[k], aow, mask, shls_slice, ao_loc)
        add_sparse(vmat[k], vtmp, mask)
        # vmat[k] += numint._dot_ao_ao(mol, ao[k], aow, mask, shls_slice, ao_loc)
    return vmat

def _gga_eval_mat_(mol, vmat, ao, wv, mask, shls_slice, ao_loc):
    wv[0] *= .5  # *.5 because vmat + vmat.T at the end
    aow = numint._scale_ao(ao[:4], wv[:4])
    tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    vtmp = tmp + tmp.T
    add_sparse(vmat[0], vtmp, mask)
    wv = cp.asarray(wv, order='C')
    vtmp = rks_grad._gga_grad_sum_(ao, wv)
    add_sparse(vmat[1:], vtmp, mask)
    return vmat

def _mgga_eval_mat_(mol, vmat, ao, wv, mask, shls_slice, ao_loc):
    wv[0] *= .5  # *.5 because vmat + vmat.T at the end
    aow = numint._scale_ao(ao[:4], wv[:4])
    tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    vtmp = tmp + tmp.T
    add_sparse(vmat[0], vtmp, mask)
    vtmp = numint._tau_dot(ao, ao, wv[4])
    add_sparse(vmat[0], vtmp, mask)
    # ! The following line should only be here, because the tau is *0.5 in the _tau_dot function
    wv[4] *= .5  # *.5 for 1/2 in tau
    wv = cp.asarray(wv, order='C')
    vtmp = rks_grad._gga_grad_sum_(ao, wv[:4])
    vtmp+= rks_grad._tau_grad_dot_(ao, wv[4])
    add_sparse(vmat[1:], vtmp, mask)
    return vmat


class Gradients(tdrhf.Gradients):
    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet, atmlst=None):
        return grad_elec(self, xy, singlet, atmlst, self.verbose)

Grad = Gradients

from gpu4pyscf import tdscf
tdscf.rks.TDA.Gradients = tdscf.rks.TDDFT.Gradients = lib.class_as_method(Gradients)
