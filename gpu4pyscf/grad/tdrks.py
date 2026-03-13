# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
import numpy as np
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, add_sparse, tag_array
from gpu4pyscf.df import int3c2e
from gpu4pyscf.df.df_jk import (
    _tag_factorize_dm, _DFHF, _make_factorized_dm, _aggregate_dm_factor_l)
from gpu4pyscf.dft import numint
from pyscf.dft.numint import NumInt as numint_cpu
from gpu4pyscf.scf import cphf
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.grad import tdrhf
from gpu4pyscf.tdscf.rhf import TDA
import os
import time


def grad_elec(td_grad, x_y, singlet=True, atmlst=None, verbose=logger.INFO,
              with_solvent=False):
    """
    Electronic part of TDA, TDDFT nuclear gradients

    Args:
        td_grad : grad.tdrhf.Gradients or grad.tdrks.Gradients object.

        x_y : a two-element list of cp arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.

    Kwargs:
        with_solvent :
            Include the response of solvent in the gradients of the electronic
            energy.
    """
    t_debug_0 = time.time()
    if singlet is None:
        singlet = True
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.init_timer(td_grad)

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    x, y = x_y
    x = cp.asarray(x)
    is_tda = isinstance(td_grad.base, TDA)
    if is_tda:
        xpy = xmy = x.reshape(nocc, nvir).T
    else:
        y = cp.asarray(y)
        xpy = (x + y).reshape(nocc, nvir).T
        xmy = (x - y).reshape(nocc, nvir).T
    orbv = mo_coeff[:, nocc:]
    orbo = mo_coeff[:, :nocc]
    dvv = contract("ai,bi->ab", xpy, xpy) + contract("ai,bi->ab", xmy, xmy)  # 2 T_{ab}
    doo = -contract("ai,aj->ij", xpy, xpy) - contract("ai,aj->ij", xmy, xmy)  # 2 T_{ij}
    dmxpy = _make_factorized_dm(orbv.dot(xpy), orbo, symmetrize=0)  # (X+Y) in ao basis
    dmxmy = _make_factorized_dm(orbv.dot(xmy), orbo, symmetrize=0)  # (X-Y) in ao basis
    dmzoo = reduce(cp.dot, (orbo, doo, orbo.T))  # T_{ij}*2 in ao basis
    dmzoo += reduce(cp.dot, (orbv, dvv, orbv.T))  # T_{ij}*2 + T_{ab}*2 in ao basis
    if with_solvent:
        td_grad._dmxpy = dmxpy
    t_debug_1 = time.time()
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    f1vo, f1oo, vxc1, k1ao = _contract_xc_kernel(td_grad, mf.xc, dmxpy, dmzoo, True, True, singlet)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    t_debug_2 = time.time()
    if with_k:
        if not isinstance(mf, _DFHF):
            dm = cp.stack([dmzoo, dmxpy, dmxmy])
            vj, vk = mf.get_jk(mol, dm, hermi=0)
            vk *= hyb
            vj = [vj[0], vj[1]*2]
            vk = [vk[0], vk[1]+vk[1].T, vk[2]-vk[2].T]
            if omega != 0:
                vk1 = mf.get_k(mol, dm, hermi=0, omega=omega)
                vk1 *= alpha - hyb
                vk[0] += vk1[0]
                vk[1] += vk1[1]+vk1[1].T
                vk[2] += vk1[2]-vk1[2].T
        else:
            dmzoo = _tag_factorize_dm(dmzoo, hermi=1)
            vj0, vk0 = mf.get_jk(mol, dmzoo, hermi=1)
            vk0 *= hyb
            if omega != 0:
                vk0 += mf.get_k(mol, dmzoo, hermi=1, omega=omega) * (alpha - hyb)
            if is_tda:
                vj, vk = mf.get_jk(mol, dmxpy, hermi=0)
                vk *= hyb
                vj = [vj0, vj*2]
                vk = [vk0, vk+vk.T, vk-vk.T]
                if omega != 0:
                    vk1 = mf.get_k(mol, dmxpy, hermi=0, omega=omega)
                    vk1 *= alpha - hyb
                    vk[1] += vk1+vk1.T
                    vk[2] += vk1-vk1.T
            else:
                dm = _aggregate_dm_factor_l([dmxpy, dmxmy])
                vj, vk = mf.get_jk(mol, dm, hermi=0)
                vk *= hyb
                vj = [vj0, vj[0]*2]
                vk = [vk0, vk[0]+vk[0].T, vk[1]-vk[1].T]
                if omega != 0:
                    vk1 = mf.get_k(mol, dm, hermi=0, omega=omega)
                    vk1 *= alpha - hyb
                    vk[1] += vk1[0]+vk1[0].T
                    vk[2] += vk1[1]-vk1[1].T
        dm = vj0 = vk0 = vk1 = None
        dmzoo = dmzoo.view(cp.ndarray)

        veff0doo = vj[0] * 2 - vk[0] + f1oo[0] + k1ao[0] * 2
        if with_solvent:
            veff0doo += td_grad.solvent_response(dmzoo)
        wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
        if singlet:
            veff = vj[1] * 2 - vk[1] + f1vo[0] * 2
        else:
            veff = f1vo[0] - vk[1]
        if with_solvent:
            veff += td_grad.solvent_response(dmxpy + dmxpy.T)
        veff0mop = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= contract("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2
        wvo += contract("ac,ai->ci", veff0mop[nocc:, nocc:], xpy) * 2
        veff = -vk[2]
        veff0mom = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= contract("ki,ai->ak", veff0mom[:nocc, :nocc], xmy) * 2
        wvo += contract("ac,ai->ci", veff0mom[nocc:, nocc:], xmy) * 2
    else:
        vj = mf.get_j(mol, cp.stack([dmzoo, dmxpy+dmxpy.T]), hermi=1)

        veff0doo = vj[0] * 2 + f1oo[0] + k1ao[0] * 2
        wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
        if singlet:
            veff = vj[1] * 2 + f1vo[0] * 2
        else:
            veff = f1vo[0]
        veff0mop = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= contract("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2
        wvo += contract("ac,ai->ci", veff0mop[nocc:, nocc:], xpy) * 2
        veff0mom = cp.zeros((nmo, nmo))

    # set singlet=None, generate function for CPHF type response kernel
    vresp = td_grad.base.gen_response(singlet=None, hermi=1)
    t_debug_3 = time.time()
    def fvind(x):
        x = orbv.dot(x.reshape(nvir,nocc)) * 2 # *2 for double occupency
        dm = _make_factorized_dm(x, orbo, symmetrize=1)
        v1ao = vresp(dm)
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()

    z1 = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        wvo,
        max_cycle=td_grad.cphf_max_cycle,
        tol=td_grad.cphf_conv_tol)[0]
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    t_debug_4 = time.time()
    z1 = z1.reshape(nvir, nocc)
    z1ao = _make_factorized_dm(orbv.dot(z1), orbo, symmetrize=1)
    veff = vresp(z1ao)

    im0 = cp.zeros((nmo, nmo))
    im0[:nocc, :nocc] = reduce(cp.dot, (orbo.T, veff0doo + veff, orbo))
    im0[:nocc, :nocc] += contract("ak,ai->ki", veff0mop[nocc:, :nocc], xpy)
    im0[:nocc, :nocc] += contract("ak,ai->ki", veff0mom[nocc:, :nocc], xmy)
    im0[nocc:, nocc:] =  contract("ci,ai->ac", veff0mop[nocc:, :nocc], xpy)
    im0[nocc:, nocc:] += contract("ci,ai->ac", veff0mom[nocc:, :nocc], xmy)
    im0[nocc:, :nocc] =  contract("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2
    im0[nocc:, :nocc] += contract("ki,ai->ak", veff0mom[:nocc, :nocc], xmy) * 2

    zeta = (mo_energy[:,cp.newaxis] + mo_energy)*0.5
    zeta[nocc:, :nocc] = mo_energy[:nocc]
    zeta[:nocc, nocc:] = mo_energy[nocc:]
    dm1 = cp.zeros((nmo, nmo))
    dm1[:nocc, :nocc] = doo
    dm1[nocc:, nocc:] = dvv
    dm1[nocc:, :nocc] = z1
    dm1[:nocc, :nocc] += cp.eye(nocc) * 2  # for ground state
    im0 = reduce(cp.dot, (mo_coeff, im0 + zeta * dm1, mo_coeff.T))

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td_grad.base._scf.nuc_grad_method()

    z1ao = orbv.dot(z1).dot(orbo.T)
    dmz1doo = z1ao + dmzoo
    if with_solvent:
        td_grad._dmz1doo = dmz1doo
    oo0 = _make_factorized_dm(orbo*2, orbo, symmetrize=0) # *2 for double occupancy
    t_debug_5 = time.time()

    if atmlst is None:
        atmlst = range(mol.natm)
    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_ground = rhf_grad.contract_h1e_dm(mol, h1, oo0, hermi=1)
    dh_td = rhf_grad.contract_h1e_dm(mol, h1, dmz1doo, hermi=0)
    ds = rhf_grad.contract_h1e_dm(mol, s1, im0, hermi=0)

    dh1e_ground = int3c2e.get_dh1e(mol, oo0)  # 1/r like terms
    if len(mol._ecpbas) > 0:
        dh1e_ground += rhf_grad.get_dh1e_ecp(mol, oo0)  # 1/r like terms
    dh1e_td = int3c2e.get_dh1e(mol, (dmz1doo + dmz1doo.T) * 0.5)  # 1/r like terms
    if len(mol._ecpbas) > 0:
        dh1e_td += rhf_grad.get_dh1e_ecp(mol, (dmz1doo + dmz1doo.T) * 0.5)  # 1/r like terms
    t_debug_6 = time.time()
    if mol._pseudo:
        raise NotImplementedError("Pseudopotential gradient not supported for molecular system yet")

    k_factor = None
    if not is_tda:
        j_factor = [2., 4.,  0.]
        if not singlet:
            j_factor[1] = 0
        if with_k:
            k_factor = np.array([2., 4., -4.])
        dms = [[oo0*.5+dmz1doo, oo0],
               [dmxpy, dmxpy + dmxpy.T],
               [dmxmy, dmxmy - dmxmy.T]]
    else:
        j_factor = [2., 8.]
        if not singlet:
            j_factor[1] = 0
        if with_k:
            k_factor = np.array([2., 8.])
        dmxpy_T = tag_array(dmxpy.T, factor_l=dmxpy.factor_r,
                            factor_r=dmxpy.factor_l)
        dms = [[oo0*.5+dmz1doo, oo0], [dmxpy, dmxpy_T]]

    if with_k:
        ejk = td_grad.jk_energies_per_atom(
            dms, j_factor, k_factor*hyb, sum_results=True)
    else:
        ejk = td_grad.jk_energies_per_atom(
            dms, j_factor, None, sum_results=True)

    if with_k and omega != 0:
        j_factor = None
        beta = alpha - hyb
        ejk += td_grad.jk_energies_per_atom(
            dms, j_factor, k_factor*beta, omega=omega, sum_results=True)
    t_debug_7 = time.time()
    time1 = log.timer('2e AO integral derivatives', *time1)

    fxcz1 = _contract_xc_kernel(td_grad, mf.xc, z1ao, None, False, False, True)[0]
    t_debug_8 = time.time()
    veff1_0 = vxc1[1:]
    veff1_1 = (f1oo[1:] + fxcz1[1:] + k1ao[1:] * 2) * 2  # *2 for dmz1doo+dmz1oo.T
    if singlet:
        veff1_2 = f1vo[1:] * 2
    else:
        veff1_2 = f1vo[1:]

    de = dh_ground + dh_td - ds + ejk
    dveff1_0 = rhf_grad.contract_h1e_dm(mol, veff1_0, oo0 + dmz1doo, hermi=0)
    dveff1_1 = rhf_grad.contract_h1e_dm(mol, veff1_1, oo0, hermi=1) * .25
    dveff1_2 = rhf_grad.contract_h1e_dm(mol, veff1_2, dmxpy, hermi=0) * 2
    de += cp.asnumpy(dh1e_ground + dh1e_td) + dveff1_0 + dveff1_1 + dveff1_2
    if atmlst is not None:
        de = de[atmlst]
    t_debug_9 = time.time()
    time_list = [t_debug_0, t_debug_1, t_debug_2, t_debug_3, t_debug_4, t_debug_5, t_debug_6, t_debug_7, t_debug_8, t_debug_9]
    time_list = [time_list[i+1] - time_list[i] for i in range(len(time_list) - 1)]
    if verbose >= logger.NOTE:
        for i, t in enumerate(time_list):
            logger.note(td_grad, f"Time for step {i}: {t:.5f}s")
    return de


# dmvo, dmoo in AO-representation
# Note spin-trace is applied for fxc, kxc
def _contract_xc_kernel(td_grad, xc_code, dmvo, dmoo=None,
            with_vxc=True, with_kxc=True, singlet=True, with_nac=False, dmvo_2=None):
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

    opt = getattr(ni, "gdftopt", None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])

    # dmvo ~ reduce(cp.dot, (orbv, Xai, orbo.T))
    dmvo = (dmvo + dmvo.T) * 0.5  # because K_{ia,jb} == K_{ia,bj}
    dmvo = opt.sort_orbitals(dmvo, axis=[0, 1])

    f1vo = cp.zeros((4, nao, nao))  # 0th-order, d/dx, d/dy, d/dz
    deriv = 2
    if dmoo is not None:
        f1oo = cp.zeros((4, nao, nao))
        dmoo = opt.sort_orbitals(dmoo, axis=[0, 1])
    else:
        f1oo = None
    if with_vxc:
        v1ao = cp.zeros((4, nao, nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = cp.zeros((4, nao, nao))
        deriv = 3
        if with_nac:
            assert dmvo_2 is not None
            dmvo_2 = (dmvo_2 + dmvo_2.T) * 0.5  # because K_{ia,jb} == K_{ia,bj}
            dmvo_2 = opt.sort_orbitals(dmvo_2, axis=[0, 1])
    else:
        k1ao = None

    if xctype == "HF":
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == "LDA":
        fmat_, ao_deriv = _lda_eval_mat_, 1
    elif xctype == "GGA":
        fmat_, ao_deriv = _gga_eval_mat_, 2
    elif xctype == "MGGA":
        fmat_, ao_deriv = _mgga_eval_mat_, 2
        logger.warn(td_grad, "TDRKS-MGGA Gradients may be inaccurate due to grids response")
    else:
        raise NotImplementedError(f"td-rks for functional {xc_code}")

    if (not td_grad.base.exclude_nlc) and mf.do_nlc():
        raise NotImplementedError("TDDFT gradient with NLC contribution is not supported yet. "
                                  "Please set exclude_nlc field of tdscf object to True, "
                                  "which will turn off NLC contribution in the whole TDDFT calculation.")

    if singlet:
        for ao, mask, weight, coords in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
            if xctype == "LDA":
                ao0 = ao[0]
            else:
                ao0 = ao
            mo_coeff_mask = mo_coeff[mask, :]
            rho = ni.eval_rho2(_sorted_mol, ao0, mo_coeff_mask, mo_occ, mask, xctype, with_lapl=False)
            # quick fix
            if deriv > 2:
                whether_use_gpu = os.environ.get('LIBXC_ON_GPU', '0') == '1'
                if not whether_use_gpu:
                    ni_cpu = numint_cpu()
                    # TODO: If the libxc is stablized, this should be gpulized
                    vxc, fxc, kxc = ni_cpu.eval_xc_eff(xc_code, rho.get(), deriv, xctype=xctype)[1:]
                    if isinstance(vxc,np.ndarray): vxc = cp.asarray(vxc)
                    if isinstance(fxc,np.ndarray): fxc = cp.asarray(fxc)
                    if isinstance(kxc,np.ndarray): kxc = cp.asarray(kxc)
                else:
                    vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]
            else:
                vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]
            dmvo_mask = dmvo[mask[:, None], mask]
            rho1 = (
                ni.eval_rho(_sorted_mol, ao0, dmvo_mask, mask, xctype, hermi=1, with_lapl=False) * 2
            )  # *2 for alpha + beta
            if xctype == "LDA":
                rho1 = rho1[cp.newaxis].copy()
            tmp = contract("yg,xyg->xg", rho1, fxc)
            wv = contract("xg,g->xg", tmp, weight)
            tmp = None
            fmat_(_sorted_mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if dmoo is not None:
                dmoo_mask = dmoo[mask[:, None], mask]
                rho2 = ni.eval_rho(_sorted_mol, ao0, dmoo_mask, mask, xctype, hermi=1, with_lapl=False) * 2
                if xctype == "LDA":
                    rho2 = rho2[cp.newaxis].copy()
                tmp = contract("yg,xyg->xg", rho2, fxc)
                wv = contract("xg,g->xg", tmp, weight)
                tmp = None
                fmat_(_sorted_mol, f1oo, ao, wv, mask, shls_slice, ao_loc)
            if with_vxc:
                fmat_(_sorted_mol, v1ao, ao, vxc * weight, mask, shls_slice, ao_loc)
            if with_kxc:
                if with_nac:
                    dmvo_2_mask = dmvo_2[mask[:, None], mask]
                    rho_dmvo_2 = (
                        ni.eval_rho(_sorted_mol, ao0, dmvo_2_mask, mask, xctype, hermi=1, with_lapl=False) * 2
                    )  # *2 for alpha + beta
                    if xctype == "LDA":
                        rho_dmvo_2 = rho_dmvo_2[cp.newaxis].copy()
                    tmp = contract("yg,xyzg->xzg", rho1, kxc)
                    tmp = contract("zg,xzg->xg", rho_dmvo_2, tmp)
                    wv = contract("xg,g->xg", tmp, weight)
                    tmp = None
                    fmat_(_sorted_mol, k1ao, ao, wv, mask, shls_slice, ao_loc)
                else:
                    tmp = contract("yg,xyzg->xzg", rho1, kxc)
                    tmp = contract("zg,xzg->xg", rho1, tmp)
                    wv = contract("xg,g->xg", tmp, weight)
                    tmp = None
                    fmat_(_sorted_mol, k1ao, ao, wv, mask, shls_slice, ao_loc)
    else:
        for ao, mask, weight, coords in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
            if xctype == "LDA":
                ao0 = ao[0]
            else:
                ao0 = ao
            mo_coeff_mask = mo_coeff[mask, :]
            rho = ni.eval_rho2(_sorted_mol, ao0, mo_coeff_mask, mo_occ, mask, xctype, with_lapl=False)
            rho *= 0.5
            rho = cp.repeat(rho[cp.newaxis], 2, axis=0)
            # quick fix
            # if deriv > 2:
            #     ni_cpu = numint_cpu()
            #     vxc, fxc, kxc = ni_cpu.eval_xc_eff(xc_code, rho.get(), deriv, xctype=xctype)[1:]
            #     if isinstance(vxc,np.ndarray): vxc = cp.asarray(vxc)
            #     if isinstance(fxc,np.ndarray): fxc = cp.asarray(fxc)
            #     if isinstance(kxc,np.ndarray): kxc = cp.asarray(kxc)
            # else:
            #     vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]
            # fxc_t couples triplet excitation amplitudes
            # 1/2 int (tia - tIA) fxc (tjb - tJB) = tia fxc_t tjb
            fxc_t = fxc[:, :, 0] - fxc[:, :, 1]
            fxc_t = fxc_t[0] - fxc_t[1]
            dmvo_mask = dmvo[mask[:, None], mask]
            rho1 = ni.eval_rho(_sorted_mol, ao0, dmvo_mask, mask, xctype, hermi=1, with_lapl=False)
            if xctype == "LDA":
                rho1 = rho1[cp.newaxis].copy()
            tmp = contract("yg,xyg->xg", rho1, fxc_t)
            wv = contract("xg,g->xg", tmp, weight)
            tmp = None
            fmat_(_sorted_mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if dmoo is not None:
                # fxc_s == 2 * fxc of spin restricted xc kernel
                # provides f1oo to couple the interaction between first order MO
                # and density response of tddft amplitudes, which is described by dmoo
                fxc_s = fxc[0, :, 0] + fxc[0, :, 1]
                dmoo_mask = dmoo[mask[:, None], mask]
                rho2 = ni.eval_rho(_sorted_mol, ao0, dmoo_mask, mask, xctype, hermi=1, with_lapl=False)
                if xctype == "LDA":
                    rho2 = rho2[cp.newaxis].copy()
                tmp = contract("yg,xyg->xg", rho2, fxc_s)
                wv = contract("xg,g->xg", tmp, weight)
                tmp = None
                fmat_(_sorted_mol, f1oo, ao, wv, mask, shls_slice, ao_loc)
            if with_vxc:
                vxc = vxc[0]
                fmat_(_sorted_mol, v1ao, ao, vxc * weight, mask, shls_slice, ao_loc)
            if with_kxc:
                # kxc in terms of the triplet coupling
                # 1/2 int (tia - tIA) kxc (tjb - tJB) = tia kxc_t tjb
                kxc = kxc[0, :, 0] - kxc[0, :, 1]
                kxc = kxc[:, :, 0] - kxc[:, :, 1]
                tmp = contract("yg,xyzg->xzg", rho1, kxc)
                tmp = contract("zg,xzg->xg", rho1, tmp)
                wv = contract("xg,g->xg", tmp, weight)
                tmp = None
                fmat_(_sorted_mol, k1ao, ao, wv, mask, shls_slice, ao_loc)

    f1vo[1:] *= -1
    f1vo = opt.unsort_orbitals(f1vo, axis=[1, 2])
    if f1oo is not None:
        f1oo[1:] *= -1
        f1oo = opt.unsort_orbitals(f1oo, axis=[1, 2])
    if v1ao is not None:
        v1ao[1:] *= -1
        v1ao = opt.unsort_orbitals(v1ao, axis=[1, 2])
    if k1ao is not None:
        k1ao[1:] *= -1
        k1ao = opt.unsort_orbitals(k1ao, axis=[1, 2])

    return f1vo, f1oo, v1ao, k1ao


def _lda_eval_mat_(mol, vmat, ao, wv, mask, shls_slice, ao_loc):
    aow = numint._scale_ao(ao[0], wv[0])
    for k in range(4):
        vtmp = numint._dot_ao_ao(mol, ao[k], aow, mask, shls_slice, ao_loc)
        add_sparse(vmat[k], vtmp, mask)
    return vmat


def _gga_eval_mat_(mol, vmat, ao, wv, mask, shls_slice, ao_loc):
    wv[0] *= 0.5  # *.5 because vmat + vmat.T at the end
    aow = numint._scale_ao(ao[:4], wv[:4])
    tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    vtmp = tmp + tmp.T
    add_sparse(vmat[0], vtmp, mask)
    wv = cp.asarray(wv, order="C")
    vtmp = rks_grad._gga_grad_sum_(ao, wv)
    add_sparse(vmat[1:], vtmp, mask)
    return vmat


def _mgga_eval_mat_(mol, vmat, ao, wv, mask, shls_slice, ao_loc):
    wv[0] *= 0.5  # *.5 because vmat + vmat.T at the end
    aow = numint._scale_ao(ao[:4], wv[:4])
    tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    vtmp = tmp + tmp.T
    add_sparse(vmat[0], vtmp, mask)
    vtmp = numint._tau_dot(ao, ao, wv[4])
    add_sparse(vmat[0], vtmp, mask)
    # ! The following line should only be here, because the tau is *0.5 in the _tau_dot function
    wv[4] *= 0.5  # *.5 for 1/2 in tau
    wv = cp.asarray(wv, order="C")
    vtmp = rks_grad._gga_grad_sum_(ao, wv[:4])
    vtmp += rks_grad._tau_grad_dot_(ao, wv[4])
    add_sparse(vmat[1:], vtmp, mask)
    return vmat


class Gradients(tdrhf.Gradients):
    grad_elec = grad_elec


Grad = Gradients
