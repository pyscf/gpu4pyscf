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
# Ref:
# J. Chem. Theory Comput. 2025, 21, 6, 3010
#

from functools import reduce
import cupy as cp
import numpy as np
from pyscf import lib
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib import logger
from gpu4pyscf.df import int3c2e
from gpu4pyscf.df.df_jk import _tag_factorize_dm, _DFHF, _make_factorized_dm
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrks as tdrks_grad
from gpu4pyscf.grad import tduhf as tduhf_grad
from gpu4pyscf.grad import tduks as tduks_grad
from gpu4pyscf.scf import ucphf
import os
from gpu4pyscf.tdscf._uhf_resp_sf import mcfun_eval_xc_adapter_sf
from gpu4pyscf.scf.jk import _VHFOpt
from gpu4pyscf.grad.tdrhf import _jk_energies_per_atom
from gpu4pyscf.tdscf.uhf import SpinFlipTDHF


# TODO: add df
def grad_elec(td_grad, x_y, atmlst=None, verbose=logger.INFO):
    """
    Electronic part of spin-flip TDA/TDDFT nuclear gradients.

    Args:
        td_grad : grad.tduks_sf.Gradients object.

        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    """
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.init_timer(td_grad)

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_occ = cp.asarray(mf.mo_occ)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_coeff = cp.asarray(mf.mo_coeff)

    occidxa = cp.where(mo_occ[0] > 0)[0]
    occidxb = cp.where(mo_occ[1] > 0)[0]
    viridxa = cp.where(mo_occ[0] == 0)[0]
    viridxb = cp.where(mo_occ[1] == 0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbva = mo_coeff[0][:, viridxa]
    orbvb = mo_coeff[1][:, viridxb]

    nmoa = nocca + nvira
    nmob = noccb + nvirb

    is_tda = not isinstance(td_grad.base, SpinFlipTDHF)

    if td_grad.base.extype == 0:
        y, x = x_y
        y = cp.asarray(y)
        dmt = _make_factorized_dm(orbob.dot(y), orbva, symmetrize=0)
        if is_tda:
            x = cp.zeros((nocca, nvirb))
        else:
            x = cp.asarray(x)
            dmt += _make_factorized_dm(orbvb.dot(x.T), orboa, symmetrize=0)
    elif td_grad.base.extype == 1:
        x, y = x_y
        x = cp.asarray(x)
        dmt = _make_factorized_dm(orbvb.dot(x.T), orboa, symmetrize=0)
        if is_tda:
            y = cp.zeros((noccb, nvira))
        else:
            y = cp.asarray(y)
            dmt += _make_factorized_dm(orbob.dot(y), orbva, symmetrize=0)

    dvva = contract('ia,ib->ab', y, y)
    dvvb = contract('ia,ib->ab', x, x)
    dooa = -contract('ia,ja->ij', x, x)
    doob = -contract('ia,ja->ij', y, y)

    dmzooa = reduce(cp.dot, (orboa, dooa, orboa.T))
    dmzooa += reduce(cp.dot, (orbva, dvva, orbva.T))
    dmzoob = reduce(cp.dot, (orbob, doob, orbob.T))
    dmzoob += reduce(cp.dot, (orbvb, dvvb, orbvb.T))

    if isinstance(mf, _DFHF):
        dmzooa = _tag_factorize_dm(dmzooa, hermi=1)
        dmzoob = _tag_factorize_dm(dmzoob, hermi=1)
        if not is_tda:
            dmt = dmt.view()  # TODO: check the reason

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

    t_debug_1 = log.timer_silent(*time0)[2]
    f1vo, f1oo, vxc1, k1ao = _contract_xc_kernel(td_grad, mf.xc, dmt, cp.stack((dmzooa, dmzoob)), True, True)
    t_debug_2 = log.timer_silent(*time0)[2]
    with_k = ni.libxc.is_hybrid_xc(mf.xc)

    # TODO: check _aggregate_dm_factor_l
    if with_k:
        dm = cp.stack([dmzooa, dmzoob])  # TODO: check
        vj0, vk0 = mf.get_jk(mol, dm, hermi=1)  # (2, nao, nao)
        vk1 = mf.get_k(mol, dmt, hermi=0) * hyb  # (nao, nao)
        vk0 = vk0 * hyb
        if omega != 0:
            vk0 += mf.get_k(mol, dm, hermi=1, omega=omega) * (alpha - hyb)
            vk1 += mf.get_k(mol, dmt, hermi=0, omega=omega) * (alpha - hyb)

        veff0doo = vj0[0] + vj0[1] - vk0 + f1oo[:, 0] + k1ao[:, 0]
        wvoa = reduce(cp.dot, (orbva.T, veff0doo[0], orboa))
        wvob = reduce(cp.dot, (orbvb.T, veff0doo[1], orbob))

        veff0mo = reduce(cp.dot, (mo_coeff[1].T, f1vo[0] - vk1, mo_coeff[0]))
        wvoa += contract('ac,ka->ck', veff0mo[noccb:, nocca:], x)
        wvoa -= contract('jk,jc->ck', veff0mo[:noccb, :nocca], y)
        wvob += contract('ac,ka->ck', veff0mo.T[nocca:, noccb:], y)
        wvob -= contract('jk,jc->ck', veff0mo.T[:nocca, :noccb], x)

        dm = vj0 = vk0 = vk1 = None
    else:
        dm = cp.stack([dmzooa, dmzoob])
        vj0 = mf.get_j(mol, dm, hermi=1)  # (2, nao, nao)
        veff0doo = vj0[0] + vj0[1] + f1oo[:, 0] + k1ao[:, 0]
        wvoa = reduce(cp.dot, (orbva.T, veff0doo[0], orboa))
        wvob = reduce(cp.dot, (orbvb.T, veff0doo[1], orbob))

        veff0mo = reduce(cp.dot, (mo_coeff[1].T, f1vo[0], mo_coeff[0]))
        wvoa += contract('ac,ka->ck', veff0mo[noccb:, nocca:], x)
        wvoa -= contract('jk,jc->ck', veff0mo[:noccb, :nocca], y)
        wvob += contract('ac,ka->ck', veff0mo.T[nocca:, noccb:], y)
        wvob -= contract('jk,jc->ck', veff0mo.T[:nocca, :noccb], x)

    t_debug_3 = log.timer_silent(*time0)[2]
    vresp = td_grad.base.gen_response(hermi=1)

    def fvind(x):
        xa = x[0, : nvira * nocca].reshape(nvira, nocca)
        xb = x[0, nvira * nocca :].reshape(nvirb, noccb)
        dma = _make_factorized_dm(orbva.dot(xa), orboa, symmetrize=1)
        dmb = _make_factorized_dm(orbvb.dot(xb), orbob, symmetrize=1)
        dm1 = cp.stack((dma, dmb))
        # dm1 = _aggregate_dm_factor_l # TODO: maybe give a unified dimension of factor_l/r for both dma and dmb
        v1 = vresp(dm1)
        v1a = reduce(cp.dot, (orbva.T, v1[0], orboa))
        v1b = reduce(cp.dot, (orbvb.T, v1[1], orbob))
        return np.hstack((v1a.ravel(), v1b.ravel()))

    z1a, z1b = ucphf.solve(
        fvind, mo_energy, mo_occ, (wvoa, wvob), max_cycle=td_grad.cphf_max_cycle, tol=td_grad.cphf_conv_tol
    )[0]
    time1 = log.timer('Z-vector using UCPHF solver', *time0)
    t_debug_4 = log.timer_silent(*time0)[2]

    z1aoaS = _make_factorized_dm(orbva.dot(z1a), orboa, symmetrize=1)
    z1aobS = _make_factorized_dm(orbvb.dot(z1b), orbob, symmetrize=1)
    z1aoS = cp.stack((z1aoaS, z1aobS))
    veff = vresp(z1aoS)

    im0a = cp.zeros((nmoa, nmoa))
    im0b = cp.zeros((nmob, nmob))
    im0a[:nocca, :nocca] = reduce(cp.dot, (orboa.T, veff0doo[0] + veff[0], orboa))
    im0b[:noccb, :noccb] = reduce(cp.dot, (orbob.T, veff0doo[1] + veff[1], orbob))
    im0a[:nocca, :nocca] += contract('al,ka->lk', veff0mo[noccb:, :nocca], x)
    im0b[:noccb, :noccb] += contract('al,ka->lk', veff0mo.T[nocca:, :noccb], y)
    im0a[nocca:, nocca:] = contract('jd,jc->dc', veff0mo[:noccb, nocca:], y)
    im0b[noccb:, noccb:] = contract('jd,jc->dc', veff0mo.T[:nocca, noccb:], x)
    im0a[:nocca, nocca:] = contract('jk,jc->kc', veff0mo[:noccb, :nocca], y) * 2
    im0b[:noccb, noccb:] = contract('jk,jc->kc', veff0mo.T[:nocca, :noccb], x) * 2

    zeta_a = (mo_energy[0][:, None] + mo_energy[0]) * 0.5
    zeta_b = (mo_energy[1][:, None] + mo_energy[1]) * 0.5
    zeta_a[nocca:, :nocca] = mo_energy[0][:nocca]
    zeta_b[noccb:, :noccb] = mo_energy[1][:noccb]
    zeta_a[:nocca, nocca:] = mo_energy[0][nocca:]
    zeta_b[:noccb, noccb:] = mo_energy[1][noccb:]
    dm1a = cp.zeros((nmoa, nmoa))
    dm1b = cp.zeros((nmob, nmob))
    dm1a[:nocca, :nocca] = dooa
    dm1b[:noccb, :noccb] = doob
    dm1a[nocca:, nocca:] = dvva
    dm1b[noccb:, noccb:] = dvvb
    dm1a[nocca:, :nocca] = z1a * 2
    dm1b[noccb:, :noccb] = z1b * 2
    dm1a[:nocca, :nocca] += cp.eye(nocca)  # for ground state
    dm1b[:noccb, :noccb] += cp.eye(noccb)
    im0a = reduce(cp.dot, (mo_coeff[0], im0a + zeta_a * dm1a, mo_coeff[0].T))
    im0b = reduce(cp.dot, (mo_coeff[1], im0b + zeta_b * dm1b, mo_coeff[1].T))
    im0 = im0a + im0b
    t_debug_5 = log.timer_silent(*time0)[2]

    dmz1dooa = 2 * z1aoaS + 2 * dmzooa
    dmz1doob = 2 * z1aobS + 2 * dmzoob
    oo0a = reduce(cp.dot, (orboa, orboa.T))
    oo0b = reduce(cp.dot, (orbob, orbob.T))
    dm_correlated = oo0a + oo0b + (dmz1dooa + dmz1doob) * 0.5
    dm_correlated = (dm_correlated + dm_correlated.T) * 0.5

    mf_grad = td_grad.base._scf.nuc_grad_method()
    h1 = cp.asarray(mf_grad.get_hcore(mol))
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_ground_and_td = rhf_grad.contract_h1e_dm(mol, h1, dm_correlated, hermi=1)
    ds = rhf_grad.contract_h1e_dm(mol, s1, im0, hermi=0)

    dh1e_ground_and_td = int3c2e.get_dh1e(mol, dm_correlated)  # 1/r like terms
    if len(mol._ecpbas) > 0:
        dh1e_ground_and_td += rhf_grad.get_dh1e_ecp(mol, oo0a + oo0b)  # 1/r like terms

    if mol._pseudo:
        raise NotImplementedError('Pseudopotential gradient not supported for molecular system yet')
    t_debug_6 = log.timer_silent(*time0)[2]

    dmz1doo = dmz1dooa + dmz1doob
    oo0 = oo0a + oo0b
    if with_k:
        dms = [[dmz1doo + oo0, oo0], [dmz1dooa + oo0a, oo0a], [dmz1doob + oo0b, oo0b], [dmt, dmt.T]]
        j_factors = [0.5, 0, 0, 0]
        k_factors = [0, hyb, hyb, 2 * hyb]
        dvhf = td_grad.jk_energies_per_atom(dms, j_factors, k_factors, sum_results=True)
    else:
        dms = [[dmz1doo + oo0, oo0]]
        j_factors = [0.5]
        k_factors = [0]
        dvhf = td_grad.jk_energies_per_atom(dms, j_factors, k_factors, sum_results=True)

    if with_k and omega != 0:
        j_factors = [0, 0, 0]
        k_factors = [alpha - hyb, alpha - hyb, 2 * (alpha - hyb)]
        dvhf += td_grad.jk_energies_per_atom(dms[1:], j_factors, k_factors, omega=omega, sum_results=True)

    t_debug_7 = log.timer_silent(*time0)[2]
    time1 = log.timer('2e AO integral derivatives', *time1)

    fxcz1 = tduks_grad._contract_xc_kernel(td_grad, mf.xc, z1aoS, None, False, False)[0]
    t_debug_8 = log.timer_silent(*time0)[2]
    veff1_0 = vxc1[:, 1:]
    veff1_1 = (f1oo[:, 1:] + fxcz1[:, 1:] + k1ao[:, 1:]) * 4
    veff1_0_a, veff1_0_b = veff1_0
    veff1_1_a, veff1_1_b = veff1_1

    de = dh_ground_and_td + cp.asnumpy(dh1e_ground_and_td) - ds + 2 * dvhf
    dveff1_0 = rhf_grad.contract_h1e_dm(mol, veff1_0_a, oo0a + dmz1dooa * 0.5, hermi=0)
    dveff1_0 += rhf_grad.contract_h1e_dm(mol, veff1_0_b, oo0b + dmz1doob * 0.5, hermi=0)
    dveff1_1 = rhf_grad.contract_h1e_dm(mol, veff1_1_a, oo0a, hermi=1) * 0.25
    dveff1_1 += rhf_grad.contract_h1e_dm(mol, veff1_1_b, oo0b, hermi=1) * 0.25
    dveff1_2 = rhf_grad.contract_h1e_dm(mol, f1vo[1:], dmt, hermi=0) * 2
    de += dveff1_0 + dveff1_1 + dveff1_2
    if atmlst is not None:
        de = de[atmlst]
    t_debug_9 = log.timer_silent(*time0)[2]
    if log.verbose >= logger.DEBUG:
        time_list = [
            0,
            t_debug_1,
            t_debug_2,
            t_debug_3,
            t_debug_4,
            t_debug_5,
            t_debug_6,
            t_debug_7,
            t_debug_8,
            t_debug_9,
        ]
        time_list = [time_list[i + 1] - time_list[i] for i in range(len(time_list) - 1)]
        for i, t in enumerate(time_list):
            logger.note(td_grad, f'Time for step {i}: {t * 1e-3:.5f}s')
    log.timer('TDUKS nuclear gradients', *time0)
    return de


def _contract_xc_kernel(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True, with_kxc=True, with_nac=False, dmvo_2=None):
    mol = td_grad.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao = mo_coeff[0].shape[0]
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])

    # dmvo ~ reduce(cp.dot, (orbv, Xai, orbo.T))
    dmvo = (dmvo + dmvo.T) * 0.5  # because K_{ia,jb} == K_{ia,bj}
    dmvo = opt.sort_orbitals(dmvo, axis=[0, 1])

    f1vo = cp.zeros((4, nao, nao))  # 4 for 0th-order, d/dx, d/dy, d/dz
    deriv = 2
    if dmoo is not None:
        f1oo = cp.zeros((2, 4, nao, nao))  # 2 for alpha, beta
        dmoo = opt.sort_orbitals(dmoo, axis=[1, 2])
    else:
        f1oo = None
    if with_vxc:
        v1ao = cp.zeros((2, 4, nao, nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = cp.zeros((2, 4, nao, nao))
        deriv = 3
        if with_nac:
            assert dmvo_2 is not None
            dmvo_2 = (dmvo_2 + dmvo_2.T) * 0.5
            dmvo_2 = opt.sort_orbitals(dmvo_2, axis=[0, 1])
    else:
        k1ao = None

    if xctype == 'HF':
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == 'LDA':
        fmat_, ao_deriv = tdrks_grad._lda_eval_mat_, 1
    elif xctype == 'GGA':
        fmat_, ao_deriv = tdrks_grad._gga_eval_mat_, 2
    elif xctype == 'MGGA':
        fmat_, ao_deriv = tdrks_grad._mgga_eval_mat_, 2
        logger.warn(td_grad, 'TDUKS-MGGA Gradients may be inaccurate due to grids response')
    else:
        raise NotImplementedError(f'td-rks for functional {xc_code}')

    if (not td_grad.base.exclude_nlc) and mf.do_nlc():
        raise NotImplementedError(
            'TDDFT gradient with NLC contribution is not supported yet. '
            'Please set exclude_nlc field of tdscf object to True, '
            'which will turn off NLC contribution in the whole TDDFT calculation.'
        )

    if td_grad.base.collinear == 'mcol':
        whether_use_gpu = os.environ.get('LIBXC_ON_GPU', '0') == '1'
        if deriv == 3:
            if whether_use_gpu:
                eval_xc_eff = mcfun_eval_xc_adapter_sf(ni, xc_code, td_grad.base.collinear_samples)
            else:
                ni_cpu = ni.to_cpu()
                eval_xc_eff = mcfun_eval_xc_adapter_sf(ni_cpu, xc_code, td_grad.base.collinear_samples)
        else:
            eval_xc_eff = mcfun_eval_xc_adapter_sf(ni, xc_code, td_grad.base.collinear_samples)
    elif td_grad.base.collinear == 'ncol':
        raise NotImplementedError('Locally collinear approach is not implemented')

    for ao, mask, weight, coords in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
        if xctype == 'LDA':
            ao0 = ao[0]
        else:
            ao0 = ao
        mo_coeff_mask_a = mo_coeff[0, mask]
        mo_coeff_mask_b = mo_coeff[1, mask]
        rho = cp.asarray(
            (
                ni.eval_rho2(_sorted_mol, ao0, mo_coeff_mask_a, mo_occ[0], mask, xctype, with_lapl=False),
                ni.eval_rho2(_sorted_mol, ao0, mo_coeff_mask_b, mo_occ[1], mask, xctype, with_lapl=False),
            )
        )
        if td_grad.base.collinear == 'mcol':
            rho_z = cp.array([rho[0] + rho[1], rho[0] - rho[1]])
            # TODO: check cpu/gpu numpy/cupy array conversion
            fxc_sf, kxc_sf = eval_xc_eff(xc_code, rho_z, deriv, xctype=xctype)[2:4]
            kxc_sf = cp.stack((kxc_sf[:, :, 0] + kxc_sf[:, :, 1], kxc_sf[:, :, 0] - kxc_sf[:, :, 1]), axis=2)
            dmvo_mask = dmvo[mask[:, None], mask]
            rho1 = ni.eval_rho(_sorted_mol, ao0, dmvo_mask, mask, xctype, hermi=1, with_lapl=False)
            if xctype == 'LDA':
                rho1 = rho1[cp.newaxis].copy()
            tmp = contract('yg,xyg->xg', rho1, 2 * fxc_sf)
            wv = contract('xg,g->xg', tmp, weight)
            tmp = None
            fmat_(_sorted_mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if with_kxc:
                tmp = contract('xg,xyczg->yczg', rho1, 2 * kxc_sf)
                if with_nac:
                    dmvo_2_mask = dmvo_2[mask[:, None], mask]
                    rho1_J = ni.eval_rho(_sorted_mol, ao0, dmvo_2_mask, mask, xctype, hermi=1, with_lapl=False)
                    if xctype == 'LDA':
                        rho1_J = rho1_J[cp.newaxis].copy()
                    tmp = contract('yg,yczg->czg', rho1_J, tmp)
                else:
                    tmp = contract('yg,yczg->czg', rho1, tmp)
                wv = contract('czg,g->czg', tmp, weight)
                tmp = None
                fmat_(_sorted_mol, k1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
                fmat_(_sorted_mol, k1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

        if dmoo is not None or with_vxc:
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv=2, spin=1)[1:]

        if dmoo is not None:
            dmoo_mask_a = dmoo[0, mask[:, None], mask]
            dmoo_mask_b = dmoo[1, mask[:, None], mask]
            rho2 = cp.asarray(
                (
                    ni.eval_rho(_sorted_mol, ao0, dmoo_mask_a, mask, xctype, hermi=1, with_lapl=False),
                    ni.eval_rho(_sorted_mol, ao0, dmoo_mask_b, mask, xctype, hermi=1, with_lapl=False),
                )
            )
            if xctype == 'LDA':
                rho2 = rho2[:, cp.newaxis].copy()
            tmp = contract('axg,axbyg->byg', rho2, fxc)
            wv = contract('byg,g->byg', tmp, weight)
            tmp = None
            fmat_(_sorted_mol, f1oo[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(_sorted_mol, f1oo[1], ao, wv[1], mask, shls_slice, ao_loc)

        if with_vxc:
            wv = vxc * weight
            fmat_(_sorted_mol, v1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(_sorted_mol, v1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

    f1vo[1:] *= -1
    f1vo = opt.unsort_orbitals(f1vo, axis=[1, 2])
    if f1oo is not None:
        f1oo[:, 1:] *= -1
        f1oo = opt.unsort_orbitals(f1oo, axis=[2, 3])
    if v1ao is not None:
        v1ao[:, 1:] *= -1
        v1ao = opt.unsort_orbitals(v1ao, axis=[2, 3])
    if k1ao is not None:
        k1ao[:, 1:] *= -1
        k1ao = opt.unsort_orbitals(k1ao, axis=[2, 3])
    return f1vo, f1oo, v1ao, k1ao


class Gradients(tduhf_grad.Gradients):
    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet=None, atmlst=None, verbose=None):
        return grad_elec(self, xy, atmlst, self.verbose)

    def jk_energies_per_atom(
        self, dm_list, j_factor=None, k_factor=None, omega=0, hermi=0, sum_results=False, verbose=None
    ):
        """
        Computes a set of first-order derivatives of J/K contributions for each
        element (density matrix or a pair of density matrices) in dm_pairs.

        This function supports evaluating multiple sets of energy derivatives in a
        single call. Additionally, for each set, the two density matrices for the
        four-index Coulomb integrals can be different.

        Args:
            dm_list :
                A list of density-matrix-pairs [[dm, dm], [dm, dm], ...].
                Each element corresponds to one set of energy derivative.
            j_factor :
                A list of factors for Coulomb (J) term
            k_factor :
                A list of factors for Coulomb (K) term
            hermi :
                No effects
            sum_results : bool
                If True, aggregate all sets of derivatives into a single result.

        Returns:
            An array of shape (*, Natm, 3) if sum_results is False; otherwise,
            an array of shape (Natm, 3).
        """
        mf = self.base._scf
        vhfopt = mf._opt_gpu.get(omega)
        if vhfopt is None:
            # For LDA and GGA, only mf._opt_jengine is initialized
            mol = mf.mol
            with mol.with_range_coulomb(omega):
                vhfopt = mf._opt_gpu[omega] = _VHFOpt(mol, mf.direct_scf_tol, tile=1).build()
        if isinstance(dm_list, cp.ndarray) and dm_list.ndim == 2:
            dm_list = dm_list[None]
        ejk = _jk_energies_per_atom(vhfopt, dm_list, j_factor, k_factor, sum_results, verbose)
        return ejk


Grad = Gradients
