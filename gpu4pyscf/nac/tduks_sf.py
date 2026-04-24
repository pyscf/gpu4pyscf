# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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
from gpu4pyscf.df import int3c2e
from gpu4pyscf.df.df_jk import _tag_factorize_dm, _DFHF, _make_factorized_dm
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.scf import ucphf
from gpu4pyscf.grad import tduks as tduks_grad
from gpu4pyscf.grad import tduks_sf as tduks_sf_grad
from gpu4pyscf.nac import tdrhf
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.tdscf.uhf import SpinFlipTDHF


def get_nacv_ee(td_nac, x_y_I, x_y_J, E_I, E_J, atmlst=None, verbose=logger.INFO):
    """
    Nonadiabatic coupling vector (NACV) between Spin-flip TDA/TDDFT excited states.
    """
    log = logger.new_logger(td_nac, verbose)
    time0 = logger.init_timer(td_nac)

    mol = td_nac.mol
    mf = td_nac.base._scf
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
    nao = orboa.shape[0]

    is_tda = not isinstance(td_nac.base, SpinFlipTDHF)

    if td_nac.base.extype == 0:
        y_I, x_I = x_y_I
        y_J, x_J = x_y_J
        y_I = cp.asarray(y_I)
        y_J = cp.asarray(y_J)
        dmt_I = _make_factorized_dm(orbob.dot(y_I), orbva, symmetrize=0)
        dmt_J = _make_factorized_dm(orbob.dot(y_J), orbva, symmetrize=0)
        if is_tda:
            x_I = cp.zeros((nocca, nvirb))
            x_J = cp.zeros((nocca, nvirb))
        else:
            x_I = cp.asarray(x_I)
            x_J = cp.asarray(x_J)
            dmt_I += _make_factorized_dm(orbvb.dot(x_I.T), orboa, symmetrize=0)
            dmt_J += _make_factorized_dm(orbvb.dot(x_J.T), orboa, symmetrize=0)
    elif td_nac.base.extype == 1:
        x_I, y_I = x_y_I
        x_J, y_J = x_y_J
        x_I = cp.asarray(x_I)
        x_J = cp.asarray(x_J)
        dmt_I = _make_factorized_dm(orbvb.dot(x_I.T), orboa, symmetrize=0)
        dmt_J = _make_factorized_dm(orbvb.dot(x_J.T), orboa, symmetrize=0)
        if is_tda:
            y_I = cp.zeros((noccb, nvira))
            y_J = cp.zeros((noccb, nvira))
        else:
            y_I = cp.asarray(y_I)
            y_J = cp.asarray(y_J)
            dmt_I += _make_factorized_dm(orbob.dot(y_I), orbva, symmetrize=0)
            dmt_J += _make_factorized_dm(orbob.dot(y_J), orbva, symmetrize=0)

    if not is_tda:
        dmt_I = dmt_I.view(cp.ndarray)
        dmt_J = dmt_J.view(cp.ndarray)

    dvva_IJ = contract('ia,ib->ab', y_I, y_J)
    dvvb_IJ = contract('ia,ib->ab', x_I, x_J)
    dooa_IJ = -contract('ia,ja->ij', x_I, x_J)
    doob_IJ = -contract('ia,ja->ij', y_I, y_J)

    dvva_JI = contract('ia,ib->ab', y_J, y_I)
    dvvb_JI = contract('ia,ib->ab', x_J, x_I)
    dooa_JI = -contract('ia,ja->ij', x_J, x_I)
    doob_JI = -contract('ia,ja->ij', y_J, y_I)

    dmzooa_IJ = reduce(cp.dot, (orboa, dooa_IJ, orboa.T))
    dmzooa_IJ += reduce(cp.dot, (orbva, dvva_IJ, orbva.T))
    dmzoob_IJ = reduce(cp.dot, (orbob, doob_IJ, orbob.T))
    dmzoob_IJ += reduce(cp.dot, (orbvb, dvvb_IJ, orbvb.T))

    dmzooa_JI = reduce(cp.dot, (orboa, dooa_JI, orboa.T))
    dmzooa_JI += reduce(cp.dot, (orbva, dvva_JI, orbva.T))
    dmzoob_JI = reduce(cp.dot, (orbob, doob_JI, orbob.T))
    dmzoob_JI += reduce(cp.dot, (orbvb, dvvb_JI, orbvb.T))

    dmzooa = (dmzooa_IJ + dmzooa_JI) * 0.5
    dmzoob = (dmzoob_IJ + dmzoob_JI) * 0.5

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

    t_debug_1 = log.timer_silent(*time0)[2]
    f1vo_I, f1oo, vxc1, k1ao = tduks_sf_grad._contract_xc_kernel(
        td_nac, mf.xc, dmt_I, cp.stack((dmzooa, dmzoob)), True, True,
        with_nac=True, dmvo_2=dmt_J)
    f1vo_J, _, _, _ = tduks_sf_grad._contract_xc_kernel(
        td_nac, mf.xc, dmt_J, None, False, False)
    t_debug_2 = log.timer_silent(*time0)[2]

    with_k = ni.libxc.is_hybrid_xc(mf.xc)

    if with_k:
        if not isinstance(mf, _DFHF):
            dm = cp.stack([dmzooa, dmzoob])
        else:
            dm = _tag_factorize_dm(cp.stack([dmzooa, dmzoob]), hermi=1)

        vj0, vk0 = mf.get_jk(mol, dm, hermi=1)
        vk1_I = mf.get_k(mol, dmt_I, hermi=0) * hyb
        vk1_J = mf.get_k(mol, dmt_J, hermi=0) * hyb
        vk0 = vk0 * hyb
        if omega != 0:
            vk0 += mf.get_k(mol, dm, hermi=1, omega=omega) * (alpha-hyb)
            vk1_I += mf.get_k(mol, dmt_I, hermi=0, omega=omega) * (alpha-hyb)
            vk1_J += mf.get_k(mol, dmt_J, hermi=0, omega=omega) * (alpha-hyb)

        veff0doo = vj0[0] + vj0[1] - vk0 + f1oo[:,0] + k1ao[:,0]
        wvoa = reduce(cp.dot, (orbva.T, veff0doo[0], orboa))
        wvob = reduce(cp.dot, (orbvb.T, veff0doo[1], orbob))
        
        veff0mo_I = reduce(cp.dot, (mo_coeff[1].T, f1vo_I[0]-vk1_I, mo_coeff[0]))
        veff0mo_J = reduce(cp.dot, (mo_coeff[1].T, f1vo_J[0]-vk1_J, mo_coeff[0]))
        wvoa += contract('ac,ka->ck', veff0mo_I[noccb:, nocca:], x_J) * 0.5
        wvoa += contract('ac,ka->ck', veff0mo_J[noccb:, nocca:], x_I) * 0.5
        wvoa -= contract('jk,jc->ck', veff0mo_I[:noccb, :nocca], y_J) * 0.5
        wvoa -= contract('jk,jc->ck', veff0mo_J[:noccb, :nocca], y_I) * 0.5
        wvob += contract('ac,ka->ck', veff0mo_I.T[nocca:, noccb:], y_J) * 0.5
        wvob += contract('ac,ka->ck', veff0mo_J.T[nocca:, noccb:], y_I) * 0.5
        wvob -= contract('jk,jc->ck', veff0mo_I.T[:nocca, :noccb], x_J) * 0.5
        wvob -= contract('jk,jc->ck', veff0mo_J.T[:nocca, :noccb], x_I) * 0.5

        dm = vj0 = vk0 = vk1_I = vk1_J = None

    else:
        vj0 = mf.get_j(mol, cp.stack((dmzooa, dmzoob)), hermi=1)
        veff0doo = vj0[0] + vj0[1] + f1oo[:, 0] + k1ao[:, 0]

        wvoa = reduce(cp.dot, (orbva.T, veff0doo[0], orboa))
        wvob = reduce(cp.dot, (orbvb.T, veff0doo[1], orbob))

        veff0mo_I = reduce(cp.dot, (mo_coeff[1].T, f1vo_I[0], mo_coeff[0]))
        veff0mo_J = reduce(cp.dot, (mo_coeff[1].T, f1vo_J[0], mo_coeff[0]))

        wvoa += contract('ac,ka->ck', veff0mo_I[noccb:, nocca:], x_J) * 0.5
        wvoa += contract('ac,ka->ck', veff0mo_J[noccb:, nocca:], x_I) * 0.5
        wvoa -= contract('jk,jc->ck', veff0mo_I[:noccb, :nocca], y_J) * 0.5
        wvoa -= contract('jk,jc->ck', veff0mo_J[:noccb, :nocca], y_I) * 0.5

        wvob += contract('ac,ka->ck', veff0mo_I.T[nocca:, noccb:], y_J) * 0.5
        wvob += contract('ac,ka->ck', veff0mo_J.T[nocca:, noccb:], y_I) * 0.5
        wvob -= contract('jk,jc->ck', veff0mo_I.T[:nocca, :noccb], x_J) * 0.5
        wvob -= contract('jk,jc->ck', veff0mo_J.T[:nocca, :noccb], x_I) * 0.5

    t_debug_3 = log.timer_silent(*time0)[2]
    vresp = td_nac.base.gen_response(hermi=1)

    def fvind(x):
        xa = x[0, : nvira * nocca].reshape(nvira, nocca)
        xb = x[0, nvira * nocca :].reshape(nvirb, noccb)
        factor_l = cp.zeros((2, nao, max(nocca, noccb)))
        factor_r = cp.zeros((2, nao, max(nocca, noccb)))
        factor_l[0, :, :nocca] = contract('ua,ai->ui', orbva, xa)
        factor_l[1, :, :noccb] = contract('ub,bi->ui', orbvb, xb)
        factor_r[0, :, :nocca] = orboa
        factor_r[1, :, :noccb] = orbob
        dm1 = _make_factorized_dm(factor_l, factor_r, symmetrize=1)
        v1 = vresp(dm1)
        v1a = reduce(cp.dot, (orbva.T, v1[0], orboa))
        v1b = reduce(cp.dot, (orbvb.T, v1[1], orbob))
        return cp.hstack((v1a.ravel(), v1b.ravel()))

    z1a, z1b = ucphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        (wvoa, wvob),
        max_cycle=td_nac.cphf_max_cycle,
        tol=td_nac.cphf_conv_tol)[0]
    time1 = log.timer('Z-vector using UCPHF solver', *time0)
    t_debug_4 = log.timer_silent(*time0)[2]

    factor_l = cp.zeros((2, nao, max(nocca, noccb)))
    factor_r = cp.zeros((2, nao, max(nocca, noccb)))
    factor_l[0, :, :nocca] = contract('ua,ai->ui', orbva, z1a)
    factor_l[1, :, :noccb] = contract('ub,bi->ui', orbvb, z1b)
    factor_r[0, :, :nocca] = orboa
    factor_r[1, :, :noccb] = orbob
    z1aoS = _make_factorized_dm(factor_l, factor_r, symmetrize=1)
    veff = vresp(z1aoS)

    im0a = cp.zeros((nmoa, nmoa))
    im0b = cp.zeros((nmob, nmob))
    im0a[:nocca, :nocca] = reduce(cp.dot, (orboa.T, veff0doo[0] + veff[0], orboa))
    im0b[:noccb, :noccb] = reduce(cp.dot, (orbob.T, veff0doo[1] + veff[1], orbob))

    im0a[:nocca, :nocca] += contract('al,ka->lk', veff0mo_I[noccb:, :nocca], x_J) * 0.5
    im0a[:nocca, :nocca] += contract('al,ka->lk', veff0mo_J[noccb:, :nocca], x_I) * 0.5
    im0b[:noccb, :noccb] += contract('al,ka->lk', veff0mo_I.T[nocca:, :noccb], y_J) * 0.5
    im0b[:noccb, :noccb] += contract('al,ka->lk', veff0mo_J.T[nocca:, :noccb], y_I) * 0.5

    im0a[nocca:, nocca:] = contract('jd,jc->dc', veff0mo_I[:noccb, nocca:], y_J) * 0.5
    im0a[nocca:, nocca:] += contract('jd,jc->dc', veff0mo_J[:noccb, nocca:], y_I) * 0.5
    im0b[noccb:, noccb:] = contract('jd,jc->dc', veff0mo_I.T[:nocca, noccb:], x_J) * 0.5
    im0b[noccb:, noccb:] += contract('jd,jc->dc', veff0mo_J.T[:nocca, noccb:], x_I) * 0.5

    im0a[:nocca, nocca:] = contract('jk,jc->kc', veff0mo_I[:noccb, :nocca], y_J)
    im0a[:nocca, nocca:] += contract('jk,jc->kc', veff0mo_J[:noccb, :nocca], y_I)
    im0b[:noccb, noccb:] = contract('jk,jc->kc', veff0mo_I.T[:nocca, :noccb], x_J)
    im0b[:noccb, noccb:] += contract('jk,jc->kc', veff0mo_J.T[:nocca, :noccb], x_I)

    zeta_a = (mo_energy[0][:, None] + mo_energy[0]) * 0.5
    zeta_b = (mo_energy[1][:, None] + mo_energy[1]) * 0.5
    zeta_a[nocca:, :nocca] = mo_energy[0][:nocca]
    zeta_b[noccb:, :noccb] = mo_energy[1][:noccb]
    zeta_a[:nocca, nocca:] = mo_energy[0][nocca:]
    zeta_b[:noccb, noccb:] = mo_energy[1][noccb:]

    dm1a = cp.zeros((nmoa, nmoa))
    dm1b = cp.zeros((nmob, nmob))
    dm1a[:nocca, :nocca] = (dooa_IJ + dooa_JI) * 0.5
    dm1b[:noccb, :noccb] = (doob_IJ + doob_JI) * 0.5
    dm1a[nocca:, nocca:] = (dvva_IJ + dvva_JI) * 0.5
    dm1b[noccb:, noccb:] = (dvvb_IJ + dvvb_JI) * 0.5
    dm1a[nocca:, :nocca] = z1a * 2
    dm1b[noccb:, :noccb] = z1b * 2

    im0a = reduce(cp.dot, (mo_coeff[0], im0a + zeta_a * dm1a, mo_coeff[0].T))
    im0b = reduce(cp.dot, (mo_coeff[1], im0b + zeta_b * dm1b, mo_coeff[1].T))
    im0 = im0a + im0b
    t_debug_5 = log.timer_silent(*time0)[2]

    dmz1dooa = 2 * z1aoS[0] + 2 * dmzooa
    dmz1doob = 2 * z1aoS[1] + 2 * dmzoob
    oo0a = _make_factorized_dm(orboa, orboa, symmetrize=0)
    oo0b = _make_factorized_dm(orbob, orbob, symmetrize=0)

    mf_grad = td_nac.base._scf.nuc_grad_method()
    h1 = cp.asarray(mf_grad.get_hcore(mol))
    s1 = mf_grad.get_ovlp(mol)
    dh_td = rhf_grad.contract_h1e_dm(mol, h1, dmz1dooa + dmz1doob, hermi=0) * 0.5
    ds = rhf_grad.contract_h1e_dm(mol, s1, im0, hermi=0)

    dh1e_td = int3c2e.get_dh1e(mol, (dmz1dooa + dmz1doob) * 0.25 + (dmz1dooa + dmz1doob).T * 0.25)  # 1/r like terms
    if len(mol._ecpbas) > 0:
        dh1e_td += rhf_grad.get_dh1e_ecp(
            mol, (dmz1dooa + dmz1doob) * 0.25 + (dmz1dooa + dmz1doob).T * 0.25)  # 1/r like terms
    t_debug_6 = log.timer_silent(*time0)[2]
    if mol._pseudo:
        raise NotImplementedError("Pseudopotential gradient not supported for molecular system yet")

    cp.get_default_memory_pool().free_all_blocks() # TODO: check what

    dmz1doo = dmz1dooa + dmz1doob
    oo0 = oo0a + oo0b
    if with_k:
        if hasattr(dmt_I, 'symmetrize'):
            dmt_I = tag_array(dmt_I.T, factor_l=dmt_I.factor_r, factor_r=dmt_I.factor_l)
            dmt_J = tag_array(dmt_J.T, factor_l=dmt_J.factor_r, factor_r=dmt_J.factor_l)
        else:
            dmt_I = dmt_I.T
            dmt_J = dmt_J.T
        dms = [[_tag_factorize_dm(dmz1doo, hermi=1), _tag_factorize_dm(oo0, hermi=1)],
              [_tag_factorize_dm(dmz1dooa, hermi=1), oo0a],
              [_tag_factorize_dm(dmz1doob, hermi=1), oo0b],
              [dmt_I, dmt_J.T],
              [dmt_J, dmt_I.T]]
        # dms = [[dmz1doo, oo0], [dmz1dooa, oo0a], [dmz1doob, oo0b], [dmt_I, dmt_J.T], [dmt_J, dmt_I.T]]
        j_factors = [0.5, 0, 0, 0, 0]
        k_factors = [0, hyb, hyb, hyb, hyb]
        dvhf = td_nac.jk_energies_per_atom(dms, j_factors, k_factors, sum_results=True)
    else:
        dms = [[dmz1doo, oo0]]
        j_factors = [0.5]
        k_factors = [0]
        dvhf = td_nac.jk_energies_per_atom(dms, j_factors, k_factors, sum_results=True)
    
    if with_k and omega != 0:
        j_factors = [0, 0, 0, 0]
        k_factors = [alpha-hyb, alpha-hyb, alpha-hyb, alpha-hyb]
        dvhf += td_nac.jk_energies_per_atom(dms[1:], j_factors, k_factors, omega=omega, sum_results=True)

    # j_factor = 1.0
    # k_factor = 0.0 # TODO: try None

    # if with_k:
    #     k_factor = hyb

    # dvhf = td_nac.get_veff(
    #     mol, cp.stack(((dmz1dooa + dmz1dooa.T) * 0.25 + oo0a,
    #                    (dmz1doob + dmz1doob.T) * 0.25 + oo0b,)),
    #     j_factor, k_factor, hermi=1)
    # dvhf -= td_nac.get_veff(
    #     mol, cp.stack(((dmz1dooa + dmz1dooa.T), (dmz1doob + dmz1doob.T))) * 0.25,
    #     j_factor, k_factor, hermi=1)
    # dvhf -= td_nac.get_veff(mol, cp.stack((oo0a, oo0b)), j_factor, k_factor, hermi=1)
    # if getattr(mf, 'with_df', None):
    #     raise NotImplementedError("DFHF special handling is intentionally skipped for now")
    # else:
    #     dvhf += td_nac.get_veff(mol, dmt_I+dmt_J.T, 0.0, k_factor, hermi=0) * 0.5
    #     dvhf -= td_nac.get_veff(mol, dmt_I-dmt_J.T, 0.0, k_factor, hermi=0) * 0.5
    #     dvhf += td_nac.get_veff(mol, dmt_I.T+dmt_J, 0.0, k_factor, hermi=0) * 0.5
    #     dvhf -= td_nac.get_veff(mol, dmt_I.T-dmt_J, 0.0, k_factor, hermi=0) * 0.5

    # if with_k and omega != 0:
    #     k_factor = alpha-hyb
    #     dvhf += td_nac.get_veff(
    #         mol, cp.stack(((dmz1dooa + dmz1dooa.T) * 0.25 + oo0a,
    #                        (dmz1doob + dmz1doob.T) * 0.25 + oo0b)),
    #         0.0, k_factor, omega=omega, hermi=1)
    #     dvhf -= td_nac.get_veff(mol,
    #             cp.stack(((dmz1dooa + dmz1dooa.T) * 0.25, (dmz1doob + dmz1doob.T) * 0.25)),
    #         0.0, k_factor, omega=omega, hermi=1)
    #     dvhf -= td_nac.get_veff(mol, cp.stack((oo0a, oo0b)), 0.0, k_factor, omega=omega, hermi=1)
    #     if getattr(mf, 'with_df', None):
    #         raise NotImplementedError("DFHF special handling is intentionally skipped for now")
    #     else:
    #         dvhf += td_nac.get_veff(mol, dmt_I+dmt_J.T, 0.0, k_factor, omega=omega, hermi=0) * 0.5
    #         dvhf -= td_nac.get_veff(mol, dmt_I-dmt_J.T, 0.0, k_factor, omega=omega, hermi=0) * 0.5
    #         dvhf += td_nac.get_veff(mol, dmt_I.T+dmt_J, 0.0, k_factor, omega=omega, hermi=0) * 0.5
    #         dvhf -= td_nac.get_veff(mol, dmt_I.T-dmt_J, 0.0, k_factor, omega=omega, hermi=0) * 0.5
    t_debug_7 = log.timer_silent(*time0)[2]
    time1 = log.timer('2e AO integral derivatives', *time1)

    z1aoS = z1aoS.view(cp.ndarray)
    fxcz1 = tduks_grad._contract_xc_kernel(td_nac, mf.xc, z1aoS, None, False, False)[0]
    t_debug_8 = log.timer_silent(*time0)[2]
    veff1_0 = vxc1[:, 1:]
    veff1_1 = (f1oo[:,1:] + fxcz1[:,1:] + k1ao[:,1:]) * 4
    veff1_0_a, veff1_0_b = veff1_0
    veff1_1_a, veff1_1_b = veff1_1

    de = dh_td - ds + 2 * dvhf
    dveff1_0  = rhf_grad.contract_h1e_dm(mol, veff1_0_a, dmz1dooa * 0.5, hermi=0)
    dveff1_0 += rhf_grad.contract_h1e_dm(mol, veff1_0_b, dmz1doob * 0.5, hermi=0)
    dveff1_1  = rhf_grad.contract_h1e_dm(mol, veff1_1_a, oo0a, hermi=1) * 0.25
    dveff1_1 += rhf_grad.contract_h1e_dm(mol, veff1_1_b, oo0b, hermi=1) * 0.25
    dveff1_2  = rhf_grad.contract_h1e_dm(mol, f1vo_J[1:], dmt_I, hermi=0)
    dveff1_2 += rhf_grad.contract_h1e_dm(mol, f1vo_I[1:], dmt_J, hermi=0)
    de += cp.asnumpy(dh1e_td) + dveff1_0 + dveff1_1 + dveff1_2

    if td_nac.base.extype == 0:
        dmzoo = dmzooa_IJ - dmzoob_IJ
    elif td_nac.base.extype == 1:
        dmzoo = dmzoob_IJ - dmzooa_IJ
    etf = rhf_grad.contract_h1e_dm(mol, s1, dmzoo-dmzoo.T, hermi=1) * 0.25

    delta_e = E_J - E_I
    de_etf = de.copy()
    de = de_etf + etf * delta_e

    if abs(delta_e) < 1e-10:
        logger.warn(td_nac, 'Energy difference is very small: %s. NAC is not energy scaled.', delta_e)
        de_scaled = de.copy()
        de_etf_scaled = de_etf.copy()
    else:
        de_scaled = de / delta_e
        de_etf_scaled = de_etf / delta_e
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
            logger.note(td_nac, f'Time for step {i}: {t * 1e-3:.5f}s')
    log.timer('TDUKS nuclear gradients', *time0)
    return de, de_scaled, de_etf, de_etf_scaled


def get_nacv_ge(td_nac, x_yI, EI, atmlst=None, verbose=logger.INFO):
    raise NotImplementedError(
        'Nonadiabatic coupling between excited and reference states is not implemented for spin-flip TDA/TDDFT.'
    )


class NAC(tdrhf.NAC):

    # get_veff = tduks_grad.Gradients.get_veff
    jk_energies_per_atom = tduks_sf_grad.Gradients.jk_energies_per_atom

    @lib.with_doc(get_nacv_ge.__doc__)
    def get_nacv_ge(self, x_yI, EI, atmlst=None, verbose=logger.INFO):
        return get_nacv_ge(self, x_yI, EI, atmlst=atmlst, verbose=verbose)

    @lib.with_doc(get_nacv_ee.__doc__)
    def get_nacv_ee(self, x_y_I, x_y_J, E_I, E_J, atmlst=None, verbose=logger.INFO):
        return get_nacv_ee(self, x_y_I, x_y_J, E_I, E_J, atmlst=atmlst, verbose=verbose)

    def kernel(self, states=None, atmlst=None):

        logger.warn(self, 'This module is under development!!')

        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        if states is None:
            states = self.states
        else:
            self.states = states
        states = sorted(states)

        nstates = len(self.base.e)
        I, J = states
        if I == J:
            raise ValueError('I and J should be different.')
        if I < 0 or J < 0:
            raise ValueError('Excited states ID should be non-negetive integers.')
        if I > nstates or J > nstates:
            raise ValueError(f'Excited state exceeds the number of states {nstates}.')

        if I == 0:
            logger.info(self, f'NACV between ground and excited state {J}.')
            xy_I = self.base.xy[J - 1]
            E_I = self.base.e[J - 1]
            self.de, self.de_scaled, self.de_etf, self.de_etf_scaled = self.get_nacv_ge(
                xy_I, E_I, atmlst=atmlst, verbose=self.verbose
            )
            self._finalize()
            return self.de, self.de_scaled, self.de_etf, self.de_etf_scaled

        logger.info(self, f'NACV between excited state {I} and {J}.')
        xy_I = self.base.xy[I - 1]
        E_I = self.base.e[I - 1]
        xy_J = self.base.xy[J - 1]
        E_J = self.base.e[J - 1]

        self.de, self.de_scaled, self.de_etf, self.de_etf_scaled = self.get_nacv_ee(
            xy_I, xy_J, E_I, E_J, atmlst=atmlst, verbose=self.verbose
        )
        self._finalize()
        return self.de, self.de_scaled, self.de_etf, self.de_etf_scaled

