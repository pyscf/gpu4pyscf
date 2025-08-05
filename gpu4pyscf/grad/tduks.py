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
from pyscf.dft.numint import NumInt as numint_cpu
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib import logger
from gpu4pyscf.df import int3c2e
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrhf
from gpu4pyscf.grad import tdrks
from gpu4pyscf.scf import ucphf
from gpu4pyscf import tdscf


#
# Given Y = 0, TDHF gradients (XAX+XBY+YBX+YAY)^1 turn to TDA gradients (XAX)^1
#
def grad_elec(td_grad, x_y, singlet=True, atmlst=None, verbose=logger.INFO):
    """
    Electronic part of TDA, TDDFT nuclear gradients

    Args:
        td_grad : grad.tdrhf.Gradients or grad.tdrks.Gradients object.

        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    """
    if singlet is not True and singlet is not None:
        raise NotImplementedError("Only for spin-conserving TDDFT")

    log = logger.new_logger(td_grad, verbose)
    time0 = logger.init_timer(td_grad)

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
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
    nao = mo_coeff[0].shape[0]
    nmoa = nocca + nvira
    nmob = noccb + nvirb

    (xa, xb), (ya, yb) = x_y
    xa = cp.asarray(xa)
    xb = cp.asarray(xb)
    ya = cp.asarray(ya)
    yb = cp.asarray(yb)
    xpya = (xa + ya).reshape(nocca, nvira).T
    xpyb = (xb + yb).reshape(noccb, nvirb).T
    xmya = (xa - ya).reshape(nocca, nvira).T
    xmyb = (xb - yb).reshape(noccb, nvirb).T

    dvva =  contract("ai,bi->ab", xpya, xpya) + contract("ai,bi->ab", xmya, xmya)
    dvvb =  contract("ai,bi->ab", xpyb, xpyb) + contract("ai,bi->ab", xmyb, xmyb)
    dooa = -contract("ai,aj->ij", xpya, xpya) - contract("ai,aj->ij", xmya, xmya)
    doob = -contract("ai,aj->ij", xpyb, xpyb) - contract("ai,aj->ij", xmyb, xmyb)
    dmxpya = reduce(cp.dot, (orbva, xpya, orboa.T))
    dmxpyb = reduce(cp.dot, (orbvb, xpyb, orbob.T))
    dmxmya = reduce(cp.dot, (orbva, xmya, orboa.T))
    dmxmyb = reduce(cp.dot, (orbvb, xmyb, orbob.T))
    dmzooa = reduce(cp.dot, (orboa, dooa, orboa.T))
    dmzoob = reduce(cp.dot, (orbob, doob, orbob.T))
    dmzooa += reduce(cp.dot, (orbva, dvva, orbva.T))
    dmzoob += reduce(cp.dot, (orbvb, dvvb, orbvb.T))
    td_grad.dmxpy = (dmxpya + dmxpyb)*0.5

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    # dm0 = mf.make_rdm1(mo_coeff, mo_occ), but it is not used when computing
    # fxc since rho0 is passed to fxc function.
    f1vo, f1oo, vxc1, k1ao = _contract_xc_kernel(
        td_grad, mf.xc, cp.stack((dmxpya, dmxpyb)), cp.stack((dmzooa, dmzoob)), True, True)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    if with_k:
        vj0, vk0 = mf.get_jk(mol, cp.stack((dmzooa, dmzoob)), hermi=0)
        vj1, vk1 = mf.get_jk(mol, cp.stack((dmxpya + dmxpya.T, dmxpyb + dmxpyb.T)), hermi=0)
        vj2, vk2 = mf.get_jk(mol, cp.stack((dmxmya - dmxmya.T, dmxmyb - dmxmyb.T)), hermi=0)
        if not isinstance(vj0, cp.ndarray):
            vj0 = cp.asarray(vj0)
        if not isinstance(vk0, cp.ndarray):
            vk0 = cp.asarray(vk0)
        if not isinstance(vj1, cp.ndarray):
            vj1 = cp.asarray(vj1)
        if not isinstance(vk1, cp.ndarray):
            vk1 = cp.asarray(vk1)
        if not isinstance(vj2, cp.ndarray):
            vj2 = cp.asarray(vj2)
        if not isinstance(vk2, cp.ndarray):
            vk2 = cp.asarray(vk2)
        vj = cp.stack((vj0, vj1, vj2), axis=1)
        vk = cp.stack((vk0, vk1, vk2), axis=1)
        vk *= hyb
        if omega != 0:
            vk0 = mf.get_k(mol, cp.stack((dmzooa, dmzoob)), hermi=0, omega=omega)
            vk1 = mf.get_k(mol, cp.stack((dmxpya + dmxpya.T, dmxpyb + dmxpyb.T)),
                hermi=0, omega=omega)
            vk2 = mf.get_k(mol, cp.stack((dmxmya - dmxmya.T, dmxmyb - dmxmyb.T)),
                hermi=0,omega=omega)
            if not isinstance(vk0, cp.ndarray):
                vk0 = cp.asarray(vk0)
            if not isinstance(vk1, cp.ndarray):
                vk1 = cp.asarray(vk1)
            if not isinstance(vk2, cp.ndarray):
                vk2 = cp.asarray(vk2)
            vk += cp.stack((vk0, vk1, vk2), axis=1) * (alpha - hyb)
        vj = vj.reshape(2, 3, nao, nao)
        vk = vk.reshape(2, 3, nao, nao)

        veff0doo = vj[0, 0] + vj[1, 0] - vk[:, 0] + f1oo[:, 0] + k1ao[:, 0] * 2
        veff0doo += td_grad.solvent_response((dmzooa + dmzoob)*0.5)
        wvoa = reduce(cp.dot, (orbva.T, veff0doo[0], orboa)) * 2
        wvob = reduce(cp.dot, (orbvb.T, veff0doo[1], orbob)) * 2
        veff = vj[0, 1] + vj[1, 1] - vk[:, 1] + f1vo[:, 0] * 2
        veff += td_grad.solvent_response((td_grad.dmxpy + td_grad.dmxpy.T))
        veff0mopa = reduce(cp.dot, (mo_coeff[0].T, veff[0], mo_coeff[0]))
        veff0mopb = reduce(cp.dot, (mo_coeff[1].T, veff[1], mo_coeff[1]))
        wvoa -= contract("ki,ai->ak", veff0mopa[:nocca, :nocca], xpya) * 2
        wvob -= contract("ki,ai->ak", veff0mopb[:noccb, :noccb], xpyb) * 2
        wvoa += contract("ac,ai->ci", veff0mopa[nocca:, nocca:], xpya) * 2
        wvob += contract("ac,ai->ci", veff0mopb[noccb:, noccb:], xpyb) * 2
        veff = -vk[:, 2]
        veff0moma = reduce(cp.dot, (mo_coeff[0].T, veff[0], mo_coeff[0]))
        veff0momb = reduce(cp.dot, (mo_coeff[1].T, veff[1], mo_coeff[1]))
        wvoa -= contract("ki,ai->ak", veff0moma[:nocca, :nocca], xmya) * 2
        wvob -= contract("ki,ai->ak", veff0momb[:noccb, :noccb], xmyb) * 2
        wvoa += contract("ac,ai->ci", veff0moma[nocca:, nocca:], xmya) * 2
        wvob += contract("ac,ai->ci", veff0momb[noccb:, noccb:], xmyb) * 2
    else:
        vj0 = mf.get_j(mol, cp.stack((dmzooa, dmzoob)), hermi=1)
        vj1 = mf.get_j(mol, cp.stack((dmxpya + dmxpya.T, dmxpyb + dmxpyb.T)), hermi=1)
        if not isinstance(vj0, cp.ndarray):
            vj0 = cp.asarray(vj0)
        if not isinstance(vj1, cp.ndarray):
            vj1 = cp.asarray(vj1)
        vj = cp.stack((vj0, vj1), axis=1)
        vj = vj.reshape(2, 2, nao, nao)

        veff0doo = vj[0, 0] + vj[1, 0] + f1oo[:, 0] + k1ao[:, 0] * 2
        wvoa = reduce(cp.dot, (orbva.T, veff0doo[0], orboa)) * 2
        wvob = reduce(cp.dot, (orbvb.T, veff0doo[1], orbob)) * 2
        veff = vj[0, 1] + vj[1, 1] + f1vo[:, 0] * 2
        veff0mopa = reduce(cp.dot, (mo_coeff[0].T, veff[0], mo_coeff[0]))
        veff0mopb = reduce(cp.dot, (mo_coeff[1].T, veff[1], mo_coeff[1]))
        wvoa -= contract("ki,ai->ak", veff0mopa[:nocca, :nocca], xpya) * 2
        wvob -= contract("ki,ai->ak", veff0mopb[:noccb, :noccb], xpyb) * 2
        wvoa += contract("ac,ai->ci", veff0mopa[nocca:, nocca:], xpya) * 2
        wvob += contract("ac,ai->ci", veff0mopb[noccb:, noccb:], xpyb) * 2
        veff0moma = cp.zeros((nmoa, nmoa))
        veff0momb = cp.zeros((nmob, nmob))

    vresp = td_grad.base.gen_response(hermi=1)

    def fvind(x):
        dm1 = cp.empty((2, nao, nao))
        xa = x[0, : nvira * nocca].reshape(nvira, nocca)
        xb = x[0, nvira * nocca :].reshape(nvirb, noccb)
        dma = reduce(cp.dot, (orbva, xa, orboa.T))
        dmb = reduce(cp.dot, (orbvb, xb, orbob.T))
        dm1[0] = dma + dma.T
        dm1[1] = dmb + dmb.T
        v1 = vresp(dm1)
        v1a = reduce(cp.dot, (orbva.T, v1[0], orboa))
        v1b = reduce(cp.dot, (orbvb.T, v1[1], orbob))
        return cp.hstack((v1a.ravel(), v1b.ravel()))

    z1a, z1b = ucphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        (wvoa, wvob),
        max_cycle=td_grad.cphf_max_cycle,
        tol=td_grad.cphf_conv_tol)[0]
    time1 = log.timer('Z-vector using UCPHF solver', *time0)
    z1ao = cp.empty((2, nao, nao))
    z1ao[0] = reduce(cp.dot, (orbva, z1a, orboa.T))
    z1ao[1] = reduce(cp.dot, (orbvb, z1b, orbob.T))
    veff = vresp((z1ao + z1ao.transpose(0, 2, 1)) * 0.5)

    im0a = cp.zeros((nmoa, nmoa))
    im0b = cp.zeros((nmob, nmob))
    im0a[:nocca, :nocca] = reduce(cp.dot, (orboa.T, veff0doo[0] + veff[0], orboa)) * 0.5
    im0b[:noccb, :noccb] = reduce(cp.dot, (orbob.T, veff0doo[1] + veff[1], orbob)) * 0.5
    im0a[:nocca, :nocca] += contract("ak,ai->ki", veff0mopa[nocca:, :nocca], xpya) * 0.5
    im0b[:noccb, :noccb] += contract("ak,ai->ki", veff0mopb[noccb:, :noccb], xpyb) * 0.5
    im0a[:nocca, :nocca] += contract("ak,ai->ki", veff0moma[nocca:, :nocca], xmya) * 0.5
    im0b[:noccb, :noccb] += contract("ak,ai->ki", veff0momb[noccb:, :noccb], xmyb) * 0.5
    im0a[nocca:, nocca:]  = contract("ci,ai->ac", veff0mopa[nocca:, :nocca], xpya) * 0.5
    im0b[noccb:, noccb:]  = contract("ci,ai->ac", veff0mopb[noccb:, :noccb], xpyb) * 0.5
    im0a[nocca:, nocca:] += contract("ci,ai->ac", veff0moma[nocca:, :nocca], xmya) * 0.5
    im0b[noccb:, noccb:] += contract("ci,ai->ac", veff0momb[noccb:, :noccb], xmyb) * 0.5
    im0a[nocca:, :nocca]  = contract("ki,ai->ak", veff0mopa[:nocca, :nocca], xpya)
    im0b[noccb:, :noccb]  = contract("ki,ai->ak", veff0mopb[:noccb, :noccb], xpyb)
    im0a[nocca:, :nocca] += contract("ki,ai->ak", veff0moma[:nocca, :nocca], xmya)
    im0b[noccb:, :noccb] += contract("ki,ai->ak", veff0momb[:noccb, :noccb], xmyb)

    zeta_a = (mo_energy[0][:, None] + mo_energy[0]) * 0.5
    zeta_b = (mo_energy[1][:, None] + mo_energy[1]) * 0.5
    zeta_a[nocca:, :nocca] = mo_energy[0][:nocca]
    zeta_b[noccb:, :noccb] = mo_energy[1][:noccb]
    zeta_a[:nocca, nocca:] = mo_energy[0][nocca:]
    zeta_b[:noccb, noccb:] = mo_energy[1][noccb:]
    dm1a = cp.zeros((nmoa, nmoa))
    dm1b = cp.zeros((nmob, nmob))
    dm1a[:nocca, :nocca] = dooa * 0.5
    dm1b[:noccb, :noccb] = doob * 0.5
    dm1a[nocca:, nocca:] = dvva * 0.5
    dm1b[noccb:, noccb:] = dvvb * 0.5
    dm1a[nocca:, :nocca] = z1a * 0.5
    dm1b[noccb:, :noccb] = z1b * 0.5
    dm1a[:nocca, :nocca] += cp.eye(nocca)  # for ground state
    dm1b[:noccb, :noccb] += cp.eye(noccb)
    im0a = reduce(cp.dot, (mo_coeff[0], im0a + zeta_a * dm1a, mo_coeff[0].T))
    im0b = reduce(cp.dot, (mo_coeff[1], im0b + zeta_b * dm1b, mo_coeff[1].T))
    im0 = im0a + im0b

    dmz1dooa = z1ao[0] + dmzooa
    dmz1doob = z1ao[1] + dmzoob
    td_grad.dmz1doo = (dmz1dooa + dmz1doob)*0.5
    oo0a = reduce(cp.dot, (orboa, orboa.T))
    oo0b = reduce(cp.dot, (orbob, orbob.T))

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td_grad.base._scf.nuc_grad_method()
    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_ground = contract("xij,ij->xi", h1, oo0a + oo0b)
    dh_td = contract("xij,ij->xi", h1, (dmz1dooa + dmz1doob) * 0.25 + (dmz1dooa + dmz1doob).T * 0.25)
    ds = contract("xij,ij->xi", s1, (im0 + im0.T) * 0.5)

    dh1e_ground = int3c2e.get_dh1e(mol, oo0a + oo0b)  # 1/r like terms
    if mol.has_ecp():
        dh1e_ground += rhf_grad.get_dh1e_ecp(mol, oo0a + oo0b)  # 1/r like terms
    dh1e_td = int3c2e.get_dh1e(mol, (dmz1dooa + dmz1doob) * 0.25 + (dmz1dooa + dmz1doob).T * 0.25)  # 1/r like terms
    if mol.has_ecp():
        dh1e_td += rhf_grad.get_dh1e_ecp(
            mol, (dmz1dooa + dmz1doob) * 0.25 + (dmz1dooa + dmz1doob).T * 0.25)  # 1/r like terms

    j_factor = 1.0
    k_factor = 0.0

    if with_k:
        k_factor = hyb

    if atmlst is None:
        atmlst = range(mol.natm)
    extra_force = cp.zeros((len(atmlst), 3))
    dvhf_all = 0
    # this term contributes the ground state contribution.
    dvhf = td_grad.get_veff(mol, cp.stack(((dmz1dooa + dmz1dooa.T) * 0.25 + oo0a,
                                           (dmz1doob + dmz1doob.T) * 0.25 + oo0b,)), j_factor, k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all += dvhf
    # this term will remove the unused-part from PP density.
    dvhf = td_grad.get_veff(mol, cp.stack(((dmz1dooa + dmz1dooa.T), (dmz1doob + dmz1doob.T))) * 0.25,
        j_factor, k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all -= dvhf
    dvhf = td_grad.get_veff(mol, cp.stack(((dmxpya + dmxpya.T), (dmxpyb + dmxpyb.T))) * 0.5,
            j_factor, k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals()) * 2)
    dvhf_all += dvhf * 2
    dvhf = td_grad.get_veff(mol, cp.stack(((dmxmya - dmxmya.T), (dmxmyb - dmxmyb.T))) * 0.5,
            j_factor=0.0, k_factor=k_factor, hermi=2)
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals()) * 2)
    dvhf_all += dvhf * 2

    if with_k and omega != 0:
        j_factor = 0.0
        k_factor = alpha-hyb  # =beta

        dvhf = td_grad.get_veff(
            mol, cp.stack(((dmz1dooa + dmz1dooa.T) * 0.25 + oo0a,
                           (dmz1doob + dmz1doob.T) * 0.25 + oo0b)),
            j_factor=0.0, k_factor = k_factor, omega=omega)
        for k, ia in enumerate(atmlst):
            extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals()))
        dvhf_all += dvhf
        dvhf = td_grad.get_veff(mol,
                cp.stack(((dmz1dooa + dmz1dooa.T) * 0.25, (dmz1doob + dmz1doob.T) * 0.25)),
                j_factor=0.0, k_factor = k_factor, omega=omega)
        for k, ia in enumerate(atmlst):
            extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
        dvhf_all -= dvhf
        dvhf = td_grad.get_veff(mol,
                cp.stack(((dmxpya + dmxpya.T) * 0.5, (dmxpyb + dmxpyb.T) * 0.5)),
                j_factor=0.0, k_factor = k_factor, omega=omega)
        for k, ia in enumerate(atmlst):
            extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals()) * 2)
        dvhf_all += dvhf * 2
        dvhf = td_grad.get_veff(mol,
                cp.stack(((dmxmya - dmxmya.T) * 0.5, (dmxmyb - dmxmyb.T) * 0.5)),
                j_factor=0.0, k_factor = k_factor, omega=omega, hermi=2)
        for k, ia in enumerate(atmlst):
            extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals()) * 2)
        dvhf_all += dvhf * 2
    time1 = log.timer('2e AO integral derivatives', *time1)

    fxcz1 = _contract_xc_kernel(td_grad, mf.xc, z1ao, None, False, False)[0]

    veff1_0 = vxc1[:, 1:]
    veff1_1 = (f1oo[:, 1:] + fxcz1[:, 1:] + k1ao[:, 1:] * 2) * 2  # *2 for dmz1doo+dmz1oo.T
    veff1_2 = f1vo[:, 1:] * 2
    veff1_0_a, veff1_0_b = veff1_0
    veff1_1_a, veff1_1_b = veff1_1
    veff1_2_a, veff1_2_b = veff1_2

    delec = 2.0 * (dh_ground + dh_td - ds)
    aoslices = mol.aoslice_by_atom()
    delec = cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])
    dveff1_0 = cp.asarray(
        [contract("xpq,pq->x", veff1_0_a[:, p0:p1], oo0a[p0:p1] + dmz1dooa[p0:p1] * 0.5) for p0, p1 in aoslices[:, 2:]])
    dveff1_0 += cp.asarray(
        [contract("xpq,pq->x", veff1_0_b[:, p0:p1], oo0b[p0:p1] + dmz1doob[p0:p1] * 0.5) for p0, p1 in aoslices[:, 2:]])
    dveff1_0 += cp.asarray(
        [contract("xpq,qp->x", veff1_0_a[:, p0:p1], oo0a[:, p0:p1] + dmz1dooa[:, p0:p1] * 0.5)
            for p0, p1 in aoslices[:, 2:]])
    dveff1_0 += cp.asarray(
        [contract("xpq,qp->x", veff1_0_b[:, p0:p1], oo0b[:, p0:p1] + dmz1doob[:, p0:p1] * 0.5)
            for p0, p1 in aoslices[:, 2:]])
    dveff1_1 = cp.asarray(
        [contract("xpq,pq->x", veff1_1_a[:, p0:p1], oo0a[p0:p1] * 0.5) for p0, p1 in aoslices[:, 2:]])
    dveff1_1 += cp.asarray(
        [contract("xpq,pq->x", veff1_1_b[:, p0:p1], oo0b[p0:p1] * 0.5) for p0, p1 in aoslices[:, 2:]])
    dveff1_2 = cp.asarray([contract("xpq,pq->x", veff1_2_a[:, p0:p1], dmxpya[p0:p1]) for p0, p1 in aoslices[:, 2:]])
    dveff1_2 += cp.asarray(
        [contract("xqp,pq->x", veff1_2_a[:, p0:p1], dmxpya[:, p0:p1]) for p0, p1 in aoslices[:, 2:]])
    dveff1_2 += cp.asarray([contract("xpq,pq->x", veff1_2_b[:, p0:p1], dmxpyb[p0:p1]) for p0, p1 in aoslices[:, 2:]])
    dveff1_2 += cp.asarray(
        [contract("xqp,pq->x", veff1_2_b[:, p0:p1], dmxpyb[:, p0:p1]) for p0, p1 in aoslices[:, 2:]])
    de = 2.0 * dvhf_all + dh1e_ground + dh1e_td + delec + extra_force + dveff1_0 + dveff1_1 + dveff1_2

    log.timer('TDUKS nuclear gradients', *time0)
    return de.get()


# dmov, dmoo in AO-representation
# Note spin-trace is applied for fxc, kxc
# TODO: to include the response of grids
def _contract_xc_kernel(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True, with_kxc=True):
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

    opt = getattr(ni, "gdftopt", None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])

    # dmvo ~ reduce(cp.dot, (orbv, Xai, orbo.T))
    dmvo = cp.array([(dmvo[0] + dmvo[0].T) * 0.5,  # because K_{ia,jb} == K_{ia,jb}
                     (dmvo[1] + dmvo[1].T) * 0.5])
    dmvo = opt.sort_orbitals(dmvo, axis=[1, 2])

    f1vo = cp.zeros((2, 4, nao, nao))
    deriv = 2
    if dmoo is not None:
        f1oo = cp.zeros((2, 4, nao, nao))
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
    else:
        k1ao = None

    if xctype == "HF":
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == "LDA":
        fmat_, ao_deriv = tdrks._lda_eval_mat_, 1
    elif xctype == "GGA":
        fmat_, ao_deriv = tdrks._gga_eval_mat_, 2
    elif xctype == "MGGA":
        fmat_, ao_deriv = tdrks._mgga_eval_mat_, 2
        logger.warn(td_grad, "TDUKS-MGGA Gradients may be inaccurate due to grids response")
    else:
        raise NotImplementedError(f"td-uks for functional {xc_code}")

    if (not td_grad.base.exclude_nlc) and mf.do_nlc():
        raise NotImplementedError("TDDFT gradient with NLC contribution is not supported yet. "
                                  "Please set exclude_nlc field of tdscf object to True, "
                                  "which will turn off NLC contribution in the whole TDDFT calculation.")

    for ao, mask, weight, coords in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
        if xctype == "LDA":
            ao0 = ao[0]
        else:
            ao0 = ao
        mo_coeff_mask_a = mo_coeff[0, mask]
        mo_coeff_mask_b = mo_coeff[1, mask]
        rho = cp.asarray((
            ni.eval_rho2(_sorted_mol, ao0, mo_coeff_mask_a, mo_occ[0], mask, xctype,with_lapl=False),
            ni.eval_rho2(_sorted_mol, ao0, mo_coeff_mask_b, mo_occ[1], mask, xctype, with_lapl=False)))
        if deriv > 2:
            ni_cpu = numint_cpu()
            # TODO: If the libxc is stablized, this should be gpulized
            # vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]
            vxc, fxc, kxc = ni_cpu.eval_xc_eff(xc_code, rho.get(), deriv, xctype=xctype)[1:]
            if isinstance(vxc, np.ndarray): vxc = cp.asarray(vxc)
            if isinstance(fxc, np.ndarray): fxc = cp.asarray(fxc)
            if isinstance(kxc, np.ndarray): kxc = cp.asarray(kxc)
        else:
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]
        dmvo_mask_a = dmvo[0, mask[:, None], mask]
        dmvo_mask_b = dmvo[1, mask[:, None], mask]
        rho1 = cp.asarray((
            ni.eval_rho(_sorted_mol, ao0, dmvo_mask_a, mask, xctype, hermi=1, with_lapl=False),
            ni.eval_rho(_sorted_mol, ao0, dmvo_mask_b, mask, xctype, hermi=1, with_lapl=False)))
        if xctype == "LDA":
            rho1 = rho1[:, cp.newaxis].copy()
        tmp = contract("axg,axbyg->byg", rho1, fxc)
        wv = contract("byg,g->byg", tmp, weight)
        tmp = None
        fmat_(_sorted_mol, f1vo[0], ao, wv[0], mask, shls_slice, ao_loc)
        fmat_(_sorted_mol, f1vo[1], ao, wv[1], mask, shls_slice, ao_loc)

        if dmoo is not None:
            dmoo_mask_a = dmoo[0, mask[:, None], mask]
            dmoo_mask_b = dmoo[1, mask[:, None], mask]
            rho2 = cp.asarray((
                ni.eval_rho(_sorted_mol, ao0, dmoo_mask_a, mask, xctype, hermi=1, with_lapl=False),
                ni.eval_rho(_sorted_mol, ao0, dmoo_mask_b, mask, xctype, hermi=1, with_lapl=False)))
            if xctype == "LDA":
                rho2 = rho2[:, cp.newaxis].copy()
            tmp = contract("axg,axbyg->byg", rho2, fxc)
            wv = contract("byg,g->byg", tmp, weight)
            tmp = None
            fmat_(_sorted_mol, f1oo[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(_sorted_mol, f1oo[1], ao, wv[1], mask, shls_slice, ao_loc)
        if with_vxc:
            wv = vxc * weight
            fmat_(_sorted_mol, v1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(_sorted_mol, v1ao[1], ao, wv[1], mask, shls_slice, ao_loc)
        if with_kxc:
            tmp = contract("axg,axbyczg->byczg", rho1, kxc)
            tmp = contract("byg, byczg->czg", rho1, tmp)
            wv = contract("czg,g->czg", tmp, weight)
            tmp = None
            fmat_(_sorted_mol, k1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(_sorted_mol, k1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

    f1vo[:, 1:] *= -1
    f1vo = opt.unsort_orbitals(f1vo, axis=[2, 3])
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


class Gradients(tdrhf.Gradients):
    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet=None, atmlst=None, verbose=logger.info):
        return grad_elec(self, xy, singlet, atmlst, self.verbose)


Grad = Gradients

tdscf.uks.TDA.Gradients = tdscf.uks.TDDFT.Gradients = lib.class_as_method(Gradients)
