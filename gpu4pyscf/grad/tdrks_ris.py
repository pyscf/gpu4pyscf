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
from pyscf import lib, gto
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.df import int3c2e
from gpu4pyscf.df.grad import tdrhf as tdrhf_df
from gpu4pyscf.dft import rks
from gpu4pyscf.scf import cphf
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrhf
from gpu4pyscf.grad import tdrks
from gpu4pyscf import tdscf
from gpu4pyscf.tdscf.ris import get_auxmol

def grad_elec(td_grad, x_y, theta=None, J_fit=None, K_fit=None, singlet=True, atmlst=None, verbose=logger.INFO):
    """
    Electronic part of TDA, TDDFT nuclear gradients

    Args:
        td_grad : grad.tdrhf.Gradients or grad.tdrks.Gradients object.

        x_y : a two-element list of cp arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    """
    if td_grad.base.Ktrunc != 0.0:
        raise NotImplementedError('Ktrunc or frozen method is not supported yet')
    if J_fit is None:
        J_fit = td_grad.base.J_fit
    if K_fit is None:
        K_fit = td_grad.base.K_fit
    if singlet is None:
        singlet = True
    log = logger.new_logger(td_grad, verbose)
    if not singlet:
        raise ValueError('TDDFT ris only supports singlet state')
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
    y = cp.asarray(y)
    xpy = (x + y).reshape(nocc, nvir).T
    xmy = (x - y).reshape(nocc, nvir).T
    orbv = mo_coeff[:, nocc:]
    orbo = mo_coeff[:, :nocc]
    if getattr(mf, 'with_solvent', None) is not None:
        raise NotImplementedError('With solvent is not supported yet')
    
    dvv = contract("ai,bi->ab", xpy, xpy) + contract("ai,bi->ab", xmy, xmy)  # 2 T_{ab}
    doo = -contract("ai,aj->ij", xpy, xpy) - contract("ai,aj->ij", xmy, xmy)  # 2 T_{ij}
    dmxpy = reduce(cp.dot, (orbv, xpy, orbo.T))  # (X+Y) in ao basis
    dmxmy = reduce(cp.dot, (orbv, xmy, orbo.T))  # (X-Y) in ao basis
    dmzoo = reduce(cp.dot, (orbo, doo, orbo.T))  # T_{ij}*2 in ao basis
    dmzoo += reduce(cp.dot, (orbv, dvv, orbv.T))  # T_{ij}*2 + T_{ab}*2 in ao basis
    td_grad.dmxpy = dmxpy

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
    if K_fit == J_fit and (omega == 0 or omega is None):
        log.info('K uese exactly same basis as J, and they share same set of Tensors')
        auxmol_K = auxmol_J
    else:
        log.info('K uese different basis as J')
        auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit)
    mf_J = rks.RKS(mol).density_fit()
    mf_J.with_df.auxmol = auxmol_J
    mf_K = rks.RKS(mol).density_fit()
    mf_K.with_df.auxmol = auxmol_K
    
    f1vo, f1oo, vxc1, k1ao = tdrks._contract_xc_kernel(td_grad, mf.xc, dmxpy, dmzoo, True, True, singlet)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    if with_k:
        vj0, vk0 = mf.get_jk(mol, dmzoo, hermi=0)
        vj1 = mf_J.get_j(mol, dmxpy + dmxpy.T, hermi=0)
        vk1 = mf_K.get_k(mol, dmxpy + dmxpy.T, hermi=0)
        vk2 = mf_K.get_k(mol, dmxmy - dmxmy.T, hermi=0)
        if not isinstance(vj0, cp.ndarray):
            vj0 = cp.asarray(vj0)
        if not isinstance(vk0, cp.ndarray):
            vk0 = cp.asarray(vk0)
        if not isinstance(vj1, cp.ndarray):
            vj1 = cp.asarray(vj1)
        if not isinstance(vk1, cp.ndarray):
            vk1 = cp.asarray(vk1)
        if not isinstance(vk2, cp.ndarray):
            vk2 = cp.asarray(vk2)
        vj = cp.stack((vj0, vj1))
        vk = cp.stack((vk0, vk1, vk2))
        vk *= hyb
        if omega != 0:
            vk0 = mf.get_k(mol, dmzoo, hermi=0, omega=omega)
            vk1 = mf_K.get_k(mol, dmxpy + dmxpy.T, hermi=0, omega=omega)
            vk2 = mf_K.get_k(mol, dmxmy - dmxmy.T, hermi=0, omega=omega)
            if not isinstance(vk0, cp.ndarray):
                vk0 = cp.asarray(vk0)
            if not isinstance(vk1, cp.ndarray):
                vk1 = cp.asarray(vk1)
            if not isinstance(vk2, cp.ndarray):
                vk2 = cp.asarray(vk2)
            vk += cp.stack((vk0, vk1, vk2)) * (alpha - hyb)
        veff0doo = vj[0] * 2 - vk[0] + f1oo[0]
        veff0doo += td_grad.solvent_response(dmzoo)
        wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
        if singlet:
            veff = vj[1] * 2 - vk[1]
        else:
            veff = - vk[1]
        veff += td_grad.solvent_response(dmxpy + dmxpy.T)
        veff0mop = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= contract("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2
        wvo += contract("ac,ai->ci", veff0mop[nocc:, nocc:], xpy) * 2
        veff = -vk[2]
        veff0mom = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= contract("ki,ai->ak", veff0mom[:nocc, :nocc], xmy) * 2
        wvo += contract("ac,ai->ci", veff0mom[nocc:, nocc:], xmy) * 2
    else:
        vj0 = mf.get_j(mol, dmzoo, hermi=1)
        vj1 = mf_J.get_j(mol, dmxpy + dmxpy.T, hermi=1)
        if not isinstance(vj0, cp.ndarray):
            vj0 = cp.asarray(vj0)
        if not isinstance(vj1, cp.ndarray):
            vj1 = cp.asarray(vj1)
        vj = cp.stack((vj0, vj1))

        veff0doo = vj[0] * 2 + f1oo[0] 
        wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
        if singlet:
            veff = vj[1] * 2
        else:
            veff = 0
        veff0mop = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
        wvo -= contract("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2
        wvo += contract("ac,ai->ci", veff0mop[nocc:, nocc:], xpy) * 2
        veff0mom = cp.zeros((nmo, nmo))

    # set singlet=None, generate function for CPHF type response kernel
    # TODO: LR-PCM TDDFT
    vresp = td_grad.base._scf.gen_response(singlet=None, hermi=1)

    def fvind(x):
        dm = reduce(cp.dot, (orbv, x.reshape(nvir, nocc) * 2, orbo.T))
        v1ao = vresp(dm + dm.T)
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()

    z1 = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        wvo,
        max_cycle=td_grad.cphf_max_cycle,
        tol=td_grad.cphf_conv_tol)[0]
    z1 = z1.reshape(nvir, nocc)
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    z1ao = reduce(cp.dot, (orbv, z1, orbo.T))
    veff = vresp(z1ao + z1ao.T)

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
    s1 = mf_grad.get_ovlp(mol)

    dmz1doo = z1ao + dmzoo
    td_grad.dmz1doo = dmz1doo
    oo0 = reduce(cp.dot, (orbo, orbo.T))

    if atmlst is None:
        atmlst = range(mol.natm)
    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_ground = contract("xij,ij->xi", h1, oo0 * 2)
    dh_td = contract("xij,ij->xi", h1, (dmz1doo + dmz1doo.T) * 0.5)
    ds = contract("xij,ij->xi", s1, (im0 + im0.T) * 0.5)

    dh1e_ground = int3c2e.get_dh1e(mol, oo0 * 2)  # 1/r like terms
    if mol.has_ecp():
        dh1e_ground += rhf_grad.get_dh1e_ecp(mol, oo0 * 2)  # 1/r like terms
    dh1e_td = int3c2e.get_dh1e(mol, (dmz1doo + dmz1doo.T) * 0.5)  # 1/r like terms
    if mol.has_ecp():
        dh1e_td += rhf_grad.get_dh1e_ecp(mol, (dmz1doo + dmz1doo.T) * 0.5)  # 1/r like terms

    j_factor = 1.0
    k_factor = 0.0
    if with_k:
        k_factor = hyb

    extra_force = cp.zeros((len(atmlst), 3))
    dvhf_all = 0
    # this term contributes the ground state contribution.
    dvhf = td_grad.get_veff(mol, (dmz1doo + dmz1doo.T) * 0.5 + oo0 * 2, j_factor=j_factor, k_factor=k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all += dvhf
    # this term will remove the unused-part from PP density.
    dvhf = td_grad.get_veff(mol, (dmz1doo + dmz1doo.T) * 0.5, j_factor=j_factor, k_factor=k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all -= dvhf
    if singlet:
        j_factor=1.0
    else:
        j_factor=0.0
    dvhf = get_veff_ris(mf_J, mf_K, mol, dmxpy + dmxpy.T, j_factor=j_factor, k_factor=k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(get_extra_force(ia, locals()) * 2)
    dvhf_all += dvhf * 2
    dvhf = get_veff_ris(mf_J, mf_K, mol, dmxmy - dmxmy.T, j_factor=0.0, k_factor=k_factor, hermi=2)
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(get_extra_force(ia, locals()) * 2)
    dvhf_all += dvhf * 2

    if with_k and omega != 0:
        j_factor = 0.0
        k_factor = alpha-hyb  # =beta

        dvhf = td_grad.get_veff(mol, (dmz1doo + dmz1doo.T) * 0.5 + oo0 * 2, 
                                j_factor=j_factor, k_factor=k_factor, omega=omega)
        for k, ia in enumerate(atmlst):
            extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals()))
        dvhf_all += dvhf
        dvhf = td_grad.get_veff(mol, (dmz1doo + dmz1doo.T) * 0.5, 
                                j_factor=j_factor, k_factor=k_factor, omega=omega)
        for k, ia in enumerate(atmlst):
            extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
        dvhf_all -= dvhf
        dvhf = get_veff_ris(mf_J, mf_K, mol, dmxpy + dmxpy.T, 
                                j_factor=j_factor, k_factor=k_factor, omega=omega)
        for k, ia in enumerate(atmlst):
            extra_force[k] += cp.asarray(get_extra_force(ia, locals()) * 2)
        dvhf_all += dvhf * 2
        dvhf = get_veff_ris(mf_J, mf_K, mol, dmxmy - dmxmy.T, 
                                j_factor=j_factor, k_factor=k_factor, omega=omega, hermi=2)
        for k, ia in enumerate(atmlst):
            extra_force[k] += cp.asarray(get_extra_force(ia, locals()) * 2)
        dvhf_all += dvhf * 2
    time1 = log.timer('2e AO integral derivatives', *time1)
    fxcz1 = tdrks._contract_xc_kernel(td_grad, mf.xc, z1ao, None, False, False, True)[0]

    veff1_0 = vxc1[1:]
    veff1_1 = (f1oo[1:] + fxcz1[1:]) * 2  # *2 for dmz1doo+dmz1oo.T

    delec = 2.0 * (dh_ground + dh_td - ds) #  - ds
    aoslices = mol.aoslice_by_atom()
    delec = cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])
    dveff1_0 = cp.asarray(
        [contract("xpq,pq->x", veff1_0[:, p0:p1], oo0[p0:p1] * 2 + dmz1doo[p0:p1]) for p0, p1 in aoslices[:, 2:]])
    dveff1_0 += cp.asarray([
            contract("xpq,pq->x", veff1_0[:, p0:p1].transpose(0, 2, 1), oo0[:, p0:p1] * 2 + dmz1doo[:, p0:p1],)
            for p0, p1 in aoslices[:, 2:]])
    dveff1_1 = cp.asarray([contract("xpq,pq->x", veff1_1[:, p0:p1], oo0[p0:p1]) for p0, p1 in aoslices[:, 2:]])
    de = 2.0 * dvhf_all + dh1e_ground + dh1e_td + delec + extra_force + dveff1_0 + dveff1_1

    return de.get()


def get_extra_force(atom_id, envs):
    return envs['dvhf'].aux[atom_id]


def get_veff_ris(mf_J, mf_K, mol=None, dm=None, j_factor=1.0, k_factor=1.0, omega=0.0, hermi=0, verbose=None):
    
    if omega != 0.0:
        vj, _, vjaux, _ = tdrhf_df.get_jk(mf_J, mol, dm, omega=omega, hermi=hermi)
        _, vk, _, vkaux = tdrhf_df.get_jk(mf_K, mol, dm, omega=omega, hermi=hermi)
    else:
        vj, _, vjaux, _ = tdrhf_df.get_jk(mf_J, mol, dm, hermi=hermi)
        _, vk, _, vkaux = tdrhf_df.get_jk(mf_K, mol, dm, hermi=hermi)
    vhf = vj * j_factor - vk * .5 * k_factor
    e1_aux = vjaux * j_factor - vkaux * .5 * k_factor
    vhf = tag_array(vhf, aux=e1_aux)
    return vhf


class Gradients(tdrhf.Gradients):
    def kernel(self, xy=None, state=None, singlet=None, atmlst=None):
        """
        Args:
            state : int
                Excited state ID.  state = 1 means the first excited state.
        """
        if self.base.Ktrunc != 0.0:
            raise NotImplementedError('Ktrunc or frozen method is not supported yet')
        log = self.base.log
        warn_message = "TDDFT-ris gradient is still in the experimental stage, \n" +\
            "and its APIs are subject to change in future releases."
        log.warn(warn_message)
        if xy is None:
            if state is None:
                state = self.state
            else:
                self.state = state

            if state == 0:
                log.warn(
                    "state=0 found in the input. Gradients of ground state is computed.",
                )
                return self.base._scf.nuc_grad_method().kernel(atmlst=atmlst)
            if self.base.xy[1] is not None:
                xy = (self.base.xy[0][state-1]*np.sqrt(0.5), self.base.xy[1][state-1]*np.sqrt(0.5))
            else:
                xy = (self.base.xy[0][state-1]*np.sqrt(0.5), self.base.xy[0][state-1]*0.0)

        if singlet is None:
            singlet = self.base.singlet
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()
        theta = self.base.theta
        de = self.grad_elec(xy, theta, singlet, atmlst, verbose=self.verbose)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de
    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, theta, singlet, atmlst=None, verbose=logger.info):
        return grad_elec(self, xy, theta, singlet=singlet, atmlst=atmlst, verbose=self.verbose)


Grad = Gradients

tdscf.ris.TDA.Gradients = tdscf.ris.TDDFT.Gradients = lib.class_as_method(Gradients)
