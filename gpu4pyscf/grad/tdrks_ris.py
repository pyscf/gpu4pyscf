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
from gpu4pyscf.df.df_jk import (
    _tag_factorize_dm, _DFHF, _make_factorized_dm, _aggregate_dm_factor_l)
from gpu4pyscf.dft import rks
from gpu4pyscf.scf import cphf
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrhf
from gpu4pyscf.grad import tdrks
from gpu4pyscf import tdscf
from gpu4pyscf.tdscf.ris import get_auxmol, rescale_spin_free_amplitudes, TDA


def gen_response_ris(mf, mf_J, mf_K, mo_coeff=None, mo_occ=None,
                      singlet=None, hermi=0):
    '''Generate a function to compute the product of RHF response function and
    RHF density matrices.

    Kwargs:
        singlet (None or boolean) : If singlet is None, response function for
            orbital hessian or CPHF will be generated. If singlet is boolean,
            it is used in TDDFT response kernel.
    '''
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    if singlet is not None:
        raise ValueError('TDDFT ris solver only supports singlet state')

    if singlet is None:
        # Without specify singlet, used in ground state orbital hessian
        def vind(dm1):
            # The singlet hessian
            v1 = cp.zeros_like(dm1)
            if hybrid:
                if hermi != 2:
                    vj = mf_J.get_j(mol, dm1, hermi=hermi)
                    vk = mf_K.get_k(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if omega > 1e-10:  # For range separated Coulomb
                        vk += mf_K.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 += vj - .5 * vk
                else:
                    vk = mf_K.get_k(mol, dm1, hermi=hermi)
                    v1 -= .5 * hyb * vk
            elif hermi != 2:
                vj = mf_J.get_j(mol, dm1, hermi=hermi)
                v1 += vj
            return v1

    return vind

def grad_elec(td_grad, x_y, singlet=True, atmlst=None, verbose=logger.INFO):
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
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.init_timer(td_grad)

    J_fit = td_grad.base.J_fit
    K_fit = td_grad.base.K_fit
    theta = td_grad.base.theta
    if singlet is None:
        singlet = True
    if not singlet:
        raise ValueError('TDDFT ris only supports singlet state')

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    orbo = mo_coeff[:, mo_occ > 0]
    orbv = mo_coeff[:, mo_occ ==0]
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]
    x, y = x_y
    x = cp.asarray(x)
    is_tda = isinstance(td_grad.base, TDA)
    if is_tda:
        xpy = xmy = x.reshape(nocc, nvir).T
    else:
        y = cp.asarray(y)
        xpy = (x + y).reshape(nocc, nvir).T
        xmy = (x - y).reshape(nocc, nvir).T
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
    t_debug_1 = log.timer_silent(*time0)[2]
    f1oo, _, vxc1, _ = tdrks._contract_xc_kernel(td_grad, mf.xc, dmzoo, None, True, False, singlet)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    t_debug_2 = log.timer_silent(*time0)[2]
    if with_k:
        dmzoo = _tag_factorize_dm(dmzoo, hermi=1)
        vj0, vk0 = mf.get_jk(mol, dmzoo, hermi=1)
        vj1 = mf_J.get_j(mol, dmxpy + dmxpy.T, hermi=0)
        vk1 = mf_K.get_k(mol, dmxpy + dmxpy.T, hermi=0)
        vk2 = mf_K.get_k(mol, dmxmy - dmxmy.T, hermi=0)
        vj = cp.stack((vj0, vj1))
        vk = cp.stack((vk0, vk1, vk2))
        vk *= hyb
        if omega != 0:
            vk0 = mf.get_k(mol, dmzoo, hermi=1, omega=omega)
            vk1 = mf_K.get_k(mol, dmxpy + dmxpy.T, hermi=0, omega=omega)
            vk2 = mf_K.get_k(mol, dmxmy - dmxmy.T, hermi=0, omega=omega)
            vk += cp.stack((vk0, vk1, vk2)) * (alpha - hyb)
        dmzoo = dmzoo.view(cp.ndarray)
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
    if td_grad.ris_zvector_solver:
        log.note('Use ris-approximated Z-vector solver')
        vresp = gen_response_ris(mf, mf_J, mf_K, singlet=None, hermi=1)
    else:
        log.note('Use standard Z-vector solver')
        vresp = td_grad.base._scf.gen_response(singlet=None, hermi=1)
    t_debug_3 = log.timer_silent(*time0)[2]
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
    t_debug_4 = log.timer_silent(*time0)[2]
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    z1 = z1.reshape(nvir, nocc)
    z1aoS = _make_factorized_dm(orbv.dot(z1), orbo, symmetrize=1)
    veff = vresp(z1aoS)

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
    t_debug_5 = log.timer_silent(*time0)[2]

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td_grad.base._scf.nuc_grad_method()

    dmz1doo = z1aoS*.5 + dmzoo
    td_grad.dmz1doo = dmz1doo
    oo0 = _make_factorized_dm(orbo*2, orbo, symmetrize=0) # *2 for double occupancy

    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dm_correlated = dmz1doo + oo0
    dh_ground_and_td = rhf_grad.contract_h1e_dm(mol, h1, dm_correlated, hermi=1)
    ds = rhf_grad.contract_h1e_dm(mol, s1, im0, hermi=0)

    dh1e_ground_and_td = int3c2e.get_dh1e(mol, dm_correlated)  # 1/r like terms
    if len(mol._ecpbas) > 0:
        dh1e_ground_and_td += rhf_grad.get_dh1e_ecp(mol, dm_correlated)  # 1/r like terms

    if mol._pseudo:
        raise NotImplementedError("Pseudopotential gradient not supported for molecular system yet")
    t_debug_6 = log.timer_silent(*time0)[2]

    dms = [[_tag_factorize_dm(oo0+dmz1doo*2., hermi=1), oo0]]
    j_factor = [1]
    k_factor = None
    if with_k:
        k_factor = np.array([1.])
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

    dms = cp.array([dmxpy + dmxpy.T, dmxmy - dmxmy.T])
    j_factor = None
    k_factor = None
    if singlet:
        j_factor = [2,  0]
    if with_k:
        k_factor = np.array([2., -2.])
        ejk += jk_energy_per_atom(mf_J, mf_K, mol, dms, j_factor, k_factor*hyb)
    else:
        ejk += jk_energy_per_atom(mf_J, mf_K, mol, dms, j_factor, None)
    if with_k and omega != 0:
        j_factor = None
        beta = alpha - hyb
        ejk += jk_energy_per_atom(mf_J, mf_K, mol, dms, j_factor, k_factor*beta, omega=omega)

    t_debug_7 = log.timer_silent(*time0)[2]
    time1 = log.timer('2e AO integral derivatives', *time1)
    fxcz1 = tdrks._contract_xc_kernel(td_grad, mf.xc, z1aoS*.5, None, False, False, True)[0]
    t_debug_8 = log.timer_silent(*time0)[2]
    veff1_0 = vxc1[1:]
    veff1_1 = (f1oo[1:] + fxcz1[1:]) * 2  # *2 for dmz1doo+dmz1oo.T

    de = dh_ground_and_td + cp.asnumpy(dh1e_ground_and_td) - ds + ejk
    dveff1_0 = rhf_grad.contract_h1e_dm(mol, veff1_0, dm_correlated, hermi=0)
    dveff1_1 = rhf_grad.contract_h1e_dm(mol, veff1_1, oo0, hermi=1) * .25
    de += dveff1_0 + dveff1_1
    if atmlst is not None:
        de = de[atmlst]
    t_debug_9 = log.timer_silent(*time0)[2]
    if log.verbose >= logger.DEBUG:
        time_list = [0, t_debug_1, t_debug_2, t_debug_3, t_debug_4, t_debug_5, t_debug_6, t_debug_7, t_debug_8, t_debug_9]
        time_list = [time_list[i+1] - time_list[i] for i in range(len(time_list) - 1)]
        for i, t in enumerate(time_list):
            logger.note(td_grad, f"Time for step {i}: {t*1e-3:.6f}s")
    return de


def get_veff_ris(mf_J, mf_K, mol, dm, j_factor=1.0, k_factor=1.0, omega=0.0, hermi=0, verbose=None):
    from gpu4pyscf.df.grad.rhf import _jk_energy_per_atom, Int3c2eOpt
    auxmol_J = mf_J.with_df.auxmol
    auxmol_K = mf_K.with_df.auxmol
    with mol.with_range_coulomb(omega), auxmol_K.with_range_coulomb(omega):
        int3c2e_opt = Int3c2eOpt(mol, auxmol_K).build()
        ejk = _jk_energy_per_atom(int3c2e_opt, dm, 0, k_factor, hermi, verbose=verbose) * .5
    if hermi != 2:
        with mol.with_range_coulomb(omega), auxmol_J.with_range_coulomb(omega):
            int3c2e_opt = Int3c2eOpt(mol, auxmol_J).build()
            ejk += _jk_energy_per_atom(int3c2e_opt, dm, j_factor, 0, hermi, verbose=verbose) * .5
    ejk *= .5
    return ejk

def jk_energy_per_atom(mf_J, mf_K, mol, dms, j_factor=None, k_factor=None, omega=0.0, hermi=0, verbose=None):
    from gpu4pyscf.df.grad.tdrhf import _jk_energy_per_atom, Int3c2eOpt
    auxmol_J = mf_J.with_df.auxmol
    auxmol_K = mf_K.with_df.auxmol
    ejk = np.zeros((mol.natm, 3))
    if k_factor is not None:
        with mol.with_range_coulomb(omega), auxmol_K.with_range_coulomb(omega):
            int3c2e_opt = Int3c2eOpt(mol, auxmol_K).build()
            ejk += _jk_energy_per_atom(int3c2e_opt, dms, None, k_factor, hermi, verbose=verbose)
    if j_factor is not None and hermi != 2:
        with mol.with_range_coulomb(omega), auxmol_J.with_range_coulomb(omega):
            int3c2e_opt = Int3c2eOpt(mol, auxmol_J).build()
            ejk += _jk_energy_per_atom(int3c2e_opt, dms, j_factor, None, hermi, verbose=verbose)
    return ejk

def jk_energies_per_atom(mf_J, mf_K, mol, dms, j_factor=None, k_factor=None,
                         omega=0.0, hermi=0, sum_results=False, verbose=None):
    from gpu4pyscf.df.grad.tdrhf import _jk_energies_per_atom, Int3c2eOpt
    auxmol_J = mf_J.with_df.auxmol
    auxmol_K = mf_K.with_df.auxmol
    ejk = np.zeros((len(dms), mol.natm, 3))
    if k_factor is not None:
        with mol.with_range_coulomb(omega), auxmol_K.with_range_coulomb(omega):
            int3c2e_opt = Int3c2eOpt(mol, auxmol_K).build()
            ejk += _jk_energies_per_atom(int3c2e_opt, dms, None, k_factor, hermi, verbose=verbose)
    if j_factor is not None:
        with mol.with_range_coulomb(omega), auxmol_J.with_range_coulomb(omega):
            int3c2e_opt = Int3c2eOpt(mol, auxmol_J).build()
            ejk += _jk_energies_per_atom(int3c2e_opt, dms, j_factor, None, hermi, verbose=verbose)
    if sum_results:
        ejk = ejk.sum(axis=0)
    return ejk


class Gradients(tdrhf.Gradients):
    """
    Analytical gradients for TDRKS using the RIS approximation.

    This class implements the analytical gradient calculation between TDRKS excited states
    (or between excited state and ground state) utilizing the Resolution of Identity (RI)
    approximation for both Coulomb and Exchange integrals.

    Attributes:
        ris_zvector_solver: Enables approximate solution for the Z-vector
            equation (Lagrangian multipliers) using the RIS approximate integrals.

            Although the integrals in TDDFT or TDA linear response are evaluated
            using the RIS approximation, the ground-state orbital response from
            the Z-vector equation requires the exact integrals used in the
            ground-state SCF procedure. Solving Z-vector equation dominates the
            cost of gradient computation. This step can be accelerated by using
            RIS approximate integrals, enabled by the ris_zvector_solver parameter.
            However, this approximation breaks strict consistency between
            excited-state energies and gradients. It should therefore be used
            with caution in geometry-optimization tasks.

    References:
        For the detailed derivation of the RIS gradient and Z-vector equation,
        please refer to the following paper:

        [1] "Analytical Excited-State Gradients and Derivative
            Couplings in TDDFT with Minimal Auxiliary Basis Set
            Approximation and GPU Acceleration",
            ArXiv:2511.18233
    """

    _keys = {'ris_zvector_solver'}

    ris_zvector_solver = False

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
            xy = rescale_spin_free_amplitudes(self.base.xy, state-1)

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
        if self.verbose >= logger.DEBUG and self.ris_zvector_solver:
            log.debug('Using ris-approximated zvector solver')

        de = self.grad_elec(xy, singlet, atmlst, verbose=self.verbose)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de

    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet, atmlst=None, verbose=logger.info):
        return grad_elec(self, xy, singlet=singlet, atmlst=atmlst, verbose=self.verbose)


Grad = Gradients
