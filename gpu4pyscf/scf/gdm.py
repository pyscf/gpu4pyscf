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
Geometric direct minimization (GDM) SCF solver
'''

import numpy as np
import cupy as cp
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import eigh

GDM_LBFGS_Y_DOT_S_ZERO_THRESHOLD = 1e-12

def pack_nocc_nvir_shape_vector(vector, nocc, nmo):
    if isinstance(nocc, int):
        assert nocc < nmo
        assert vector.shape == (nocc, nmo - nocc)
        return vector.flatten(order = "C")
    else:
        nocca, noccb = nocc
        assert isinstance(nocca, int) and isinstance(noccb, int)
        assert nocca < nmo and noccb < nmo
        va, vb = vector
        assert va.shape == (nocca, nmo - nocca)
        assert vb.shape == (noccb, nmo - noccb)
        return cp.concatenate([va.flatten(order = "C"), vb.flatten(order = "C")])

def unpack_nocc_nvir_shape_vector(vector, nocc, nmo):
    if isinstance(nocc, int):
        assert nocc < nmo
        assert vector.size == nocc * (nmo - nocc)
        return vector.reshape(nocc, nmo - nocc)
    else:
        nocca, noccb = nocc
        assert isinstance(nocca, int) and isinstance(noccb, int)
        assert nocca < nmo and noccb < nmo
        assert vector.size == nocca * (nmo - nocca) + noccb * (nmo - noccb)
        va = vector[:nocca * (nmo - nocca)]
        vb = vector[nocca * (nmo - nocca):]
        return (va.reshape(nocca, nmo - nocca), vb.reshape(noccb, nmo - noccb))

def apply_exp_U(mo_coeff, nocc, kappa):
    """
        Computes C @ U where
        U = exp( 0 & kappa \\ -kappa^T & 0 )
    """
    new_mo_coeff = cp.zeros_like(mo_coeff)
    nmo = mo_coeff.shape[-1]
    if isinstance(nocc, int):
        assert nocc < nmo
        assert mo_coeff.ndim == 2
        assert kappa.shape == (nocc, nmo - nocc)

        mocc = mo_coeff[:, :nocc]
        mvir = mo_coeff[:, nocc:]
        U, sigma, VT = cp.linalg.svd(kappa, full_matrices=False)
        V = VT.T
        del VT

        cos_sigma = cp.cos(sigma) - 1
        sin_sigma = cp.sin(sigma)
        moccU = mocc @ U
        mvirV = mvir @ V

        new_mo_coeff[:, :nocc] = moccU @ (cos_sigma * U).T + mocc - mvirV @ (sin_sigma * U).T
        new_mo_coeff[:, nocc:] = moccU @ (sin_sigma * V).T + mvirV @ (cos_sigma * V).T + mvir
    else:
        nocca, noccb = nocc
        assert isinstance(nocca, int) and isinstance(noccb, int)
        assert nocca < nmo and noccb < nmo
        assert mo_coeff.ndim == 3

        kappaa = kappa[0]
        assert kappaa.shape == (nocca, nmo - nocca)
        mocc = mo_coeff[0, :, :nocca]
        mvir = mo_coeff[0, :, nocca:]
        U, sigma, VT = cp.linalg.svd(kappaa, full_matrices=False)
        V = VT.T
        del VT
        cos_sigma = cp.cos(sigma) - 1
        sin_sigma = cp.sin(sigma)
        moccU = mocc @ U
        mvirV = mvir @ V
        new_mo_coeff[0, :, :nocca] = moccU @ (cos_sigma * U).T + mocc - mvirV @ (sin_sigma * U).T
        new_mo_coeff[0, :, nocca:] = moccU @ (sin_sigma * V).T + mvirV @ (cos_sigma * V).T + mvir

        kappab = kappa[1]
        assert kappab.shape == (noccb, nmo - noccb)
        mocc = mo_coeff[1, :, :noccb]
        mvir = mo_coeff[1, :, noccb:]
        U, sigma, VT = cp.linalg.svd(kappab, full_matrices=False)
        V = VT.T
        del VT
        cos_sigma = cp.cos(sigma) - 1
        sin_sigma = cp.sin(sigma)
        moccU = mocc @ U
        mvirV = mvir @ V
        new_mo_coeff[1, :, :noccb] = moccU @ (cos_sigma * U).T + mocc - mvirV @ (sin_sigma * U).T
        new_mo_coeff[1, :, noccb:] = moccU @ (sin_sigma * V).T + mvirV @ (cos_sigma * V).T + mvir

    return new_mo_coeff

def packed_apply_exp_U(mo_coeff, nocc, kappa):
    nmo = mo_coeff.shape[-1]
    return apply_exp_U(mo_coeff, nocc, unpack_nocc_nvir_shape_vector(kappa, nocc, nmo))

def pseudo_canonicalize(fock, mo_coeff, nocc, kappas):
    '''
        Computes Coo^T X Cvv
        where Coo and Cvv diagonalize the oo and vv block of the Fock matrix

        This is equivalent to Eq 28 e^(H^T) V e^H, where V = ( 0 & kappa \\ -kappa^T & 0 ), e^H = ( Coo & 0 \\ 0 & Cvv )
    '''
    new_mo_coeff = cp.zeros_like(mo_coeff)
    nmo = mo_coeff.shape[-1]
    if isinstance(nocc, int):
        assert nocc < nmo
        assert mo_coeff.ndim == 2
        assert fock.ndim == 2

        mocc = mo_coeff[:, :nocc]
        mvir = mo_coeff[:, nocc:]

        Foo = mocc.T @ fock @ mocc
        eoo, Coo = eigh(Foo)
        Fvv = mvir.T @ fock @ mvir
        evv, Cvv = eigh(Fvv)

        new_mo_energy = cp.zeros(nmo)
        new_mo_energy[:nocc] = eoo
        new_mo_energy[nocc:] = evv

        new_mo_coeff[:, :nocc] = mocc @ Coo
        new_mo_coeff[:, nocc:] = mvir @ Cvv

        transformed_kappas = []
        for kappa in kappas:
            assert kappa.shape == (nocc, nmo - nocc)
            kappa = Coo.T @ kappa @ Cvv
            transformed_kappas.append(kappa)

        return new_mo_energy, new_mo_coeff, transformed_kappas
    else:
        nocca, noccb = nocc
        assert isinstance(nocca, int) and isinstance(noccb, int)
        assert nocca < nmo and noccb < nmo
        assert mo_coeff.ndim == 3
        assert fock.ndim == 3

        mocca = mo_coeff[0, :, :nocca]
        mvira = mo_coeff[0, :, nocca:]
        Fooa = mocca.T @ fock[0] @ mocca
        eooa, Cooa = eigh(Fooa)
        Fvva = mvira.T @ fock[0] @ mvira
        evva, Cvva = eigh(Fvva)

        moccb = mo_coeff[1, :, :noccb]
        mvirb = mo_coeff[1, :, noccb:]
        Foob = moccb.T @ fock[1] @ moccb
        eoob, Coob = eigh(Foob)
        Fvvb = mvirb.T @ fock[1] @ mvirb
        evvb, Cvvb = eigh(Fvvb)

        new_mo_energy = cp.zeros((2, nmo))
        new_mo_energy[0, :nocca] = eooa
        new_mo_energy[0, nocca:] = evva
        new_mo_energy[1, :noccb] = eoob
        new_mo_energy[1, noccb:] = evvb

        new_mo_coeff[0, :, :nocca] = mocca @ Cooa
        new_mo_coeff[0, :, nocca:] = mvira @ Cvva
        new_mo_coeff[1, :, :noccb] = moccb @ Coob
        new_mo_coeff[1, :, noccb:] = mvirb @ Cvvb

        transformed_kappas = []
        for (kappaa, kappab) in kappas:
            assert kappaa.shape == (nocca, nmo - nocca)
            assert kappab.shape == (noccb, nmo - noccb)
            kappaa = Cooa.T @ kappaa @ Cvva
            kappab = Coob.T @ kappab @ Cvvb
            transformed_kappas.append((kappaa, kappab))
        return new_mo_energy, new_mo_coeff, transformed_kappas

def packed_pseudo_canonicalize(fock, mo_coeff, nocc, packed_kappas):
    nmo = mo_coeff.shape[-1]
    assert not any([kappa is None for kappa in packed_kappas])
    kappas = [unpack_nocc_nvir_shape_vector(kappa, nocc, nmo) for kappa in packed_kappas]
    new_mo_energy, new_mo_coeff, new_kappas = pseudo_canonicalize(fock, mo_coeff, nocc, kappas)
    packed_new_kappas = [pack_nocc_nvir_shape_vector(kappa, nocc, nmo) for kappa in new_kappas]
    return new_mo_energy, new_mo_coeff, packed_new_kappas

def approximate_optimal_diagonal_hessian(mo_energy, nocc, delta_E, minimal_diagonal_hessian):
    nmo = mo_energy.shape[-1]
    if isinstance(nocc, int):
        assert nocc < nmo
        assert mo_energy.shape == (nmo,)

        mo_energy_gap = mo_energy[None, nocc:] - mo_energy[:nocc, None] # (nocc, nvir)
        # Factor of two explained in kappa_gradient() function
        mo_energy_gap = cp.maximum(2.0 * (mo_energy_gap + delta_E), minimal_diagonal_hessian)
        return mo_energy_gap
    else:
        nocca, noccb = nocc
        assert isinstance(nocca, int) and isinstance(noccb, int)
        assert nocca < nmo and noccb < nmo
        assert mo_energy.shape == (2, nmo)

        mo_energy_gapa = mo_energy[0, nocca:][None, :] - mo_energy[0, :nocca][:, None]
        mo_energy_gapa = cp.maximum(2.0 * (mo_energy_gapa + delta_E), minimal_diagonal_hessian)
        mo_energy_gapb = mo_energy[1, noccb:][None, :] - mo_energy[1, :noccb][:, None]
        mo_energy_gapb = cp.maximum(2.0 * (mo_energy_gapb + delta_E), minimal_diagonal_hessian)
        return mo_energy_gapa, mo_energy_gapb

def packed_approximate_optimal_diagonal_hessian(mo_energy, nocc, delta_E, minimal_diagonal_hessian):
    nmo = mo_energy.shape[-1]
    h = approximate_optimal_diagonal_hessian(mo_energy, nocc, delta_E, minimal_diagonal_hessian)
    return pack_nocc_nvir_shape_vector(h, nocc, nmo)

def kappa_gradient(mo_coeff, nocc, fock):
    nmo = mo_coeff.shape[-1]
    if isinstance(nocc, int):
        assert nocc < nmo
        assert mo_coeff.ndim == 2
        assert fock.ndim == 2

        mocc = mo_coeff[:, :nocc]
        mvir = mo_coeff[:, nocc:]
        Fov = mocc.T @ fock @ mvir
        # The following factor of -2.0 comes from:
        # dE(alpha * kappa)/dalpha = - Tr(F_{ov} kappa^T) - Tr(F_{vo} kappa) = -2 Re( Tr(F_{vo} kappa) )
        g = -2.0 * Fov
        return g
    else:
        nocca, noccb = nocc
        assert isinstance(nocca, int) and isinstance(noccb, int)
        assert nocca < nmo and noccb < nmo
        assert mo_coeff.ndim == 3
        assert fock.ndim == 3

        mocc = mo_coeff[0, :, :nocca]
        mvir = mo_coeff[0, :, nocca:]
        Fov = mocc.T @ fock[0] @ mvir
        ga = -2.0 * Fov
        mocc = mo_coeff[1, :, :noccb]
        mvir = mo_coeff[1, :, noccb:]
        Fov = mocc.T @ fock[1] @ mvir
        gb = -2.0 * Fov
        return ga, gb

def packed_kappa_gradient(mo_coeff, nocc, fock):
    nmo = mo_coeff.shape[-1]
    g = kappa_gradient(mo_coeff, nocc, fock)
    return pack_nocc_nvir_shape_vector(g, nocc, nmo)

def get_nocc(mo_occ):
    nmo = mo_occ.shape[-1]
    if mo_occ.ndim == 1:
        assert mo_occ.shape == (nmo,)
        nocc = int(cp.sum(mo_occ == 2))
        nvir = int(cp.sum(mo_occ == 0))
        assert nocc + nvir == nmo, "Only integer occupations are supported for GDM yet."
        assert cp.all(mo_occ[:nocc] == 2), "Occupation number must be sorted."
        return nocc
    else:
        assert mo_occ.shape == (2, nmo)
        nocca = int(cp.sum(mo_occ[0] == 1))
        nvira = int(cp.sum(mo_occ[0] == 0))
        assert nocca + nvira == nmo, "Only integer occupations are supported for GDM yet."
        assert cp.all(mo_occ[0, :nocca] == 1), "Occupation number must be sorted."
        noccb = int(cp.sum(mo_occ[1] == 1))
        nvirb = int(cp.sum(mo_occ[1] == 0))
        assert noccb + nvirb == nmo, "Only integer occupations are supported for GDM yet."
        assert cp.all(mo_occ[1, :noccb] == 1), "Occupation number must be sorted."
        return (nocca, noccb)

def get_energy_and_fock(mf, mo_coeff, mo_occ, h1e, s1e):
    assert isinstance(mo_coeff, cp.ndarray)
    assert isinstance(mo_occ, cp.ndarray)
    assert isinstance(h1e, cp.ndarray)
    assert isinstance(s1e, cp.ndarray)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mf.mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    fock = mf.get_fock(h1e, s1e, vhf, dm)
    return e_tot, fock

def lbfgs_get_direction(gradient, lbfgs_s_y, initial_inverse_hessian = None):
    '''
        Notations follow https://en.wikipedia.org/wiki/Limited-memory_BFGS

        lbfgs_s_y order is older first (smaller index)
    '''
    q = gradient.copy()
    assert q.ndim == 1
    alphas = []
    rhos = []
    for s, y in reversed(lbfgs_s_y):
        assert s.shape == q.shape and y.shape == q.shape
        y_dot_s = float(cp.vdot(y, s).real)
        nonzero_y_dot_s = y_dot_s > GDM_LBFGS_Y_DOT_S_ZERO_THRESHOLD * float(cp.linalg.norm(y)) * float(cp.linalg.norm(s))
        rho = 1 / y_dot_s if nonzero_y_dot_s else 0.0

        alpha = rho * float(cp.vdot(s, q).real)
        q -= alpha * y
        rhos.append(rho)
        alphas.append(alpha)
    rhos = reversed(rhos)
    alphas = reversed(alphas)
    # From now on, rhos and alphas orders are the same as lbfgs_s_y order

    if initial_inverse_hessian is None:
        z = q # initial_inverse_hessian = cp.eye(q.size)
    else:
        z = initial_inverse_hessian * q
    assert z.shape == q.shape

    for (s, y), alpha, rho in zip(lbfgs_s_y, alphas, rhos):
        beta = rho * float(cp.vdot(y, z).real)
        z += s * (alpha - beta)

    return -z

def _kernel(mf, dm0,
            minimal_diagonal_hessian,
            min_n_lbfgs_record,
            max_n_lbfgs_record,
            max_step_norm,
            line_search_c,
            line_search_tau,
            max_n_line_search_attempt,
            line_search_initial_scale):
    """
        Geometric Direct Minimization
        Implementation follows:
        Van Voorhis, T.; Head-Gordon, M. A geometric approach to direct minimization.
        Molecular Physics 2002, 100, 1713-1721. https://doi.org/10.1080/00268970110103642

        Except for the two-loop-recursion LBFGS and backtracking line search,
        which follow wikipedia pages:
        https://en.wikipedia.org/wiki/Limited-memory_BFGS
        https://en.wikipedia.org/wiki/Backtracking_line_search
    """
    if dm0 is None:
        dm0 = mf.get_init_guess(mf.mol, mf.init_guess)
    dm0 = cp.asarray(dm0)

    h1e = cp.asarray(mf.get_hcore())
    s1e = cp.asarray(mf.get_ovlp())
    vhf0 = mf.get_veff(mf.mol, dm0)
    fock0 = mf.get_fock(h1e, s1e, vhf0, dm0)
    mo_energy, mo_coeff = mf.eig(fock0, s1e) # The initial density is converted to orbitals by one Roothaan diagonalization.
    del vhf0, fock0
    mo_occ = cp.asarray(mf.get_occ(mo_energy, mo_coeff))
    nocc = get_nocc(mo_occ)
    e_tot, fock = get_energy_and_fock(mf, mo_coeff, mo_occ, h1e, s1e)

    verbose = mf.verbose
    log = logger.new_logger(mf, verbose)
    if mf.conv_tol_grad is None:
        mf.conv_tol_grad = mf.conv_tol**0.5
        log.info('Set gradient conv threshold to %g', mf.conv_tol_grad)

    if mo_coeff.ndim == 2:
        log.info(" Cycle               Energy   Gradient Norm    Step")
    else:
        log.info(" Cycle               Energy   Gradient Norm      <S^2>    Step")

    lbfgs_s_y = []
    previous_gradient = previous_step = None
    previous_e_tot = None
    step_type = "Uninitialized error! Please report a bug if you see this line."

    converged = False
    assert mf.max_cycle > 1

    for cycle in range(mf.max_cycle):
        flatten_to_transform = [s_or_y for sy in lbfgs_s_y for s_or_y in sy]
        if previous_gradient is not None and previous_step is not None:
            flatten_to_transform.extend([previous_gradient, previous_step])
        mo_energy, mo_coeff, flatten_transformed = packed_pseudo_canonicalize(fock, mo_coeff, nocc, flatten_to_transform)
        del flatten_to_transform

        if previous_gradient is not None and previous_step is not None:
            previous_step = flatten_transformed.pop()
            previous_gradient = flatten_transformed.pop()

        transformed_s_y = []
        for i in range(0, len(flatten_transformed), 2):
            s, y = flatten_transformed[i], flatten_transformed[i + 1]
            s_dot_y = float(cp.vdot(s, y).real)
            if not s_dot_y > GDM_LBFGS_Y_DOT_S_ZERO_THRESHOLD * float(cp.linalg.norm(s)) * float(cp.linalg.norm(y)):
                continue
            transformed_s_y.append((s, y))
        del flatten_transformed
        lbfgs_s_y = transformed_s_y
        del transformed_s_y

        gradient = packed_kappa_gradient(mo_coeff, nocc, fock)
        gradient_norm = float(cp.linalg.norm(gradient))

        if mo_coeff.ndim == 2:
            spin_text = ""
        else:
            assert len(nocc) == 2
            ss, _ = mf.spin_square((mo_coeff[0, :, :nocc[0]], mo_coeff[1, :, :nocc[1]]))
            spin_text = f" {ss:10.4f}"

        if previous_e_tot is None:
            assert cycle == 0
            log.info(f"{cycle + 1:6d} {e_tot:20.12f} {gradient_norm:15.3e}{spin_text}  Roothaan")
        else:
            delta_E = e_tot - previous_e_tot
            if gradient_norm < mf.conv_tol_grad and abs(delta_E) < mf.conv_tol:
                converged = True
                log.info(f"{cycle + 1:6d} {e_tot:20.12f} {gradient_norm:15.3e}{spin_text}  Converged")
                break
            else:
                log.info(f"{cycle + 1:6d} {e_tot:20.12f} {gradient_norm:15.3e}{spin_text}  {step_type}")

        if previous_gradient is not None:
            y = gradient - previous_gradient
            s = previous_step
            s_dot_y = float(cp.vdot(s, y).real)
            if s_dot_y > GDM_LBFGS_Y_DOT_S_ZERO_THRESHOLD * float(cp.linalg.norm(s)) * float(cp.linalg.norm(y)):
                lbfgs_s_y.append((s, y))
                lbfgs_s_y = lbfgs_s_y[-max_n_lbfgs_record:]

        delta_E = 0.0 if previous_e_tot is None else e_tot - previous_e_tot
        approx_hess_diag = packed_approximate_optimal_diagonal_hessian(mo_energy, nocc, delta_E, minimal_diagonal_hessian)

        if len(lbfgs_s_y) < min_n_lbfgs_record:
            step = -gradient / approx_hess_diag
            step_type = "Descent"
        else:
            sqrt_approx_hess_diag = cp.sqrt(approx_hess_diag)
            weighted_lbfgs_s_y = [ (s * sqrt_approx_hess_diag, y / sqrt_approx_hess_diag) for s, y in lbfgs_s_y ]
            weighted_gradient = gradient / sqrt_approx_hess_diag
            weighted_step = lbfgs_get_direction(weighted_gradient, weighted_lbfgs_s_y)
            step = weighted_step / sqrt_approx_hess_diag
            step_type = "L-BFGS"
            if float(cp.vdot(gradient, step).real) >= 0.0:
                lbfgs_s_y.clear()
                step = -gradient / approx_hess_diag
                step_type = "Descent Reset"

        step_norm = float(cp.linalg.norm(step))
        if step_norm > max_step_norm:
            step *= max_step_norm / step_norm
            step_type = "Clipped " + step_type

        # Backtracking line search
        # Notation follows https://en.wikipedia.org/wiki/Backtracking_line_search
        assert 0.0 < line_search_tau and line_search_tau < 1.0
        assert 0.0 < line_search_c and line_search_c < 1.0

        line_search_succeed = False
        line_search_slope = float(cp.vdot(gradient, step).real) # It's negative!
        line_search_scale = line_search_initial_scale
        for i_line_search_attempt in range(max_n_line_search_attempt + 1):
            line_search_step = line_search_scale * step
            line_search_mo_coeff = packed_apply_exp_U(mo_coeff, nocc, line_search_step)
            line_search_energy, line_search_fock = get_energy_and_fock(mf, line_search_mo_coeff, mo_occ, h1e, s1e)
            if e_tot - line_search_energy >= -line_search_c * line_search_scale * line_search_slope:
                line_search_succeed = True
                break
            line_search_scale *= line_search_tau

        if i_line_search_attempt > 0:
            step_type = f"Line search ({i_line_search_attempt})"

        if not line_search_succeed:
            lbfgs_s_y.clear()
            log.info("Line search failed! However we accept the last (smallest) line search step and keep going. LBFGS record cleared.")

        previous_e_tot = e_tot
        e_tot = line_search_energy
        mo_coeff = line_search_mo_coeff
        fock = line_search_fock
        previous_step = line_search_step
        previous_gradient = gradient

    mo_energy, mo_coeff = mf.canonicalize(mo_coeff, mo_occ, fock)
    e_tot, fock = get_energy_and_fock(mf, mo_coeff, mo_occ, h1e, s1e)

    mf.converged = converged
    mf.cycles = cycle + 1
    mf.e_tot = e_tot
    mf.mo_energy = mo_energy
    mf.mo_coeff = mo_coeff
    mf.mo_occ = mo_occ

    return e_tot

class GDM:
    _keys = {}

    def __init__(self, mf):
        self.mf = mf

        # Approximated hessian parameters
        self.minimal_diagonal_hessian = 1e-4

        # L-BFGS parameters
        self.min_n_lbfgs_record = 1
        self.max_n_lbfgs_record = 20
        self.max_step_norm = 0.5

        # Line search parameters
        self.line_search_c = 1e-4
        self.line_search_tau = 0.5
        self.max_n_line_search_attempt = 8
        self.line_search_initial_scale = 1.0

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        raise NotImplementedError("GDM not implemented in CPU mode yet")

    def build(self, mol=None):
        self.mf.build(mol)
        return self

    def reset(self, mol=None):
        self.mf.reset(mol)
        return self

    def undo_gdm(self):
        return self.mf

    def error_undo_first(self):
        raise ValueError("Please call undo_gdm() first to get the mean-field object back!")
    nuc_grad_method = error_undo_first
    Gradients = error_undo_first
    Hessian = error_undo_first

    def kernel(self, dm0=None):
        return _kernel(self.mf, dm0,
                       self.minimal_diagonal_hessian,
                       self.min_n_lbfgs_record,
                       self.max_n_lbfgs_record,
                       self.max_step_norm,
                       self.line_search_c,
                       self.line_search_tau,
                       self.max_n_line_search_attempt,
                       self.line_search_initial_scale)

def gdm(mf):
    if isinstance(mf, GDM):
        return mf

    from gpu4pyscf.scf.hf import SCF
    assert isinstance(mf, SCF)

    if mf.istype('ROHF'):
        raise NotImplementedError("GDM does not support ROHF yet")
    elif mf.istype('UHF'):
        pass
    elif mf.istype('GHF'):
        raise NotImplementedError("GDM does not support GHF yet")
    elif mf.istype('RDHF'):
        raise NotImplementedError("GDM does not support RDHF yet")
    elif mf.istype('DHF'):
        raise NotImplementedError("GDM does not support DHF yet")
    else: # RHF
        pass

    return GDM(mf)
