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
Second order SCF solver for SCF with fractional occupation number
'''

import math
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import expm
from pyscf import lib
from pyscf.scf import chkfile
from pyscf.scf.addons import _fermi_smearing_occ
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import hf
from gpu4pyscf.scf.hf import SCF
from gpu4pyscf.scf.smearing import _SmearingSCF, _smearing_optimize
from gpu4pyscf.scf.soscf import _CIAH_SOSCF, _davidson_cc

def helmert_matrix(n):
    # Return a n*n orthonormal matrix, where the last column is 1/sqrt(n) * vector(1)
    assert n > 0
    H = np.zeros((n, n), dtype = np.float64)
    H[:, -1] = 1 / np.sqrt(n)
    for i in range(1, n):
        H[:i, i - 1] = 1 / np.sqrt(i * (i + 1))
        H[i, i - 1] = -i / np.sqrt(i * (i + 1))

    H = cp.asarray(H)
    return H

def get_occupation_sum_constraint_projector(n):
    return helmert_matrix(n)[:, :-1]

def get_active_occupation_index(f, zero_occupation_threshold, another_f = None):
    assert isinstance(f, cp.ndarray)
    assert cp.all(f >= 0.0)
    assert cp.all(f <= 1.0)
    mask = (f > zero_occupation_threshold) & (f < 1.0 - zero_occupation_threshold)

    if another_f is not None:
        assert isinstance(another_f, cp.ndarray)
        assert f.shape == another_f.shape
        assert cp.all(another_f >= 0.0)
        assert cp.all(another_f <= 1.0)
        mask |= (another_f > zero_occupation_threshold) & (another_f < 1.0 - zero_occupation_threshold)

    if f.ndim != 1:
        raise NotImplementedError

    return cp.where(mask)

def get_active_rotation_index(f, equal_occupation_threshold):
    assert isinstance(f, cp.ndarray)
    assert cp.all(f >= 0.0)
    assert cp.all(f <= 1.0)

    nmo = f.shape[-1]
    idx_is, idx_js = cp.tril_indices(nmo, -1) # Strictly lower-triangular part

    if f.ndim != 1:
        raise NotImplementedError

    mask = cp.abs(f[idx_is] - f[idx_js]) > equal_occupation_threshold
    idx_is, idx_js = idx_is[mask], idx_js[mask]

    return idx_is, idx_js

class ActiveOccupationRotationSpace:
    def __init__(self, nmo, rotation_idx_is, rotation_idx_js, occupation_idx):
        assert isinstance(nmo, int)
        self.nmo = nmo
        assert isinstance(rotation_idx_is, cp.ndarray) and rotation_idx_is.dtype == np.int64
        assert isinstance(rotation_idx_js, cp.ndarray) and rotation_idx_js.dtype == np.int64
        assert rotation_idx_is.ndim == 1 and rotation_idx_js.ndim == 1
        assert rotation_idx_is.shape == rotation_idx_js.shape
        self.rotation_idx_is = rotation_idx_is
        self.rotation_idx_js = rotation_idx_js
        assert isinstance(occupation_idx, cp.ndarray) and occupation_idx.dtype == np.int64
        assert occupation_idx.ndim == 1
        self.occupation_idx = occupation_idx

        n_active_f = int(occupation_idx.size)
        occupation_sum_constraint_projector = get_occupation_sum_constraint_projector(n_active_f)
        assert occupation_sum_constraint_projector.shape == (n_active_f, n_active_f - 1)
        self.occupation_sum_constraint_projector = occupation_sum_constraint_projector

    @classmethod
    def from_f(cls, f, equal_occupation_threshold, zero_occupation_threshold, another_f=None):
        nmo = f.shape[-1]
        rotation_idx_is, rotation_idx_js = get_active_rotation_index(f, equal_occupation_threshold)
        occupation_idx = get_active_occupation_index(f, zero_occupation_threshold, another_f)[0]
        return cls(nmo, rotation_idx_is, rotation_idx_js, occupation_idx)

    def __eq__(self, other):
        if not isinstance(other, ActiveOccupationRotationSpace):
            return NotImplemented
        return (
            self.nmo == other.nmo
            and bool(cp.array_equal(self.rotation_idx_is, other.rotation_idx_is))
            and bool(cp.array_equal(self.rotation_idx_js, other.rotation_idx_js))
            and bool(cp.array_equal(self.occupation_idx, other.occupation_idx))
        )

    @property
    def n_active_kappa(self):
        return int(self.rotation_idx_is.size)

    @property
    def n_active_f(self):
        if self.occupation_sum_constraint_projector is None:
            return int(self.occupation_idx.size)
        else:
            return int(self.occupation_idx.size) - 1

    def apply_occupation_sum_constraint(self, f):
        if self.occupation_sum_constraint_projector is None:
            return f
        else:
            return self.occupation_sum_constraint_projector.T @ f

    def undo_occupation_sum_constraint(self, f):
        if self.occupation_sum_constraint_projector is None:
            return f
        else:
            return self.occupation_sum_constraint_projector @ f

    def pack(self, kappa, f, if_apply_occupation_sum_constraint = True):
        assert isinstance(kappa, cp.ndarray) and kappa.ndim == 1
        assert isinstance(f, cp.ndarray) and f.ndim == 1
        assert kappa.dtype == np.float64
        assert f.dtype == np.float64
        assert kappa.size == self.n_active_kappa

        if if_apply_occupation_sum_constraint:
            assert f.size == self.occupation_idx.size
            f = self.apply_occupation_sum_constraint(f)
        else:
            assert f.size == self.n_active_f
        return cp.concatenate((kappa, f))

    def unpack(self, vector):
        assert isinstance(vector, cp.ndarray) and vector.ndim == 1
        assert vector.dtype == np.float64
        assert vector.size == self.n_active_kappa + self.n_active_f

        kappa = vector[:self.n_active_kappa]
        f = vector[self.n_active_kappa:]
        f = self.undo_occupation_sum_constraint(f)
        return kappa, f

    def unpack_to_full(self, vector):
        kappa, f = self.unpack(vector)

        rotation_generator = cp.zeros((self.nmo, self.nmo), dtype=kappa.dtype)
        rotation_generator[self.rotation_idx_is, self.rotation_idx_js] = kappa
        rotation_generator[self.rotation_idx_js, self.rotation_idx_is] = -kappa.conj()

        occupation = cp.zeros(self.nmo, dtype=kappa.dtype)
        occupation[self.occupation_idx] = f

        return rotation_generator, occupation

def occupation_two(mf):
    if mf.istype('RHF'):
        return 2.0
    else:
        return 1.0

def get_mo_energy_occupation_from_fock_mo(mf, fock_mo):
    # The purpose of computing mo_energy and obtain a mo_occ from mo_energy is to unlock frozen occupations.
    nelectron = mf.mol.nelectron
    nelectron = nelectron / 2
    assert isinstance(nelectron, float)

    if mf.smearing_method.lower() == "fermi":
        f_occ = _fermi_smearing_occ
    else:
        raise NotImplementedError
        # f_occ = _gaussian_smearing_occ

    # The diagonalization is necessary, because if two MOs are degenerate, the rotation within degenerate space is missing.
    # The diagonalization serves as canonicalization.
    mo_energy, U = cp.linalg.eigh(fock_mo)
    mo_energy = mo_energy.real

    mu, f = _smearing_optimize(f_occ, mo_energy, nelectron, mf.sigma)

    return mo_energy, U, f, mu

def reset_active_rotation_occupation_space_from_distribution(mf, fock_mo, ref_f):
    _, _, f_from_distribution, _ = get_mo_energy_occupation_from_fock_mo(mf, fock_mo)
    assert f_from_distribution.dtype == np.float64
    assert f_from_distribution.ndim == 1

    occupation_order = cp.argsort(-ref_f, kind = "stable")
    f_from_distribution_order_corrected = cp.empty_like(ref_f)
    f_from_distribution_order_corrected[occupation_order] = cp.asarray(f_from_distribution)
    f_from_distribution = f_from_distribution_order_corrected
    del f_from_distribution_order_corrected, occupation_order
    active_rotation_occupation_space = ActiveOccupationRotationSpace.from_f(ref_f, mf.equal_occupation_threshold, mf.zero_occupation_threshold, f_from_distribution)

    return active_rotation_occupation_space

def gen_g_hop_rhf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None, with_nlc=False, active_rotation_occupation_space=None):
    '''Orbital gradient and Hessian action; NLC is optional for SOSCF.'''
    assert mo_coeff.dtype == np.float64
    assert mo_occ.dtype == np.float64
    assert mo_occ.ndim == 1
    f = mo_occ * (1 / occupation_two(mf))

    if fock_ao is None:
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, None, None, dm0)
    fock = mo_coeff.conj().T.dot(fock_ao).dot(mo_coeff)
    fock_diag = fock.diagonal().real

    if active_rotation_occupation_space is None:
        active_rotation_occupation_space = reset_active_rotation_occupation_space_from_distribution(mf, fock, f)

    mf._active_rotation_occupation_space = active_rotation_occupation_space
    p = active_rotation_occupation_space.rotation_idx_is
    q = active_rotation_occupation_space.rotation_idx_js
    fq_minus_fp = f[q] - f[p]

    g_kappa = 2 * fq_minus_fp * fock[p, q]
    active_f = cp.clip(
        f[active_rotation_occupation_space.occupation_idx],
        0.0 + mf.zero_occupation_threshold,
        1.0 - mf.zero_occupation_threshold,
    )
    g_f = fock_diag[active_rotation_occupation_space.occupation_idx] + mf.sigma * (cp.log(active_f) - cp.log1p(-active_f))
    g = active_rotation_occupation_space.pack(g_kappa, g_f)

    vind = mf.gen_response(mo_coeff, mo_occ, singlet=None, hermi=1, with_nlc=with_nlc)

    def h_op(x):
        rotation_generator, eta = active_rotation_occupation_space.unpack_to_full(x)
        dm1 = 2 * (rotation_generator * f[None, :] - f[:, None] * rotation_generator + cp.diag(eta))
        dm1_ao = mo_coeff @ dm1 @ mo_coeff.conj().T
        v1_ao = vind(dm1_ao)
        v1 = mo_coeff.conj().T @ v1_ao @ mo_coeff

        # The Hessian needs to be symmetric.
        # Using the diagonal approximation of MO Fock matrix here is an easy way to get a symmetric approximated Hessian.
        fock_rotation_commutator = (fock_diag[p] - fock_diag[q]) * rotation_generator[p, q]
        x_kappa = 2 * (fq_minus_fp * (fock_rotation_commutator + v1[p, q]) + (eta[q] - eta[p]) * fock[p, q])

        # Cannot use the diagonal approximation of MO Fock matrix, because the diagonal of commutator is zero.
        fock_rotation_commutator = fock @ rotation_generator - rotation_generator @ fock
        x_f = cp.diag(fock_rotation_commutator + v1).real.copy()
        x_f = x_f[active_rotation_occupation_space.occupation_idx]
        x_f += mf.sigma * eta[active_rotation_occupation_space.occupation_idx] / (active_f * (1. - active_f))
        return active_rotation_occupation_space.pack(x_kappa, x_f)

    hdiag_kappa = 2 * fq_minus_fp * (fock_diag[p] - fock_diag[q])
    hdiag_f = mf.sigma / (active_f * (1.0 - active_f))
    if active_rotation_occupation_space.occupation_sum_constraint_projector is not None:
        hdiag_f = cp.einsum("ij,i->j", active_rotation_occupation_space.occupation_sum_constraint_projector**2, hdiag_f)
    h_diag = active_rotation_occupation_space.pack(hdiag_kappa, hdiag_f, if_apply_occupation_sum_constraint=False)

    return g, h_op, h_diag

# def gen_g_hop_uhf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None, with_nlc=False):
#     '''Orbital gradient and Hessian action; NLC is optional for SOSCF.'''
#     assert mo_coeff[0].dtype == np.float64
#     occidxa = cp.nonzero(mo_occ[0] >  0)[0]
#     occidxb = cp.nonzero(mo_occ[1] >  0)[0]
#     viridxa = cp.nonzero(mo_occ[0] == 0)[0]
#     viridxb = cp.nonzero(mo_occ[1] == 0)[0]
#     orboa = mo_coeff[0][:,occidxa]
#     orbob = mo_coeff[1][:,occidxb]
#     orbva = mo_coeff[0][:,viridxa]
#     orbvb = mo_coeff[1][:,viridxb]
#     nmo = mo_occ[0].size
#     nocca, noccb = mf.nelec
#     nvira = nmo - nocca
#     nvirb = nmo - noccb

#     if fock_ao is None:
#         dm0 = mf.make_rdm1(mo_coeff, mo_occ)
#         fock_ao = mf.get_fock(h1e, None, None, dm0)
#     focka = mo_coeff[0].conj().T.dot(fock_ao[0]).dot(mo_coeff[0])
#     fockb = mo_coeff[1].conj().T.dot(fock_ao[1]).dot(mo_coeff[1])
#     fooa = focka[occidxa[:,None],occidxa]
#     fvva = focka[viridxa[:,None],viridxa]
#     foob = fockb[occidxb[:,None],occidxb]
#     fvvb = fockb[viridxb[:,None],viridxb]

#     g = cp.hstack((focka[viridxa[:,None],occidxa].ravel(),
#                    fockb[viridxb[:,None],occidxb].ravel()))
#     h_diaga = fvva.diagonal().real[:,None] - fooa.diagonal().real
#     h_diagb = fvvb.diagonal().real[:,None] - foob.diagonal().real
#     h_diag = cp.hstack((h_diaga.reshape(-1), h_diagb.reshape(-1)))

#     vind = mf.gen_response(mo_coeff, mo_occ, hermi=1, with_nlc=with_nlc)
#     # Pad the two spin factors to retain their tags in a single UHF density.
#     nao = mo_coeff[0].shape[0]
#     nocc = max(nocca, noccb)
#     orbo = cp.zeros((2, nao, nocc), dtype=mo_coeff[0].dtype)
#     orbo[0, :, :nocca] = orboa.conj()
#     orbo[1, :, :noccb] = orbob.conj()

#     def h_op(x):
#         x1a = x[:nvira*nocca].reshape(nvira,nocca)
#         x1b = x[nvira*nocca:].reshape(nvirb,noccb)
#         x2a = contract('pr,rq->pq', fvva, x1a)
#         x2a-= contract('sq,ps->pq', fooa, x1a)
#         x2b = contract('pr,rq->pq', fvvb, x1b)
#         x2b-= contract('sq,ps->pq', foob, x1b)

#         dm_factor = cp.zeros_like(orbo)
#         dm_factor[0, :, :nocca] = orbva.dot(x1a)
#         dm_factor[1, :, :noccb] = orbvb.dot(x1b)
#         dm1 = _make_factorized_dm(dm_factor, orbo)
#         v1 = vind(dm1)
#         x2a += orbva.conj().T.dot(v1[0]).dot(orboa)
#         x2b += orbvb.conj().T.dot(v1[1]).dot(orbob)
#         return cp.hstack((x2a.ravel(), x2b.ravel()))

#     return g, h_op, h_diag

def scale_ah_step_preserve_occupation_range(mf, dxi, mo_occ, accumulated_dx, active_rotation_occupation_space):
    scale = 1.0
    dxmax = abs(dxi).max()
    if dxmax > mf.max_stepsize:
        scale = min(scale, float(mf.max_stepsize / dxmax))

    _, df = active_rotation_occupation_space.unpack_to_full(dxi)
    maximum = abs(df).max()
    if maximum > mf.max_occupation_step:
        scale = min(scale, float(mf.max_occupation_step / maximum))

    assert mo_occ.ndim == 1
    assert mo_occ.dtype == np.float64
    f = mo_occ * (1 / occupation_two(mf))
    assert isinstance(accumulated_dx, (float, cp.ndarray))
    if isinstance(accumulated_dx, cp.ndarray):
        _, accumulated_df = active_rotation_occupation_space.unpack_to_full(accumulated_dx)
        f = f + accumulated_df

    assert 0.0 < mf.occupation_fraction_to_boundary < 1.0
    positive_mask = df > 0
    negative_mask = df < 0
    if cp.any(positive_mask):
        boundary_distance_ratio = cp.min((1.0 - f[positive_mask]) / df[positive_mask])
        scale = min(scale, float(mf.occupation_fraction_to_boundary * boundary_distance_ratio))
    if cp.any(negative_mask):
        boundary_distance_ratio = cp.min(-f[negative_mask] / df[negative_mask])
        scale = min(scale, float(mf.occupation_fraction_to_boundary * boundary_distance_ratio))

    scale = max(0., scale)
    return scale

def _rotate_orb_cc(mf, h1e, s1e, conv_tol_grad=None, verbose=None):
    log = logger.new_logger(mf, verbose)

    if conv_tol_grad is None:
        conv_tol_grad = (mf.conv_tol*.1)**.5
        #TODO: dynamically adjust max_stepsize, as done in mc1step.py

    def precond(trial_vector, eigenvalues):
        hdiagd = h_diag - (eigenvalues - mf.ah_level_shift)
        hdiagd[cp.abs(hdiagd) < 1e-8] = 1e-8
        trial_vector = trial_vector / hdiagd
        return trial_vector

    t3m = log.init_timer()
    u = eta = g_kf = g_orb = norm_gorb = dxi = kfcount = jkcount = None
    dm0 = vhf0 = None
    active_rotation_occupation_space = None
    g_op = lambda: g_orb
    while True:
        mo_coeff, mo_occ, dm0, vhf0, e_tot = (yield u, eta, g_kf, kfcount, jkcount, dm0, vhf0)
        fock_ao = mf.get_fock(h1e, s1e, vhf0, dm0)

        previous_active_rotation_occupation_space = active_rotation_occupation_space
        g_kf, h_op, h_diag = mf.gen_g_hop(mo_coeff, mo_occ, fock_ao)
        active_rotation_occupation_space = mf._active_rotation_occupation_space
        if g_orb is not None and not active_rotation_occupation_space == previous_active_rotation_occupation_space:
            log.debug('    Restart AH because the FON active_rotation_occupation_space changed')
            g_orb = dxi = None
        norm_gkf = cp.linalg.norm(g_kf)
        if g_orb is None:
            log.debug('    |g|= %4.3g (keyframe)', norm_gkf)
            kf_trust_region = mf.kf_trust_region
            x0_guess = g_kf
        else:
            norm_dg = cp.linalg.norm(g_kf-g_orb)
            log.debug('    |g|= %4.3g (keyframe), |g-correction|= %4.3g',
                      norm_gkf, norm_dg)
            kf_trust_region = min(max(norm_gorb/(norm_dg+1e-9), mf.kf_trust_region), 10)
            log.debug1('Set  kf_trust_region = %g', kf_trust_region)
            x0_guess = dxi
        g_orb = g_kf
        norm_gorb = norm_gkf

        problem_size = g_orb.size
        if cp.iscomplexobj(g_orb):
            # With real Davidson coefficients, N complex rotations span up to 2N
            # real directions. Using N would stop at an incomplete subspace.
            problem_size *= 2

        ah_conv_tol = min(norm_gorb**2, mf.ah_conv_tol)
        # increase the AH accuracy when approach convergence
        ah_start_cycle = mf.ah_start_cycle
        imic = 0
        dr = 0.
        u = 1.
        eta = 0.
        ukf = None
        eta_kf = None
        jkcount = 0
        kfcount = 0
        ikf = 0
        ihop = 0

        for ah_end, ihop, w, dxi, hdxi, residual, seig \
                in _davidson_cc(h_op, g_op, precond, x0_guess,
                                tol=ah_conv_tol, max_cycle=mf.ah_max_cycle,
                                lindep=mf.ah_lindep, verbose=log):
            norm_residual = cp.linalg.norm(residual)
            ah_start_tol = min(norm_gorb*5, mf.ah_start_tol)
            if (ah_end or ihop == mf.ah_max_cycle or # make sure to use the last step
                ((norm_residual < ah_start_tol) and (ihop >= ah_start_cycle)) or
                (seig < mf.ah_lindep)):
                imic += 1
                dxmax = abs(dxi).max()
                if eta_kf is not None:
                    mo_occ_kf = mf.accumulate_occ(mo_occ, eta_kf)
                else:
                    mo_occ_kf = mo_occ
                scale = scale_ah_step_preserve_occupation_range(mf, dxi, mo_occ_kf, dr, active_rotation_occupation_space)
                if ihop == problem_size:
                    log.debug1('... Hx=g fully converged for small systems')
                if scale < 1:
                    log.debug1('... scale rotation/occupation size %g', scale)
                    dxi *= scale
                    hdxi *= scale

                dr = dr + dxi
                g_orb = g_orb + hdxi
                norm_dr = cp.linalg.norm(dr)
                norm_gorb = cp.linalg.norm(g_orb)
                norm_dxi = cp.linalg.norm(dxi)
                log.debug('    imic %d(%d)  |g|= %4.3g  |dxi|= %4.3g  '
                          'max(|x|)= %4.3g  |dr|= %4.3g  eig= %4.3g  seig= %4.3g',
                          imic, ihop, norm_gorb, norm_dxi,
                          dxmax, norm_dr, w, seig)

                max_cycle = max(mf.max_cycle_inner,
                                mf.max_cycle_inner-int(math.log(norm_gkf+1e-9)*2))
                log.debug1('Set ah_start_tol %g, ah_start_cycle %d, max_cycle %d',
                           ah_start_tol, ah_start_cycle, max_cycle)
                ikf += 1
                if imic > 3 and norm_gorb > norm_gkf*mf.ah_grad_trust_region:
                    g_orb = g_orb - hdxi
                    dr -= dxi
                    norm_gorb = cp.linalg.norm(g_orb)
                    log.debug('|g| >> keyframe, Restore previouse step')
                    break

                elif (imic >= max_cycle or norm_gorb < conv_tol_grad/mf.ah_grad_trust_region):
                    break

                elif (ikf > 2 and # avoid frequent keyframe
                      #TODO: replace it with keyframe_scheduler
                      (ikf >= max(mf.kf_interval, mf.kf_interval-math.log(norm_dr+1e-9)) or
                       # Insert keyframe if the keyframe and the estimated g_orb are too different
                       norm_gorb < norm_gkf/kf_trust_region)):
                    ikf = 0
                    du, deta = mf._active_rotation_occupation_space.unpack_to_full(dr)
                    u = mf.update_rotate_matrix(du, mo_occ, mo_coeff=mo_coeff)
                    eta = deta
                    if ukf is not None:
                        u = mf.rotate_mo(ukf, u)
                    ukf = u
                    if eta_kf is not None:
                        eta += eta_kf
                    eta_kf = eta
                    dr[:] = 0
                    mo1 = mf.rotate_mo(mo_coeff, u)
                    mo_occ1 = mf.accumulate_occ(mo_occ, eta_kf)
                    dm = mf.make_rdm1(mo1, mo_occ1)
                    # use mf._scf.get_veff to avoid density-fit mf polluting get_veff
                    vhf0 = mf._scf.get_veff(mf._scf.mol, dm, dm_last=dm0, vhf_last=vhf0)
                    dm0 = dm
                    # Use API to compute fock instead of "fock=h1e+vhf0". This is because get_fock
                    # is the hook being overloaded in many places.
                    fock_ao = mf.get_fock(h1e, s1e, vhf0, dm0)
                    g_kf1 = mf.gen_g_hop(mo1, mo_occ1, fock_ao, active_rotation_occupation_space=active_rotation_occupation_space)[0]
                    norm_gkf1 = cp.linalg.norm(g_kf1)
                    norm_dg = cp.linalg.norm(g_kf1-g_orb)
                    jkcount += 1
                    kfcount += 1
                    if log.verbose >= logger.DEBUG:
                        e_tot, e_last = mf._scf.energy_tot(dm, h1e, vhf0), e_tot
                        log.debug('Adjust keyframe g_orb to |g|= %4.3g  '
                                  '|g-correction|=%4.3g  E=%.12g dE=%.5g',
                                  norm_gkf1, norm_dg, e_tot, e_tot-e_last)

                    if (norm_dg < norm_gorb*mf.ah_grad_trust_region  # kf not too diff
                        #or norm_gkf1 < norm_gkf  # grad is decaying
                        # close to solution
                        or norm_gkf1 < conv_tol_grad*mf.ah_grad_trust_region):
                        kf_trust_region = min(max(norm_gorb/(norm_dg+1e-9), mf.kf_trust_region), 10)
                        log.debug1('Set kf_trust_region = %g', kf_trust_region)
                        g_orb = g_kf = g_kf1
                        norm_gorb = norm_gkf = norm_gkf1
                    else:
                        g_orb = g_orb - hdxi
                        dr -= dxi
                        norm_gorb = cp.linalg.norm(g_orb)
                        log.debug('Out of trust region. Restore previouse step')
                        break

        if ihop > 0:
            du, deta = mf._active_rotation_occupation_space.unpack_to_full(dr)
            u = mf.update_rotate_matrix(du, mo_occ, mo_coeff=mo_coeff)
            if ukf is not None:
                u = mf.rotate_mo(ukf, u)
            eta = deta
            if eta_kf is not None:
                eta += eta_kf
            dr[:] = 0
            mo1 = mf.rotate_mo(mo_coeff, u)
            mo_occ1 = mf.accumulate_occ(mo_occ, eta)
            dm = mf.make_rdm1(mo1, mo_occ1)
            jkcount += ihop + 1
            log.debug('    tot inner=%d  %d JK  |g|= %4.3g  |u-1|= %4.3g',
                      imic, jkcount, norm_gorb, cp.linalg.norm(dr))
        h_op = h_diag = None
        t3m = log.timer('aug_hess in %d inner iters' % imic, *t3m)

def get_free_energy(mf, e_tot, mo_occ):
    if mf.smearing_method.lower() == 'fermi':
        f = mo_occ * (1 / occupation_two(mf))
        f = f[(f>0) & (f<1)]
        entropy = -cp.sum(f * cp.log(f) + (1.0 - f) * cp.log1p(-f))
        entropy *= occupation_two(mf)
    else:
        raise NotImplementedError
        #     entropy = (numpy.exp(-((mo_energy-mu)/self.sigma)**2).sum()
        #                / (2*numpy.sqrt(numpy.pi)))
    mf.entropy = float(entropy)
    mf.e_free = e_tot - mf.sigma * mf.entropy
    mf.e_zero = e_tot - mf.sigma * mf.entropy * 0.5
    return mf.e_free

def kernel(mf, mo_coeff=None, mo_occ=None, dm=None,
           conv_tol=1e-10, conv_tol_grad=None, max_cycle=50, dump_chk=True,
           callback=None, verbose=logger.NOTE):
    log = logger.new_logger(mf, verbose)
    cput0 = log.init_timer()
    mol = mf._scf.mol
    assert mol is mf.mol

    if conv_tol_grad is None:
        conv_tol_grad = conv_tol**.5
        log.info('Set conv_tol_grad to %g', conv_tol_grad)

    # call mf._scf.get_hcore, mf._scf.get_ovlp because they might be overloaded
    h1e = mf._scf.get_hcore(mol)
    s1e = mf._scf.get_ovlp(mol)
    x_orth = mf._scf.check_linear_dependency(s1e, log)
    s1e = None

    if mo_coeff is not None and mo_occ is not None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        # call mf._scf.get_veff, to avoid "newton().density_fit()" polluting get_veff
        vhf = mf._scf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        mo_energy, mo_tmp = mf.eig(fock, s1e, x=x_orth)
        mf.get_occ(mo_energy, mo_tmp)
        mo_tmp = None
    else:
        if dm is None:
            dm = mf.get_init_guess(mol, mf.init_guess)
        vhf = mf._scf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        mo_energy, mo_coeff = mf.eig(fock, s1e, x=x_orth)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf._scf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)

    # Save mo_coeff and mo_occ because they are needed by function rotate_mo
    mf.mo_coeff, mf.mo_occ = mo_coeff, mo_occ

    e_tot = mf._scf.energy_tot(dm, h1e, vhf)
    e_free = get_free_energy(mf, e_tot, mo_occ)
    fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
    gradient = mf.gen_g_hop(mo_coeff, mo_occ, fock)[0]
    g_kappa, g_f = mf._active_rotation_occupation_space.unpack(gradient)
    norm_gorb = cp.linalg.norm(g_kappa)
    norm_gocc = cp.linalg.norm(g_f)
    log.info('Initial guess E= %.15g  Free energy= %.15g  |g_orb|= %g  |g_occ|= %g',
             e_tot, e_free, norm_gorb, norm_gocc)

    if dump_chk and mf.chkfile:
        chkfile.save_mol(mol, mf.chkfile)
    # Copy the integral file to soscf object to avoid the integrals being
    # cached twice.
    if mol is mf.mol and not getattr(mf, 'with_df', None):
        mf._eri = mf._scf._eri

    rotaiter = _rotate_orb_cc(mf, h1e, s1e, conv_tol_grad, verbose=log)
    next(rotaiter)  # start the iterator
    kftot = jktot = 0
    norm_gorb = 0.
    scf_conv = False
    cput1 = log.timer('initializing second order scf', *cput0)

    for imacro in range(max_cycle):
        u, eta, g_orb, kfcount, jkcount, dm_last, vhf = \
                rotaiter.send((mo_coeff, mo_occ, dm, vhf, e_tot))
        kftot += kfcount + 1
        jktot += jkcount + 1

        last_e_free = e_free
        norm_gorb = cp.linalg.norm(g_orb)
        mo_coeff = mf.rotate_mo(mo_coeff, u, log)
        mo_occ = mf.accumulate_occ(mo_occ, eta)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf._scf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        gradient = mf.gen_g_hop(mo_coeff, mo_occ, fock)[0]
        g_kappa, g_f = mf._active_rotation_occupation_space.unpack(gradient)
        norm_gorb = cp.linalg.norm(gradient)
        norm_gorb = cp.linalg.norm(g_kappa)
        norm_gocc = cp.linalg.norm(g_f)
        # NOTE: DO NOT change the initial guess mo_occ, mo_coeff
        if mf.verbose >= logger.DEBUG:
            mo_energy, mo_tmp = mf.eig(fock, s1e, x=x_orth)
            mf.get_occ(mo_energy, mo_tmp)
            # call mf._scf.energy_tot for dft, because the (dft).get_veff step saved _exc in mf._scf
        e_tot = mf._scf.energy_tot(dm, h1e, vhf)
        e_free = get_free_energy(mf, e_tot, mo_occ)

        log.info('macro= %d  E= %.15g  Free energy= %.15g  delta_free_energy= %g  |g_orb|= %g  |g_occ|= %g  %d KF %d JK',
                 imacro, e_tot, e_free, e_free-last_e_free, norm_gorb, norm_gocc, kfcount+1, jkcount)
        cput1 = log.timer('cycle= %d'%(imacro+1), *cput1)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_free-last_e_free) < conv_tol and norm_gorb < conv_tol_grad:
            # Intentionally not checking occupation number convergence
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        if scf_conv:
            break

    if callable(callback):
        callback(locals())

    rotaiter.close()
    if not mf.canonicalization:
        log.info("For SOSCF with smearing, canonicalization is turned on regardless of mf.canonicalization option.")
    fock_mo = mo_coeff.conj().T @ fock @ mo_coeff
    mo_energy, U, f, _ = get_mo_energy_occupation_from_fock_mo(mf, fock_mo)
    mo_coeff = mo_coeff @ U
    mo_occ = f * occupation_two(mf)
    log.info('macro X = %d  E=%.15g  Free energy=%.15g  |g_orb|= %g  |g_occ|= %g  total %d KF %d JK',
             imacro+1, e_tot, e_free, norm_gorb, norm_gocc, kftot+1, jktot+1)

    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

# A tag to label the derived SCF class
class _CIAH_SOSCF_FON(_CIAH_SOSCF):
    equal_occupation_threshold = 1e-8
    zero_occupation_threshold = 1e-8
    max_occupation_step = 0.1
    occupation_fraction_to_boundary = 0.95

    _keys = {
        'equal_occupation_threshold',
        'zero_occupation_threshold',
        'max_occupation_step',
        'occupation_fraction_to_boundary',
        *_CIAH_SOSCF._keys,
    }

    def kernel(self, mo_coeff=None, mo_occ=None, dm0=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        cput0 = logger.init_timer(self)
        self.build(self.mol)
        self.dump_flags()

        self.converged, self.e_tot, \
                self.mo_energy, self.mo_coeff, self.mo_occ = \
                kernel(self, mo_coeff, mo_occ, dm0, conv_tol=self.conv_tol,
                       conv_tol_grad=self.conv_tol_grad,
                       max_cycle=self.max_cycle,
                       callback=self.callback, verbose=self.verbose)

        logger.timer(self, 'Second order SCF with smearing', *cput0)
        self._finalize()
        return self.e_tot

    gen_g_hop = gen_g_hop_rhf

    def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
        nmo = mo_occ.shape[-1]
        assert dx.shape == (nmo, nmo)
        u = expm(dx)
        if isinstance(u0, cp.ndarray):
            u = u0.dot(u)
        return u

    def rotate_mo(self, mo_coeff, u, log=None):
        return mo_coeff.dot(u)

    def accumulate_occ(self, mo_occ, eta):
        f = mo_occ * 0.5 + eta
        if cp.any(f < 0) or cp.any(f > 1):
            raise RuntimeError('Occupation update left the interval [0, 1]')
        return 2 * f

    def to_cpu(self):
        raise NotImplementedError

    def approx_hessian(self, auxbasis=None, with_df=None, only_dfj=False):
        '''Approximate the orbital Hessian using density fitting integrals.'''
        import gpu4pyscf.df.df_jk
        logger.debug(self, 'Approximate the orbital hessian using DF integrals')
        return gpu4pyscf.df.df_jk.density_fit(self, auxbasis, with_df, only_dfj)


# class _SecondOrderUHF(_CIAH_SOSCF):
#     gen_g_hop = gen_g_hop_uhf

#     def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
#         occidxa = mo_occ[0] > 0
#         occidxb = mo_occ[1] > 0
#         viridxa = ~occidxa
#         viridxb = ~occidxb

#         nmo = len(occidxa)
#         dr = cp.zeros((2,nmo,nmo), dtype=dx.dtype)
#         uniq = cp.array((viridxa[:,None] & occidxa,
#                          viridxb[:,None] & occidxb))
#         dr[uniq] = dx
#         dr = dr - dr.conj().transpose(0,2,1)

#         if isinstance(u0, int) and u0 == 1:
#             return cp.asarray((expm(dr[0]), expm(dr[1])))
#         else:
#             return cp.asarray((u0[0].dot(expm(dr[0])),
#                                u0[1].dot(expm(dr[1]))))

#     def rotate_mo(self, mo_coeff, u, log=None):
#         mo = cp.asarray((mo_coeff[0].dot(u[0]),
#                          mo_coeff[1].dot(u[1])))
#         return mo

#     def kernel(self, mo_coeff=None, mo_occ=None, dm0=None):
#         if isinstance(mo_coeff, cp.ndarray) and mo_coeff.ndim == 2:
#             mo_coeff = cp.stack((mo_coeff, mo_coeff))
#         if isinstance(mo_occ, cp.ndarray) and mo_occ.ndim == 1:
#             mo_occ = cp.asarray((mo_occ > 0, mo_occ == 2), dtype=np.float64)
#         return _CIAH_SOSCF.kernel(self, mo_coeff, mo_occ, dm0)

class _SecondOrderFONRHF(_CIAH_SOSCF_FON):
    gen_g_hop = gen_g_hop_rhf

def newton(mf):
    if isinstance(mf, _CIAH_SOSCF_FON):
        return mf

    assert isinstance(mf, SCF)
    assert isinstance(mf, _SmearingSCF) and hasattr(mf, "sigma")

    if mf.istype('ROHF'):
        raise NotImplementedError
    elif mf.istype('UHF'):
        # cls = _SecondOrderUHF
        raise NotImplementedError
    elif mf.istype('GHF'):
        raise NotImplementedError
    elif mf.istype('RDHF'):
        raise NotImplementedError
    elif mf.istype('DHF'):
        raise NotImplementedError
    else:
        cls = _SecondOrderFONRHF
    return lib.set_class(cls(mf), (cls, mf.__class__))
