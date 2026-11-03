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

import numpy as np
import cupy as cp
from cupyx.scipy.linalg import block_diag, expm
from gpu4pyscf.lib import logger
from gpu4pyscf.scf.soscf import gen_g_hop_rhf, gen_g_hop_uhf
from gpu4pyscf.scf.hf import KohnShamDFT
from gpu4pyscf.fci.direct_spin1 import davidson1
from gpu4pyscf.scf.gdm import get_nocc, apply_exp_U
from gpu4pyscf.df.df_jk import _make_factorized_dm
from gpu4pyscf.solvent._attach_solvent import SCFWithSolvent
from gpu4pyscf import __config__

STAB_NROOTS = getattr(__config__, 'stab_nroots', 3)
STAB_TOL = getattr(__config__, 'stab_tol', 1e-6)
STAB_HESS_NUMERICAL_ZERO = 1e-8

def rhf_stability(mf, internal=True, external=False, verbose=None, return_status=False,
                  nroots=STAB_NROOTS, tol=STAB_TOL):
    '''
    Stability analysis for RHF/RKS method.

    Args:
        mf : RHF or RKS object

    Kwargs:
        internal : bool
            Internal stability, within the RHF space.
        external : bool
            External stability. Including the RHF -> UHF and real -> complex
            stability analysis.
        return_status: bool
            Whether to return `stable_i` and `stable_e`
        nroots : int
            Number of roots solved by Davidson solver
        tol : float
            Convergence threshold for Davidson solver

    Returns:
        If return_status is False (default), the return value includes
        two set of orbitals, which are more close to the stable condition.
        The first corresponds to the internal stability
        and the second corresponds to the external stability.

        Else, another two boolean variables (indicating current status:
        stable or unstable) are returned.
        The first corresponds to the internal stability
        and the second corresponds to the external stability.
    '''
    mo_i = mo_e = None
    stable_i = stable_e = None
    if internal:
        mo_i, stable_i = rhf_internal(mf, verbose=verbose, return_status=True, nroots=nroots, tol=tol)
    if external:
        mo_e, stable_e = rhf_external(mf, verbose=verbose, return_status=True, nroots=nroots, tol=tol)
    if return_status:
        return mo_i, mo_e, stable_i, stable_e
    else:
        return mo_i, mo_e

def uhf_stability(mf, internal=True, external=False, verbose=None, return_status=False,
                  nroots=STAB_NROOTS, tol=STAB_TOL):
    '''
    Stability analysis for UHF/UKS method.

    Args:
        mf : UHF or UKS object

    Kwargs:
        internal : bool
            Internal stability, within the UHF space.
        external : bool
            External stability. Including the UHF -> GHF and real -> complex
            stability analysis.
        return_status: bool
            Whether to return `stable_i` and `stable_e`
        nroots : int
            Number of roots solved by Davidson solver
        tol : float
            Convergence threshold for Davidson solver

    Returns:
        If return_status is False (default), the return value includes
        two set of orbitals, which are more close to the stable condition.
        The first corresponds to the internal stability
        and the second corresponds to the external stability.

        Else, another two boolean variables (indicating current status:
        stable or unstable) are returned.
        The first corresponds to the internal stability
        and the second corresponds to the external stability.
    '''
    mo_i = mo_e = None
    stable_i = stable_e = None
    if internal:
        mo_i, stable_i = uhf_internal(mf, verbose=verbose, return_status=True, nroots=nroots, tol=tol)
    if external:
        mo_e, stable_e = uhf_external(mf, verbose=verbose, return_status=True, nroots=nroots, tol=tol)
    if return_status:
        return mo_i, mo_e, stable_i, stable_e
    else:
        return mo_i, mo_e

def rohf_stability(mf, internal=True, external=False, verbose=None, return_status=False,
                   nroots=STAB_NROOTS, tol=STAB_TOL):
    raise NotImplementedError('ROHF stability analysis is not available in GPU4PySCF yet.')

def ghf_stability(mf, verbose=None, return_status=False,
                  nroots=STAB_NROOTS, tol=STAB_TOL):
    raise NotImplementedError('GHF stability analysis is not available in GPU4PySCF yet.')

def dhf_stability(mf, verbose=None, return_status=False,
                  nroots=STAB_NROOTS, tol=STAB_TOL):
    raise NotImplementedError('DHF stability analysis is not available in GPU4PySCF yet.')

def dump_status(log, stable, method_class, stab_type):
    if not stable:
        log.note(method_class + f' wavefunction has an {stab_type} instability')
    else:
        log.note(method_class + f' wavefunction is stable in the {stab_type} '
                 'stability analysis')

def get_davidson_initial_guess(hdiag, nroots, nguess):
    assert nroots > 0
    assert hdiag.ndim == 1
    nvar = hdiag.size
    assert nvar >= nroots
    assert nguess >= nroots

    nroots = min(nroots, nvar)
    nguess = min(nguess, nvar)
    order = cp.argsort(hdiag)
    x0 = cp.zeros((nguess, nvar), dtype=hdiag.dtype)
    x0[0, cp.abs(hdiag) > STAB_HESS_NUMERICAL_ZERO] = 1. / hdiag[cp.abs(hdiag) > STAB_HESS_NUMERICAL_ZERO]
    for i_guess in range(nguess):
        x0[i_guess, order[i_guess]] = 1
    return x0

def _rotate_mo(mo_coeff, mo_occ, v):
    assert mo_coeff.ndim == 2 and mo_occ.ndim == 1
    nmo = mo_occ.shape[-1]
    nocc = get_nocc(mo_occ)
    kappa = -v.reshape((nmo - nocc, nocc)).T # The kappa from gdm is the upper right corner (nocc, nvir), the kappa from soscf is the lower left corner (nvir, nocc).
    new_mo = apply_exp_U(mo_coeff, nocc, kappa)
    return new_mo

def _rotate_mo_uhf(mo_coeff, mo_occ, v):
    assert mo_coeff.ndim == 3 and mo_occ.ndim == 2
    assert mo_coeff.shape[0] == 2 and mo_occ.shape[0] == 2
    nmo = mo_occ.shape[-1]
    nocca, noccb = get_nocc(mo_occ)
    va = v[:nocca * (nmo - nocca)]
    vb = v[nocca * (nmo - nocca):]
    kappaa = -va.reshape((nmo - nocca, nocca)).T
    kappab = -vb.reshape((nmo - noccb, noccb)).T
    new_mo = apply_exp_U(mo_coeff, (nocca, noccb), (kappaa, kappab))
    return new_mo

def rhf_internal(mf, with_symmetry=False, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    assert not with_symmetry
    log = logger.new_logger(mf, verbose)
    g, hop, hdiag = gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ, with_nlc=True)
    hdiag *= 2
    def precond(residuals, eigenvalues):
        hdiagd = hdiag[None, :] - eigenvalues[:, None]
        hdiagd[cp.abs(hdiagd) < STAB_HESS_NUMERICAL_ZERO] = STAB_HESS_NUMERICAL_ZERO
        return residuals / hdiagd
    # The results of hop(x) corresponds to a displacement that reduces
    # gradients g.  It is the vir-occ block of the matrix vector product
    # (Hessian*x). The occ-vir block equals to x2.T.conj(). The overall
    # Hessian for internal rotation is x2 + x2.T.conj(). This is
    # the reason we apply (.real * 2) below
    def hessian_x(xs):
        return cp.stack([hop(x).real * 2 for x in xs])

    x0 = get_davidson_initial_guess(hdiag, nroots, nroots * 2)
    converged, e, v = davidson1(hessian_x, x0, precond, tol=tol, verbose=log, nroots=nroots)
    log.info('rhf_internal: lowest eigs of H = %s', e)
    if not converged[0]:
        raise RuntimeError("rhf_internal stability analysis Davidson solver doesn't converge even the lowest energy root")
    if not np.all(converged):
        log.warn(f"rhf_internal stability analysis Davidson solver doesn't converge all roots ({converged})")

    e, v = e[0], v[0]

    stable = not (e < -1e-5)
    dump_status(log, stable, f'{mf.__class__}', 'internal')
    if stable:
        mo = mf.mo_coeff
    else:
        mo = _rotate_mo(mf.mo_coeff, mf.mo_occ, v)
    if return_status:
        return mo, stable
    else:
        return mo

def _gen_hop_rhf_external(mf, with_symmetry=False, verbose=None):
    assert not with_symmetry
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nocc = get_nocc(mo_occ)
    nvir = mo_coeff.shape[-1] - nocc
    mocc = mo_coeff[:, :nocc]
    mvir = mo_coeff[:, nocc:]

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    if isinstance(mf, SCFWithSolvent):
        fock_ao = mf.get_fock(dm_or_wfn=dm0)
    else:
        fock_ao = mf.get_fock(dm=dm0)
    Foo = mocc.conj().T @ fock_ao @ mocc
    Fvv = mvir.conj().T @ fock_ao @ mvir

    hdiag = Fvv.diagonal()[:, None] - Foo.diagonal()
    hdiag = hdiag.ravel()

    vrespz = mf.gen_response(singlet=None, hermi=2)
    def hop_real2complex(x1):
        x1 = x1.reshape(nvir, nocc)
        x2 = cp.einsum('ps,sq->pq', Fvv, x1)
        x2-= cp.einsum('ps,rp->rs', Foo, x1)

        dm1 = _make_factorized_dm(2 * (mvir @ x1), mocc.conj(),
                                  symmetrize=2)
# No Coulomb and fxc contribution for anti-hermitian DM
        v1 = vrespz(dm1)
        x2 += mvir.conj().T @ v1 @ mocc
        return x2.ravel()

    vresp1 = mf.gen_response(singlet=False, hermi=1)
    def hop_rhf2uhf(x1):
        # See also rhf.TDA triplet excitation
        x1 = x1.reshape(nvir,nocc)
        x2 = cp.einsum('ps,sq->pq', Fvv, x1)
        x2-= cp.einsum('ps,rp->rs', Foo, x1)

        dm1 = _make_factorized_dm(2 * (mvir @ x1), mocc.conj())
        v1ao = vresp1(dm1)
        x2 += mvir.conj().T @ v1ao @ mocc
        return x2.real.ravel()

    return hop_real2complex, hdiag, hop_rhf2uhf, hdiag

def rhf_external(mf, with_symmetry=False, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    assert not with_symmetry
    log = logger.new_logger(mf, verbose)
    hop1, hdiag1, hop2, hdiag2 = _gen_hop_rhf_external(mf, with_symmetry)

    def precond1(residuals, eigenvalues):
        hdiagd = hdiag1[None, :] - eigenvalues[:, None]
        hdiagd[cp.abs(hdiagd) < STAB_HESS_NUMERICAL_ZERO] = STAB_HESS_NUMERICAL_ZERO
        return residuals / hdiagd
    def hessian1(xs):
        return cp.stack([hop1(x) for x in xs])

    x0_1 = get_davidson_initial_guess(hdiag1, nroots, nroots * 2)

    converged1, e1, v1 = davidson1(hessian1, x0_1, precond1, tol=tol, verbose=log, nroots=nroots)
    log.info('rhf_real2complex: lowest eigs of H = %s', e1)
    if not converged1[0]:
        raise RuntimeError("rhf_external real2complex stability analysis Davidson solver doesn't converge even the lowest energy root")
    if not np.all(converged1):
        log.warn(f"rhf_external real2complex stability analysis Davidson solver doesn't converge all roots ({converged1})")

    e1, v1 = e1[0], v1[0]
    stable1 = not (e1 < -1e-5)
    dump_status(log, stable1, f'{mf.__class__}', 'real -> complex')

    def precond2(residuals, eigenvalues):
        hdiagd = hdiag2[None, :] - eigenvalues[:, None]
        hdiagd[cp.abs(hdiagd) < STAB_HESS_NUMERICAL_ZERO] = STAB_HESS_NUMERICAL_ZERO
        return residuals / hdiagd
    def hessian2(xs):
        return cp.stack([hop2(x) for x in xs])

    x0_2 = get_davidson_initial_guess(hdiag2, nroots, nroots * 2)

    converged3, e3, v3 = davidson1(hessian2, x0_2, precond2, tol=tol, verbose=log, nroots=nroots)
    log.info('rhf_rhf2uhf: lowest eigs of H = %s', e3)
    if not converged3[0]:
        raise RuntimeError("rhf_external rhf2uhf stability analysis Davidson solver doesn't converge even the lowest energy root")
    if not np.all(converged3):
        log.warn(f"rhf_external rhf2uhf stability analysis Davidson solver doesn't converge all roots ({converged3})")

    e3, v3 = e3[0], v3[0]
    stable3 = not (e3 < -1e-5)
    dump_status(log, stable3, f'{mf.__class__}', 'RHF/RKS -> UHF/UKS')

    if stable3:
        mo = (mf.mo_coeff, mf.mo_coeff)
    else:
        mo = (_rotate_mo(mf.mo_coeff, mf.mo_occ, v3), mf.mo_coeff)
    mo = cp.stack(mo)
    if return_status:
        return mo, stable3
    else:
        return mo

def uhf_internal(mf, with_symmetry=False, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    assert not with_symmetry
    log = logger.new_logger(mf, verbose)
    g, hop, hdiag = gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ, with_nlc=True)
    hdiag *= 2
    def precond(residuals, eigenvalues):
        hdiagd = hdiag[None, :] - eigenvalues[:, None]
        hdiagd[cp.abs(hdiagd) < STAB_HESS_NUMERICAL_ZERO] = STAB_HESS_NUMERICAL_ZERO
        return residuals / hdiagd
    def hessian_x(xs): # See comments in function rhf_internal
        return cp.stack([hop(x).real * 2 for x in xs])

    x0 = get_davidson_initial_guess(hdiag, nroots, nroots * 2)
    converged, e, v = davidson1(hessian_x, x0, precond, tol=tol, verbose=log, nroots=nroots)
    log.info('uhf_internal: lowest eigs of H = %s', e)
    if not converged[0]:
        raise RuntimeError("uhf_internal stability analysis Davidson solver doesn't converge even the lowest energy root")
    if not np.all(converged):
        log.warn(f"uhf_internal stability analysis Davidson solver doesn't converge all roots ({converged})")

    e, v = e[0], v[0]

    stable = not (e < -1e-5)
    dump_status(log, stable, f'{mf.__class__}', 'internal')
    if stable:
        mo = mf.mo_coeff
    else:
        mo = _rotate_mo_uhf(mf.mo_coeff, mf.mo_occ, v)
    if return_status:
        return mo, stable
    else:
        return mo

def _gen_hop_uhf_external(mf, with_symmetry=False, verbose=None):
    assert not with_symmetry
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nocca, noccb = get_nocc(mo_occ)
    nmo = mo_coeff.shape[-1]
    nvira, nvirb = nmo - nocca, nmo - noccb
    mocca = mo_coeff[0, :, :nocca]
    moccb = mo_coeff[1, :, :noccb]
    mvira = mo_coeff[0, :, nocca:]
    mvirb = mo_coeff[1, :, noccb:]

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    if isinstance(mf, SCFWithSolvent):
        fock_ao = mf.get_fock(dm_or_wfn=dm0)
    else:
        fock_ao = mf.get_fock(dm=dm0)
    fooa = mocca.conj().T @ fock_ao[0] @ mocca
    fvva = mvira.conj().T @ fock_ao[0] @ mvira
    foob = moccb.conj().T @ fock_ao[1] @ moccb
    fvvb = mvirb.conj().T @ fock_ao[1] @ mvirb

    h_diaga = fvva.diagonal()[:, None] - fooa.diagonal()
    h_diagb = fvvb.diagonal()[:, None] - foob.diagonal()
    hdiag1 = cp.hstack((h_diaga.ravel(), h_diagb.ravel()))

    vrespz = mf.gen_response(with_j=False, hermi=2)
    nao = mo_coeff.shape[-2]
    mocc = cp.zeros((2, nao, max(nocca, noccb)), dtype=mo_coeff.dtype)
    mocc[0, :, :nocca] = mocca.conj()
    mocc[1, :, :noccb] = moccb.conj()
    def hop_real2complex(x1):
        x1a = x1[:nvira*nocca].reshape(nvira,nocca)
        x1b = x1[nvira*nocca:].reshape(nvirb,noccb)
        x2a = fvva @ x1a - x1a @ fooa
        x2b = fvvb @ x1b - x1b @ foob

        dm_factor = cp.zeros_like(mocc)
        dm_factor[0, :, :nocca] = mvira @ x1a
        dm_factor[1, :, :noccb] = mvirb @ x1b
        dm1 = _make_factorized_dm(dm_factor, mocc, symmetrize=2)

        v1 = vrespz(dm1)
        x2a += mvira.conj().T @ v1[0] @ mocca
        x2b += mvirb.conj().T @ v1[1] @ moccb
        x2 = cp.hstack((x2a.ravel(), x2b.ravel()))
        return x2

    hdiagab = fvva.diagonal()[:, None] - foob.diagonal()
    hdiagba = fvvb.diagonal()[:, None] - fooa.diagonal()
    hdiag2 = cp.hstack((hdiagab.ravel(), hdiagba.ravel()))

    if isinstance(mf, KohnShamDFT):
        gmf = mf.to_gks()
        gmf.collinear = 'mcol'
    else:
        gmf = mf.to_ghf()
    vresp1 = gmf.gen_response(with_j=False, hermi=1)
    mocc_ghf = block_diag(mocca, moccb)
    def hop_uhf2ghf(x1):
        x1ab = x1[:nvira*noccb].reshape(nvira,noccb)
        x1ba = x1[nvira*noccb:].reshape(nvirb,nocca)
        x2ab = fvva @ x1ab - x1ab @ foob
        x2ba = fvvb @ x1ba - x1ba @ fooa

        # A rotation between alpha-beta blocks
        dmo = cp.zeros_like(mocc_ghf)
        dmo[:nao, nocca:] = mvira @ x1ab
        dmo[nao:, :nocca] = mvirb @ x1ba
        dm1 = _make_factorized_dm(dmo, mocc_ghf.conj())
        v1 = vresp1(dm1)
        x2ab += mvira.conj().T @ v1[:nao, nao:] @ moccb
        x2ba += mvirb.conj().T @ v1[nao:, :nao] @ mocca
        x2 = cp.hstack((x2ab.real.ravel(), x2ba.real.ravel()))
        return x2

    return hop_real2complex, hdiag1, hop_uhf2ghf, hdiag2

def uhf_external(mf, with_symmetry=False, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    assert not with_symmetry
    log = logger.new_logger(mf, verbose)
    hop1, hdiag1, hop2, hdiag2 = _gen_hop_uhf_external(mf, with_symmetry)

    def precond1(residuals, eigenvalues):
        hdiagd = hdiag1[None, :] - eigenvalues[:, None]
        hdiagd[cp.abs(hdiagd) < STAB_HESS_NUMERICAL_ZERO] = STAB_HESS_NUMERICAL_ZERO
        return residuals / hdiagd
    def hessian1(xs):
        return cp.stack([hop1(x) for x in xs])

    x0_1 = get_davidson_initial_guess(hdiag1, nroots, nroots * 2)

    converged1, e1, v1 = davidson1(hessian1, x0_1, precond1, tol=tol, verbose=log, nroots=nroots)
    log.info('uhf_real2complex: lowest eigs of H = %s', e1)
    if not converged1[0]:
        raise RuntimeError("uhf_external real2complex stability analysis Davidson solver doesn't converge even the lowest energy root")
    if not np.all(converged1):
        log.warn(f"uhf_external real2complex stability analysis Davidson solver doesn't converge all roots ({converged1})")

    e1, v1 = e1[0], v1[0]
    stable1 = not (e1 < -1e-5)
    dump_status(log, stable1, f'{mf.__class__}', 'real -> complex')

    def precond2(residuals, eigenvalues):
        hdiagd = hdiag2[None, :] - eigenvalues[:, None]
        hdiagd[cp.abs(hdiagd) < STAB_HESS_NUMERICAL_ZERO] = STAB_HESS_NUMERICAL_ZERO
        return residuals / hdiagd
    def hessian2(xs):
        return cp.stack([hop2(x) for x in xs])

    x0_2 = get_davidson_initial_guess(hdiag2, nroots, nroots * 2)

    converged3, e3, v3 = davidson1(hessian2, x0_2, precond2, tol=tol, verbose=log, nroots=nroots)
    log.info('uhf_uhf2ghf: lowest eigs of H = %s', e3)
    if not converged3[0]:
        raise RuntimeError("uhf_external uhf2ghf stability analysis Davidson solver doesn't converge even the lowest energy root")
    if not np.all(converged3):
        log.warn(f"uhf_external uhf2ghf stability analysis Davidson solver doesn't converge all roots ({converged3})")

    e3, v3 = e3[0], v3[0]
    stable3 = not (e3 < -1e-5)
    dump_status(log, stable3, f'{mf.__class__}', 'UHF/UKS -> GHF/GKS')

    mo = block_diag(mf.mo_coeff[0], mf.mo_coeff[1])
    nmo = mf.mo_occ.shape[-1]
    nocca, noccb = get_nocc(mf.mo_occ)
    nvira, nvirb = nmo - nocca, nmo - noccb
    if not stable3:
        dx = cp.zeros((nmo * 2, nmo * 2))
        dx[nocca:nmo, nmo:nmo + noccb] = v3[:nvira*noccb].reshape(nvira, noccb)
        dx[nmo + noccb:nmo * 2, 0:nocca] = v3[nvira*noccb:].reshape(nvirb, nocca)
        U = expm(dx - dx.conj().T)
        mo = mo @ U
    mo = cp.hstack([mo[:,:nocca], mo[:,nmo:nmo+noccb], mo[:,nocca:nmo], mo[:,nmo+noccb:]])
    if return_status:
        return mo, stable3
    else:
        return mo
