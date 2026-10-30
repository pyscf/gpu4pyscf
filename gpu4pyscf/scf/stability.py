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
from gpu4pyscf.lib import logger
from gpu4pyscf.scf.soscf import gen_g_hop_rhf, gen_g_hop_uhf
from gpu4pyscf.fci.direct_spin1 import davidson1
from gpu4pyscf.scf.gdm import get_nocc, apply_exp_U
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
    nmo = mo_occ.shape[-1]
    nocc = get_nocc(mo_occ)
    kappa = -v.reshape((nmo - nocc, nocc)).T # The kappa from gdm is the upper right corner (nocc, nvir), the kappa from soscf is the lower left corner (nvir, nocc).
    new_mo = apply_exp_U(mo_coeff, nocc, kappa)
    return new_mo

def rhf_internal(mf, with_symmetry=False, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    assert not with_symmetry
    log = logger.new_logger(mf, verbose)
    g, hop, hdiag = gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ)
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
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nocc = get_nocc(mo_occ)
    nvir = mo_coeff.shape[-1] - nocc
    mocc = mo_coeff[:, :nocc]
    mvir = mo_coeff[:, nocc:]

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    Fock_ao = mf.get_fock(dm = dm0)
    Foo = mocc.conj().T @ Fock_ao @ mocc
    Fvv = mvir.conj().T @ Fock_ao @ mvir

    hdiag = Fvv.diagonal()[:, None] - Foo.diagonal()
    hdiag = hdiag.ravel()

    vrespz = mf.gen_response(singlet=None, hermi=2)
    def hop_real2complex(x1):
        x1 = x1.reshape(nvir, nocc)
        x2 = cp.einsum('ps,sq->pq', Fvv, x1)
        x2-= cp.einsum('ps,rp->rs', Foo, x1)

        d1 = 2 * mvir @ x1 @ mocc.conj().T
        dm1 = d1 - d1.conj().T
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

        d1 = 2 * mvir @ x1 @ mocc.conj().T
        dm1 = d1 + d1.conj().T
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

def uhf_internal(mf, with_symmetry=True, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    log = logger.new_logger(mf, verbose)
    g, hop, hdiag = newton_ah.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ,
                                            with_symmetry=with_symmetry)
    hdiag *= 2
    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    def hessian_x(x): # See comments in function rhf_internal
        return hop(x).real * 2

    x0 = numpy.zeros_like(g)
    x0[g!=0] = 1. / hdiag[g!=0]
    if not with_symmetry:  # allow to break point group symmetry
        x0[numpy.argmin(hdiag)] = 1
    e, v = lib.davidson(hessian_x, x0, precond, tol=tol, verbose=log, nroots=nroots)
    log.info('uhf_internal: lowest eigs of H = %s', e)
    if nroots != 1:
        e, v = e[0], v[0]
    stable = not (e < -1e-5)
    dump_status(log, stable, f'{mf.__class__}', 'internal')
    if stable:
        mo = mf.mo_coeff
    else:
        nocca = numpy.count_nonzero(mf.mo_occ[0]> 0)
        nvira = numpy.count_nonzero(mf.mo_occ[0]==0)
        mo = (_rotate_mo(mf.mo_coeff[0], mf.mo_occ[0], v[:nocca*nvira]),
              _rotate_mo(mf.mo_coeff[1], mf.mo_occ[1], v[nocca*nvira:]))
    if return_status:
        return mo, stable
    else:
        return mo

def _gen_hop_uhf_external(mf, with_symmetry=True, verbose=None):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    mocca = mo_coeff[0][:,occidxa]
    moccb = mo_coeff[1][:,occidxb]
    mvira = mo_coeff[0][:,viridxa]
    mvirb = mo_coeff[1][:,viridxb]

    if with_symmetry and mol.symmetry:
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        sym_forbida = orbsyma[viridxa].reshape(-1,1) != orbsyma[occidxa]
        sym_forbidb = orbsymb[viridxb].reshape(-1,1) != orbsymb[occidxb]
        sym_forbid1 = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

    h1e = mf.get_hcore()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    fock_ao = h1e + mf.get_veff(mol, dm0)
    focka = reduce(numpy.dot, (mo_coeff[0].conj().T, fock_ao[0], mo_coeff[0]))
    fockb = reduce(numpy.dot, (mo_coeff[1].conj().T, fock_ao[1], mo_coeff[1]))
    fooa = focka[occidxa[:,None],occidxa]
    fvva = focka[viridxa[:,None],viridxa]
    foob = fockb[occidxb[:,None],occidxb]
    fvvb = fockb[viridxb[:,None],viridxb]

    h_diaga =(focka[viridxa,viridxa].reshape(-1,1) - focka[occidxa,occidxa])
    h_diagb =(fockb[viridxb,viridxb].reshape(-1,1) - fockb[occidxb,occidxb])
    hdiag1 = numpy.hstack((h_diaga.reshape(-1), h_diagb.reshape(-1)))
    if with_symmetry and mol.symmetry:
        hdiag1[sym_forbid1] = 0

    vrespz = mf.gen_response(with_j=False, hermi=2)
    def hop_real2complex(x1):
        if with_symmetry and mol.symmetry:
            x1 = x1.copy()
            x1[sym_forbid1] = 0
        x1a = x1[:nvira*nocca].reshape(nvira,nocca)
        x1b = x1[nvira*nocca:].reshape(nvirb,noccb)
        x2a = numpy.einsum('pr,rq->pq', fvva, x1a)
        x2a-= numpy.einsum('sq,ps->pq', fooa, x1a)
        x2b = numpy.einsum('pr,rq->pq', fvvb, x1b)
        x2b-= numpy.einsum('qs,ps->pq', foob, x1b)

        d1a = reduce(numpy.dot, (mvira, x1a, mocca.conj().T))
        d1b = reduce(numpy.dot, (mvirb, x1b, moccb.conj().T))
        dm1 = numpy.array((d1a-d1a.conj().T, d1b-d1b.conj().T))

        v1 = vrespz(dm1)
        x2a += reduce(numpy.dot, (mvira.conj().T, v1[0], mocca))
        x2b += reduce(numpy.dot, (mvirb.conj().T, v1[1], moccb))
        x2 = numpy.hstack((x2a.ravel(), x2b.ravel()))
        if with_symmetry and mol.symmetry:
            x2[sym_forbid1] = 0
        return x2

    if with_symmetry and mol.symmetry:
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        sym_forbidab = orbsyma[viridxa].reshape(-1,1) != orbsymb[occidxb]
        sym_forbidba = orbsymb[viridxb].reshape(-1,1) != orbsyma[occidxa]
        sym_forbid2 = numpy.hstack((sym_forbidab.ravel(), sym_forbidba.ravel()))
    hdiagab = fvva.diagonal().reshape(-1,1) - foob.diagonal()
    hdiagba = fvvb.diagonal().reshape(-1,1) - fooa.diagonal()
    hdiag2 = numpy.hstack((hdiagab.ravel(), hdiagba.ravel()))
    if with_symmetry and mol.symmetry:
        hdiag2[sym_forbid2] = 0

    vresp1 = mf.gen_response(with_j=False, hermi=0)
    # Spin flip GHF solution is not considered
    def hop_uhf2ghf(x1):
        if with_symmetry and mol.symmetry:
            x1 = x1.copy()
            x1[sym_forbid2] = 0
        x1ab = x1[:nvira*noccb].reshape(nvira,noccb)
        x1ba = x1[nvira*noccb:].reshape(nvirb,nocca)
        x2ab = numpy.einsum('pr,rq->pq', fvva, x1ab)
        x2ab-= numpy.einsum('sq,ps->pq', foob, x1ab)
        x2ba = numpy.einsum('pr,rq->pq', fvvb, x1ba)
        x2ba-= numpy.einsum('qs,ps->pq', fooa, x1ba)

        d1ab = reduce(numpy.dot, (mvira, x1ab, moccb.conj().T))
        d1ba = reduce(numpy.dot, (mvirb, x1ba, mocca.conj().T))
        dm1 = numpy.array((d1ab+d1ba.conj().T, d1ba+d1ab.conj().T))
        v1 = vresp1(dm1)
        x2ab += reduce(numpy.dot, (mvira.conj().T, v1[0], moccb))
        x2ba += reduce(numpy.dot, (mvirb.conj().T, v1[1], mocca))
        x2 = numpy.hstack((x2ab.real.ravel(), x2ba.real.ravel()))
        if with_symmetry and mol.symmetry:
            x2[sym_forbid2] = 0
        return x2

    return hop_real2complex, hdiag1, hop_uhf2ghf, hdiag2


def uhf_external(mf, with_symmetry=True, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    log = logger.new_logger(mf, verbose)
    hop1, hdiag1, hop2, hdiag2 = _gen_hop_uhf_external(mf, with_symmetry)

    def precond(dx, e, x0):
        hdiagd = hdiag1 - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    x0 = numpy.zeros_like(hdiag1)
    x0[hdiag1>1e-5] = 1. / hdiag1[hdiag1>1e-5]
    if not with_symmetry:  # allow to break point group symmetry
        x0[numpy.argmin(hdiag1)] = 1
    e1, v = lib.davidson(hop1, x0, precond, tol=tol, verbose=log, nroots=nroots)
    log.info('uhf_real2complex: lowest eigs of H = %s', e1)
    if nroots != 1:
        e1, v = e1[0], v[0]
    stable1 = not (e1 < -1e-5)
    dump_status(log, stable1, f'{mf.__class__}', 'real -> complex')

    def precond(dx, e, x0):
        hdiagd = hdiag2 - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    x0 = numpy.zeros_like(hdiag2)
    x0[hdiag2>1e-5] = 1. / hdiag2[hdiag2>1e-5]
    if not with_symmetry:  # allow to break point group symmetry
        x0[numpy.argmin(hdiag2)] = 1
    e3, v = lib.davidson(hop2, x0, precond, tol=tol, verbose=log, nroots=nroots)
    log.info('uhf_external: lowest eigs of H = %s', e3)
    if nroots != 1:
        e3, v = e3[0], v[0]
    stable = not (e3 < -1e-5)
    dump_status(log, stable, f'{mf.__class__}', 'UHF/UKS -> GHF/GKS')
    mo = scipy.linalg.block_diag(*mf.mo_coeff)
    if not stable:
        occidxa = numpy.where(mf.mo_occ[0]> 0)[0]
        viridxa = numpy.where(mf.mo_occ[0]==0)[0]
        occidxb = numpy.where(mf.mo_occ[1]> 0)[0]
        viridxb = numpy.where(mf.mo_occ[1]==0)[0]
        nocca = len(occidxa)
        nvira = len(viridxa)
        noccb = len(occidxb)
        nvirb = len(viridxb)
        nmo = nocca + nvira
        dx = numpy.zeros((nmo*2,nmo*2))
        dx[viridxa[:,None],nmo+occidxb] = v[:nvira*noccb].reshape(nvira,noccb)
        dx[nmo+viridxb[:,None],occidxa] = v[nvira*noccb:].reshape(nvirb,nocca)
        u = newton_ah.expmat(dx - dx.conj().T)
        mo = numpy.dot(mo, u)
        mo = numpy.hstack([mo[:,:nocca], mo[:,nmo:nmo+noccb],
                           mo[:,nocca:nmo], mo[:,nmo+noccb:]])
    if return_status:
        return mo, stable
    else:
        return mo
