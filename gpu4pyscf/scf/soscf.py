#!/usr/bin/env python
#
# Copyright 2024 The PySCF Developers. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Second order SCF solver
'''

import sys
import math
import numpy as np
import cupy as cp
import scipy.linalg
from cupyx.scipy.linalg import expm
from pyscf import lib
from pyscf.scf import chkfile
from pyscf.soscf import ciah
from pyscf.soscf.newton_ah import _CIAH_SOSCF as _SOSCF_cpu
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import hf, rohf, uhf
from gpu4pyscf.lib.cupy_helper import transpose_sum, contract
from gpu4pyscf.lib import utils

def gen_g_hop_rhf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None):
    assert mo_coeff.dtype == np.float64
    occidx = cp.nonzero(mo_occ==2)[0]
    viridx = cp.nonzero(mo_occ==0)[0]
    orbo = mo_coeff[:,occidx]
    orbv = mo_coeff[:,viridx]
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]

    if fock_ao is None:
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
    fock = mo_coeff.conj().T.dot(fock_ao).dot(mo_coeff)
    foo = fock[occidx[:,None],occidx]
    fvv = fock[viridx[:,None],viridx]

    g = fock[viridx[:,None],occidx] * 2
    h_diag = (fvv.diagonal().real[:,None] - foo.diagonal().real) * 2

    vind = mf.gen_response(mo_coeff, mo_occ, singlet=None, hermi=1)

    def h_op(x):
        x = x.reshape(nvir,nocc)
        x2 = contract('ps,sq->pq', fvv, x)
        x2-= contract('ps,rp->rs', foo, x)

        # *2 for double occupancy
        dm1 = orbv.dot(x*2).dot(orbo.conj().T)
        dm1 = transpose_sum(dm1)
        v1 = vind(dm1)
        x2 += orbv.conj().T.dot(v1).dot(orbo)
        return x2.ravel() * 2

    return g.reshape(-1), h_op, h_diag.reshape(-1)

def gen_g_hop_rohf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None):
    if getattr(fock_ao, 'focka', None) is None:
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
    fock_ao = fock_ao.focka, fock_ao.fockb
    mo_occa = occidxa = mo_occ > 0
    mo_occb = occidxb = mo_occ ==2
    ug, uh_op, uh_diag = gen_g_hop_uhf(
        mf, (mo_coeff,)*2, (mo_occa,mo_occb), fock_ao, None)

    viridxa = ~occidxa
    viridxb = ~occidxb
    uniq_var_a = viridxa[:,None] & occidxa
    uniq_var_b = viridxb[:,None] & occidxb
    uniq_ab = uniq_var_a | uniq_var_b
    nmo = mo_coeff.shape[-1]
    nocca, noccb = mf.nelec
    nvira = nmo - nocca

    def sum_ab(x):
        x1 = cp.zeros((nmo,nmo), dtype=x.dtype)
        x1[uniq_var_a]  = x[:nvira*nocca]
        x1[uniq_var_b] += x[nvira*nocca:]
        return x1[uniq_ab]

    g = sum_ab(ug)
    h_diag = sum_ab(uh_diag)
    def h_op(x):
        x1 = cp.zeros((nmo,nmo), dtype=x.dtype)
        # unpack ROHF rotation parameters
        x1[uniq_ab] = x
        x1 = cp.hstack((x1[uniq_var_a],x1[uniq_var_b]))
        return sum_ab(uh_op(x1))

    return g, h_op, h_diag

def gen_g_hop_uhf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None):
    assert mo_coeff[0].dtype == np.float64
    occidxa = cp.nonzero(mo_occ[0] >  0)[0]
    occidxb = cp.nonzero(mo_occ[1] >  0)[0]
    viridxa = cp.nonzero(mo_occ[0] == 0)[0]
    viridxb = cp.nonzero(mo_occ[1] == 0)[0]
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]
    nmo = mo_occ[0].size
    nocca, noccb = mf.nelec
    nvira = nmo - nocca
    nvirb = nmo - noccb

    if fock_ao is None:
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
    focka = mo_coeff[0].conj().T.dot(fock_ao[0]).dot(mo_coeff[0])
    fockb = mo_coeff[1].conj().T.dot(fock_ao[1]).dot(mo_coeff[1])
    fooa = focka[occidxa[:,None],occidxa]
    fvva = focka[viridxa[:,None],viridxa]
    foob = fockb[occidxb[:,None],occidxb]
    fvvb = fockb[viridxb[:,None],viridxb]

    g = cp.hstack((focka[viridxa[:,None],occidxa].ravel(),
                   fockb[viridxb[:,None],occidxb].ravel()))
    h_diaga = fvva.diagonal().real[:,None] - fooa.diagonal().real
    h_diagb = fvvb.diagonal().real[:,None] - foob.diagonal().real
    h_diag = cp.hstack((h_diaga.reshape(-1), h_diagb.reshape(-1)))

    vind = mf.gen_response(mo_coeff, mo_occ, hermi=1)

    def h_op(x):
        x1a = x[:nvira*nocca].reshape(nvira,nocca)
        x1b = x[nvira*nocca:].reshape(nvirb,noccb)
        x2a = contract('pr,rq->pq', fvva, x1a)
        x2a-= contract('sq,ps->pq', fooa, x1a)
        x2b = contract('pr,rq->pq', fvvb, x1b)
        x2b-= contract('sq,ps->pq', foob, x1b)

        d1a = orbva.dot(x1a).dot(orboa.conj().T)
        d1b = orbvb.dot(x1b).dot(orbob.conj().T)
        dm1 = cp.array([transpose_sum(d1a),
                        transpose_sum(d1b)])
        v1 = vind(dm1)
        x2a += orbva.conj().T.dot(v1[0]).dot(orboa)
        x2b += orbvb.conj().T.dot(v1[1]).dot(orbob)
        return cp.hstack((x2a.ravel(), x2b.ravel()))

    return g, h_op, h_diag


def _rotate_orb_cc(mf, h1e, s1e, conv_tol_grad=None, verbose=None):
    log = logger.new_logger(mf, verbose)

    if conv_tol_grad is None:
        conv_tol_grad = (mf.conv_tol*.1)**.5
        #TODO: dynamically adjust max_stepsize, as done in mc1step.py

    def precond(x, e):
        hdiagd = h_diag-(e-mf.ah_level_shift)
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        x = x/hdiagd
        return x

    t3m = log.init_timer()
    u = g_kf = g_orb = norm_gorb = dxi = kfcount = jkcount = None
    dm0 = vhf0 = None
    g_op = lambda: g_orb
    while True:
        mo_coeff, mo_occ, dm0, vhf0, e_tot = (yield u, g_kf, kfcount, jkcount, dm0, vhf0)
        fock_ao = mf.get_fock(h1e, s1e, vhf0, dm0)

        g_kf, h_op, h_diag = mf.gen_g_hop(mo_coeff, mo_occ, fock_ao)
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

        ah_conv_tol = min(norm_gorb**2, mf.ah_conv_tol)
        # increase the AH accuracy when approach convergence
        ah_start_cycle = mf.ah_start_cycle
        imic = 0
        dr = 0.
        u = 1.
        ukf = None
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
                if ihop == problem_size:
                    log.debug1('... Hx=g fully converged for small systems')
                elif dxmax > mf.max_stepsize:
                    scale = mf.max_stepsize / dxmax
                    log.debug1('... scale rotation size %g', scale)
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
                    u = mf.update_rotate_matrix(dr, mo_occ, mo_coeff=mo_coeff)
                    if ukf is not None:
                        u = mf.rotate_mo(ukf, u)
                    ukf = u
                    dr[:] = 0
                    mo1 = mf.rotate_mo(mo_coeff, u)
                    dm = mf.make_rdm1(mo1, mo_occ)
                    # use mf._scf.get_veff to avoid density-fit mf polluting get_veff
                    vhf0 = mf._scf.get_veff(mf._scf.mol, dm, dm_last=dm0, vhf_last=vhf0)
                    dm0 = dm
                    # Use API to compute fock instead of "fock=h1e+vhf0". This is because get_fock
                    # is the hook being overloaded in many places.
                    fock_ao = mf.get_fock(h1e, s1e, vhf0, dm0)
                    g_kf1 = mf.get_grad(mo1, mo_occ, fock_ao)
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
            u = mf.update_rotate_matrix(dr, mo_occ, mo_coeff=mo_coeff)
            if ukf is not None:
                u = mf.rotate_mo(ukf, u)
            jkcount += ihop + 1
            log.debug('    tot inner=%d  %d JK  |g|= %4.3g  |u-1|= %4.3g',
                      imic, jkcount, norm_gorb, cp.linalg.norm(dr))
        h_op = h_diag = None
        t3m = log.timer('aug_hess in %d inner iters' % imic, *t3m)

def _davidson_cc(h_op, g_op, precond, x0, tol=1e-10, xs=[], ax=[],
                 max_cycle=30, lindep=1e-14, verbose=logger.WARN):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    toloose = tol**.5
    # the first trial vector is (1,0,0,...), which is not included in xs
    xs = list(xs)
    ax = list(ax)
    nx = len(xs)

    problem_size = x0.size
    max_cycle = min(max_cycle, problem_size)
    heff = np.zeros((max_cycle+nx+1,max_cycle+nx+1), dtype=x0.dtype)
    ovlp = np.eye(max_cycle+nx+1, dtype=x0.dtype)
    if nx == 0:
        xs.append(x0)
        ax.append(h_op(x0))
    else:
        for i in range(1, nx+1):
            for j in range(1, i+1):
                heff[i,j] = xs[i-1].conj().dot(ax[j-1])
                ovlp[i,j] = xs[i-1].conj().dot(xs[j-1])
            heff[1:i,i] = heff[i,1:i].conj()
            ovlp[1:i,i] = ovlp[i,1:i].conj()

    w_t = 0
    for istep in range(max_cycle):
        g = g_op()
        nx = len(xs)
        for i in range(nx):
            heff[i+1,0] = xs[i].conj().dot(g)
            heff[nx,i+1] = xs[nx-1].conj().dot(ax[i])
            ovlp[nx,i+1] = xs[nx-1].conj().dot(xs[i])
        heff[0,:nx+1] = heff[:nx+1,0].conj()
        heff[1:nx,nx] = heff[nx,1:nx].conj()
        ovlp[1:nx,nx] = ovlp[nx,1:nx].conj()
        nvec = nx + 1
        #s0 = scipy.linalg.eigh(ovlp[:nvec,:nvec])[0][0]
        #if s0 < lindep:
        #    yield True, istep, w_t, xtrial, hx, dx, s0
        #    break
        wlast = w_t
        xtrial, w_t, v_t, index, seig = \
                _regular_step(heff[:nvec,:nvec], ovlp[:nvec,:nvec], xs,
                              lindep, log)
        s0 = seig[0]
        hx = _dgemv(v_t[1:], ax)
        # note g*v_t[0], as the first trial vector is (1,0,0,...)
        dx = hx + g*v_t[0] - w_t * v_t[0]*xtrial
        norm_dx = np.linalg.norm(dx)
        log.debug1('... AH step %d  index= %d  |dx|= %.5g  eig= %.5g  v[0]= %.5g  lindep= %.5g',
                   istep+1, index, norm_dx, w_t, v_t[0].real, s0)
        hx *= 1/v_t[0] # == h_op(xtrial)
        if ((abs(w_t-wlast) < tol and norm_dx < toloose) or
            s0 < lindep or
            istep+1 == problem_size):
            # Avoid adding more trial vectors if hessian converged
            yield True, istep+1, w_t, xtrial, hx, dx, s0
            if s0 < lindep or norm_dx < lindep:# or np.linalg.norm(xtrial) < lindep:
                # stop the iteration because eigenvectors would be barely updated
                break
        else:
            yield False, istep+1, w_t, xtrial, hx, dx, s0
            x0 = precond(dx, w_t)
            xs.append(x0)
            ax.append(h_op(x0))

def _regular_step(heff, ovlp, xs, lindep, log, root_id=0):
    w, v, seig = lib.safe_eigh(heff, ovlp, lindep)
    #if e[0] < -.1:
    #    sel = 0
    #else:
    # There exists systems that the first eigenvalue of AH is -inf.
    # Dynamically choosing the eigenvectors may be better.
    idx = np.nonzero(abs(v[0]) > 0.1)[0]
    sel = idx[root_id]
    log.debug1('CIAH eigen-sel %s', sel)
    w_t = w[sel]

    if w_t < 1e-4:
        try:
            e, c = scipy.linalg.eigh(heff[1:,1:], ovlp[1:,1:])
        except scipy.linalg.LinAlgError:
            e, c = lib.safe_eigh(heff[1:,1:], ovlp[1:,1:], lindep)[:2]
        if np.any(e < -1e-5):
            log.debug('Negative hessians found %s', e[e<0])

    xtrial = _dgemv(v[1:,sel]/v[0,sel], xs)
    return xtrial, w_t, v[:,sel], sel, seig

def _dgemv(v, m):
    vm = v[0] * m[0]
    for i,vi in enumerate(v[1:]):
        vm += vi * m[i+1]
    return vm


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

    if mo_coeff is not None and mo_occ is not None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        # call mf._scf.get_veff, to avoid "newton().density_fit()" polluting get_veff
        vhf = mf._scf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        mo_energy, mo_tmp = mf.eig(fock, s1e)
        mf.get_occ(mo_energy, mo_tmp)
        mo_tmp = None

    else:
        if dm is None:
            dm = mf.get_init_guess(mol, mf.init_guess)
        vhf = mf._scf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf._scf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)

    # Save mo_coeff and mo_occ because they are needed by function rotate_mo
    mf.mo_coeff, mf.mo_occ = mo_coeff, mo_occ

    e_tot = mf._scf.energy_tot(dm, h1e, vhf)
    fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
    log.info('Initial guess E= %.15g  |g|= %g', e_tot,
             cp.linalg.norm(mf._scf.get_grad(mo_coeff, mo_occ, fock)))

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
        u, g_orb, kfcount, jkcount, dm_last, vhf = \
                rotaiter.send((mo_coeff, mo_occ, dm, vhf, e_tot))
        kftot += kfcount + 1
        jktot += jkcount + 1

        last_hf_e = e_tot
        norm_gorb = cp.linalg.norm(g_orb)
        mo_coeff = mf.rotate_mo(mo_coeff, u, log)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf._scf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        # NOTE: DO NOT change the initial guess mo_occ, mo_coeff
        if mf.verbose >= logger.DEBUG:
            mo_energy, mo_tmp = mf.eig(fock, s1e)
            mf.get_occ(mo_energy, mo_tmp)
            # call mf._scf.energy_tot for dft, because the (dft).get_veff step saved _exc in mf._scf
        e_tot = mf._scf.energy_tot(dm, h1e, vhf)

        log.info('macro= %d  E= %.15g  delta_E= %g  |g|= %g  %d KF %d JK',
                 imacro, e_tot, e_tot-last_hf_e, norm_gorb,
                 kfcount+1, jkcount)
        cput1 = log.timer('cycle= %d'%(imacro+1), *cput1)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
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
    mo_energy, mo_coeff1 = mf._scf.canonicalize(mo_coeff, mo_occ, fock)
    if mf.canonicalization:
        log.info('Canonicalize SCF orbitals')
        mo_coeff = mo_coeff1
        if dump_chk:
            mf.dump_chk(locals())
    log.info('macro X = %d  E=%.15g  |g|= %g  total %d KF %d JK',
             imacro+1, e_tot, norm_gorb, kftot+1, jktot+1)

    if cp.any(mo_occ==0):
        homo = mo_energy[mo_occ>0].max()
        lumo = mo_energy[mo_occ==0].min()
        if homo > lumo:
            log.warn('canonicalized orbital HOMO %s > LUMO %s ', homo, lumo)
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

# A tag to label the derived SCF class
class _CIAH_SOSCF:
    '''
    Attributes for Newton solver:
        max_cycle_inner : int
            AH iterations within eacy macro iterations. Default is 10
        max_stepsize : int
            The step size for orbital rotation.  Small step is prefered.  Default is 0.05.
        canonicalization : bool
            To control whether to canonicalize the orbitals optimized by
            Newton solver.  Default is True.
    '''

    __name_mixin__ = 'SecondOrder'

    max_cycle_inner = _SOSCF_cpu.max_cycle_inner
    max_stepsize = _SOSCF_cpu.max_stepsize
    canonicalization = _SOSCF_cpu.canonicalization

    ah_start_tol = _SOSCF_cpu.ah_start_tol
    ah_start_cycle = _SOSCF_cpu.ah_start_cycle
    ah_level_shift = _SOSCF_cpu.ah_level_shift
    ah_conv_tol = _SOSCF_cpu.ah_conv_tol
    ah_lindep = _SOSCF_cpu.ah_lindep
    ah_max_cycle = _SOSCF_cpu.ah_max_cycle
    ah_grad_trust_region = _SOSCF_cpu.ah_grad_trust_region
    kf_interval = _SOSCF_cpu.kf_interval
    kf_trust_region = _SOSCF_cpu.kf_trust_region

    _keys = _SOSCF_cpu._keys

    to_gpu = utils.to_gpu
    device = utils.device
    to_cpu = utils.to_cpu

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self._scf = mf

    def undo_soscf(self):
        '''Remove the SOSCF Mixin'''
        from gpu4pyscf.df.df_jk import _DFHF
        if isinstance(self, _DFHF) and not isinstance(self._scf, _DFHF):
            # where density fitting is only applied on the SOSCF hessian
            mf = self.undo_df()
        else:
            mf = self
        obj = lib.view(mf, lib.drop_class(mf.__class__, _CIAH_SOSCF))
        del obj._scf
        # When both self and self._scf are DF objects, they may be different df
        # objects. The DF object of the base scf object should be used.
        if hasattr(self._scf, 'with_df'):
            obj.with_df = self._scf.with_df
        return obj

    undo_newton = undo_soscf

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        super().dump_flags(verbose)
        log.info('******** %s Newton solver flags ********', self._scf.__class__)
        log.info('max_cycle_inner = %d',  self.max_cycle_inner)
        log.info('max_stepsize = %g',     self.max_stepsize)
        log.info('ah_start_tol = %g',     self.ah_start_tol)
        log.info('ah_level_shift = %g',   self.ah_level_shift)
        log.info('ah_conv_tol = %g',      self.ah_conv_tol)
        log.info('ah_lindep = %g',        self.ah_lindep)
        log.info('ah_start_cycle = %d',   self.ah_start_cycle)
        log.info('ah_max_cycle = %d',     self.ah_max_cycle)
        log.info('ah_grad_trust_region = %g', self.ah_grad_trust_region)
        log.info('kf_interval = %d', self.kf_interval)
        log.info('kf_trust_region = %d', self.kf_trust_region)
        log.info('canonicalization = %s', self.canonicalization)
        return self

    build = _SOSCF_cpu.build
    reset = _SOSCF_cpu.reset

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

        logger.timer(self, 'Second order SCF', *cput0)
        self._finalize()
        return self.e_tot

    from_dm = _SOSCF_cpu.from_dm

    gen_g_hop = gen_g_hop_rhf

    def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
        nmo = len(mo_occ)
        x1 = cp.zeros((nmo,nmo), dtype=dx.dtype)
        occidxa = mo_occ>0
        occidxb = mo_occ==2
        viridxa = ~occidxa
        viridxb = ~occidxb
        mask = (viridxa[:,None] & occidxa) | (viridxb[:,None] & occidxb)
        x1[mask] = dx
        dr = x1 - x1.conj().T
        u = expm(dr)
        if isinstance(u0, cp.ndarray):
            u = u0.dot(u)
        return u

    def rotate_mo(self, mo_coeff, u, log=None):
        return mo_coeff.dot(u)

class _SecondOrderROHF(_CIAH_SOSCF):
    gen_g_hop = gen_g_hop_rohf

class _SecondOrderUHF(_CIAH_SOSCF):
    gen_g_hop = gen_g_hop_uhf

    def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
        occidxa = mo_occ[0] > 0
        occidxb = mo_occ[1] > 0
        viridxa = ~occidxa
        viridxb = ~occidxb

        nmo = len(occidxa)
        dr = cp.zeros((2,nmo,nmo), dtype=dx.dtype)
        uniq = cp.array((viridxa[:,None] & occidxa,
                         viridxb[:,None] & occidxb))
        dr[uniq] = dx
        dr = dr - dr.conj().transpose(0,2,1)

        if isinstance(u0, int) and u0 == 1:
            return cp.asarray((expm(dr[0]), expm(dr[1])))
        else:
            return cp.asarray((u0[0].dot(expm(dr[0])),
                               u0[1].dot(expm(dr[1]))))

    def rotate_mo(self, mo_coeff, u, log=None):
        mo = cp.asarray((mo_coeff[0].dot(u[0]),
                         mo_coeff[1].dot(u[1])))
        return mo

    def kernel(self, mo_coeff=None, mo_occ=None, dm0=None):
        if isinstance(mo_coeff, cp.ndarray) and mo_coeff.ndim == 2:
            mo_coeff = (mo_coeff, mo_coeff)
        if isinstance(mo_occ, cp.ndarray) and mo_occ.ndim == 1:
            mo_occ = (cp.asarray(mo_occ >0, dtype=np.float64),
                      cp.asarray(mo_occ==2, dtype=np.float64))
        return _CIAH_SOSCF.kernel(self, mo_coeff, mo_occ, dm0)

class _SecondOrderRHF(_CIAH_SOSCF):
    gen_g_hop = gen_g_hop_rhf

def newton(mf):
    if isinstance(mf, _CIAH_SOSCF):
        return mf

    assert isinstance(mf, hf.SCF)

    if mf.istype('ROHF'):
        cls = _SecondOrderROHF
    elif mf.istype('UHF'):
        cls = _SecondOrderUHF
    elif mf.istype('GHF'):
        raise NotImplementedError
    elif mf.istype('RDHF'):
        raise NotImplementedError
    elif mf.istype('DHF'):
        raise NotImplementedError
    else:
        cls = _SecondOrderRHF
    return lib.set_class(cls(mf), (cls, mf.__class__))
