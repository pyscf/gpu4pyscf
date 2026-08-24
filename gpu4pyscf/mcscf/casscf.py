# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
import math

import cupy
from cupyx.scipy.linalg import expm
from pyscf import gto
from pyscf import lib
from pyscf import scf as cpu_scf
from pyscf.mcscf import mc1step as cpu_mc1step

from gpu4pyscf import scf
from gpu4pyscf.fci import direct_spin1 as gpu_direct_spin1
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import utils
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.mcscf.casci import h1e_for_cas
from gpu4pyscf.scf import hf


class _ERIS:
    def __init__(self, casscf, mo, paaa, hcore=None):
        ncore = casscf.ncore
        nocc = ncore + casscf.ncas
        self.paaa = cupy.asarray(paaa)
        self.aaaa = self.paaa[ncore:nocc].copy()
        if hcore is None:
            hcore = cupy.asarray(casscf.get_hcore())
        self.hcore = hcore
        self.h1eff, self.ecore, self.vhf_c = casscf.h1e_for_cas(
            mo, hcore=hcore, return_corevhf=True)


def gen_g_hdiag(casscf, mo, casdm1, casdm2, eris):
    ncore = casscf.ncore
    ncas = casscf.ncas
    nocc = ncore + ncas
    nmo = mo.shape[1]

    core = slice(0, ncore)
    act = slice(ncore, nocc)
    vir = slice(nocc, nmo)
    mo_act = mo[:, act]

    fcore = reduce(cupy.dot, (mo.T, eris.hcore + eris.vhf_c, mo))

    dm_act = reduce(cupy.dot, (mo_act, casdm1, mo_act.T))
    jact, kact = casscf.get_jk(casscf.mol, dm_act)
    fact = reduce(cupy.dot, (mo.T, 2*jact - kact, mo))

    y = fcore[:, act] @ casdm1
    z = contract('puvw,tuvw->pt', eris.paaa, casdm2, alpha=.5)

    g_ia = 4 * fcore[core, vir] + 2 * fact[core, vir]
    g_ta = 2 * y[vir, :].T + 4 * z[vir, :].T
    g_it = 4 * fcore[core, act] + 2 * fact[core, act]
    g_it += -2 * y[core, :] - 4 * z[core, :]

    f_diag = (4 * fcore + 2 * fact).diagonal()
    fcore_diag = fcore.diagonal()
    fact_diag = fact.diagonal()
    gamma_diag = casdm1.diagonal()
    y_diag = y[act].diagonal()
    z_diag = z[act].diagonal()

    d_ia = f_diag[vir][None, :] - f_diag[core][:, None]
    d_ta = 2 * gamma_diag[:, None] * fcore_diag[vir][None, :]
    d_ta += gamma_diag[:, None] * fact_diag[vir][None, :]
    d_ta += (-2 * y_diag - 4 * z_diag)[:, None]
    d_it = f_diag[act][None, :] - f_diag[core][:, None]
    d_it += 2 * gamma_diag[None, :] * fcore_diag[core][:, None]
    d_it += gamma_diag[None, :] * fact_diag[core][:, None]
    d_it += (-2 * y_diag - 4 * z_diag)[None, :]

    return g_ia, g_ta, g_it, d_ia, d_ta, d_it


def build_rotation_matrix(casscf, mo, terms, denom_floor=1e-8,
                          max_abs_step=.03):
    ncore = casscf.ncore
    nocc = ncore + casscf.ncas
    nmo = mo.shape[1]
    core = slice(0, ncore)
    act = slice(ncore, nocc)
    vir = slice(nocc, nmo)
    g_ia, g_ta, g_it, d_ia, d_ta, d_it = terms

    if max_abs_step is not None and max_abs_step <= 0:
        raise ValueError('max_abs_step must be positive')

    # Apply one level shift to keep the diagonal model positive.  The orbital
    # step limit is applied afterwards and does not make this shift more
    # conservative.
    level_shift = cupy.asarray(0., dtype=mo.dtype)
    gd_pairs = ((g_ia, d_ia), (g_ta, d_ta), (g_it, d_it))
    for _, hdiag in gd_pairs:
        if hdiag.size:
            level_shift = cupy.maximum(level_shift, cupy.max(denom_floor - hdiag))

    s_ia = g_ia / (d_ia + level_shift)
    s_ta = g_ta / (d_ta + level_shift)
    s_it = g_it / (d_it + level_shift)
    if max_abs_step is not None:
        s_ia = cupy.clip(s_ia, -max_abs_step, max_abs_step)
        s_ta = cupy.clip(s_ta, -max_abs_step, max_abs_step)
        s_it = cupy.clip(s_it, -max_abs_step, max_abs_step)

    s = cupy.zeros((nmo, nmo), dtype=mo.dtype)
    s[core, vir] = s_ia
    s[vir, core] = -s_ia.T
    s[act, vir] = s_ta
    s[vir, act] = -s_ta.T
    s[core, act] = s_it
    s[act, core] = -s_it.T
    return s


def kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=None, ci0=None,
           callback=None, verbose=logger.NOTE, dump_chk=True):
    log = logger.new_logger(casscf, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    if callback is None:
        callback = casscf.callback
    if ci0 is None:
        ci0 = casscf.ci

    mo = cupy.asarray(mo_coeff)
    if conv_tol_grad is None:
        conv_tol_grad = math.sqrt(tol)
        logger.info(casscf, 'Set conv_tol_grad to %g', conv_tol_grad)

    conv = False
    e_last = None
    e_tot = e_cas = fcivec = eris = casdm1 = None
    denom_floor = casscf.denom_floor
    max_stepsize = casscf.max_stepsize
    timing = casscf.timing = {
        'macro_cycles': 0,
        'hcore_wall': 0.,
        'ao2mo_wall': 0.,
        'h1e_wall': 0.,
        'fci_wall': 0.,
        'fci_setup_wall': 0.,
        'fci_davidson_wall': 0.,
        'fci_iterations': 0,
        'rdm_wall': 0.,
        'orbital_derivatives_wall': 0.,
        'orbital_rotation_wall': 0.,
    }
    t0 = log.init_timer()
    hcore = cupy.asarray(casscf.get_hcore())
    timing['hcore_wall'] = log.timer_silent(*t0)[2] * 1e-3

    for istep in range(1, casscf.max_cycle_macro + 1):
        t0 = log.init_timer()
        eris = casscf.ao2mo(mo, hcore=hcore)
        timing['ao2mo_wall'] += log.timer_silent(*t0)[2] * 1e-3

        t0 = log.init_timer()
        e_tot, e_cas, fcivec = casscf.casci(
            mo, ci0, eris, log, locals())
        timing['fci_wall'] += log.timer_silent(*t0)[2] * 1e-3
        fci_timing = getattr(casscf.fcisolver, 'timing', {})
        timing['fci_setup_wall'] += fci_timing.get('setup_wall', 0.)
        timing['fci_davidson_wall'] += fci_timing.get('davidson_wall', 0.)
        timing['fci_iterations'] += fci_timing.get('davidson_iterations', 0)

        t0 = log.init_timer()
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, casscf.ncas,
                                                     casscf.nelecas)
        casdm1 = cupy.asarray(casdm1)
        casdm2 = cupy.asarray(casdm2)
        timing['rdm_wall'] += log.timer_silent(*t0)[2] * 1e-3

        t0 = log.init_timer()
        terms = gen_g_hdiag(casscf, mo, casdm1, casdm2, eris)
        g_norm = max((float(cupy.abs(x).max().get())
                      for x in terms[:3] if x.size), default=0.)
        timing['orbital_derivatives_wall'] += (
            log.timer_silent(*t0)[2] * 1e-3)
        timing['macro_cycles'] = istep
        de = e_tot - e_last if e_last is not None else e_tot
        log.info('cycle %3d  E = %#.15g  de = %.6g  |g| = %.6g',
                 istep, e_tot, de, g_norm)
        if max_stepsize is not None:
            max_stepsize = casscf.max_stepsize_scheduler(locals())
        if callable(callback):
            callback(locals())

        if e_last is not None and abs(de) < tol and g_norm < conv_tol_grad:
            conv = True
            break
        e_last = e_tot
        ci0 = fcivec
        if istep == casscf.max_cycle_macro:
            break

        t0 = log.init_timer()
        s = build_rotation_matrix(casscf, mo, terms, denom_floor, max_stepsize)
        mo = mo @ expm(s)
        timing['orbital_rotation_wall'] += log.timer_silent(*t0)[2] * 1e-3
        if dump_chk and casscf.chkfile:
            chk_env = locals().copy()
            chk_env['mo'] = cupy.asnumpy(mo)
            chk_env['casdm1'] = cupy.asnumpy(casdm1)
            if casscf.chk_ci:
                chk_env['fcivec'] = cupy.asnumpy(fcivec)
            casscf.dump_chk(chk_env)

    if conv:
        log.info('Diagonal-Hessian CASSCF converged in %3d steps', istep)
    else:
        log.info('Diagonal-Hessian CASSCF not converged in %3d steps', istep)
    if dump_chk and casscf.chkfile:
        chk_env = locals().copy()
        chk_env['mo'] = cupy.asnumpy(mo)
        chk_env['casdm1'] = cupy.asnumpy(casdm1)
        if casscf.chk_ci:
            chk_env['fcivec'] = cupy.asnumpy(fcivec)
        casscf.dump_chk(chk_env)
    log.timer('Diagonal-Hessian CASSCF', *cput0)
    return conv, e_tot, e_cas, fcivec, mo, None


class _CASSCF(cpu_mc1step.CASSCF):
    _keys = cpu_mc1step.CASSCF._keys.union({'denom_floor'})
    canonicalization = False
    denom_floor = 1e-8
    max_stepsize = .04

    get_h1eff = h1e_for_cas
    h1e_for_cas = h1e_for_cas
    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    def __init__(self, mf_or_mol, ncas=0, nelecas=0, ncore=None, frozen=None):
        if frozen is not None:
            raise NotImplementedError('GPU CASSCF frozen orbitals are not implemented')
        if isinstance(mf_or_mol, gto.MoleBase):
            mf_or_mol = scf.RHF(mf_or_mol)
        elif (hasattr(mf_or_mol, 'istype') and
              any(mf_or_mol.istype(x) for x in ('UHF', 'ROHF', 'GHF'))):
            raise NotImplementedError(
                'GPU CASSCF supports restricted HF/KS objects only')
        elif not getattr(mf_or_mol, '__module__', '').startswith('gpu4pyscf'):
            if not isinstance(mf_or_mol, cpu_scf.hf.RHF):
                raise NotImplementedError(
                    'GPU CASSCF supports restricted HF/KS objects only')
            mf_or_mol = mf_or_mol.to_gpu()

        if not isinstance(mf_or_mol, hf.RHF):
            raise NotImplementedError(
                'GPU CASSCF supports restricted HF/KS objects only')

        super().__init__(mf_or_mol, ncas, nelecas, ncore, frozen)
        fcisolver = gpu_direct_spin1.FCISolver(self.mol)
        fcisolver.__dict__.update(self.fcisolver.__dict__)
        self.fcisolver = fcisolver
        self.canonicalization = False

    def ao2mo(self, mo_coeff=None, hcore=None):
        raise NotImplementedError('CASSCF integral backend is not configured')

    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        max_memory = max(400, self.max_memory - lib.current_memory()[0])
        e_tot, fcivec = self.fcisolver.kernel(
            eris.h1eff, eris.aaaa, self.ncas, self.nelecas,
            ci0=ci0, verbose=verbose, max_memory=max_memory,
            ecore=eris.ecore)
        return e_tot, e_tot - eris.ecore, fcivec

    def kernel(self, mo_coeff=None, ci0=None, callback=None):
        if self.canonicalization:
            raise NotImplementedError('GPU CASSCF canonicalization is not implemented')
        if self.natorb:
            raise NotImplementedError('GPU CASSCF natural orbitals are not implemented')
        self.timing = {}
        wall0 = logger.perf_counter()

        if mo_coeff is None:
            if self.mo_coeff is None and self._scf.mol.nelectron > 0:
                self._scf.run()
                self.mo_coeff = self._scf.mo_coeff
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None:
            ci0 = self.ci
        if callback is None:
            callback = self.callback

        self.check_sanity()
        self.dump_flags()
        (self.converged, self.e_tot, self.e_cas, self.ci,
         self.mo_coeff, self.mo_energy) = kernel(
             self, mo_coeff, tol=self.conv_tol,
             conv_tol_grad=self.conv_tol_grad, ci0=ci0,
             callback=callback, verbose=self.verbose)
        logger.note(self, 'CASSCF energy = %#.15g', self.e_tot)
        self._finalize()

        total_wall = logger.perf_counter() - wall0
        accounted_wall = sum(self.timing[key] for key in (
            'hcore_wall', 'ao2mo_wall', 'fci_wall', 'rdm_wall',
            'orbital_derivatives_wall', 'orbital_rotation_wall'))
        self.timing['total_wall'] = total_wall
        self.timing['other_wall'] = max(0., total_wall - accounted_wall)
        log = logger.new_logger(self)
        log.debug(
            'CASSCF timing: total %.3f s in %d cycles; hcore %.3f s; '
            'AO2MO %.3f s; FCI %.3f s; RDM %.3f s; orbital derivatives '
            '%.3f s; orbital rotation %.3f s; other %.3f s',
            total_wall, self.timing['macro_cycles'],
            self.timing['hcore_wall'],
            self.timing['ao2mo_wall'], self.timing['fci_wall'],
            self.timing['rdm_wall'],
            self.timing['orbital_derivatives_wall'],
            self.timing['orbital_rotation_wall'], self.timing['other_wall'])
        return (self.e_tot, self.e_cas, self.ci, self.mo_coeff,
                self.mo_energy)
