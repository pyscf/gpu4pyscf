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

"""Periodic finite-field SCF mixin using the ordinary KRKS driver."""
import warnings
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib import logger
from .field_state import FieldState
from .operator import orbital_gradient


def apply(mf, field=(0., 0., 0.), solver='scf', singular_tol=1e-10,
          orbital_optimizer=None):
    from gpu4pyscf.pbc.dft.krks import KRKS
    if not isinstance(mf, KRKS):
        raise TypeError("Periodic external_field requires a GPU KRKS instance")
    if not mf.istype('KRKS') or mf.istype('KUKS') or mf.cell.spin != 0:
        raise NotImplementedError('Only nonmagnetic closed-shell KRKS is supported')
    if getattr(mf, 'smearing_method', None) is not None:
        raise NotImplementedError('Smearing is unsupported')
    if solver not in ('scf', 'cg'):
        raise ValueError('Use solver=scf/cg')
    from ..berry.kpt_string import KPointMesh
    KPointMesh(mf.cell, mf.kpts)
    base = mf.undo_external_field() if isinstance(mf, ExternalField) else mf
    return lib.set_class(ExternalFieldSCF(base, field, solver, singular_tol,
                                         orbital_optimizer),
                         (ExternalFieldSCF, base.__class__))


class ExternalField:
    __name_mixin__ = 'ExternalField'


class ExternalFieldSCF(ExternalField):
    _keys = {'field', 'solver', 'singular_tol', 'conv_tol_dm',
             'orbital_optimizer', 'max_stepsize', 'e_ks', 'e_field'}

    def __init__(self, mf, field, solver, singular_tol, orbital_optimizer):
        self.__dict__.update(mf.__dict__)
        self._scf = mf.copy()
        self.field = np.asarray(field, dtype=float)
        if self.field.shape != (3,) or not np.all(np.isfinite(self.field)):
            raise ValueError('field must be a finite Cartesian vector')
        self.solver = solver
        self.singular_tol = singular_tol
        self.orbital_optimizer = orbital_optimizer
        self.conv_tol_grad = getattr(mf, 'conv_tol_grad', None) or np.sqrt(mf.conv_tol)
        self.conv_tol_dm = max(np.sqrt(mf.conv_tol), 1e-7)
        self.max_stepsize = .15
        self._field_state = self._functional = None
        self._branch = self._initial_branch = None
        self.history = []
        self.converged = False
        self.e_ks = self.e_field = None
        self._field_running = False
        self.time_reversal_symmetry = False
        self._user_check_convergence = getattr(mf, "check_convergence", None)

    def undo_external_field(self):
        """Return the saved zero-field snapshot, discarding subsequent field
        results and parameter changes made on this wrapper.
        """
        return self._scf.copy()

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, 'Periodic electric field (a.u.) = %s; solver = %s',
                    self.field, self.solver)
        return self

    def reset(self, cell=None):
        super().reset(cell)
        self._scf.reset(cell)
        if self._field_state is not None:
            self._field_state.reset()
        self._field_state = self._functional = None
        self._branch = None
        self.__dict__.pop('ao_links', None)
        self.converged = False
        return self

    def _initialize_field(self, coeff, occ):

        occ = cp.asnumpy(occ)
        if occ.ndim != 2 or not np.all((abs(occ) < 1e-7) | (abs(occ-2) < 1e-7)):
            raise ValueError('Only integer closed-shell occupations are supported')
        counts = (occ > 1).sum(axis=1)
        if not counts[0] or np.any(counts != counts[0]):
            raise ValueError('Occupied band count must be positive and constant')
        self.nocc = int(counts[0])
        c = cp.asarray(coeff, dtype=cp.complex128)
        c = cp.stack([c[k][:, np.r_[np.flatnonzero(occ[k] > 1),
                                     np.flatnonzero(occ[k] < 1)]] for k in range(len(c))])
        self.mo_occ = cp.zeros((len(c), c.shape[-1]))
        self.mo_occ[:, :self.nocc] = 2
        self.mo_coeff = c
        self.overlap = cp.asarray(super().get_ovlp())
        se, su = cp.linalg.eigh(self.overlap)
        if float(cp.asnumpy(se.min())) < 1e-9:
            raise ValueError('Near-linear-dependent AO basis; finite field requires a nonsingular metric')
        self.x = su / cp.sqrt(se)[:, None, :]
        self.hcore = cp.asarray(super().get_hcore())
        if self._field_state is None:
            self._field_state = FieldState(self.cell, self.kpts, self.field,
                singular_tol=self.singular_tol, ao_links=getattr(self, 'ao_links', None))
        self._functional = self._field_state
        if getattr(self, '_restart_path', None) is not None:
            self._field_state.load_branch(self._restart_path)
            self._restart_path = None
        if self._field_state.branch_nocc not in (None, self.nocc):
            raise ValueError('Branch state occupied band count does not match')
        self._initial_branch = self._field_state.branch
        energies = cp.asnumpy(self._scf.mo_energy) if self._scf.mo_energy is not None else None
        if energies is not None and self.nocc < energies.shape[-1]:
            gap = energies[:, self.nocc:].min()-energies[:, :self.nocc].max()
            drops = self._field_state.topology.kmesh * abs(self._field_state.lattice @ self.field)
            if gap <= 0 or np.any(drops >= gap):
                raise ValueError('Zero-field state is metallic or field exceeds discrete Zener scale')
        return c

    def evaluate_orbitals(self, coeff, reference=None, with_operator=False):
        """Energy and vo gradient callback for an external orbital optimizer.

        Trial evaluation does not commit a Berry branch or diagonalize H_E.
        Return (enthalpy, KS energy, density, KS Fock, field result, vo gradient).
        """
        dm = super().make_rdm1(coeff, self.mo_occ)
        v = super().get_veff(self.cell, dm)
        f = self.hcore + v
        e_ks = float(super().energy_tot(dm, self.hcore, v))
        result = self._field_state.evaluate(coeff[:, :, :self.nocc], self.overlap,
            reference=reference, with_hamiltonian=with_operator)
        g = orbital_gradient(coeff, self.nocc, f, result.gradient)
        return e_ks+result.enthalpy, e_ks, dm, f, result, g

    @staticmethod
    def reference_state(state):
        return state[4].phases

    def _evaluate(self, c, reference=None, hamiltonian=True):
        return self.evaluate_orbitals(c, reference, hamiltonian)

    def _accept(self, coeff, state):
        self.mo_coeff = coeff
        self.e_tot, self.e_ks, dm, f, self._result, g = state
        self.e_field = self._result.enthalpy
        self._field_state.accept(self._result)
        self._branch = self._result.phases
        self.mo_energy = cp.real(cp.diagonal(
            coeff.conj().swapaxes(-1, -2) @ f @ coeff, axis1=-2, axis2=-1))
        self.scf_summary.update(e_ks=self.e_ks, e_field=self.e_field)
        self.cycles = len(self.history)

    def kernel(self, dm0=None, mo_coeff=None, mo_occ=None, **kwargs):
        self.history = []
        self.converged = False
        if mo_coeff is None:
            if not self._scf.converged:
                self._scf.kernel(dm0=dm0)
            if not self._scf.converged:
                raise RuntimeError('Zero-field bootstrap did not converge')
            mo_coeff, mo_occ = self._scf.mo_coeff, self._scf.mo_occ
        elif mo_occ is None:
            raise ValueError('mo_occ is required with mo_coeff')
        if self.max_cycle < 1 and np.any(self.field):
            raise ValueError('max_cycle must be positive for nonzero field')
        c = self._initialize_field(mo_coeff, mo_occ)
        state = self.evaluate_orbitals(c, self._initial_branch, True)
        self._accept(c, state)
        if not np.any(self.field):
            self.converged = self._scf.converged
            return self.e_tot
        if self.solver == 'cg' or self.orbital_optimizer is not None:
            from .optimizer import minimize, LineSearchError
            optimizer = self.orbital_optimizer or minimize
            try:
                c, state = optimizer(self, c)
            except LineSearchError as error:
                self._accept(error.coeff, error.state)
                raise
            self._accept(c, state)
        else:
            # Reuse the GPU KRKS driver, DIIS, logging,
            # density construction, checkpoints and callback lifecycle.
            callback = self.callback
            self.callback = lambda env: self._scf_callback(env, callback)
            self._field_running = True
            self._last_dm = super().make_rdm1(c, self.mo_occ)
            self._density_norm = 0.
            conv_check = self.conv_check
            try:
                self.conv_check = False
                super().scf(dm0=self._last_dm, **kwargs)
            finally:
                self.callback = callback
                self._field_running = False
                self.conv_check = conv_check
            final = self.evaluate_orbitals(self.mo_coeff, self._branch, True)
            self._accept(self.mo_coeff, final)
            self.converged = bool(self.converged and self.history and
                abs(self.history[-1]['delta_e']) < self.conv_tol and
                self.history[-1]['gradient_norm'] < self.conv_tol_grad and
                self.history[-1]['density_norm'] < self.conv_tol_dm)
        if not self.converged:
            warnings.warn(f'Finite-field {self.solver} did not converge in {self.max_cycle} cycles')
        return self.e_tot

    scf = kernel

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        dm = super().make_rdm1(mo_coeff, mo_occ, **kwargs)
        if self._field_running and mo_coeff is not None:
            self.mo_coeff, self.mo_occ = mo_coeff, mo_occ
            self._density_norm = float(cp.asnumpy(cp.linalg.norm(dm-self._last_dm)))
            self._last_dm = dm
            result = self._field_state.evaluate(mo_coeff[:, :, :self.nocc], self.overlap,
                                                reference=self._branch)
            self._field_state.accept(result)
            self._result, self._branch = result, result.phases
        return dm

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, *args, **kwargs):
        if self._field_state is None:
            return super().get_fock(h1e, s1e, vhf, dm, *args, **kwargs)
        if h1e is None:
            h1e = self.hcore
        if self._result.hamiltonian is None:
            from .operator import hamiltonian
            self._result.hamiltonian = hamiltonian(
                self.mo_coeff[:, :, :self.nocc], self.overlap, self._result.gradient)
        # Pass H_E as a one-electron addition to base DIIS/damping machinery.
        return super().get_fock(h1e+self._result.hamiltonian, s1e, vhf, dm,
                                *args, **kwargs)

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        e = float(super().energy_tot(dm, h1e, vhf))
        if self._field_state is not None:
            self.e_ks, self.e_field = e, self._result.enthalpy
            return e+self.e_field
        return e

    def eig(self, h, s, **kwargs):
        if not self._field_running:
            return super().eig(h, s, **kwargs)
        f = self.x.conj().swapaxes(-1, -2) @ h @ self.x
        e, u = cp.linalg.eigh((f+f.conj().swapaxes(-1, -2))*.5)
        return e, self.x @ u

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if not self._field_running:
            return super().get_grad(mo_coeff, mo_occ, fock)
        if fock is None:
            fock = self.get_fock()
        g = 4/len(mo_coeff) * mo_coeff[:, :, self.nocc:].conj().swapaxes(-1, -2) @ (fock @ mo_coeff[:, :, :self.nocc])
        # GPU base drivers do not all expose check_convergence. Augment their
        # residual with density convergence, conservatively using the same norm.
        density = self._density_norm*self.conv_tol_grad/self.conv_tol_dm
        return cp.concatenate((g.ravel(), cp.asarray([density])))

    def check_convergence(self, env):
        user_check = self._user_check_convergence
        return ((user_check is None or user_check(env)) and
                abs(env['e_tot']-env['last_hf_e']) < self.conv_tol and
                float(cp.asnumpy(env['norm_gorb'])) < self.conv_tol_grad and
                float(cp.asnumpy(env['norm_ddm'])) < self.conv_tol_dm)

    def _scf_callback(self, env, callback):
        f = self.hcore + env['vhf']
        g = orbital_gradient(self.mo_coeff, self.nocc, f, self._result.gradient)
        row = dict(cycle=env['cycle']+1, enthalpy=float(env['e_tot']),
                   e_ks=self.e_ks, delta_e=float(env['e_tot']-env['last_hf_e']),
                   gradient_norm=float(cp.asnumpy(cp.linalg.norm(g))),
                   density_norm=float(cp.asnumpy(env['norm_ddm'])),
                   polarization=self.polarization().tolist())
        self.history.append(row)
        if callback:
            callback(env)
        if callable(getattr(self, 'field_callback', None)):
            self.field_callback(row)

    def _record(self, cycle, state, old):
        row = dict(cycle=cycle+1, enthalpy=float(state[0]), e_ks=state[1],
                   delta_e=float(state[0]-old[0]),
                   gradient_norm=float(cp.asnumpy(cp.linalg.norm(state[5]))),
                   density_norm=float(cp.asnumpy(cp.linalg.norm(state[2]-old[2]))),
                   polarization=state[4].polarization.tolist())
        self.history.append(row)
        logger.info(self, 'finite-field cg cycle %d H=%.12g |g|=%.3g |dD|=%.3g',
                    cycle+1, row['enthalpy'], row['gradient_norm'], row['density_norm'])
        if self.callback:
            self.callback(row)

    def _is_converged(self, state, old):
        return (abs(state[0]-old[0]) < self.conv_tol and
                float(cp.asnumpy(cp.linalg.norm(state[5]))) < self.conv_tol_grad and
                float(cp.asnumpy(cp.linalg.norm(state[2]-old[2]))) < self.conv_tol_dm)

    def to_gpu(self):
        raise NotImplementedError("Convert the ordinary mean field before apply")

    def to_cpu(self):
        raise NotImplementedError("Convert the ordinary mean field before apply")

    def get_bands(self, *args, **kwargs):
        # Interpolating H_E to arbitrary new k points is nontrivial: it depends
        # on occupied orbitals and neighboring Berry links, not just density.
        raise NotImplementedError("Finite-field arbitrary k-point bands are unsupported")

    def as_scanner(self):
        raise NotImplementedError("Finite-field geometry scanning needs link-cache reset")

    def Hessian(self):
        raise NotImplementedError("Finite-field Hessian is not implemented")

    def polarization(self):
        if self._branch is None:
            raise RuntimeError('Run kernel before requesting polarization')
        return self._result.polarization.copy()

    def save_branch(self, path):
        self._field_state.save_branch(path, self.nocc)

    def load_branch(self, path):
        self._restart_path = path
        # Validate identity immediately without computing geometry links.
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text())
        for key, actual in (('lattice', self.cell.lattice_vectors()),
                            ('kpts', cp.asnumpy(self.kpts)), ('atoms', self.cell.atom_coords()),
                            ('charges', self.cell.atom_charges())):
            stored = np.asarray(data[key])
            if stored.shape != actual.shape or not np.allclose(stored, actual, rtol=0, atol=1e-10):
                raise ValueError(f'Branch state {key} does not match this calculation')
        return self

    def Gradients(self):
        raise NotImplementedError('Finite-field forces require Berry-link Pulay terms')
    nuc_grad_method = Gradients

    def newton(self):
        raise NotImplementedError('Ordinary SOSCF lacks the Berry Hessian; use an orbital optimizer')

    def stress(self, *args, **kwargs):
        raise NotImplementedError('Finite-field stress is not implemented')
