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

"""Topology, branch continuity and geometry-cache lifetime for finite fields."""
from dataclasses import dataclass
import numpy as np
import cupy as cp
from ..berry.kpt_string import KPointMesh
from ..berry.berry_phase import berry_phase
from ..berry.polarization import continuous_phases
from .operator import (evaluate as evaluate_gradient,
                       hamiltonian as field_hamiltonian,
                       enthalpy as field_enthalpy)

@dataclass
class FieldResult:
    phases: tuple
    electronic: np.ndarray
    ionic: np.ndarray
    polarization: np.ndarray
    enthalpy: float
    gradient: object
    hamiltonian: object
    links: tuple
    dual_forward: tuple
    dual_backward: tuple


class FieldState:
    """Geometry-only AO links are cached; orbital-dependent links are rebuilt.

    Supply ao_links explicitly for model tests. GPU Fourier overlaps use
    the same convention as existing Berry postprocessing.
    """
    def __init__(self, cell, kpts, field, kmesh=None, ao_links=None,
                 singular_tol=1e-10):
        self.cell = cell
        self.topology = KPointMesh(cell, cp.asnumpy(kpts), kmesh)
        self.field = np.array(field, dtype=float)
        if self.field.shape != (3,) or not np.all(np.isfinite(self.field)):
            raise ValueError('field must be a finite Cartesian vector in Hartree atomic units')
        if cell.dimension != 3 or cell.charge != 0:
            raise NotImplementedError('Only neutral three-dimensional cells are supported')
        self.lattice = np.asarray(cell.lattice_vectors())
        self.singular_tol = singular_tol
        self.neighbors = tuple(self.topology.neighbors(d)[0] for d in range(3))
        self.previous = tuple(np.argsort(n) for n in self.neighbors)
        self.gamma = -1j * self.topology.kmesh * (self.lattice @ self.field) / (4*np.pi)
        self.active = np.flatnonzero(np.abs(self.lattice @ self.field) > 1e-15)
        self.ao_links = cp.asarray(ao_links) if ao_links is not None else self._build_links()
        self.branch = self.result = None
        self.branch_nocc = None

    def _build_links(self):
        from gpu4pyscf.pbc.df import ft_ao
        from ..berry.overlap import _commensurate_bvk_mesh
        top = self.topology
        steps = np.array([top.reciprocal_step(d) for d in range(3)])
        opt = ft_ao.FTOpt(self.cell, _commensurate_bvk_mesh(self.cell, top.kpts, top.kmesh))
        opt.permutation_symmetry = False
        raw = opt.gen_ft_kernel()(-steps, q=np.zeros(3), kpts=top.kpts)
        return raw.conj().swapaxes(-1, -2).swapaxes(0, 1)

    def evaluate(self, coeff, overlap, reference=None, spin_factor=2,
                 with_hamiltonian=True):
        if self.ao_links is None:
            self.ao_links = self._build_links()

        c = cp.asarray(coeff)
        s = cp.asarray(overlap)
        nk, nao, nocc = c.shape
        if nk != len(self.topology.kpts) or s.shape != (nk, nao, nao) or nocc == 0:
            raise ValueError('Invalid occupied coefficients or overlap shape')
        orth_error = float(cp.asnumpy(cp.max(cp.abs(c.conj().swapaxes(-1, -2) @ s @ c - cp.eye(nocc)))))
        if orth_error > 1e-7:
            raise ValueError(f'Occupied orbitals are not S-orthonormal: {orth_error:.3g}')
        data = evaluate_gradient(self, c)
        phases = []
        for d, m in enumerate(data.links):
            principal = cp.asnumpy(berry_phase(m, self.topology.strings(d)))
            phase = continuous_phases(principal, self.topology.kmesh, d,
                                      None if reference is None else reference[d])
            phases.append(phase)
        electronic = spin_factor/(2*np.pi*self.cell.vol) * np.array([p.mean() for p in phases]) @ self.lattice
        ionic = self.cell.atom_charges() @ self.cell.atom_coords() / self.cell.vol
        polarization = electronic + ionic
        energy = field_enthalpy(self.cell, self.field, polarization)
        h = field_hamiltonian(c, s, data.gradient) if with_hamiltonian else None
        return FieldResult(tuple(phases), electronic, ionic, polarization, energy,
                           data.gradient, h, data.links, data.dual_forward,
                           data.dual_backward)

    def accept(self, result):
        """Commit only accepted orbitals; trial evaluation never moves branch."""
        self.branch = tuple(p.copy() for p in result.phases)
        self.result = result

    def reset(self):
        """Release geometry links and branch when geometry or k mesh changes."""
        self.ao_links = None
        self.branch = self.result = None

    def save_branch(self, path, nocc):
        if self.branch is None:
            raise RuntimeError('Run kernel before saving a branch')
        import json
        from pathlib import Path
        data = self.identity()
        data.update(nocc=nocc, phases=[p.tolist() for p in self.branch])
        Path(path).write_text(json.dumps(data, indent=2) + '\n')

    def identity(self):
        return dict(lattice=self.cell.lattice_vectors().tolist(),
                    kpts=self.topology.kpts.tolist(),
                    atoms=self.cell.atom_coords().tolist(),
                    charges=self.cell.atom_charges().tolist())

    def load_branch(self, path):
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text())
        for key, actual in self.identity().items():
            stored, actual = np.asarray(data[key]), np.asarray(actual)
            if stored.shape != actual.shape or not np.allclose(
                    stored, actual, rtol=0, atol=1e-10):
                raise ValueError(f'Branch state {key} does not match this calculation')
        self.branch = tuple(np.asarray(p) for p in data['phases'])
        self.branch_nocc = data['nocc']
        return self
