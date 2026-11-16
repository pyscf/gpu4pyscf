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

"""Occupied links, field gradients, Hermitian operator and electric enthalpy."""
from dataclasses import dataclass
import numpy as np
import cupy as cp

@dataclass
class LinkGradient:
    links: tuple
    inverse: tuple
    dual_forward: tuple
    dual_backward: tuple
    gradient: object

def evaluate(state, coeff):
    """Return M, inverse M, Q+/- and G_E for occupied coefficients.

    The covariant action G_E contributes to F C. For closed-shell orbitals,
    dH = 4/nk Re sum(dC.conj() * G_E). No Hamiltonian is constructed.
    """

    c = cp.asarray(coeff)
    links, inverse, forward, backward = [], [], [], []
    gradient = cp.zeros_like(c, dtype=cp.complex128)
    for d in range(3):
        t = cp.asarray(state.ao_links[d])
        n, prev = state.neighbors[d], state.previous[d]
        y = t @ c[n]
        m = c.conj().swapaxes(-1, -2) @ y
        minimum = float(cp.asnumpy(cp.linalg.svd(m, compute_uv=False).min()))
        if not np.isfinite(minimum) or minimum < state.singular_tol:
            raise np.linalg.LinAlgError(
                f'Near-singular occupied link in direction {d}: {minimum:.3e}')
        inv = cp.linalg.solve(m, cp.broadcast_to(cp.eye(c.shape[-1]), m.shape))
        qp = y @ inv
        ym = t[prev].conj().swapaxes(-1, -2) @ c[prev]
        qm = ym @ inv[prev].conj().swapaxes(-1, -2)
        if d in state.active:
            gradient += state.gamma[d] * (qm - qp)
        links.append(m)
        inverse.append(inv)
        forward.append(qp)
        backward.append(qm)
    return LinkGradient(tuple(links), tuple(inverse), tuple(forward),
                        tuple(backward), gradient)

def orbital_gradient(coeff, nocc, fock, field_gradient):
    """Full electric-enthalpy derivative in occupied/virtual rotation space."""
    return 4 / len(coeff) * (coeff[:, :, nocc:].conj().swapaxes(-1, -2) @
                            (fock @ coeff[:, :, :nocc] + field_gradient))

def hamiltonian(coeff, overlap, gradient):
    """Hermitian H_E satisfying H_E C_occ = G_E on the tangent space."""
    sc = overlap @ coeff
    return gradient @ sc.conj().swapaxes(-1, -2) + sc @ gradient.conj().swapaxes(-1, -2)

def enthalpy(cell, field, polarization):
    """Return -Omega E.P in Hartree, with E and P in atomic units."""
    return -cell.vol * float(field @ polarization)
