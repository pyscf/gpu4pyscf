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

"""Electric-enthalpy CG with branch-aware trials and orbital rotations."""
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import expm
from gpu4pyscf.lib.cupy_helper import contract

def adjoint(a):
    return a.conj().swapaxes(-1, -2)


# TODO: Reuse shared rotation helpers from pbc/scf/soscf.py when available.
# Keep local implementations while that module is being developed.
def update_rotate_matrix(dx, mo_occ, u0=None):
    """Exponentiate virtual/occupied rotations for closed-shell k-point orbitals."""
    nkpts, nmo = mo_occ.shape
    omask = mo_occ > 0
    vmask = mo_occ == 0
    dr = cp.zeros((nkpts, nmo, nmo), dtype=dx.dtype)
    dr[vmask[:,:,None] & omask[:,None]] = dx
    dr = dr - dr.conj().transpose(0,2,1)
    for k in range(nkpts):
        dr[k] = expm(dr[k])
    if u0 is not None:
        dr = contract('kpq,kqr->kpr', u0, dr)
    return dr


def rotate_mo(mo_coeff, u):
    """Apply a unitary orbital rotation independently at each k-point."""
    return contract('kpq,kqr->kpr', mo_coeff, u)


class LineSearchError(RuntimeError):
    """Failed search carrying the last accepted orbitals and functional state."""
    def __init__(self,coeff,state):
        super().__init__('Orbital CG line search failed; inspect the gradient and trial energies')
        self.coeff,self.state=coeff,state


def rotation(c, direction, step, nocc):
    mo_occ = cp.zeros((len(c), c.shape[-1]))
    mo_occ[:, :nocc] = 2
    u = update_rotate_matrix((step*direction).ravel(), mo_occ)
    return rotate_mo(c, u), u


def minimize(controller,c):
    """Minimize through an energy/gradient callback, without diagonalization.

    controller supplies evaluate_orbitals, reference_state, convergence and
    iteration callbacks. No Berry links or field operators are imported.
    """
    nocc = controller.nocc
    state = controller.evaluate_orbitals(c, reference=getattr(controller, "_initial_branch", None))
    previous_g = previous_z = previous_d = None
    for cycle in range(controller.max_cycle):
        old = state
        g = old[5]
        fmo = adjoint(c) @ old[3] @ c
        eps = cp.real(cp.diagonal(fmo,axis1=-2,axis2=-1))
        denom = cp.maximum(eps[:,nocc:,None]-eps[:,None,:nocc],.05)
        z = g / (4/len(c)*denom)
        direction = -z
        if previous_g is not None:
            numerator = float(cp.asnumpy(cp.vdot(g,z-previous_z).real))
            denominator = float(cp.asnumpy(cp.vdot(previous_g,previous_z).real))
            beta = max(0.,min(numerator/max(denominator,1e-30),1.))
            direction += beta*previous_d
        slope = float(cp.asnumpy(cp.vdot(g,direction).real))
        if slope >= 0:
            direction = -z
            slope = -float(cp.asnumpy(cp.vdot(g,z).real))
        maxnorm = float(cp.asnumpy(cp.max(cp.abs(direction)))) if direction.size else 0.
        step = min(1., controller.max_stepsize/max(maxnorm,1e-30))
        accepted = False
        for attempt in range(25):
            trial,u = rotation(c,direction,step,nocc)
            try:
                candidate = controller.evaluate_orbitals(trial, controller.reference_state(old))
            except (ValueError,np.linalg.LinAlgError):
                step *= .5
                continue
            # Roundoff floor for extensive KS energy; never use conv_tol as
            # line-search slack, which would permit physical uphill steps.
            slack = 8*np.finfo(float).eps*max(1.,abs(old[0]))
            if candidate[0] <= old[0] + 1e-4*step*slope + slack:
                accepted = True
                break
            step *= .5
        if not accepted:
            raise LineSearchError(c,state)
        # Transport tangent matrices into the new local orbital coordinates.
        def transport(v):
            a = cp.zeros_like(u)
            a[:,nocc:,:nocc] = v
            a[:,:nocc,nocc:] = -adjoint(v)
            return (adjoint(u) @ a @ u)[:,nocc:,:nocc]
        previous_g,previous_z,previous_d = transport(g),transport(z),transport(direction)
        c,state = trial,candidate
        controller._record(cycle,state,old)
        if controller._is_converged(state,old):
            controller.converged = True
            break
    return c,state
