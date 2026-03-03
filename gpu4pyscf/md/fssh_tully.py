# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
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
from gpu4pyscf.md.fssh import FSSH, PES

class FSSH_Tully(FSSH):

    '''
    Implementation of Tully's seminal non-adiabatic models for Fewest Switches Surface Hopping (FSSH).
    
    Supported Models:
    - 'sac': Simple Avoided Crossing
    - 'dac': Dual Avoided Crossing
    - 'ecr': Extended Coupling Region

    Args:
    - model (str): The Tully model to use. Must be one of 'sac', 'dac', or 'ecr'.
    - mass (float): The mass of the particle in the system.
    '''

    def __init__(self, model, mass):
        super().__init__(pyscf.M(), [0,1])

        self.mass = np.array(mass).reshape(-1)
        self.model = model.lower()

    def _adiabatic_transform(self, V11, V22, V12, dV11, dV22, dV12):

        '''
        Transforms diabatic potential energy matrix elements and their derivatives 
        into adiabatic quantities (energies, forces, and NACVs).

        Args:
            Vii, dVii: Diabatic potentials and gradients (diagonal elements).
            V12, dV12: Diabatic coupling and gradient (off-diagonal elements).

        Returns:
            energy: Adiabatic eigenvalues [E1, E2].
            force:  Adiabatic forces [F1, F2], where F = -dE/dx.
            d12:    Non-adiabatic coupling vector (NACV) <1|d/dx|2>.
        '''

        v_bar = (V11 + V22) * 0.5
        v_diff = (V11 - V22) * 0.5
        e_sqrt = np.sqrt(v_diff**2 + V12**2 + 1e-16)
        energy = np.array([v_bar - e_sqrt, v_bar + e_sqrt])

        g_bar = (dV11 + dV22) * 0.5
        g_diff = (dV11 - dV22) * 0.5
        g_sqrt = (v_diff * g_diff + V12 * dV12) / e_sqrt
        force = np.array([-(g_bar - g_sqrt), -(g_bar + g_sqrt)])

        d12 = (V12 * g_diff - v_diff * dV12) / (2.0 * e_sqrt**2)
        
        return energy, force, d12

    def evaluate_pes(self, position, cur_state, with_nacv=True):
        '''
        Evaluates the Potential Energy Surface (PES) properties at a given nuclear position.
        
        Args:
            position: Nuclear coordinates (array-like).
            cur_state: Index of the current electronic state for force return.
            
        Returns:
            PES object containing adiabatic energies, force on the current state, and the NACV matrix.
        '''

        x = np.asarray(position).item()

        if self.model == 'sac':

            A = 0.01
            B = 1.6
            C = 0.005
            D = 1.0

            V11 = A * (1- np.exp(-B * np.abs(x))) * np.sign(x)
            V22 = -V11
            V12 = C * np.exp(-D * x**2)
            dV11 = A * B * np.exp(-B * np.abs(x))
            dV22 = -dV11
            dV12 = -2 * C * D * x * np.exp(-D * x**2)

        elif self.model == 'dac':

            A = 0.1
            B = 0.28
            C = 0.015
            D = 0.06
            E0 = 0.05

            V11 = 0
            V22 = -A * np.exp(-B * x**2) + E0
            V12 = C * np.exp(-D * x**2)
            dV11 = 0
            dV22 = 2 * A * B * x * np.exp(-B * x**2)
            dV12 = -2 * C * D * x * np.exp(-D * x**2)

        elif self.model == 'ecr':

            B = 0.9
            C = 0.1
            E0 = -0.0006

            V11 = E0
            V22 = -E0
            V12 = C * (np.exp(B * x) * np.heaviside(-x,0) + (2 - np.exp(-B * x))* np.heaviside(x,0))

            dV11 = 0
            dV22 = 0
            dV12 = C * B * np.exp(B * x) * np.heaviside(-x,0) + C * B * np.exp(-B * x) * np.heaviside(x,0)

        else:
            raise ValueError('model must be sac, dac or ecr')
        
        E, F, D12 = self._adiabatic_transform(V11, V22, V12, dV11, dV22, dV12)
        
        
        D = np.array([[0.0, D12.item()],
                      [-D12.item(), 0.0]])

        return PES(energy=E, force=F[cur_state], nacv=D.reshape(2,2,1,1))
