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
from pyscf.data.nist import BOLTZMANN, HARTREE2J, AVOGADRO, AMU2AU

def maxwell_boltzmann_velocities(masses, temperature=300.0, force_temp=False):
    '''
    Maxwell-Boltzmann distribution within a given temperature for velocity
    initialization

    masses : ndarray
        masses in g/mol unit
    temperature : float
        in Kelvin
    force_temp: bool
        rescale velocities so that the kinetic energy is exactly 3/2 N k T.

    Returns:
        An (N, 3) array for velocities in atomic units (1 au = 21.877 A/fs)
    '''
    k_b = BOLTZMANN / HARTREE2J # = 8.314/2626e3
    N = len(masses)
    masses_amu = np.array(masses) * AMU2AU
    std_dev = np.sqrt(k_b * temperature / masses_amu[:, np.newaxis])
    velocities = np.random.normal(0, std_dev, (N, 3))
    if force_temp:
        total_momentum = np.sum(velocities * masses_amu[:, np.newaxis], axis=0)
        # remove center-of-mass velocity, thereby removing 3 degrees of freedom
        velocities -= total_momentum / np.sum(masses_amu)
        degree_freedom = 3 * N - 3
        kinetic_energy = .5 * np.einsum( 'm,mx,mx->', masses_amu, velocities, velocities)
        target_energy = .5 * degree_freedom * k_b * temperature
        velocities *= np.sqrt(target_energy / kinetic_energy)
    return velocities
