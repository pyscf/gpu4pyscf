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
from pyscf.data.nist import BOLTZMANN, AVOGADRO

def maxwell_boltzmann_velocities(masses, temperature=300.0):
    '''
    Maxwell-Boltzmann distribution within a given temperature for velocity
    initialization

    Returns:
        An (N, 3) array for velocities in Angstrom/fs
    '''
    k_b = (BOLTZMANN * AVOGADRO) * 1e-3 # 8.314e-3
    N = len(masses)
    masses_kg = masses * 1e-3
    std_dev = np.sqrt(k_b * temperature / masses_kg[:, np.newaxis])
    velocities = np.random.normal(0, std_dev, (N, 3))
    total_momentum = np.sum(velocities * masses_kg[:, np.newaxis], axis=0)
    velocities -= total_momentum / np.sum(masses_kg)
    kinetic_energy = 0.5 * np.sum(masses_kg * np.sum(velocities**2, axis=1))
    target_energy = 1.5 * N * k_b * temperature
    velocities *= np.sqrt(target_energy / kinetic_energy)
    # Convert unit to Angstrom/fs
    velocities *= 0.01
    return velocities
