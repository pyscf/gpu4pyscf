# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
from scipy.special import laguerre
from pyscf.data.nist import HARTREE2WAVENUMBER, AMU2AU, BOHR, PLANCK, BOLTZMANN, LIGHT_SPEED_SI

wavenumber2Kelvin = 100 * (PLANCK * LIGHT_SPEED_SI) / BOLTZMANN

def wignerfunc(mu, temp, max_pop=0.9999, max_nlevel=150, seed=None):
    ex = mu / temp * wavenumber2Kelvin  # vibrational temperature: ex=h*c*mu/(kb*T), 0.69503 convert cm-1 to K
    pop = []

    while np.sum(pop) < max_pop and len(pop) < max_nlevel:
        _pop = float(np.exp(-1 * ex * len(pop)) * (1 - np.exp(-1 * ex)))
        pop.append(_pop)
    pop.append(1 - np.sum(pop))

    if seed is not None:
        np.random.seed(seed)

    while True:
        random_state = np.random.choice(len(pop), p=pop)  # random generate a state
        q = np.random.uniform(0, 1) * 10.0 - 5.0
        p = np.random.uniform(0, 1) * 10.0 - 5.0
        rho2 = 2 * (q ** 2 + p ** 2)
        w = (-1) ** random_state * laguerre(random_state)(rho2) * np.exp(-0.5 * rho2)
        if np.random.uniform(0, 1) < w < 1:
            break
    return float(q), float(p)


def wigner(temp, freqs, xyz, vib, seed=None):
    nfreq = len(freqs)
    natom = len(xyz)

    mu_to_hartree = 1./HARTREE2WAVENUMBER  # 1 cm-1  = h*c/Eh = 4.55633518e-6 au
    ma_to_amu = AMU2AU  # 1 g/mol = 1/Na*me*1000 = 1822.88852 amu

    q_p = np.array([wignerfunc(i, temp, seed=seed) for i in freqs])  # generates update coordinates and momenta pairs Q and P

    q = q_p[:, 0].reshape((nfreq, 1))  # first column is Q

    q *= 1 / np.sqrt(freqs[:,None] * mu_to_hartree * ma_to_amu)  # convert coordinates from m to Bohr
    qvib = np.array([np.ones((natom, 3)) * i for i in q])  # generate identity array to expand Q
    qvib = np.sum(vib * qvib, axis=0)  # sum sampled structure over all modes
    newc = (xyz + qvib)  # cartesian coordinates in Bohr

    p = q_p[:, 1].reshape((nfreq, 1))  # second column is P
    p *= np.sqrt(freqs[:,None] * mu_to_hartree / ma_to_amu)  # convert velocity from m/s to Bohr/au
    pvib = np.array([np.ones((natom, 3)) * i for i in p])  # generate identity array to expand P
    velo = np.sum(vib * pvib, axis=0)  # sum sampled velocity over all modes in Bohr/au

    return newc, velo

def wigner_samples(temp, freqs, xyz, vib, samples, seed=None):
    '''
    Wigner sampling to produce initial conditions for MD simulations.
    Ref. https://pubs.rsc.org/en/content/articlelanding/2023/cp/d3cp01007d

    Args:
        temp:
            Temperature in Kelvin
        freqs:
            Vibrational freqencies in cm-1
        xyz:
            Coordinates in atomic unit
        vib:
            Normal modes in atomic unit
        samples:
            Number of initial samples to generate

    Returns:
        A list of (positions, velocities)
    '''
    if seed is not None:
        np.random.seed(seed)
    valid = []
    temperature = 300
    while len(valid) < samples:
        initcond = wigner(temperature, freqs, xyz, vib)
        p, v = initcond
        distance = np.linalg.norm(p[:,None] - p, axis=-1)
        distance = distance[np.tril_indices(len(p), -1)]
        # filter out unreasonable geometries
        if distance.min() > 0.7:
            valid.append(initcond)
    return valid
