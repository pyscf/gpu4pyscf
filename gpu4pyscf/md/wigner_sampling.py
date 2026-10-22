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
from pyscf.data.nist import HARTREE2WAVENUMBER, AMU2AU, BOHR, PLANCK, BOLTZMANN, LIGHT_SPEED_SI
from scipy.special import eval_laguerre

# hc/kB for a wavenumber expressed in cm^-1:
#   1 cm^-1 = 1.438776... K
WAVENUMBER2KELVIN = (100.0 * PLANCK * LIGHT_SPEED_SI / BOLTZMANN)


def _sample_vibrational_state(freq_cm, temp, rng, max_pop=0.9999, max_nlevel=300):
    """
    Sample vibrational quantum number using the Boltzmann distribution.

    The distribution is truncated once its cumulative population
    reaches max_pop, and the retained states are sampled with the
    corresponding conditional distribution.
    """
    if freq_cm <= 0:
        raise ValueError(f"Frequency must be positive, got {freq_cm} cm^-1")

    if temp < 0:
        raise ValueError(f"Temperature must be non-negative, got {temp} K")

    if temp == 0:
        return 0
    
    exponent = freq_cm * WAVENUMBER2KELVIN / temp
    z = np.exp(-exponent)
    probs = []
    cumulative = 0.0
    p_n = 1.0 - z

    while cumulative < max_pop:
        probs.append(p_n)
        cumulative += p_n
        p_n *= z

    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()

    n = int(rng.choice(len(probs), p=probs))
    if n > max_nlevel:
        n = max_nlevel

    return n


def wignerfunc(freq_cm, temp, rng=None, max_pop=0.9999, max_nlevel=300, max_trials=10000):
    """
    Wigner sampling for one harmonic vibrational mode.

    Parameters
    ----------
    freq_cm: float                         Vibrational frequency in cm^-1.
    temp: float                            Temperature in K.
    rng: numpy.random.Generator, optional  Random-number generator.
    max_pop: float                         Cumulative Boltzmann population threshold.
    max_nlevel: int                        Maximum vibrational quantum number.
    max_trials: int                        Safety limit for rejection sampling.

    Returns
    -------
    q, p: float                            Dimensionless normal coordinate and momentum.
    """

    tol = 1e-12

    if rng is None:
        rng = np.random.default_rng()

    n = _sample_vibrational_state(freq_cm, temp, rng, max_pop=max_pop, max_nlevel=max_nlevel)

    for _ in range(max_trials):

        q = rng.uniform(-5.0, 5.0)
        p = rng.uniform(-5.0, 5.0)

        if n == 0:
            w = np.exp(-(q**2 + p**2))
        else:
            rho2 = 2.0 * (q**2 + p**2)
            w = ((-1.0)**n * eval_laguerre(n, rho2) * np.exp(-0.5 * rho2))

        if not np.isfinite(w):
            continue

        if w < -tol or w > 1.0 + tol:
            continue

        w = np.clip(w, 0.0, 1.0)

        if rng.random() < w:
            return float(q), float(p)

    raise RuntimeError(f"Wigner rejection sampling failed after {max_trials} "
                       f"trials for freq={freq_cm} cm^-1, T={temp} K, n={n}")


def wigner(temp, freqs, xyz, vib, rng=None):
    """
    Generate one Wigner initial condition.

    Important
    ---------
    This implementation assumes vib has shape(nmodes, natoms, 3)
    and uses the same convention as the original implementation:
    the Cartesian modes are mass-normalized such that the atom-mass
    dependence is already contained in vib.
    """
    if rng is None:
        rng = np.random.default_rng()

    omega = freqs / HARTREE2WAVENUMBER

    freqs = np.asarray(freqs, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    vib = np.asarray(vib, dtype=float)
    nfreq = len(freqs)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (natoms, 3), got {xyz.shape}")

    if vib.shape != (nfreq, len(xyz), 3):
        raise ValueError(f"vib must have shape ({nfreq}, {len(xyz)}, 3), got {vib.shape}")

    if np.any(freqs <= 0):
        raise ValueError("All sampled vibrational frequencies must be positive")

    q_p = np.array([wignerfunc(freq, temp, rng=rng) for freq in freqs])
    Q = q_p[:, 0]
    P = q_p[:, 1]
    q_scale = (Q / np.sqrt(omega * AMU2AU))
    p_scale = (P * np.sqrt(omega / AMU2AU))
    dq = np.einsum('i,iad->ad', q_scale, vib)
    velocity = np.einsum('i,iad->ad', p_scale, vib)
    position = xyz + dq

    return position, velocity

def wigner_samples(temp, freqs, xyz, vib, samples, seed=None, min_freq=10.0, min_dis=None,):
    """
    Generate Wigner initial conditions using the harmonic-oscillator sampling procedure.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")

    freqs = np.asarray(freqs, dtype=float)
    vib = np.asarray(vib, dtype=float)
    xyz = np.asarray(xyz, dtype=float)

    mask = freqs >= min_freq
    freqs_used = freqs[mask]
    vib_used = vib[mask]

    if len(freqs_used) == 0:
        raise ValueError(f"No vibrational modes >= {min_freq} cm^-1")

    rng = np.random.default_rng(seed)
    result = []

    while len(result) < samples:
        pos, vel = wigner(temp, freqs_used, xyz, vib_used, rng=rng)
        if min_dis is not None:
            dis = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
            pair_dis = dis[np.tril_indices(len(pos), -1)]
            if pair_dis.min() <= min_dis:
                continue
        result.append((pos, vel))

    return result
