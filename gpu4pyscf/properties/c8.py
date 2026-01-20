# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
from gpu4pyscf.lib import logger
from gpu4pyscf.properties.c6 import _solve_full_spectrum


import numpy as np
import cupy as cp
from gpu4pyscf.lib import logger

import numpy as np

# def _get_g_osc(td):
#     e_exc = td.e
#     q = td.transition_quadrupole()
    
#     q_xx = q[:, 0, 0]
#     q_yy = q[:, 1, 1]
#     q_zz = q[:, 2, 2]
#     q_xy = q[:, 0, 1]
#     q_xz = q[:, 0, 2]
#     q_yz = q[:, 1, 2]
    

#     q_sum_sq_cartesian = (q_xx**2 + q_yy**2 + q_zz**2 + 
#                           q_xy**2 + q_xz**2 + q_yz**2)
    
#     g_osc = (2.0 / 6.0) * e_exc * q_sum_sq_cartesian
    
#     return g_osc

# def _get_g_osc(td):
#     e_exc = td.e
#     q = td.transition_quadrupole()

#     # trace = q[:, 0, 0] + q[:, 1, 1] + q[:, 2, 2]
#     # q = (3*q - trace[:, None, None]*np.eye(3)[None,])
    
#     q_sum = 0.0
#     for i in range(3):
#         for j in range(3):
#             for k in range(3):
#                 for l in range(3):
#                     q_sum += q[:, i, j] * q[:, k, l]
    
#     g_osc = (2.0 / 6.0) * e_exc * q_sum
    
#     return g_osc

def _get_g_osc(td):
    e_exc = td.e
    q = td.transition_quadrupole()
    # trace = q[:, 0, 0] + q[:, 1, 1] + q[:, 2, 2]
    # q = (3 * q - trace[:, None, None]*np.eye(3)[None,])
    
    # q_xx = q[:, 0, 0]
    # q_yy = q[:, 1, 1]
    # q_zz = q[:, 2, 2]
    # q_xy = q[:, 0, 1]
    # q_xz = q[:, 0, 2]
    # q_yz = q[:, 1, 2]
    # coeff_m1 = np.sqrt(15 / (8 * np.pi))
    # coeff_m2 = 0.25 * np.sqrt(15 / (2 * np.pi))

    # q20 = 0.25 * np.sqrt(5 / np.pi) * (2 * q_zz - q_xx - q_yy)
    # q21  = -coeff_m1 * (q_xz + 1j * q_yz)
    # q21_m = coeff_m1 * (q_xz - 1j * q_yz)
    # q22  = coeff_m2 * ( (q_xx - q_yy) + 2j * q_xy )
    # q22_m = coeff_m2 * ( (q_xx - q_yy) - 2j * q_xy )
    # q_sum_sq_cartesian = q20*q20.conj() + q21*q21.conj() + q22*q22.conj() + q21_m*q21_m.conj() + q22_m*q22_m.conj()

    # q20 = q_zz 
    # q21c = 2/np.sqrt(3) * q_xz
    # q21s = 2/np.sqrt(3) * q_yz
    # q22c = 1/np.sqrt(3) * (q_xx - q_yy)
    # q22s = 2/np.sqrt(3) * q_xy

    # q_sum_sq_cartesian = q20*q20.conj() + q21c*q21c.conj() + q22c*q22c.conj() + q21s*q21s.conj() + q22s*q22s.conj()

    g_osc = (2.0 / 5.0) * e_exc * q_sum_sq_cartesian.real#*4*np.pi/3
    
    return g_osc

# def _get_g_osc(td):
#     e_exc = td.e
#     q = td.transition_quadrupole()

#     trace = q[:, 0, 0] + q[:, 1, 1] + q[:, 2, 2]
#     q = (3*q - trace[:, None, None]*np.eye(3)[None,])*0.5
    
#     q_xx = q[:, 0, 0]
#     q_yy = q[:, 1, 1]
#     q_zz = q[:, 2, 2]
#     q_xy = q[:, 0, 1]
#     q_xz = q[:, 0, 2]
#     q_yz = q[:, 1, 2]
    
#     q_sum_sq_cartesian = (q_xx*q_xx + q_yy*q_yy + q_zz*q_zz + 
#                           2*q_xy*q_xy + 2*q_xz*q_xz + 2*q_yz*q_yz )
    
#     g_osc = (2.0 / 5.0) * e_exc * q_sum_sq_cartesian
    
#     return g_osc

# def _get_g_osc(td):
#     e_exc = td.e
#     q = td.transition_quadrupole()

#     trace = q[:, 0, 0] + q[:, 1, 1] + q[:, 2, 2]
#     q = (3*q - trace[:, None, None]*np.eye(3)[None,])
    
#     q_xx = q[:, 0, 0]
#     q_yy = q[:, 1, 1]
#     q_zz = q[:, 2, 2]
#     q_xy = q[:, 0, 1]
#     q_xz = q[:, 0, 2]
#     q_yz = q[:, 1, 2]
    
#     q_sum_sq_cartesian = (q_xx*q_xx + q_yy*q_yy + q_zz*q_zz + 
#                           q_xy*q_xy*2 + q_xz*q_xz*2 + q_yz*q_yz*2 - 1/3*(q_xx*q_yy + q_xx*q_zz + q_yy*q_zz))
    
#     g_osc = (2.0 / 5.0) * e_exc * q_sum_sq_cartesian
    
#     return g_osc

# def _get_g_osc(td):
#     e_exc = td.e
#     q = td.transition_quadrupole() 
    
#     trace = q[:, 0, 0] + q[:, 1, 1] + q[:, 2, 2]

#     eye3 = np.eye(3)[None, :, :]
#     theta = 0.5 * (3.0 * q - trace[:, None, None] * eye3)

#     q_sum_sq = np.sum(theta**2, axis=(1, 2))
#     g_osc = (2.0 / 3.0) * e_exc * q_sum_sq
    
#     return g_osc


# def _get_g_osc(td):
#     e_exc = td.e
#     q = td.transition_quadrupole()
    
#     # m = 0: (1/2) * (3z^2 - r^2) -> 0.5 * (2*Qzz - Qxx - Qyy)
#     q0 = 0.5 * (2 * q[:, 2, 2] - q[:, 0, 0] - q[:, 1, 1])
    
#     # m = 1c, 1s: sqrt(3)*xz, sqrt(3)*yz
#     q1c = np.sqrt(3.0) * q[:, 0, 2]
#     q1s = np.sqrt(3.0) * q[:, 1, 2]
    
#     # m = 2c: (sqrt(3)/2) * (x^2 - y^2) -> (sqrt(3)/2) * (Qxx - Qyy)
#     q2c = (np.sqrt(3.0) / 2.0) * (q[:, 0, 0] - q[:, 1, 1])
    
#     # m = 2s: sqrt(3)*xy
#     q2s = np.sqrt(3.0) * q[:, 0, 1]
    
#     q_sum_sq_sph = q0**2 + q1c**2 + q1s**2 + q2c**2 + q2s**2
    
#     g_osc = (2.0 / 5.0) * e_exc * q_sum_sq_sph
    
#     return g_osc

def calc_c8(td_a, td_b, n_grid=20):
    """
    Calculate the C8 dispersion coefficient using TDDFT polarizabilities.
    """
    log = logger.new_logger(td_a)
    log.info('\n' + '*' * 40)
    log.info('GPU4PySCF C8 Calculation (Isotropic)')
    log.info('*' * 40)
    
    # 1. Frequency grid setup (Gauss-Legendre quadrature)
    x, w_leg = np.polynomial.legendre.leggauss(n_grid)
    w0 = 0.5  # Scaling parameter for the semi-infinite interval
    freqs_im = w0 * (1 + x) / (1 - x)
    weights = w_leg * w0 * 2 / ((1 - x)**2)
    
    # 2. Solve full spectrum for System A
    log.info('Solving for System A...')
    _solve_full_spectrum(td_a)
    f_osc_a = td_a.oscillator_strength()  # Dipole (L=1)
    g_osc_a = _get_g_osc(td_a)            # Quadrupole (L=2)
    e_exc_a = td_a.e
    
    # 3. Solve full spectrum for System B
    log.info('Solving for System B...')
    _solve_full_spectrum(td_b)
    f_osc_b = td_b.oscillator_strength()  # Dipole (L=1)
    g_osc_b = _get_g_osc(td_b)            # Quadrupole (L=2)
    e_exc_b = td_b.e
    
    # 4. Construct dynamic polarizabilities alpha(iw)
    # alpha(iw) = sum_I [ oscillator_strength / (omega_I^2 + w^2) ]
    denom_a = e_exc_a[:, None]**2 + freqs_im[None, :]**2
    alpha1_a = np.sum(f_osc_a[:, None] / denom_a, axis=0)
    alpha2_a = np.sum(g_osc_a[:, None] / denom_a, axis=0)
    print(f"System A static alpha1: {alpha1_a[0]:.4f}")
    print(f"System A static alpha2: {alpha2_a[0]:.4f}")
    
    denom_b = e_exc_b[:, None]**2 + freqs_im[None, :]**2
    alpha1_b = np.sum(f_osc_b[:, None] / denom_b, axis=0)
    alpha2_b = np.sum(g_osc_b[:, None] / denom_b, axis=0)
    
    # 5. Casimir-Polder integration
    # C8 = (15/2pi) * integral_0^inf [a1_a*a2_b + a2_a*a1_b] dw
    integrand = (alpha1_a * alpha2_b + alpha2_a * alpha1_b)
    c8_val = (15.0 / (2.0 * np.pi)) * np.sum(integrand * weights)
    
    log.info(f'Calculated C8 coefficient: {c8_val:.6f} a.u.')
    log.info('*' * 40 + '\n')
    
    return float(c8_val.real)