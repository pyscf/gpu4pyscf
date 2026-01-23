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


def _get_g_osc(td):
    e_exc = td.e
    q = td.transition_quadrupole()
    trace = q[:, 0, 0] + q[:, 1, 1] + q[:, 2, 2]
    q = 0.5*(3 * q - trace[:, None, None]*np.eye(3)[None,]) # 1/2 
    
    q_xx = q[:, 0, 0]
    q_yy = q[:, 1, 1]
    q_zz = q[:, 2, 2]
    q_xy = q[:, 0, 1]
    q_xz = q[:, 0, 2]
    q_yz = q[:, 1, 2]
    coeff_m1 = 2 / np.sqrt(3)
    coeff_m2 = 1 / np.sqrt(3)

    q20 = q_zz
    q21c  = coeff_m1 * q_xz
    q21s = coeff_m1 * q_yz
    q22c = coeff_m2 * (q_xx - q_yy)
    q22s = coeff_m1 * q_xy

    q_sum_sq_cartesian = q20**2 + q21c**2 + q22c**2 + q21s**2 + q22s**2
    g_osc = (2.0 / 5.0) * e_exc * q_sum_sq_cartesian.real
    
    return g_osc


def calc_c8(td_a, td_b, n_grid=20):
    """
    Calculate the C8 dispersion coefficient using TDDFT polarizabilities.
    """
    log = logger.new_logger(td_a)
    log.info('\n' + '*' * 40)
    log.info('GPU4PySCF C8 Calculation (Isotropic)')
    log.info('*' * 40)
    
    x, w_leg = np.polynomial.legendre.leggauss(n_grid)
    w0 = 0.5
    freqs_im = w0 * (1 + x) / (1 - x)
    weights = w_leg * w0 * 2 / ((1 - x)**2)
    
    log.info('Solving for System A...')
    _solve_full_spectrum(td_a)
    f_osc_a = td_a.oscillator_strength() 
    g_osc_a = _get_g_osc(td_a)           
    e_exc_a = td_a.e
    
    log.info('Solving for System B...')
    _solve_full_spectrum(td_b)
    f_osc_b = td_b.oscillator_strength()  
    g_osc_b = _get_g_osc(td_b)            
    e_exc_b = td_b.e
    
    denom_a = e_exc_a[:, None]**2 + freqs_im[None, :]**2
    alpha1_a = np.sum(f_osc_a[:, None] / denom_a, axis=0)
    alpha2_a = np.sum(g_osc_a[:, None] / denom_a, axis=0)
    print(f"System A static alpha1: {alpha1_a[0]:.4f}")
    print(f"System A static alpha2: {alpha2_a[0]:.4f}")
    
    denom_b = e_exc_b[:, None]**2 + freqs_im[None, :]**2
    alpha1_b = np.sum(f_osc_b[:, None] / denom_b, axis=0)
    alpha2_b = np.sum(g_osc_b[:, None] / denom_b, axis=0)
    
    integrand = (alpha1_a * alpha2_b + alpha2_a * alpha1_b)
    c8_val = (15.0 / (2.0 * np.pi)) * np.sum(integrand * weights)
    
    log.info(f'Calculated C8 coefficient: {c8_val:.6f} a.u.')
    log.info('*' * 40 + '\n')
    
    return float(c8_val.real)