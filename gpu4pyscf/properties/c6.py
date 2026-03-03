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
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.tdscf import rhf as tdhf_gpu
from gpu4pyscf.tdscf import rks as tdrks_gpu


def _solve_full_spectrum(td):
    log = logger.new_logger(td)
    mf = td._scf
    log.info('Constructing A and B matrices (GPU)...')
    a_mat, b_mat = td.get_ab(mf)
    
    # Ensure matrices are real for RKS/RHF to avoid numerical noise creating complex components
    a_mat = np.asarray(a_mat).real
    b_mat = np.asarray(b_mat).real

    nocc, nvir = a_mat.shape[:2]
    nov = nocc * nvir

    a_mat = a_mat.reshape(nov, nov)
    b_mat = b_mat.reshape(nov, nov)
    
    is_tda = isinstance(td, (tdhf_gpu.TDA, tdrks_gpu.TDA))
    
    e_exc = None
    xy_vectors = []

    if is_tda:
        log.info('Solving full TDA eigenvalue problem...')
        w, x_mat = np.linalg.eigh(a_mat)
        
        for i in range(len(w)):
            # TDA normalization
            x_vec = x_mat[:, i].reshape(nocc, nvir)
            x_vec *= np.sqrt(0.5)
            xy_vectors.append((x_vec, 0))
            
        e_exc = w

    else:
        log.info('Solving full Casida eigenvalue problem...')
        try:
            amb = a_mat - b_mat
            apb = a_mat + b_mat
            
            l_mat = np.linalg.cholesky(amb)
            
            # M = L.T * (A+B) * L
            # M * v = w^2 * v
            h_eff = np.dot(l_mat.T, np.dot(apb, l_mat))
            w2, v = np.linalg.eigh(h_eff)
            
            mask = w2 > 1e-6
            w_pos = np.sqrt(w2[mask])
            v_pos = v[:, mask]
            
            # D = (L^T)^-1 * v
            # Z = (1/w) * L * v 
            d_vecs = np.linalg.solve(l_mat.T, v_pos)
            z_vecs = np.dot(l_mat, v_pos) / w_pos[None, :]
            
            x_all = 0.5 * (z_vecs + d_vecs)
            y_all = 0.5 * (z_vecs - d_vecs)
            
            for i in range(len(w_pos)):
                x = x_all[:, i]
                y = y_all[:, i]
                
                norm_x = np.linalg.norm(x)
                norm_y = np.linalg.norm(y)
                norm_diff = norm_x**2 - norm_y**2
                
                if abs(norm_diff) < 1e-9:
                    scale = 1.0
                else:
                    scale = np.sqrt(0.5 / abs(norm_diff))
                
                x_vec = (x * scale).reshape(nocc, nvir)
                y_vec = (y * scale).reshape(nocc, nvir)
                
                xy_vectors.append((x_vec, y_vec))
            
            e_exc = w_pos

        except np.linalg.LinAlgError:
            log.warn('Ground state unstable (A-B not positive definite). Fallback to non-symmetric diagonalization.')
            
            h_mat = np.empty((2 * nov, 2 * nov), dtype=a_mat.dtype)
            h_mat[:nov, :nov] = a_mat
            h_mat[:nov, nov:] = b_mat
            h_mat[nov:, :nov] = -b_mat.conj()
            h_mat[nov:, nov:] = -a_mat.conj()
            
            w, v = np.linalg.eig(h_mat)
            
            sorted_indices = np.argsort(w.real)
            w = w[sorted_indices]
            v = v[:, sorted_indices]
            
            mask = w.real > 1e-3
            w_pos = w[mask]
            v_pos = v[:, mask]
            
            for i in range(len(w_pos)):
                xy_vec_c = v_pos[:, i]
                idx_max = np.argmax(np.abs(xy_vec_c))
                phase = np.angle(xy_vec_c[idx_max])
                xy_vec = (xy_vec_c * np.exp(-1j * phase)).real
                
                x = xy_vec[:nov]
                y = xy_vec[nov:]
                
                # Normalize: X^2 - Y^2 = 0.5 (PySCF convention for RHF/RKS)
                norm_x = np.linalg.norm(x)
                norm_y = np.linalg.norm(y)
                norm_diff = norm_x**2 - norm_y**2
                
                if abs(norm_diff) < 1e-9:
                    scale = 1.0
                else:
                    scale = np.sqrt(0.5 / abs(norm_diff))
                
                x_vec = (x * scale).reshape(nocc, nvir)
                y_vec = (y * scale).reshape(nocc, nvir)
                
                xy_vectors.append((x_vec, y_vec))
            
            e_exc = w_pos.real

    td.e = e_exc
    td.xy = xy_vectors
    td.converged = [True] * len(e_exc)
    
    return td

def calc_c6(td_a, td_b, n_grid=20):
    log = logger.new_logger(td_a)
    log.info('\n' + '*' * 40)
    log.info('GPU4PySCF C6 Calculation (Full Spectrum)')
    log.info('*' * 40)
    
    x, w_leg = np.polynomial.legendre.leggauss(n_grid)
    w0 = 0.5  # TODO: hard coded
    freqs_im = w0 * (1 + x) / (1 - x)
    weights = w_leg * w0 * 2 / ((1 - x)**2)
    
    log.info('Solving for System A')
    _solve_full_spectrum(td_a)
    f_osc_a = td_a.oscillator_strength()
    e_exc_a = td_a.e
    
    log.info('Solving for System B')
    _solve_full_spectrum(td_b)
    f_osc_b = td_b.oscillator_strength() # in length gauge
    e_exc_b = td_b.e
    
    # alpha(iw) = sum_I f_I / (w_I^2 + w^2)
    denom_a = e_exc_a[:, None]**2 + freqs_im[None, :]**2
    alpha_a = np.sum(f_osc_a[:, None] / denom_a, axis=0)
    
    denom_b = e_exc_b[:, None]**2 + freqs_im[None, :]**2
    alpha_b = np.sum(f_osc_b[:, None] / denom_b, axis=0)
    
    integrand = alpha_a * alpha_b
    c6_val = (3.0 / np.pi) * np.sum(integrand * weights)
    
    log.info(f'Calculated C6 coefficient: {c6_val:.6f} a.u.')
    log.info('*' * 40 + '\n')
    
    return float(c6_val.real)