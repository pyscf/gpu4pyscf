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
from gpu4pyscf import lib as gpu_lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import eigh, asarray
import types

"""
References:
1. 10.1021/ct300442w
2. 10.1007/s00894-020-04571-6
"""

def estimate_eigenvalues_gershgorin(h):
    """
    Estimate the maximum and minimum eigenvalues using Gershgorin disk theorem.
    
    Args:
        h (cp.ndarray): Hamiltonian matrix
    
    Returns:
        tuple: (e_max, e_min), estimated maximum and minimum eigenvalues
    """
    diag = cp.diag(h)
    row_sums = cp.sum(cp.abs(h), axis=1) - cp.abs(diag)

    e_min_est = cp.min(diag - row_sums)
    e_max_est = cp.max(diag + row_sums)
    return float(e_max_est.get()), float(e_min_est.get())


def get_eigenvalues_diag(h):
    """
    Get exact maximum and minimum eigenvalues using diagonalization.
    
    Args:
        h (cp.ndarray): Hamiltonian matrix
    
    Returns:
        tuple: (e_max, e_min), exact maximum and minimum eigenvalues
    """
    e, _ = eigh(h)
    return float(e[-1].get()), float(e[0].get())


class SP2Purification:
    
    def __init__(self, mol, conv_tol=1e-10, max_cycle=100,
                 eig_method='gershgorin'):
        """
        Initialize SP2 purification.
        
        Args:
            mol: PySCF molecule object
            conv_tol (float): Convergence tolerance for idempotency error,
                               default 1e-10 (This can be modified)
            max_cycle (int): Maximum number of purification cycles
            eig_method (str): Method for estimating eigenvalues,
                               'gershgorin' or 'diag'
        """
        self.mol = mol
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.eig_method = eig_method
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self._nocc = None
    
    @property
    def nocc(self):
        """Number of occupied orbitals"""
        if self._nocc is None:
            self._nocc = self.mol.nelectron // 2
        return self._nocc
    
    @nocc.setter
    def nocc(self, value):
        self._nocc = value
    
    def _scale_hamiltonian(self, h):
        """
        Scale the Hamiltonian matrix to [0, 1] range for SP2 algorithm.
        
        Args:
            h (cp.ndarray): Hamiltonian matrix
        
        Returns:
            cp.ndarray: Scaled matrix X0
        """
        if self.eig_method == 'diag':
            self.warn("Diagonalization method is not recommended for SP2 purification.")
            e_max, e_min = get_eigenvalues_diag(h)
        else:
            e_max, e_min = estimate_eigenvalues_gershgorin(h)
        
        if e_max == e_min:
            raise ValueError("All eigenvalues are the same")
        
        n = h.shape[0]
        identity = cp.eye(n, dtype=h.dtype)
        x0 = (e_max * identity - h) / (e_max - e_min)
        return x0
    
    def purify(self, h, nocc=None):
        """
        Purify the density matrix using the SP2 algorithm.
        
        Args:
            h (cp.ndarray): Hamiltonian matrix
            nocc (int): Number of occupied orbitals
        
        Returns:
            cp.ndarray: Purified density matrix
        """
        if nocc is None:
            nocc = self.nocc
        
        log = logger.new_logger(self, self.verbose)
        
        h = h.astype(cp.float64)
        
        log.info('\n')
        log.info('******** SP2 Density Matrix Purification ********')
        log.info('Convergence tolerance = %g', self.conv_tol)
        log.info('Maximum cycles = %d', self.max_cycle)
        log.info('Eigenvalue estimation method = %s', self.eig_method)
        
        x = self._scale_hamiltonian(h)
        ne = 2.0 * nocc
        trace_x = float(cp.trace(x).get())
        
        log.debug('Initial trace = %g', trace_x)
        
        idemp_history = []

        x_sq = cp.empty_like(x)
        x_tmp = cp.empty_like(x)
        
        for cycle in range(self.max_cycle):
            #* Only ONE matrix multiplication per cycle
            cp.matmul(x, x, out=x_sq)
            cp.subtract(x, x_sq, out=x_tmp)
            
            trace_tmp = float(cp.trace(x_tmp).get())

            abs_trace_tmp = abs(trace_tmp)
            idemp_history.append(abs_trace_tmp)
            
            # Polynomial selection (19)
            if abs(2.0 * trace_x - 2.0 * trace_tmp - ne) <= abs(2.0 * trace_x + 2.0 * trace_tmp - ne):
                cp.copyto(x, x_sq)
                trace_x = trace_x - trace_tmp
            else:
                x += x_tmp
                trace_x = trace_x + trace_tmp
            cp.add(x, x.T, out=x_tmp)
            cp.multiply(x_tmp, 0.5, out=x) # stable the result
            
            # Check convergence
            if cycle >= 2:
                idemp_error = abs_trace_tmp
                
                log.debug('Cycle %d: trace = %g, idempotency error = %g',
                           cycle + 1, trace_x, idemp_error)
                
                if idemp_error < self.conv_tol:
                    log.info('SP2 converged in %d cycles', cycle + 1)
                    break
                
                # Relative convergence criterion based on (18)
                if idemp_history[-1] >= idemp_history[-3] and idemp_error < 1e-4:
                    log.info('SP2 stopped (error stabilized) in %d cycles', cycle + 1)
                    break
        else:
            log.warn('SP2 did not converge in %d cycles', self.max_cycle)
        
        # Final formal idempotency check
        # TODO: can be removed or changed based on trace
        final_idemp_error = float(cp.linalg.norm(x @ x - x, ord='fro').get())
        
        log.info('Final trace = %g (target = %d)', trace_x, ne / 2)
        log.info('Final idempotency error = %g', final_idemp_error)
        
        # Verify electron count
        electron_error = abs(2.0 * trace_x - ne)
        if electron_error > 1e-6:
            log.warn('Electron count error = %g', electron_error)
        
        density = 2.0 * x
        
        return density


def purify(mf, conv_tol=1e-10, max_cycle=100, eig_method='gershgorin'):
    """
    Apply SP2 purification to an SCF object.
    
    This function replaces the entire SCF kernel with a custom SP2-driven SCF loop,
    completely bypassing any molecular orbital (MO) diagonalization routines.
    
    Args:
        mf: SCF object
        conv_tol, max_cycle, eig_method: SP2 parameters
    
    Returns:
        Modified SCF object with SP2 purification kernel
    """
    
    # Create SP2 purification instance and store it
    sp2 = SP2Purification(mf.mol, conv_tol, max_cycle, eig_method)
    mf.with_purification = sp2

    def sp2_scf_kernel(self, dm0=None, callback=None, **kwargs):
        """
        Custom SCF kernel strictly driven by SP2 density matrix purification.
        Adapted from the semi-empirical `_kernel` to correctly handle DIIS and logic flow.
        All references to molecular orbitals (mo_coeff, mo_energy, mo_occ) are eliminated.
        """
        conv_tol = self.conv_tol
        mol = self.mol
        verbose = self.verbose
        log = logger.new_logger(self, verbose)
        t0 = t1 = log.init_timer()
        
        conv_tol_grad = getattr(self, 'conv_tol_grad', None)
        if conv_tol_grad is None:
            conv_tol_grad = conv_tol**.5
            log.info('Set gradient conv threshold to %g', conv_tol_grad)

        if dm0 is None:
            dm0 = self.get_init_guess(mol, self.init_guess)
            t1 = log.timer_debug1('generating initial guess', *t1)

        # Drop attributes like mo_coeff, mo_occ. SP2 only operates on the density matrix.
        dm0 = asarray(dm0, order='C')

        h1e = cp.asarray(self.get_hcore())
        t1 = log.timer_debug1('hcore', *t1)

        dm = asarray(dm0, order='C')
        vhf = self.get_veff(mol, dm)
        e_tot = self.energy_tot(dm, h1e, vhf)
        log.info('init E= %.15g', e_tot)
        
        t1 = log.timer('SCF initialization', *t0)
        self.converged = False

        if self.max_cycle <= 0:
            self.e_tot = e_tot
            self.mo_coeff = None
            self.mo_energy = None
            self.mo_occ = None
            
            self._sp2_dm = dm
            return self.e_tot

        if isinstance(self.diis, gpu_lib.diis.DIIS):
            mf_diis = self.diis
        elif self.diis:
            assert issubclass(self.DIIS, gpu_lib.diis.DIIS)
            mf_diis = self.DIIS(self, self.diis_file)
            mf_diis.space = self.diis_space
            mf_diis.rollback = self.diis_space_rollback
        else:
            mf_diis = None

        dump_chk = self.chkfile is not None
        if dump_chk:
            self.chkfile.save_mol(mol, self.chkfile)

        fock_last = None
        self.cycles = 0
        for cycle in range(self.max_cycle):
            t0 = log.init_timer()
            dm_last = dm
            last_hf_e = e_tot

            fock = self.get_fock(h1e, None, vhf, dm, cycle, mf_diis, fock_last=fock_last)
            t1 = log.timer_debug1('DIIS', *t0)
            
            # Pure diagonalization replaced by SP2 Purification
            dm_new = self.with_purification.purify(fock)
            
            if self.damp is not None:
                fock_last = fock
            fock = None
            t1 = log.timer_debug1('eig (SP2 purify)', *t1)

            dm = asarray(dm_new)
            vhf = self.get_veff(mol, dm, dm_last, vhf)
            t1 = log.timer_debug1('veff', *t1)

            fock = self.get_fock(h1e, None, vhf, dm) 
            e_tot = self.energy_tot(dm, h1e, vhf)
            
            grad = fock @ dm - dm @ fock
            norm_gorb = cp.linalg.norm(grad)

            norm_ddm = cp.linalg.norm(dm - dm_last)
            t1 = log.timer(f'cycle={cycle+1}', *t0)

            log.info('cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                     cycle+1, float(e_tot), float(e_tot-last_hf_e), float(norm_gorb), float(norm_ddm))

            if dump_chk:
                self.dump_chk(locals())

            if callable(callback):
                callback(locals())

            e_diff = abs(e_tot - last_hf_e)
            if (e_diff < conv_tol and norm_gorb < conv_tol_grad):
                self.converged = True
                break
        else:
            log.warn("SCF failed to converge")

        self.cycles = cycle + 1

        self.e_tot = e_tot
        self.mo_coeff = None
        self.mo_energy = None
        self.mo_occ = None
        
        self._sp2_dm = dm
        
        return self.e_tot

    mf.kernel = types.MethodType(sp2_scf_kernel, mf)
    
    # Override make_rdm1 to expose the purified density matrix
    if not hasattr(mf, '_original_make_rdm1'):
        mf._original_make_rdm1 = mf.make_rdm1
        
    def make_rdm1_sp2(self, *args, **kwargs):
        if getattr(self, '_sp2_dm', None) is not None:
            return self._sp2_dm
        return self._original_make_rdm1(*args, **kwargs)
        
    mf.make_rdm1 = types.MethodType(make_rdm1_sp2, mf)
    
    return mf


def purification(self, conv_tol=1e-10, max_cycle=100, eig_method='gershgorin'):
    """
    Enable SP2 purification for this RHF calculation.
    
    Args:
        conv_tol (float): Convergence tolerance
        max_cycle (int): Maximum number of purification cycles
        eig_method (str): 'gershgorin' or 'diag' for eigenvalue estimation
    
    Returns:
        Self with SP2 purification enabled
    """
    return purify(self, conv_tol, max_cycle, eig_method)


try:
    from gpu4pyscf.sem.scf.hf import RHF as SemRHF
    SemRHF.purification = purification
except ImportError:
    pass