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

import pyscf, gpu4pyscf
import numpy as np
import cupy as cp
import gc, sys, os, h5py
import cupyx.scipy.linalg as cpx_linalg

from concurrent.futures import ThreadPoolExecutor
from pyscf import gto, lib
from gpu4pyscf import scf
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib.cupy_helper import asarray as cuasarray

from gpu4pyscf.tdscf import parameter, math_helper, spectralib, _krylov_tools
from gpu4pyscf.tdscf.math_helper import gpu_mem_info, release_memory, get_avail_gpumem
from pyscf.data.nist import HARTREE2EV
from gpu4pyscf.lib import logger
from gpu4pyscf.df import int3c2e
from gpu4pyscf.df import int3c2e_bdiv

logger.TIMER_LEVEL = 5

contract_to_out = contract

DEBUG = True
if DEBUG:
    contract = cp.einsum


CITATION_INFO = """
Please cite the TDDFT-ris method:

    1.  Zhou, Zehao, Fabio Della Sala, and Shane M. Parker.
        Minimal auxiliary basis set approach for the electronic excitation spectra
        of organic molecules. The Journal of Physical Chemistry Letters
        14, no. 7 (2023): 1968-1976.
        (must cite)

    2.  Zhou, Zehao, and Shane M. Parker.
        Converging Time-Dependent Density Functional Theory Calculations in Five Iterations
        with Minimal Auxiliary Preconditioning. Journal of Chemical Theory and Computation
        20, no. 15 (2024): 6738-6746.
        (for efficient orbital truncation technique)

    3.  Giannone, Giulia, and Fabio Della Sala.
        Minimal auxiliary basis set for time-dependent density functional theory and
        comparison with tight-binding approximations: Application to silver nanoparticles.
        The Journal of Chemical Physics 153, no. 8 (2020).
        (TDDFT-ris is for hybrid functionals, originates from TDDFT-as with pure functional)
"""

LINEAR_EPSILON = 1e-8

class RisBase(lib.StreamObject):
    def __init__(self, mf,
                theta: float = 0.2, J_fit: str = 'sp', K_fit: str = 's', excludeHs=False,
                Ktrunc: float = 40.0, full_K_diag: bool = False, a_x: float = None, omega: float = None,
                alpha: float = None, beta: float = None, conv_tol: float = 1e-5,
                nstates: int = 5, max_iter: int = 25, extra_init=8, restart_subspace=None, spectra: bool = False,
                out_name: str = '', print_threshold: float = 0.05, gram_schmidt: bool = True,
                single: bool = True, store_Tpq_J: bool = True, store_Tpq_K: bool = False,
                tensor_in_ram: bool = False, krylov_in_ram: bool = False,
                verbose=None, citation=True):
        """
        Args:
            mf (object): Mean field object, typically obtained from a ground - state calculation.
            theta (float, optional): Global scaling factor for the fitting basis exponent.
                                The relationship is defined as `alpha = theta/R_A^2`,
                                where `alpha` is the Gaussian exponent
                                and `R_A` is tabulated semi-empirical radii for element A. Defaults to 0.2.
            J_fit (str, optional): Fitting basis for the J matrix (`iajb` integrals).
                                   's' means only one s orbital per atom,
                                   'sp' means adding one extra p orbital per non Hydrogen atom.
                                   Defaults to 'sp', becasue more accurate than s.
            K_fit (str, optional): Fitting basis for the K matrix (`ijab` and `ibja` integrals).
                                  's' means only one s orbital per atom,
                                  'sp' means adding one extra p orbital per atom.
                                   Defaults to 's', becasue 'sp' has no accuracy improvement.
            Ktrunc (float, optional): Truncation threshold for the K matrix. Orbitals are discarded if:
                                    - Occupied orbitals with energies < e_LUMO - Ktrunc
                                    - Virtual orbitals with energies > e_HOMO + Ktrunc. Defaults to 40.0.
            a_x (float, optional): Hartree-Fock component. By default, it will be assigned according
                                    to the `mf.xc` attribute.
                                    Will override the default value if provided.
            omega (float, optional): Range-separated hybrid functional parameter. By default, it will be
                                    assigned according to the `mf.xc` attribute.
                                    Will override the default value if provided.
            alpha (float, optional): Range-separated hybrid functional parameter. By default, it will be
                                    assigned according to the `mf.xc` attribute.
                                    Will override the default value if provided.
            beta (float, optional): Range-separated hybrid functional parameter. By default, it will be
                                    assigned according to the `mf.xc` attribute.
            conv_tol (float, optional): Convergence tolerance for the Davidson iteration. Defaults to 1e-3.
            nstates (int, optional): Number of excited states to be calculated. Defaults to 5.
            max_iter (int, optional): Maximum number of iterations for the Davidson iteration. Defaults to 25.
            extra_init (int, optional): Number of extra initial vectors to be used in the Davidson iteration.
                                    Defaults to 8.
            restart_subspace (int, optional): size of the Davidson/Krylov method subspace to restart the iteration.
                                    Defaults to None, which will restart according to allocatbale memory.
            spectra (bool, optional): Whether to calculate and dump the excitation spectra in G16 & Multiwfn style.
                                     Defaults to False.
            out_name (str, optional): Output file name for the excitation spectra. Defaults to ''.
            print_threshold (float, optional): Threshold for printing the transition coefficients. Defaults to 0.05.
            gram_schmidt (bool, optional): Whether to calculate the ground state. Defaults to False.
            single (bool, optional): Whether to use single precision. Defaults to True.
            store_Tpq_J (bool, optional): Whether to store Tpq_J tensors in RAM. Defaults to True.
                                   For large system, it is recommended to be True.
            store_Tpq_K (bool, optional): Whether to store Tpq_K tensors in RAM. Defaults to True.
            tensor_in_ram (bool, optional): Whether to store Tpq tensors in RAM. Defaults to False.
                                  For large system with limited GPU memory, it is recommended to be True.
            krylov_in_ram (bool, optional): Whether to store Krylov vectors in RAM. Defaults to False.
                                  For large system with limited GPU memory, it is recommended to be True.
            verbose (optional): Verbosity level of the logger. If None, it will use the verbosity of `mf`.
            citation (bool, optional): Whether to print the citation information. Defaults to True.
        """
        self.single = single

        if single:
            self.dtype = cp.dtype(cp.float32)
        else:
            self.dtype = cp.dtype(cp.float64)

        self._scf = mf
        # self.chkfile = mf.chkfile
        self.singlet = True # TODO: add R-T excitation.
        self.exclude_nlc = False # TODO: exclude nlc functional
        self.xy = None

        self.theta = theta
        self.J_fit = J_fit
        self.K_fit = K_fit

        self.Ktrunc = Ktrunc
        self._excludeHs = excludeHs
        self._full_K_diag = full_K_diag
        self.a_x = a_x
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.conv_tol = conv_tol
        self.nstates = nstates
        self.max_iter = max_iter
        self.extra_init = extra_init
        self.restart_subspace = restart_subspace
        self.mol = mf.mol
        self.spectra = spectra
        self.out_name = out_name
        self.print_threshold = print_threshold
        self.gram_schmidt = gram_schmidt

        self.verbose = verbose if verbose else mf.verbose

        self.device = mf.device
        self.converged = None
        self._store_Tpq_J = store_Tpq_J
        # self._store_Tpq_K = store_Tpq_K

        self._tensor_in_ram = tensor_in_ram
        self._krylov_in_ram = krylov_in_ram

        logger.WARN = 6
        pyscf.lib.logger.WARN=6

        self.log = logger.new_logger(verbose=self.verbose)

        ''' following attributes will be initialized in self.build() '''
        self.n_occ = None
        self.n_vir = None
        self.rest_occ = None
        self.rest_vir = None

        self.C_occ_notrunc = None
        self.C_vir_notrunc = None
        self.C_occ_Ktrunc = None
        self.C_vir_Ktrunc = None

        self.delta_hdiag = None
        self.hdiag = None
        if self.mol.cart:
            self.eri_tag = '_cart'
        else:
            self.eri_tag = '_sph'

        self.auxmol_J = None
        self.auxmol_K = None
        self.lower_inv_eri2c_J = None
        self.lower_inv_eri2c_K = None

        self.RKS = True
        self.UKS = False
        self._citation = citation


    @property
    def e_tot(self):
        '''Excited state energies'''
        return self._scf.e_tot + self.energies/HARTREE2EV

    def get_ab(self, mf=None):
        if mf is None:
            mf = self._scf
        J_fit = self.J_fit
        K_fit = self.K_fit
        theta = self.theta
        return get_ab(self, mf, J_fit, K_fit, theta, singlet=True)

    def build(self):
        log = self.log
        log.warn("TDA&TDDFT-ris is still in the experimental stage, APIs may subject to change in future releases.")

        log.info(f'nstates: {self.nstates}')
        log.info(f'N atoms:{self.mol.natm}')
        log.info(f'conv_tol: {self.conv_tol}')
        log.info(f'max_iter: {self.max_iter}')
        log.info(f'Ktrunc: {self.Ktrunc}')
        log.info(f'calculate and print UV-vis spectra info: {self.spectra}')
        log.info(gpu_mem_info('  after init of RisBase'))

        if self.spectra:
            log.info(f'spectra files will be written and their name start with: {self.out_name}')

        if self._store_Tpq_J:
            log.info(f'will calc Tia_J. In CPU RAM? {self._tensor_in_ram}')
        else:
            log.info('will calc J on-the-fly')

        log.info(f'will calc Tia_J (if full TDDFT) Tij_K Tab_K. In CPU RAM? {self._tensor_in_ram}')


        if self.a_x or self.omega or self.alpha or self.beta:
            ''' user wants to define some XC parameters '''
            if self.a_x:
                if self.a_x == 0:
                    log.info('use pure XC functional, a_x = 0')
                elif self.a_x > 0 and self.a_x < 1:
                    log.info(f'use hybrid XC functional, a_x = {self.a_x}')
                    if self.single:
                        self.a_x = cp.float32(self.a_x)
                elif self.a_x == 1:
                    log.info('use HF, a_x = 1')
                else:
                    log.info('a_x > 1, weird')

            elif self.omega and self.alpha and self.beta:
                log.info('use range-separated hybrid XC functional')
            else:
                raise ValueError('Please dounble check the XC functional parameters')
        else:
            ''' use default XC parameters
                note: the definition of a_x, α and β is kind of weird in pyscf/libxc
            '''
            log.info(f'auto detect functional: {self._scf.xc}')

            omega, alpha_libxc, hyb_libxc = self._scf._numint.rsh_and_hybrid_coeff(self._scf.xc,
                                                                                  spin=self._scf.mol.spin)
            log.info(f'omega, alpha_libxc, hyb_libxc: {omega}, {alpha_libxc}, {hyb_libxc}')

            if omega > 0:
                log.info('use range-separated hybrid XC functional')
                self.a_x = 1
                self.omega = omega
                self.alpha = hyb_libxc
                self.beta = alpha_libxc - hyb_libxc

            elif omega == 0:
                self.a_x = alpha_libxc
                if self.a_x == 0:
                    log.info('use pure XC functional, a_x = 0')
                elif self.a_x > 0 and self.a_x < 1:
                    log.info(f'use hybrid XC functional, a_x = {self.a_x}')
                elif self.a_x == 1:
                    log.info('use HF, a_x = 1')
                else:
                    log.info('a_x > 1, weird')

        log.info(f'omega: {self.omega}')
        log.info(f'alpha: {self.alpha}')
        log.info(f'beta: {self.beta}')
        log.info(f'a_x: {self.a_x}')
        log.info(f'gram_schmidt: {self.gram_schmidt}')
        log.info(f'single: {self.single}')

        if self.J_fit == self.K_fit:
            log.info(f'use same J and K fitting basis: {self.J_fit}')
        else:
            log.info(f'use different J and K fitting basis: J with {self.J_fit} and K with {self.K_fit}')


        log.info(f'cartesian or spherical electron integral: {self.eri_tag}')

        log.info(gpu_mem_info('  before process mo_coeff'))

        if self._scf.mo_coeff.ndim == 2:
            self.RKS = True
            self.UKS = False
            n_occ = int(sum(self._scf.mo_occ>0))
            n_vir = int(sum(self._scf.mo_occ==0))
            self.n_occ = n_occ
            self.n_vir = n_vir

            self.C_occ_notrunc = cuasarray(self._scf.mo_coeff[:,:n_occ], dtype=self.dtype, order='F')
            self.C_vir_notrunc = cuasarray(self._scf.mo_coeff[:,n_occ:], dtype=self.dtype, order='F')
            mo_energy = self._scf.mo_energy
            log.info(f'mo_energy.shape: {mo_energy.shape}')

            occ_ene = mo_energy[:n_occ].reshape(n_occ,1)
            vir_ene = mo_energy[n_occ:].reshape(1,n_vir)

            delta_hdiag = cp.repeat(vir_ene, n_occ, axis=0) - cp.repeat(occ_ene, n_vir, axis=1)
            if self.single:
                delta_hdiag = cuasarray(delta_hdiag, dtype=cp.float32)

            self.delta_hdiag = delta_hdiag
            self.hdiag = cuasarray(delta_hdiag.reshape(-1))

            log.info(f'n_occ = {n_occ}, E_HOMO ={occ_ene[-1,0]}')
            log.info(f'n_vir = {n_vir}, E_LOMO ={vir_ene[0,0]}')
            log.info(f'H-L gap = {(vir_ene[0,0] - occ_ene[-1,0])*HARTREE2EV:.2f} eV')

            if self.Ktrunc > 0:
                log.info(f' MO truncation in K with threshold {self.Ktrunc} eV above HOMO and below LUMO')

                trunc_tol_au = self.Ktrunc/HARTREE2EV

                homo_vir_delta_ene = delta_hdiag[-1,:]
                occ_lumo_delta_ene = delta_hdiag[:,0]

                rest_occ = int(cp.sum(occ_lumo_delta_ene <= trunc_tol_au))
                rest_vir = int(cp.sum(homo_vir_delta_ene <= trunc_tol_au))

                assert rest_occ > 0
                assert rest_vir > 0


            elif self.Ktrunc == 0:
                log.info('no MO truncation in K')
                rest_occ = n_occ
                rest_vir = n_vir


            log.info(f'rest_occ = {rest_occ}')
            log.info(f'rest_vir = {rest_vir}')

            self.C_occ_Ktrunc = cuasarray(self._scf.mo_coeff[:,n_occ-rest_occ:n_occ], dtype=self.dtype, order='F')
            self.C_vir_Ktrunc = cuasarray(self._scf.mo_coeff[:,n_occ:n_occ+rest_vir], dtype=self.dtype, order='F')

            self.rest_occ = rest_occ
            self.rest_vir = rest_vir

        elif self._scf.mo_coeff.ndim == 3:
            raise NotImplementedError('Does not support UKS method yet')
            ''' TODO UKS method '''
            self.RKS = False
            self.UKS = True
            self.n_occ_a = sum(self._scf.mo_occ[0]>0)
            self.n_vir_a = sum(self._scf.mo_occ[0]==0)
            self.n_occ_b = sum(self._scf.mo_occ[1]>0)
            self.n_vir_b = sum(self._scf.mo_occ[1]==0)
            log.info('n_occ for alpha spin = {self.n_occ_a}')
            log.info('n_vir for alpha spin = {self.n_vir_a}')
            log.info('n_occ for beta spin = {self.n_occ_b}')
            log.info('n_vir for beta spin = {self.n_vir_b}')

        auxmol_J = get_auxmol(mol=self.mol, theta=self.theta, fitting_basis=self.J_fit)
        log.info(f'n_bf in auxmol_J = {auxmol_J.nao_nr()}')
        self.auxmol_J = auxmol_J
        self.lower_inv_eri2c_J = get_eri2c_inv_lower(self.auxmol_J, omega=0)
        byte_T_ia_J = self.auxmol_J.nao_nr() * self.n_occ * self.n_vir * self.dtype.itemsize
        log.info(f'FYI, storing T_ia_J will take {byte_T_ia_J / 1024**2:.0f} MB memory')


        if self.a_x != 0:

            auxmol_K = get_auxmol(mol=self.mol, theta=self.theta, fitting_basis=self.K_fit, excludeHs=self._excludeHs)

            log.info(f'n_bf in auxmol_K = {auxmol_K.nao_nr()}')
            self.auxmol_K = auxmol_K

            self.lower_inv_eri2c_K = get_eri2c_inv_lower(auxmol_K, omega=self.omega, alpha=self.alpha, beta=self.beta)

            byte_T_ij_K = auxmol_K.nao_nr() * (self.rest_occ * (self.rest_occ +1) //2 )* self.dtype.itemsize
            byte_T_ab_K = auxmol_K.nao_nr() * (self.rest_vir * (self.rest_vir +1) //2 )* self.dtype.itemsize
            log.info(f'T_ij_K will take {byte_T_ij_K / 1024**2:.0f} MB memory')
            log.info(f'T_ab_K will take {byte_T_ab_K / 1024**2:.0f} MB memory')

            byte_T_ia_K = auxmol_K.nao_nr() * self.rest_occ * self.rest_vir * self.dtype.itemsize
            log.info(f'(if full TDDFT) T_ia_K will take {byte_T_ia_K / 1024**2:.0f} MB memory')

        log.info(gpu_mem_info('  built ris obj'))
        self.log = log

    def get_T_J(self):
        log = self.log
        log.info('==================== RIJ ====================')
        cpu0 = log.init_timer()

        T_ia_J = get_Tpq(mol=self.mol, auxmol=self.auxmol_J, lower_inv_eri2c=self.lower_inv_eri2c_J,
                        C_p=self.C_occ_notrunc, C_q=self.C_vir_notrunc, calc="J", omega=0,
                        in_ram=self._tensor_in_ram, single=self.single, log=log)

        log.timer('build T_ia_J', *cpu0)
        log.info(gpu_mem_info('after T_ia_J'))
        return T_ia_J

    def get_2T_K(self):
        log = self.log
        log.info('==================== RIK ====================')
        cpu1 = log.init_timer()

        T_ij_K, T_ab_K = get_Tpq(mol=self.mol, auxmol=self.auxmol_K, lower_inv_eri2c=self.lower_inv_eri2c_K,
                                C_p=self.C_occ_Ktrunc, C_q=self.C_vir_Ktrunc, calc='K',
                                omega=self.omega, alpha=self.alpha,beta=self.beta,
                                in_ram=self._tensor_in_ram, single=self.single,log=log)

        log.timer('get_2T_K', *cpu1)
        log.info(gpu_mem_info('after get_2T_K'))
        return T_ij_K, T_ab_K

    def get_3T_K(self):
        log = self.log
        log.info('==================== RIK ====================')
        cpu1 = log.init_timer()
        T_ia_K, T_ij_K, T_ab_K = get_Tpq(mol=self.mol, auxmol=self.auxmol_K, lower_inv_eri2c=self.lower_inv_eri2c_K,
                                C_p=self.C_occ_Ktrunc, C_q=self.C_vir_Ktrunc, calc='JK',
                                omega=self.omega, alpha=self.alpha,beta=self.beta,
                                in_ram=self._tensor_in_ram, single=self.single,log=log)

        log.timer('get_3T_K', *cpu1)
        log.info(gpu_mem_info('after get_3T_K'))
        return T_ia_K, T_ij_K, T_ab_K

    def Gradients(self):
        raise NotImplementedError

    def nuc_grad_method(self):
        return self.Gradients()

    def NAC(self):
        raise NotImplementedError

    def nac_method(self):
        return self.NAC()

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    # as_scanner = as_scanner
    def as_scanner(self):
        return as_scanner(self)

    def transition_dipole(self):
        '''
        transition dipole u
        '''
        int_r = self.mol.intor_symmetric('int1e_r' + self.eri_tag)
        int_r = cuasarray(int_r, dtype=self.dtype)
        if self.RKS:
            P = get_inter_contract_C(int_tensor=int_r, C_occ=self.C_occ_notrunc, C_vir=self.C_vir_notrunc)
        else:
            ''' TODO '''
            P_alpha = get_inter_contract_C(int_tensor=int_r, C_occ=self.C_occ[0], C_vir=self.C_vir[0])
            P_beta = get_inter_contract_C(int_tensor=int_r, C_occ=self.C_occ[1], C_vir=self.C_vir[1])
            P = cp.vstack((P_alpha, P_beta))
        return P

    def transition_magnetic_dipole(self):
        '''
        magnatic dipole m
        '''
        int_rxp = self.mol.intor('int1e_cg_irxp' + self.eri_tag, comp=3, hermi=2)
        int_rxp = cuasarray(int_rxp, dtype=self.dtype)

        if self.RKS:
            mdpol = get_inter_contract_C(int_tensor=int_rxp, C_occ=self.C_occ_notrunc, C_vir=self.C_vir_notrunc)
        else:
            ''' TODO '''
            mdpol_alpha = get_inter_contract_C(int_tensor=int_rxp, C_occ=self.C_occ[0], C_vir=self.C_vir[0])
            mdpol_beta = get_inter_contract_C(int_tensor=int_rxp, C_occ=self.C_occ[1], C_vir=self.C_vir[1])
            mdpol = cp.vstack((mdpol_alpha, mdpol_beta))
        return mdpol

    def get_nto(self,state_id, save_fch=False, save_cube=False, save_h5=False, resolution=None):

        ''' dump NTO coeff in h5 file or fch file or cube file'''
        orbo = self.C_occ_notrunc
        orbv = self.C_vir_notrunc
        nocc = self.n_occ
        nvir = self.n_vir

        log = self.log
        # X
        cis_t1 = self.xy[0][state_id-1, :].copy()
        log.info(f'state_id {state_id}')
        log.info(f'X norm {cp.linalg.norm(cis_t1):.3f}')
        # TDDFT (X,Y) has X^2-Y^2=1.
        # Renormalizing X (X^2=1) to map it to CIS coefficients
        # cis_t1 *= 1. / cp.linalg.norm(cis_t1)

        cis_t1 = cis_t1.reshape(nocc, nvir)

        nto_o, w, nto_vT = cp.linalg.svd(cis_t1)
        '''each column of nto_o and nto_v corresponds to one NTO pair
        usually the first (few) NTO pair have significant weights
        '''

        w_squared = w**2
        dominant_weight = float(w_squared[0]) # usually ~1.0
        log.info(f"Dominant NTO weight: {dominant_weight:.4f} (should be close to 1.0)")

        hole_nto = nto_o[:, 0]      # shape: (nocc,)
        particle_nto = nto_vT[0, :].T  # shape: (nvir,)

        # Phase convention: max abs coeff positive, and consistent phase between hole/particle
        if hole_nto[cp.argmax(cp.abs(hole_nto))] < 0:
            hole_nto = -hole_nto
            particle_nto = -particle_nto

        nto_hole = orbo.dot(hole_nto)    # shape: (nao,)
        nto_electron = orbv.dot(particle_nto) # shape: (nao,)

        nto_coeff = cp.hstack((nto_hole[:,None], nto_electron[:,None]))


        if save_fch:
            cpu0 = log.init_timer()
            '''save nto_coeff to fch file'''
            try:
                from mokit.lib.py2fch_direct import fchk
                from mokit.lib.rwwfn import del_dm_in_fch
            except ImportError:
                info = 'mokit is not installed. Please install mokit to save nto_coeff to fch file.'
                info += 'https://gitlab.com/jxzou/mokit'
                raise ImportError(info)
            nto_mf = self._scf.copy().to_cpu()
            nto_mf.mo_coeff = nto_coeff.get()
            nto_mf.mo_energy = cp.asarray([dominant_weight, dominant_weight]).get()

            fchfilename = f'ntopair_{state_id}.fch'
            if os.path.exists(fchfilename):
                os.remove(fchfilename)
            fchk(nto_mf, fchfilename)
            del_dm_in_fch(fchname=fchfilename,itype=1)
            log.info(f'nto_coeff saved to {fchfilename}')
            log.info('Please cite MOKIT: https://gitlab.com/jxzou/mokit')
            log.timer(' save nto_coeff', *cpu0)
        if save_h5:
            cpu0 = log.init_timer()

            '''save nto_coeff to h5 file'''
            h5filename = f'nto_coeff_{state_id}.h5'
            with h5py.File(h5filename, 'w') as f:
                f.create_dataset('nto_coeff', data=nto_coeff.get(), dtype='f4')
                f.create_dataset('dominant_weight', data=dominant_weight, dtype='f4')
                f.create_dataset('state_id', data=state_id, dtype='i4')
            log.info(f'nto_coeff saved to {h5filename}')
            log.timer(' save nto_coeff', *cpu0)

        if save_cube:
            cpu0 = log.init_timer()
            from pyscf.tools import cubegen
            '''save nto_coeff to cube file'''
            cubegen.orbital(self.mol, f'nto_{state_id}_hole.cube', nto_hole.get(), resolution=resolution)
            log.timer(' save nto_coeff hole', *cpu0)
            cpu0 = log.init_timer()
            cubegen.orbital(self.mol, f'nto_{state_id}_electron.cube', nto_electron.get(), resolution=resolution)
            log.timer(' save nto_coeff electron', *cpu0)

            log.info(f'nto density saved to nto_{state_id}_hole.cube and nto_{state_id}_electron.cube')

        return dominant_weight, nto_coeff

    def get_lowdin_charge(self,state_id):
        ''' TODO: what is the normization factor of X in RKS? currently is 1.0, maybe wrong'''
        nocc = self.n_occ
        nvir = self.n_vir
        mo_coeff = cuasarray(self._scf.mo_coeff)
        mol = self._scf.mol
        log = self.log
        S_sqrt = math_helper.matrix_power(self._scf.get_ovlp(), 0.5)

        ortho_C_matrix = S_sqrt.dot(mo_coeff)
        orbo = ortho_C_matrix[:,:nocc,]
        orbv = ortho_C_matrix[:,nocc:]

        ''' Dpq MO basis -> Duv AO basis
        in general:
        Cup                  Dpq             Cqv
        |----|--------|  |----|--------|  |------------|
        |    |        |  | I  |   X    |  |   orbo.T   |
        |orbo|  orbv  |  |----|--------|  |------------|
        |    |        |  | Y  |   0    |  |            |
        |    |        |  |    |        |  |   orbv.T   |
        |----|--------|  |----|--------|  |------------|

        when X !=0 (Y=0), excited state (TDA)
        =
        |----|     |----------------|
        |    |     |orbo.T+ X*orbv.T|
        |orbo|     |----------------|
        |    |
        |    |
        |----|

        excited state density matrix
        = orbo * orbo.T (ground state dm) + orbo * X *orbv.T (transition dm)
        '''
        cis_t1 = self.xy[0][state_id-1, :].copy()
        cis_t1 = cis_t1.reshape(nocc, nvir) # Xia
        X_orbv = cis_t1.dot(orbv.T)
        # cis_dm = orbo.dot(cis_t1).dot(orbv.T) # Xuv, large, dont build it
        aoslice = mol.aoslice_by_atom()

        gs_diag = 2*cp.sum(orbo*orbo, axis=1)
        transition_diag = 2*cp.sum(orbo*X_orbv.T, axis=1)

        q_atoms = cp.empty([mol.natm,2], dtype=cp.float32)

        for atom_id in range(mol.natm):
            _shst, _shend, atstart, atend = aoslice[atom_id]
            q_atoms[atom_id, 0] = cp.sum(gs_diag[atstart:atend,])
            q_atoms[atom_id, 1] = cp.sum(transition_diag[atstart:atend,])

        cp.savetxt(f'q_atoms_{state_id}.txt', q_atoms, fmt='%.5f')
        log.info(f'q_atoms saved to {f"q_atoms_{state_id}.txt"}')
        log.info(f'first column is ground state charge, second column is excited state {state_id} transition charge')
        return q_atoms


def get_minimal_auxbasis(auxmol_basis_keys, theta, fitting_basis, excludeHs=False):
    '''
    Args:
        auxmol_basis_keys: (['C1', 'H2', 'O3', 'H4', 'H5', 'H6'])
        theta: float 0.2
        fitting_basis: str ('s','sp','spd')

    return:
        aux_basis:
        C1 [[0, [0.1320292535005648, 1.0]]]
        H2 [[0, [0.1999828038466018, 1.0]]]
        O3 [[0, [0.2587932305664396, 1.0]]]
        H4 [[0, [0.1999828038466018, 1.0]]]
        H5 [[0, [0.1999828038466018, 1.0]]]
        H6 [[0, [0.1999828038466018, 1.0]]]
    '''
    aux_basis = {}

    for atom_index in auxmol_basis_keys:
        atom = ''.join([char for char in atom_index if char.isalpha()])

        if excludeHs:
            if atom == 'H':
                continue
        '''
        exponent_alpha = theta/R^2
        '''
        exp_alpha = parameter.ris_exp[atom] * theta

        if 's' in fitting_basis:
            aux_basis[atom_index] = [[0, [exp_alpha, 1.0]]]

        if atom != 'H':
            if 'p' in fitting_basis:
                aux_basis[atom_index].append([1, [exp_alpha, 1.0]])
            if 'd' in fitting_basis:
                aux_basis[atom_index].append([2, [exp_alpha, 1.0]])

    return aux_basis

def get_auxmol(mol, theta=0.2, fitting_basis='s', excludeHs=False):
    """
    Assigns a minimal auxiliary basis set to the molecule.

    Args:
        mol: The input molecule object.
        theta: The scaling factor for the exponents.
        fitting_basis: Basis set type ('s', 'sp', 'spd').

    Returns:
        auxmol: The molecule object with assigned auxiliary basis.
    """


    '''
    parse_arg = False
    turns off PySCF built-in parsing function
    '''
    auxmol = mol.copy()
    auxmol.verbose=0
    auxmol_basis_keys = mol._basis.keys()
    auxmol.basis = get_minimal_auxbasis(auxmol_basis_keys, theta, fitting_basis,excludeHs=excludeHs)
    auxmol.build(dump_input=False, parse_arg=False)
    return auxmol


'''
            n_occ          n_vir
       -|-------------||-------------|
        |             ||             |
  n_occ |   3c2e_ij   ||  3c2e_ia    |
        |             ||             |
        |             ||             |
       =|=============||=============|
        |             ||             |
  n_vir |             ||  3c2e_ab    |
        |             ||             |
        |             ||             |
       -|-------------||-------------|
'''

def get_uvPCupCvq_to_Ppq(eri3c: cp.ndarray, C_pT: cp.ndarray, C_q: cp.ndarray, in_ram: bool = False):
    '''
    eri3c : (uv|P) , P = naux
    C_pT = C_p.T
    C_p and C_q:  C[:, :n_occ] or C[:, n_occ:], can be both

    Ppq = einsum("uvP,up,vq->Ppq", eri3c, Cp, C_q)
    '''
    nao, nao, naux = eri3c.shape
    size_p, nao = C_pT.shape
    nao, size_q = C_q.shape

    # tmp = contract('uvP,up->Ppv', eri3c, C_p)
    # Ppq = contract('Ppv,vq->Ppq', tmp, C_q)
    # del tmp

    eri3c = eri3c.reshape(nao, nao*naux)

    pvP = C_pT.dot(eri3c)  # (size_p, nao*naux)
    pvP = pvP.reshape(size_p, nao, naux)  # (size_p, nao, naux)
    Ppq = contract('pvP,vq->Ppq', pvP, C_q)

    if in_ram:
        Ppq = Ppq.get()
    release_memory()
    return Ppq


def einsum2dot(_, PQ, Pia):
    PQT = PQ.T
    naux, nocc, nvir = Pia.shape
    Pia = Pia.reshape(naux, nocc*nvir)
    Tia = PQT.dot(Pia)
    Tia = Tia.reshape(naux, nocc, nvir)
    del PQT, Pia, PQ
    return Tia


def get_Tpq(mol, auxmol, lower_inv_eri2c, C_p, C_q,
           calc='JK',omega=None, alpha=None, beta=None,
           log=None, in_ram=True, single=True):
    """
    (3c2e_{Puv}, C_{up}, C_{vq} -> Ppq)。

    Parameters:
        mol: pyscf.gto.Mole
        auxmol: pyscf.gto.Mole
        C_p: cupy.ndarray (nao, p)
        C_q: cupy.ndarray  (nao, q)

        lower_inv_eri2c is the inverse of the lower part of the 2-center Coulomb integral
        in the case of RSH, lower_inv_eri2c already includes the RSH factor when parsed into this function
        thus lower_inv_eri2c do not need specific processing

    Returns:
        Tpq: cupy.ndarray (naux, nao, nao)
    """
    cpu0 = log.init_timer()
    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()
    tmp = int3c2e_opt.aux_coeff
    aux_coeff = cp.empty_like(tmp)

    log.timer('int3c2e_opt', *cpu0)

    nao  = mol.nao
    ''' naux here is not auxmol.nao '''
    naux = aux_coeff.shape[0]
    # assert naux == auxmol.nao

    log.info(f'mol.nao: {mol.nao}')
    log.info(f'mol.cart: {mol.cart}')

    log.info(f'int3c2e_opt.mol.nao: {int3c2e_opt.mol.nao}')
    log.info(f'int3c2e_opt.mol.cart: {int3c2e_opt.mol.cart}')

    siz_p = C_p.shape[1]
    siz_q = C_q.shape[1]

    C_pT = C_p.T
    C_qT = C_q.T

    xp = np if in_ram else cp
    log.info(f'xp {xp}')

    P_dtype = xp.dtype(xp.float32 if single else xp.float64)
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)
    log.info(f'cp_int3c_dtype: {cp_int3c_dtype}')


    if 'J' in calc:
        Pia = xp.empty((naux, siz_p, siz_q), dtype=P_dtype)

    if 'K' in calc:
        '''only store lower triangle of Tij and Tab'''
        n_tri_p = (siz_p * (siz_p + 1)) // 2
        n_tri_q = (siz_q * (siz_q + 1)) // 2
        Pij = xp.empty((naux, n_tri_p), dtype=P_dtype)
        Pab = xp.empty((naux, n_tri_q), dtype=P_dtype)

        tril_indices_p = cp.tril_indices(siz_p)
        tril_indices_q = cp.tril_indices(siz_q)

    # ao_pair_mapping = int3c2e_opt.pair_and_diag_indices(cart=mol.cart)[0]
    ao_pair_mapping = int3c2e_opt.pair_and_diag_indices()[0]

    rows, cols = divmod(ao_pair_mapping, nao)
    naopair = len(ao_pair_mapping)
    log.info(f' number of AO pairs: {naopair}')
    log.info(f'compression ratio: {naopair / (nao * nao):.2f}')


    byte_eri3c = nao * nao * cp_int3c_dtype.itemsize

    available_gpu_memory = get_avail_gpumem()
    n_eri3c_per_aux = naopair * 2
    n_eri3c_unzip_per_aux = nao * nao * 1
    n_Ppq_per_aux = siz_p * nao  + siz_p * siz_q * 1.5


    bytes_per_aux = ( n_eri3c_per_aux + n_eri3c_unzip_per_aux + n_Ppq_per_aux) * cp_int3c_dtype.itemsize
    batch_size = min(naux, max(1, int(available_gpu_memory * 0.5 // bytes_per_aux)) )

    DEBUG = False
    if DEBUG:
        batch_size = 2

    log.info(f'eri3c per aux dimension will take {byte_eri3c / 1024**2:.0f} MB memory')
    log.info(f'batch_size for int3c2e_evaluator (in aux dimension): {batch_size}')
    log.info(f'eri3c per aux batch will take {byte_eri3c * batch_size / 1024**2:.0f} MB memory')
    log.info(gpu_mem_info('before int3c2e_evaluator'))


    if omega is None or omega == 0:
        eval_j3c, aux_sorting, _ao_pair_offsets, aux_offsets = int3c2e_opt.int3c2e_evaluator(
                                                                        reorder_aux=True,
                                                                        # cart=mol.cart,
                                                                        aux_batch_size=batch_size)[:4]
    else:
        eval_j3c, aux_sorting, _ao_pair_offsets, aux_offsets = int3c2e_opt.int3c2e_evaluator(
                                                                        reorder_aux=True,
                                                                        # cart=mol.cart,
                                                                        aux_batch_size=batch_size,
                                                                        omega=omega, lr_factor=alpha+beta,
                                                                        sr_factor=alpha)[:4]

    ''' aux_coeff is contraction coeff of GTO, not the MO coeff
        nomatter whether RSH is used, aux_coeff is the same
    '''

    aux_coeff[aux_sorting] = tmp
    del tmp, aux_sorting
    log.timer('before batching', *cpu0)

    cpu00 = log.init_timer()
    for i, (p0, p1) in enumerate(zip(aux_offsets[:-1], aux_offsets[1:])):
        cpu1 = log.init_timer()
        eri3c_batch_tmp = eval_j3c(aux_batch_id=i)
        log.timer(f'eval_j3c {i}', *cpu1)

        eri3c_batch = eri3c_batch_tmp.astype(cp_int3c_dtype, copy=False)
        del eri3c_batch_tmp
        naopair, aux_batch_size = eri3c_batch.shape
        release_memory()

        eri3c_unzip_batch = cp.zeros((nao, nao, aux_batch_size), dtype=cp_int3c_dtype, order='F')
        eri3c_unzip_batch[rows, cols, :] = eri3c_batch
        eri3c_unzip_batch[cols, rows, :] = eri3c_batch

        log.timer(f'eri3c_unzip_batch {i}', *cpu1)
        DEBUG = False
        if DEBUG:
            ''' cannot exactly control the sllicing batch size, please only test in in K'''
            from pyscf.df import incore
            ref = incore.aux_e2(mol, auxmol)
            ref = ref.astype(P_dtype)
            log.info(f'eri3c_unzip_batch.shape {eri3c_unzip_batch.shape}')
            if omega and omega != 0:
                # mol_omega = mol.copy()
                # auxmol_omega = auxmol.copy()
                # mol_omega.omega = omega
                # ref_omega = incore.aux_e2(mol_omega, auxmol_omega)
                # ref = alpha * ref + beta * ref_omega
                # log.info(f'eref.shape {ref.shape}')
                mol_a = mol.copy()
                mol_a.omega = 0
                auxmol.omega = 0
                a_omega = incore.aux_e2(mol_a, auxmol)

                mol_b = mol.copy()
                mol_b.omega = omega
                auxmol.omega = omega
                b_omega = incore.aux_e2(mol_b, auxmol)

                ref = alpha * a_omega + beta * b_omega
            # out = contract('uvP,PQ->uvQ', eri3c_unzip_batch, aux_coeff)
            out = eri3c_unzip_batch.dot(aux_coeff)
            log.info(f'-------------eri3c DEBUG: out vs .incore.aux_e2(mol, auxmol) {abs(out.get()-ref).max()}')
            assert abs(out.get()-ref).max() < 1e-10

        '''Puv -> Ppq, AO->MO transform '''
        if 'J' in calc:
            cpu0 = log.init_timer()
            Pia_tmp = get_uvPCupCvq_to_Ppq(eri3c_unzip_batch,C_pT,C_q, in_ram=False)
            if in_ram:
                Pia_tmp = Pia_tmp.get()
            Pia[p0:p1,:,:] = Pia_tmp
            log.timer(f'Pia get_uvPCupCvq_to_Ppq {i}', *cpu0)

        if 'K' in calc:
            cpu0 = log.init_timer()
            Pij_tmp = get_uvPCupCvq_to_Ppq(eri3c_unzip_batch,C_pT,C_p, in_ram=False)
            log.timer(f'Pij_tmp get_uvPCupCvq_to_Ppq {i}', *cpu0)

            Pij_lower = Pij_tmp[:, tril_indices_p[0], tril_indices_p[1]].reshape(Pij_tmp.shape[0], -1)
            del Pij_tmp
            release_memory()

            if in_ram:
                Pij_lower = Pij_lower.get()
            Pij[p0:p1,:] = Pij_lower
            del Pij_lower
            release_memory()

            Pab_tmp = get_uvPCupCvq_to_Ppq(eri3c_unzip_batch,C_qT,C_q, in_ram=False)
            Pab_lower = Pab_tmp[:, tril_indices_q[0], tril_indices_q[1]].reshape(Pab_tmp.shape[0], -1)
            del Pab_tmp
            release_memory()

            if in_ram:
                Pab_lower = Pab_lower.get()
            Pab[p0:p1,:] = Pab_lower
            del Pab_lower
            release_memory()

        last_reported = 0
        progress = int(100.0 * (i+1) / (len(aux_offsets)-1))

        if progress % 20 == 0 and progress != last_reported:
            log.last_reported = progress
            log.info(f'get_Tpq batch {p1} / {naux} done ({progress} percent). aux_batch_size: {aux_batch_size}')

    assert p1 == naux

    log.info(f' get_Tpq {calc} all batches processed')
    log.info(gpu_mem_info('after generate Ppq'))
    log.timer('total batching', *cpu00)

    aux_coeff_lower_inv_eri2c = aux_coeff.dot(lower_inv_eri2c)

    # if 'K' in calc:
    eri2c_inv = contract('QR,PR->QP', aux_coeff_lower_inv_eri2c, aux_coeff_lower_inv_eri2c)
    eri2c_inv = eri2c_inv.astype(cp_int3c_dtype, copy=False)
    aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.astype(cp_int3c_dtype, copy=False)
    #     if in_ram:
    #         eri2c_inv = eri2c_inv.get()
    #         aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.get()
    # else:
    #     aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.astype(cp_int3c_dtype, copy=False)
    #     if in_ram:
    #         aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.get()

    # if in_ram:
    #     tmp_contract = einsum2dot
    # else:
    #     tmp_contract = contract

    if calc == 'J':
        cpu0 = log.init_timer()
        Pia = cuasarray(Pia)
        Tia = contract('PQ,Pia->Qia', aux_coeff_lower_inv_eri2c, Pia)
        log.timer(f'Tia tmp_contract {i}', *cpu0)
        if in_ram:
            Tia = Tia.get()
        release_memory()
        return Tia

    if calc == 'K':
        cpu0 = log.init_timer()
        Pij = cuasarray(Pij)
        Tij = eri2c_inv.dot(Pij, out=Pij)
        log.timer(f'Tij eri2c_inv.dot(Pij, out=Pij) {i}', *cpu0)
        if in_ram:
            Tij = Tij.get()
        release_memory()
        Tab = Pab
        return Tij, Tab

    if calc == 'JK':

        cpu0 = log.init_timer()
        Pia = cuasarray(Pia)

        Tia = contract('PQ,Pia->Qia',aux_coeff_lower_inv_eri2c, Pia)
        log.timer(f'Tia tmp_contract {i}', *cpu0)

        cpu0 = log.init_timer()
        Pij = cuasarray(Pij)
        Tij = eri2c_inv.dot(Pij, out=Pij)
        log.timer(f'Tij eri2c_inv.dot(Pij, out=Pij) {i}', *cpu0)

        Tab = Pab
        return Tia, Tij, Tab



def get_eri2c_inv_lower(auxmol, omega=0, alpha=None, beta=None, dtype=cp.float64):

    eri2c = auxmol.intor('int2c2e')

    if omega and omega != 0:

        with auxmol.with_range_coulomb(omega):
            eri2c_erf = auxmol.intor('int2c2e')

        eri2c = alpha * eri2c + beta * eri2c_erf

    eri2c = cuasarray(eri2c)

    try:
        ''' we want lower_inv_eri2c = X
                X X.T = eri2c^-1
                (X X.T)^-1 = eri2c
                (X.T)^-1 X^-1 = eri2c = L L.T
                (X.T)^-1 = L
        need to solve  L_inv = L^-1
                X = L_inv.T

        '''
        L = cp.linalg.cholesky(eri2c)
        L_inv = cpx_linalg.solve_triangular(L, cp.eye(L.shape[0]), lower=True)
        lower_inv_eri2c = L_inv.T

    except cp.linalg.LinAlgError:
        ''' lower_inv_eri2c = eri2c ** -0.5
            LINEAR_EPSILON = 1e-8 to remove the linear dependency, sometimes the aux eri2c is not full rank.
        '''
        lower_inv_eri2c = math_helper.matrix_power(eri2c,-0.5,epsilon=LINEAR_EPSILON)

    lower_inv_eri2c = cuasarray(lower_inv_eri2c, dtype=dtype, order='C')
    return lower_inv_eri2c

def get_inter_contract_C(int_tensor, C_occ, C_vir):

    ''' 3 for xyz three directions.
        reshape is helpful when calculating oscillator strength and polarizability.
    '''
    tmp = contract('Puv,up->Ppv', int_tensor, C_occ)
    Pia = contract('Ppv,vq->Ppq', tmp, C_vir)
    del tmp
    release_memory()
    Pia = Pia.reshape(3,-1)
    return Pia


def gen_hdiag_MVP(hdiag, n_occ, n_vir):
    def hdiag_MVP(V):
        m = V.shape[0]
        V = cuasarray(V)
        V = V.reshape(m, n_occ*n_vir)
        hdiag_v = hdiag[None,:] * V
        hdiag_v = hdiag_v.reshape(m, n_occ, n_vir)
        return hdiag_v

    return hdiag_MVP


# def gen_iajb_MVP_bdiv1(mol, auxmol, lower_inv_eri2c, C_p, C_q,  single, log=None):

#     '''
#     (ia|jb)V = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
#              = Σ_P [ T_left_ia^P Σ_jb(T_right_jb^P V_jb^m) ]
#     (ia|jb) in RKS

#     V in shape (m, n_occ * n_vir)
#     '''

#     int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()
#     nao = mol.nao
#     print(f'nao: {nao}')
#     # naux = auxmol.nao
#     naux = int3c2e_opt.aux_coeff.shape[0]
#     log.info(f'int3c2e_opt.aux_coeff.shape: {int3c2e_opt.aux_coeff.shape}')
#     cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)

#     C_pT = C_p.T
#     C_qT = C_q.T
#     print(f'C_pT.shape: {C_pT.shape}')
#     print(f'C_qT.shape: {C_qT.shape}')
#     release_memory()
#     print(f'mol.cart: {mol.cart}')
#     # ao_pair_mapping, pair_diag = int3c2e_opt.pair_and_diag_indices(cart=mol.cart)
#     ao_pair_mapping, pair_diag = int3c2e_opt.pair_and_diag_indices()

#     rows, cols = divmod(ao_pair_mapping, nao)
#     naopair = len(ao_pair_mapping)
#     print(f'naopair: {naopair}')
#     log.info(gpu_mem_info('before generate iajb_MVP function'))

#     def iajb_MVP(X, factor=2, out=None):
#         '''
#         Optimized calculation of (ia|jb) = Σ_Pjb (T_left_ia^P T_right_jb^P X_jb^m)
#         by chunking along the auxao dimension to reduce memory usage.

#         Parameters:
#             X   (cupy.ndarray): Input tensor of shape (m, n_occ, n_vir).
#             out (cupy.ndarray): output holder of shape (m, n_occ, n_vir).
#             results are accumulated in out if provided.

#         Returns:
#             iajb_X (cupy.ndarray): Result tensor of shape (m, n_occ, n_vir).
#         '''
#         n_state, n_occ, n_vir = X.shape
#         if out is None:
#             out = cp.zeros_like(X)

#         log.info(gpu_mem_info('  iajb_X before build dm_sparse'))

#         cpu0 = log.init_timer()
#         ''' build dm_sparse '''
#         dm_sparse = cp.empty((n_state, naopair),dtype=cp_int3c_dtype)
#         log.info( f'     dm_sparse {dm_sparse.nbytes/1024**3:.2f} GB')

#         X_buffer    = cp.empty((n_occ, n_vir), dtype=cp_int3c_dtype) #ia
#         temp_buffer = cp.empty((nao, n_vir), dtype=cp_int3c_dtype) # ua
#         dms_buffer  = cp.empty((nao, nao), dtype=cp_int3c_dtype) # uv

#         for i in range(n_state):
#             X_buffer[:,:] = cuasarray(X[i,:,:])
#             cp.dot(C_p, X_buffer, out=temp_buffer)
#             dms_buffer = cp.dot(temp_buffer, C_qT, out=dms_buffer)
#             dm_sparse[i,:]  = dms_buffer[rows, cols]
#             dm_sparse[i,:] += dms_buffer[cols, rows]
#             release_memory()

#         del X_buffer, temp_buffer, dms_buffer
#         release_memory()
#         # cp.cuda.Stream.null.synchronize()
#         log.info(gpu_mem_info('  iajb_X after del buffers'))

#         dm_sparse[:,pair_diag] *= 0.5
#         log.timer(' dm_sparse', *cpu0)

#         cpu0 = log.init_timer()

#         available_gpu_memory = get_avail_gpumem()
#         available_gpu_memory -= naopair * n_state * cp_int3c_dtype.itemsize
#         bytes_per_aux = ( naopair*3 + n_state) * cp_int3c_dtype.itemsize
#         batch_size = min(naux, max(1, int(available_gpu_memory * 0.8 // bytes_per_aux)) )
#         log.info(f'   iajb_MVP: int3c2e_evaluator batch_size: {batch_size}')

#         DEBUG = False
#         if DEBUG:
#             batch_size=None

#         eval_j3c, aux_sorting, _ao_pair_offsets, aux_offsets = int3c2e_opt.int3c2e_evaluator(
#                                                                         reorder_aux=True,
#                                                                         # cart=mol.cart,
#                                                                         aux_batch_size=batch_size)[:4]

#         tmp = int3c2e_opt.aux_coeff
#         aux_coeff = cp.empty_like(tmp)
#         aux_coeff[aux_sorting] = tmp
#         del tmp, aux_sorting

#         aux_coeff_lower_inv_eri2c = aux_coeff.dot(lower_inv_eri2c)
#         # eri2c_inv = contract('QR,PR->QP', aux_coeff_lower_inv_eri2c, aux_coeff_lower_inv_eri2c)
#         eri2c_inv = aux_coeff_lower_inv_eri2c.dot(aux_coeff_lower_inv_eri2c.T)
#         print(f'eri2c_inv.dtype: {eri2c_inv.dtype}')

#         eri2c_inv = eri2c_inv.astype(cp_int3c_dtype, copy=False)
#         print(f'eri2c_inv.dtype: {eri2c_inv.dtype}')
#         ''' eri2c_inv is the int2c2e_inv that also absorbs two aux_coeff on both sides
#             might not in shape of (naux, naux)
#         '''

#         ''' (z|Q)X_mz mQ '''
#         T_right = cp.empty((n_state, naux),dtype=cp_int3c_dtype)
#         log.info( f'     T_right {T_right.nbytes/1024**2:.2f} MB')

#         for i, (p0, p1) in enumerate(zip(aux_offsets[:-1], aux_offsets[1:])):
#             eri3c_batch = eval_j3c(aux_batch_id=i)
#             # cpu1 = log.init_timer()
#             eri3c_batch = cuasarray(eri3c_batch, dtype=cp_int3c_dtype, order='F')
#             release_memory()

#             aopair, aux_batch_size = eri3c_batch.shape
#             tmp = dm_sparse.dot(eri3c_batch)
#             T_right[:, p0:p1] = tmp

#             del eri3c_batch, tmp
#             release_memory()
#             # log.timer('eri3c_batch', *cpu1)

#         del dm_sparse
#         release_memory()
#         cp.cuda.Stream.null.synchronize()
#         log.timer('T_right', *cpu0)

#         DEBUG = True
#         if DEBUG:
#             from pyscf.df import incore
#             eri3c = incore.aux_e2(mol, auxmol)
#             eri3c = cuasarray(eri3c, dtype=cp_int3c_dtype, order='F')
#             dm = cp.einsum('ui,mia,av->muv', C_p, X, C_qT)  # mia -> mua
#             ref = cp.einsum('uvP,muv->mP', eri3c, dm)

#             aux_coeff = aux_coeff.astype(cp_int3c_dtype, copy=False)
#             dat = T_right.dot(aux_coeff)
#             print('check norm of difference: ', cp.linalg.norm(dat - ref))
#             # assert cp.allclose(dat, ref)

#         T_right = cp.dot(T_right, eri2c_inv) #mP,PQ->mQ   PQ symmetry
#         T_right = cuasarray(T_right.T, order='F') #Pm
#         T_right *= factor
#         #(z|P) @ (Pm),slice over P
#         T_left = cp.zeros((len(ao_pair_mapping), n_state),dtype=cp_int3c_dtype)
#         log.info( f'     T_left {T_left.nbytes/1024**3:.2f} GB')


#         # (z|P)Pm -> zm  i.e.(uv|m)

#         for i, (p0, p1) in enumerate(zip(aux_offsets[:-1], aux_offsets[1:])):
#             eri3c_batch = eval_j3c(aux_batch_id=i)
#             eri3c_batch = cuasarray(eri3c_batch, dtype=cp_int3c_dtype, order='C')
#             release_memory()
#              #(uv|P) @ (Pm) -> (uv|m)
#             # T_left = contract('zP,mP->zm',eri3c_batch, T_right[:, p0:p1], alpha=1, beta=1, out=T_left)
#             T_left += eri3c_batch.dot(T_right[p0:p1,:])
#             del eri3c_batch
#             release_memory()

#         del T_right
#         release_memory()

#         # Cui Cva (uv|m) -> mia
#         J_buffer = cp.empty((nao, nao), dtype=cp_int3c_dtype)
#         temp_buffer = cp.empty((n_occ, nao), dtype=cp_int3c_dtype)

#         for i in range(n_state):
#             #(uv|m)
#             J_buffer.fill(0)
#             J_buffer[rows, cols] = T_left[:, i]
#             J_buffer[cols, rows] = T_left[:, i]
#             temp_buffer = cp.dot(C_pT, J_buffer, out=temp_buffer) # iu,uv->iv

#             # contract('iu,ua->ia',temp_buffer, C_q, alpha=factor, beta=1, out=out[i, :, :])
#             out[i, :, :] += cp.dot(temp_buffer, C_q)

#         del T_left, temp_buffer
#         release_memory()
#         log.info(gpu_mem_info('  iajb_MVP done'))
#         return out

#     log.info(gpu_mem_info('after generate iajb_MVP'))
#     return iajb_MVP


def gen_iajb_MVP_bdiv(mol, auxmol, lower_inv_eri2c, C_p, C_q,  single, log=None):

    '''
    (ia|jb)V = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
             = Σ_P [ T_left_ia^P Σ_jb(T_right_jb^P V_jb^m) ]
    (ia|jb) in RKS

    V in shape (m, n_occ * n_vir)
    '''

    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()
    aux_coeff_unsorted = int3c2e_opt.aux_coeff
    nao = mol.nao
    # naux = auxmol.nao
    naux = int3c2e_opt.aux_coeff.shape[0]
    log.info(f'int3c2e_opt.aux_coeff.shape: {int3c2e_opt.aux_coeff.shape}')
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)

    C_pT = C_p.T
    C_qT = C_q.T
    release_memory()
    # ao_pair_mapping, pair_diag = int3c2e_opt.pair_and_diag_indices(cart=mol.cart)
    ao_pair_mapping, pair_diag = int3c2e_opt.pair_and_diag_indices()

    rows, cols = divmod(ao_pair_mapping, nao)
    naopair = len(ao_pair_mapping)
    log.info(f'naopair: {naopair}')
    log.info(gpu_mem_info('before generate iajb_MVP function'))

    # eri2c = cuasarray(auxmol.intor('int2c2e'))
    # eri2c_inv = cp.linalg.inv(eri2c)
    # eri2c_inv = aux_coeff.dot(eri2c_inv).dot(aux_coeff.T)
    # log.info(f'eri2c_inv.dtype: {eri2c_inv.dtype}')

    # eri2c_inv = eri2c_inv.astype(cp_int3c_dtype, copy=False)
    ''' eri2c_inv is the int2c2e_inv that also absorbs two aux_coeff on both sides
        might not in shape of (naux, naux)
    '''

    def iajb_MVP(X, factor=2, out=None):
        '''
        Optimized calculation of (ia|jb) = Σ_Pjb (T_left_ia^P T_right_jb^P X_jb^m)
        by chunking along the auxao dimension to reduce memory usage.

        Parameters:
            X   (cupy.ndarray): Input tensor of shape (m, n_occ, n_vir).
            out (cupy.ndarray): output holder of shape (m, n_occ, n_vir).
            results are accumulated in out if provided.

        Returns:
            iajb_X (cupy.ndarray): Result tensor of shape (m, n_occ, n_vir).
        '''
        n_state, n_occ, n_vir = X.shape
        if out is None:
            out = cp.zeros_like(X)

        log.info(gpu_mem_info('  iajb_X before build dm_sparse'))

        cpu0 = log.init_timer()
        ''' build dm_sparse '''
        dm_sparse = cp.empty((n_state, naopair)) # always float64
        log.info( f'     dm_sparse {dm_sparse.nbytes/1024**3:.2f} GB')

        X_buffer    = cp.empty((n_occ, n_vir), dtype=cp_int3c_dtype) #ia
        temp_buffer = cp.empty((nao, n_vir), dtype=cp_int3c_dtype) # ua
        dms_buffer  = cp.empty((nao, nao), dtype=cp_int3c_dtype) # uv

        for i in range(n_state):
            X_buffer[:,:] = cuasarray(X[i,:,:])
            cp.dot(C_p, X_buffer, out=temp_buffer)
            dms_buffer = cp.dot(temp_buffer, C_qT, out=dms_buffer)
            dm_sparse[i,:]  = dms_buffer[rows, cols]
            dm_sparse[i,:] += dms_buffer[cols, rows]
            release_memory()

        del X_buffer, temp_buffer, dms_buffer
        release_memory()
        # cp.cuda.Stream.null.synchronize()
        log.info(gpu_mem_info('  iajb_X after del buffers'))

        dm_sparse[:,pair_diag] *= 0.5
        log.timer(' dm_sparse', *cpu0)
        # dm_sparse *=100
        cpu0 = log.init_timer()

        ''' (z|Q)X_mz mQ '''
        # T_right = cp.empty((n_state, naux),dtype=cp_int3c_dtype)
        T_right = cp.empty((n_state, naux))

        log.info( f'     T_right {T_right.nbytes/1024**2:.2f} MB')


        available_gpu_memory = get_avail_gpumem()
        available_gpu_memory -= naopair * n_state * cp_int3c_dtype.itemsize
        bytes_per_aux = ( naopair*3 ) * cp_int3c_dtype.itemsize
        batch_size = min(naux, max(1, int(available_gpu_memory * 0.8 // bytes_per_aux)) )
        log.info(f'   iajb_MVP: int3c2e_evaluator batch_size: {batch_size}')

        DEBUG = False
        if DEBUG:
            batch_size=None

        eval_j3c, aux_sorting, _ao_pair_offsets, aux_offsets = int3c2e_opt.int3c2e_evaluator(
                                                                        reorder_aux=True,
                                                                        aux_batch_size=batch_size)[:4]

        aux_coeff = cp.empty_like(aux_coeff_unsorted)
        aux_coeff[aux_sorting] = aux_coeff_unsorted
        del aux_sorting
        aux_coeff_lower_inv_eri2c = aux_coeff.dot(lower_inv_eri2c)
        eri2c_inv = contract('QR,PR->QP', aux_coeff_lower_inv_eri2c, aux_coeff_lower_inv_eri2c)
        del aux_coeff_lower_inv_eri2c

        for i, (p0, p1) in enumerate(zip(aux_offsets[:-1], aux_offsets[1:])):
            eri3c_batch = eval_j3c(aux_batch_id=i)
            # cpu1 = log.init_timer()
            # eri3c_batch = cuasarray(eri3c_batch, dtype=cp_int3c_dtype, order='F')  # convert to float32, problem

            tmp = dm_sparse.dot(eri3c_batch)
            T_right[:, p0:p1] = tmp

            del eri3c_batch, tmp
            release_memory()
            # log.timer('eri3c_batch', *cpu1)
        # T_right *= 0.01
        del dm_sparse
        release_memory()
        cp.cuda.Stream.null.synchronize()
        log.timer('T_right', *cpu0)

        DEBUG = False
        if DEBUG:
            from pyscf.df import incore
            eri3c = incore.aux_e2(mol, auxmol)
            # eri3c = cuasarray(eri3c, dtype=cp_int3c_dtype, order='F')
            dm = cp.einsum('ui,mia,av->muv', C_p, X, C_qT)  # mia -> mua
            ref = cp.einsum('uvP,muv->mP', eri3c, dm)
            log.info(f'ref.dtype: {ref.dtype}')
            # aux_coeff1 = aux_coeff.astype(cp_int3c_dtype, copy=False)
            aux_coeff1 = aux_coeff
            dat = T_right.dot(aux_coeff1)
            log.info(f'dat.dtype: {dat.dtype}')
            log.info(f'check norm of difference: {cp.linalg.norm(dat - ref)}')
            log.info(f'check max difference: {cp.max(cp.abs(dat - ref))}')
            # assert cp.allclose(dat, ref)

        T_right = cp.dot(T_right, eri2c_inv) #mP,PQ->mQ   PQ symmetry
        T_right = cuasarray(T_right.T, order='F') #Pm
        T_right *= factor
        #(z|P) @ (Pm),slice over P
        # T_left = cp.zeros((len(ao_pair_mapping), n_state),dtype=cp_int3c_dtype)
        T_left = cp.zeros((len(ao_pair_mapping), n_state))

        log.info( f'     T_left {T_left.nbytes/1024**3:.2f} GB')
        cpu0 = log.init_timer()
        # (z|P)Pm -> zm  i.e.(uv|m)
        for i, (p0, p1) in enumerate(zip(aux_offsets[:-1], aux_offsets[1:])):
            eri3c_batch = eval_j3c(aux_batch_id=i)
            # eri3c_batch = cuasarray(eri3c_batch, dtype=cp_int3c_dtype, order='C')
            release_memory()
            #(uv|P) @ (Pm) -> (uv|m)
            # T_left = contract('zP,mP->zm',eri3c_batch, T_right[:, p0:p1], alpha=1, beta=1, out=T_left)
            T_left += eri3c_batch.dot(T_right[p0:p1,:])
            del eri3c_batch
            release_memory()

        del T_right
        release_memory()
        log.timer('T_left', *cpu0)
        # Cui Cva (uv|m) -> mia
        J_buffer = cp.empty((nao, nao), dtype=cp_int3c_dtype)
        temp_buffer = cp.empty((n_occ, nao), dtype=cp_int3c_dtype)

        cpu0 = log.init_timer()
        for i in range(n_state):
            #(uv|m)
            J_buffer.fill(0)
            J_buffer[rows, cols] = T_left[:, i]
            J_buffer[cols, rows] = T_left[:, i]
            temp_buffer = cp.dot(C_pT, J_buffer, out=temp_buffer) # iu,uv->iv

            contract_to_out('iu,ua->ia',temp_buffer, C_q, alpha=1, beta=1, out=out[i, :, :])
            # out[i, :, :] += cp.dot(temp_buffer, C_q)

        del T_left, temp_buffer
        release_memory()
        log.timer('T_left to out', *cpu0)

        log.info(gpu_mem_info('  iajb_MVP done'))
        return out

    log.info(gpu_mem_info('after generate iajb_MVP'))
    return iajb_MVP


def gen_iajb_MVP_Tpq(T_ia, log=None):
    '''
    (ia|jb)V = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
             = Σ_P [ T_left_ia^P Σ_jb(T_right_jb^P V_jb^m) ]
    (ia|jb) in RKS

    V in shape (m, n_occ * n_vir)
    '''

    # def iajb_MVP(V):
    #     T_right_jb_V = einsum("Pjb,mjb->Pm", T_right, V)
    #     iajb_V = einsum("Pia,Pm->mia", T_left, T_right_jb_V)
    #     return iajb_V

    def iajb_MVP(V, factor=2, out=None):
        '''
        Optimized calculation of (ia|jb) = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
        by chunking along the auxao dimension to reduce memory usage.

        Parameters:
            V (cupy.ndarray): Input tensor of shape (m, n_occ * n_vir).
            results are accumulated in out if provided.

        Returns:
            iajb_V (cupy.ndarray): Result tensor of shape (m, n_occ, n_vir).
        '''
        # Get the shape of the tensors
        naux, n_occ, n_vir = T_ia.shape
        n_state, n_occ, n_vir = V.shape
        # Initialize result tensor
        if out is None:
            out = cp.zeros_like(V)

        # 1 denotes one auxao, we are slucing the auxao dimension.
        n_Tia_chunk = 1 * n_occ * n_vir
        n_TjbVjb_chunk = 1 * n_state
        n_iajb_V_chunk = n_state * n_occ * n_vir

        estimated_chunk_size_bytes = (n_Tia_chunk + n_TjbVjb_chunk + n_iajb_V_chunk) * T_ia.itemsize

        # Estimate the optimal chunk size based on available GPU memory
        aux_batch_size = int(get_avail_gpumem() * 0.8 // estimated_chunk_size_bytes)

        # Ensure the chunk size is at least 1 and doesn't exceed the total number of auxao
        aux_batch_size = max(1, min(naux, aux_batch_size))

        # Iterate over chunks of the auxao dimension
        for aux_start in range(0, naux, aux_batch_size):
            aux_end = min(aux_start + aux_batch_size, naux)
            T_ia_slice = T_ia[aux_start:aux_end, :, :]
            Tjb_chunk = cuasarray(T_ia_slice)
            del T_ia_slice

            Tjb_Vjb_chunk = contract("Pjb,mjb->Pm", Tjb_chunk, V)

            Tia_chunk = Tjb_chunk  # Shape: (aux_range, n_occ, n_vir)
            out = contract_to_out("Pia,Pm->mia", Tia_chunk, Tjb_Vjb_chunk, alpha=factor, beta=1, out=out)

            # Release intermediate variables and clean up memory, must!
            del Tjb_chunk, Tia_chunk, Tjb_Vjb_chunk
            release_memory()

        return out

    return iajb_MVP


def gen_ijab_MVP_Tpq(T_ij, T_ab, log=None):
    '''
    (ij|ab)V = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
             = Σ_P [T_ij^P Σ_jb(T_ab^P V_jb^m)]
    V in shape (m, n_occ * n_vir)
    '''

    naux, n_tri_ij = T_ij.shape
    naux, n_tri_ab = T_ab.shape
    log.info(f'T_ij.dtype {T_ij.dtype}')
    log.info(f'T_ab.dtype {T_ab.dtype}')
    n_occ = int((-1 + (1 + 8 * n_tri_ij)**0.5) / 2)
    n_vir = int((-1 + (1 + 8 * n_tri_ab)**0.5) / 2)

    tril_indices_occ = cp.tril_indices(n_occ)
    tril_indices_vir = cp.tril_indices(n_vir)

    def ijab_MVP(X, a_x, out=None):
        '''
        Optimized calculation of (ij|ab) = Σ_Pjb (T_ij^P T_ab^P X_jb^m)
        by chunking along the P (naux) dimension for both T_ij and T_ab,
        uploading chunks to GPU to reduce memory usage.

        Parameters:
            X (cupy.ndarray): Input tensor of shape (n_state, n_occ, n_vir).
            a_x (float): Scaling factor.
            out (cupy.ndarray, optional): Output tensor of shape (n_state, n_occ, n_vir).

        Returns:
            ijab_X (cupy.ndarray): Result tensor of shape (n_state, n_occ, n_vir).
        '''


        n_state, n_occ, n_vir = X.shape    # Dimensions of X

        # Initialize result tensor
        if out is None:
            out = cp.zeros_like(X)

        # Get free memory and dynamically calculate chunk size
        available_gpu_memory = get_avail_gpumem()

        log.info(gpu_mem_info('          ijab_MVP start'))

        # Memory estimation for one P index
        n_T_ab_chunk = 1 * n_vir * n_vir * 1.5  # T_ab chunk: (1, n_vir, n_vir)
        n_T_ij_chunk = 1 * n_occ * n_occ * 1.5 # T_ij chunk: (1, n_occ, n_occ)
        n_T_ab_chunk_X = 1 * n_state * n_occ * n_vir  # T_ab_X chunk: (1, n_state, n_occ)
        n_ijab_chunk_X = n_state * n_occ * n_vir  # Full output size (accumulated)

        bytes_per_P = max(n_T_ab_chunk + n_T_ab_chunk_X,  n_T_ij_chunk + n_T_ab_chunk_X ) * T_ab.itemsize
        # log.info(f'bytes_per_P {bytes_per_P}')
        P_chunk_size = int((available_gpu_memory * 0.7 - n_ijab_chunk_X * T_ab.itemsize) // bytes_per_P)
        P_chunk_size = min(naux, max(1, P_chunk_size))
        log.info(f'    ijab with Tij Tab, P_chunk_size = {P_chunk_size}')
        # Iterate over chunks of the P (naux) dimension
        for P_start in range(0, naux, P_chunk_size):
            P_end = min(P_start + P_chunk_size, naux)

            # log.info(gpu_mem_info(f'  ijab {P_start,P_end}'))
            # Extract and upload the corresponding chunks of T_ab and T_ij to GPU
            # T_ab_slice = T_ab[P_start:P_end, :, :]
            T_ab_chunk_lower = cuasarray(T_ab[P_start:P_end, :])  # Shape: (P_chunk_size, (n_vir*n_vir+1)//2 )

            # Compute T_ab_X for the current chunk
            T_ab_chunk = cp.empty((T_ab_chunk_lower.shape[0], n_vir, n_vir),dtype=T_ab_chunk_lower.dtype)
            T_ab_chunk[:, tril_indices_vir[0], tril_indices_vir[1]] = T_ab_chunk_lower
            T_ab_chunk[:, tril_indices_vir[1], tril_indices_vir[0]] = T_ab_chunk_lower
            del T_ab_chunk_lower
            release_memory()
            T_ab_chunk_X = contract("Pab,mjb->Pamj", T_ab_chunk, X)
            del T_ab_chunk
            release_memory()

            T_ij_chunk_lower = cuasarray(T_ij[P_start:P_end, :])  # Shape: (P_chunk_size, (n_occ*n_occ+1)//2 )
            # T_ij_slice = T_ij[P_start:P_end, :]
            T_ij_chunk = cp.empty((T_ij_chunk_lower.shape[0], n_occ, n_occ),dtype=T_ij_chunk_lower.dtype)
            T_ij_chunk[:, tril_indices_occ[0], tril_indices_occ[1]] = T_ij_chunk_lower
            T_ij_chunk[:, tril_indices_occ[1], tril_indices_occ[0]] = T_ij_chunk_lower
            del T_ij_chunk_lower
            gc.collect()

            # Compute ijab_X for the current chunk and accumulate
            out = contract_to_out("Pij,Pamj->mia", T_ij_chunk, T_ab_chunk_X, -a_x, 1, out=out)
            del T_ij_chunk, T_ab_chunk_X
            release_memory()
            # Release intermediate variables and clean up memory

        log.info(gpu_mem_info('          ijab_MVP done'))
        return out

    return ijab_MVP

def gen_ibja_MVP_Tpq(T_ia, log=None):
    '''
    the exchange (ib|ja) in B matrix
    (ib|ja) = Σ_Pjb (T_ib^P T_ja^P V_jb^m)
            = Σ_P [T_ja^P Σ_jb(T_ib^P V_jb^m)]
    '''
    # def ibja_MVP(V):
    #     T_ib_V = einsum("Pib,mjb->Pimj", T_ia, V)
    #     ibja_V = einsum("Pja,Pimj->mia", T_ia, T_ib_V)
    #     return ibja_V

    def ibja_MVP(V, a_x, out=None):
        '''
        Optimized calculation of (ib|ja) = Σ_Pjb (T_ib^P T_ja^P V_jb^m)
        by chunking along the n_occ dimension to reduce memory usage.

        Parameters:
            V (cupy.ndarray): Input tensor of shape (n_state, n_occ, n_vir).
            occ_chunk_size (int): Chunk size for splitting the n_occ dimension.

        Returns:
            ibja_V (cupy.ndarray): Result tensor of shape (n_state, n_occ, n_vir).
        '''
        naux, n_occ, n_vir = T_ia.shape
        n_state, n_occ, n_vir = V.shape

        available_gpu_memory = get_avail_gpumem()

        bytes_per_aux = (n_occ * n_vir * 1 + n_state * n_occ * n_vir ) * T_ia.itemsize

        batch_size = max(1, int(available_gpu_memory * 0.8 // bytes_per_aux))

        if out is None:
            out = cp.zeros_like(V)
        # Iterate over chunks of the n_occ dimension
        for p0 in range(0, naux, batch_size):
            p1 = min(p0+batch_size, naux)

            # Extract the corresponding chunk of T_ia
            T_ib_chunk = cuasarray(T_ia[p0:p1, :, :])  # Shape: (batch_size, n_occ, n_vir)
            T_jb_chunk = T_ib_chunk

            T_ib_V_chunk = contract("Pib,mjb->mPij", T_ib_chunk, V)

            out = contract_to_out("Pja,mPij->mia", T_jb_chunk, T_ib_V_chunk, alpha=-a_x, beta=1, out=out)
            # out = contract("Pja,mPij->mia", T_jb_chunk, T_ib_V_chunk, alpha=1, beta=1, out=out)

            # out -= a_x * contract("Pja,mPij->mia", T_jb_chunk, T_ib_V_chunk)

            release_memory()

        return out

    return ibja_MVP


def get_ab(td, mf, J_fit, K_fit, theta, mo_energy=None, mo_coeff=None, mo_occ=None, singlet=True):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ai||jb)
    B[i,a,j,b] = (ai||bj)

    Ref: Chem Phys Lett, 256, 454
    '''

    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mo_energy = cuasarray(mo_energy)
    mo_coeff = cuasarray(mo_coeff)
    mo_occ = cuasarray(mo_occ)
    mol = mf.mol
    nao, nmo = mo_coeff.shape
    occidx = cp.where(mo_occ==2)[0]
    viridx = cp.where(mo_occ==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = cp.hstack((orbo,orbv))

    e_ia = mo_energy[viridx] - mo_energy[occidx,None]
    a = cp.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = cp.zeros_like(a)
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
    if K_fit == J_fit and (omega == 0 or omega is None):
        auxmol_K = auxmol_J
    else:
        auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit)

    def get_erimo(auxmol_i):
        naux = auxmol_i.nao
        int3c = int3c2e.get_int3c2e(mol, auxmol_i)
        int2c2e = auxmol_i.intor('int2c2e')
        int3c = cuasarray(int3c)
        int2c2e = cuasarray(int2c2e)
        df_coef = cp.linalg.solve(int2c2e, int3c.reshape(nao*nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        eri = contract('ijP,Pkl->ijkl', int3c, df_coef)
        eri_mo = contract('pjkl,pi->ijkl', eri, orbo.conj())
        eri_mo = contract('ipkl,pj->ijkl', eri_mo, mo)
        eri_mo = contract('ijpl,pk->ijkl', eri_mo, mo.conj())
        eri_mo = contract('ijkp,pl->ijkl', eri_mo, mo)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        return eri_mo
    def get_erimo_omega(auxmol_i, omega):
        naux = auxmol_i.nao
        int3c = int3c2e.get_int3c2e(mol, auxmol_i, omega=omega)
        with auxmol_i.with_range_coulomb(omega):
            int2c2e = auxmol_i.intor('int2c2e')
        int3c = cuasarray(int3c)
        int2c2e = cuasarray(int2c2e)
        df_coef = cp.linalg.solve(int2c2e, int3c.reshape(nao*nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        eri = contract('ijP,Pkl->ijkl', int3c, df_coef)
        eri_mo = contract('pjkl,pi->ijkl', eri, orbo.conj())
        eri_mo = contract('ipkl,pj->ijkl', eri_mo, mo)
        eri_mo = contract('ijpl,pk->ijkl', eri_mo, mo.conj())
        eri_mo = contract('ijkp,pl->ijkl', eri_mo, mo)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        return eri_mo
    def add_hf_(a, b, hyb=1):
        eri_mo_J = get_erimo(auxmol_J)
        eri_mo_K = get_erimo(auxmol_K)
        if singlet:
            a += cp.einsum('iabj->iajb', eri_mo_J[:nocc,nocc:,nocc:,:nocc]) * 2
            a -= cp.einsum('ijba->iajb', eri_mo_K[:nocc,:nocc,nocc:,nocc:]) * hyb
            b += cp.einsum('iajb->iajb', eri_mo_J[:nocc,nocc:,:nocc,nocc:]) * 2
            b -= cp.einsum('jaib->iajb', eri_mo_K[:nocc,nocc:,:nocc,nocc:]) * hyb
        else:
            a -= cp.einsum('ijba->iajb', eri_mo_K[:nocc,:nocc,nocc:,nocc:]) * hyb
            b -= cp.einsum('jaib->iajb', eri_mo_K[:nocc,nocc:,:nocc,nocc:]) * hyb

    if getattr(td, 'with_solvent', None):
        raise NotImplementedError("PCM TDDFT RIS is not supported")

    if isinstance(mf, scf.hf.KohnShamDFT):
        add_hf_(a, b, hyb)
        if omega != 0:  # For RSH
            eri_mo_K = get_erimo_omega(auxmol_K, omega)
            k_fac = alpha - hyb
            a -= cp.einsum('ijba->iajb', eri_mo_K[:nocc,:nocc,nocc:,nocc:]) * k_fac
            b -= cp.einsum('jaib->iajb', eri_mo_K[:nocc,nocc:,:nocc,nocc:]) * k_fac

        if mf.do_nlc():
            raise NotImplementedError('vv10 nlc not implemented in get_ab(). '
                                      'However the nlc contribution is small in TDDFT, '
                                      'so feel free to take the risk and comment out this line.')
    else:
        add_hf_(a, b)

    return a.get(), b.get()

def rescale_spin_free_amplitudes(xy, state_id):
    '''
    Rescales spin-free excitation amplitudes in TDDFT-ris to the normalization
    convention used in standard RKS-TDDFT.

    The original RKS-TDDFT formulation uses excitation amplitudes corresponding to
    the spin-up components only. The TDDFT-RIS implementation employs spin-free
    amplitudes that are not equivalent to the spin-up components and are
    normalized to 1.
    '''
    x, y = xy
    x = x[state_id] * .5**.5
    if y is not None: # TDDFT
        y = y[state_id] * .5**.5
    else: # TDA
        y = cp.zeros_like(x)
    return x, y

def as_scanner(td):
    if isinstance(td, lib.SinglePointScanner):
        return td

    logger.info(td, 'Set %s as a scanner', td.__class__)
    name = td.__class__.__name__ + TD_Scanner.__name_mixin__
    return lib.set_class(TD_Scanner(td), (TD_Scanner, td.__class__), name)


class TD_Scanner(lib.SinglePointScanner):
    def __init__(self, td):
        self.__dict__.update(td.__dict__)
        self._scf = td._scf.as_scanner()

    def __call__(self, mol_or_geom, **kwargs):
        assert self.device == 'gpu'
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)
        mf_scanner = self._scf
        mf_e = mf_scanner(mol)
        self.n_occ = None
        self.n_vir = None
        self.rest_occ = None
        self.rest_vir = None
        self.C_occ_notrunc = None
        self.C_vir_notrunc = None
        self.C_occ_Ktrunc = None
        self.C_vir_Ktrunc = None
        self.delta_hdiag = None
        self.hdiag = None
        # self.eri_tag = None
        self.auxmol_J = None
        self.auxmol_K = None
        self.lower_inv_eri2c_J = None
        self.lower_inv_eri2c_K = None
        self.RKS = True
        self.UKS = False
        self.mo_coeff = cuasarray(self._scf.mo_coeff, dtype=self.dtype)
        self.build()
        self.kernel()
        return mf_e + self.energies/HARTREE2EV


class TDA(RisBase):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, **kwargs)
        log = self.log
        log.info('TDA-ris initialized')

    ''' ===========  RKS hybrid =========== '''
    def get_RKS_TDA_hybrid_MVP(self):
        ''' TDA RKS hybrid '''
        log = self.log

        T_ij_K, T_ab_K = self.get_2T_K()
        ijab_MVP = gen_ijab_MVP_Tpq(T_ij=T_ij_K, T_ab=T_ab_K, log=log)

        if self._store_Tpq_J:
            T_ia_J = self.get_T_J()
            iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J, log=log)
        else:
            iajb_MVP = gen_iajb_MVP_bdiv(mol=self.mol, auxmol=self.auxmol_J, lower_inv_eri2c=self.lower_inv_eri2c_J,
                                        C_p=self.C_occ_notrunc, C_q=self.C_vir_notrunc, log=log, single=self.single)

        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag , n_occ=self.n_occ, n_vir=self.n_vir)

        def RKS_TDA_hybrid_MVP(X):
            ''' hybrid or range-sparated hybrid, a_x > 0
                return AX
                AX = hdiag_MVP(X) + 2*iajb_MVP(X) - a_x*ijab_MVP(X)
                for RSH, a_x = 1


                With MO truncation, most the occ and vir orbitals (transition pair) are neglected in the exchange part

                As shown below, * denotes the included transition pair
                         -------------------
                       /                  /
       original X =   /                  /  nstates
                     -------------------
                    |******************|
             n_occ  |******************|
                    |******************|
                    |******************|
                    |------------------|
                            n_vir
        becomes:
                         -------------------
                       /                  /
                X' =  /                  /  nstates
                     -------------------
                    |                  |
     n_occ-rest_occ |                  |
                    |-----|------------|
                    |*****|            |
         rest_occ   |*****|            |
                    |-----|------------|
                  rest_vir

                (If no MO truncation, then n_occ-rest_occ=0 and rest_vir=n_vir)
            '''
            nstates = X.shape[0]
            X = X.reshape(nstates, self.n_occ, self.n_vir)

            cpu0 = log.init_timer()
            log.info(gpu_mem_info('       TDA MVP before hdiag_MVP'))
            out = hdiag_MVP(X)
            log.timer('--hdiag_MVP', *cpu0)

            cpu0 = log.init_timer()

            X_trunc = cuasarray(X[:,self.n_occ-self.rest_occ:,:self.rest_vir])
            ijab_MVP(X_trunc, a_x=self.a_x, out=out[:,self.n_occ-self.rest_occ:,:self.rest_vir])
            del X_trunc

            log.timer('--ijab_MVP', *cpu0)

            cpu0 = log.init_timer()

            iajb_MVP(X, out=out)

            gc.collect()
            cp.cuda.Stream.null.synchronize()
            release_memory()

            log.timer('--iajb_MVP', *cpu0)
            log.info(gpu_mem_info('       TDA MVP after iajb'))


            out = out.reshape(nstates, self.n_occ*self.n_vir)
            return out

        return RKS_TDA_hybrid_MVP, self.hdiag


    ''' ===========  RKS pure =========== '''
    def get_RKS_TDA_pure_MVP(self):
        '''hybrid RKS TDA'''
        log = self.log
        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)

        if self._store_Tpq_J:
            T_ia_J = self.get_T_J()
            iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J, log=log)
        else:
            iajb_MVP = gen_iajb_MVP_bdiv(mol=self.mol, auxmol=self.auxmol_J, lower_inv_eri2c=self.lower_inv_eri2c_J,
                                        C_p=self.C_occ_notrunc, C_q=self.C_vir_notrunc, log=log, single=self.single)

        def RKS_TDA_pure_MVP(X):
            ''' pure functional, a_x = 0
                return AX
                AV = hdiag_MVP(V) + 2*iajb_MVP(V)
            '''
            nstates = X.shape[0]
            X = X.reshape(nstates, self.n_occ, self.n_vir)
            out = hdiag_MVP(X)
            cpu0 = log.init_timer()
            # AX += 2 * iajb_MVP(X)
            iajb_MVP(X, out=out)
            log.timer('--iajb_MVP', *cpu0)
            out = out.reshape(nstates, self.n_occ*self.n_vir)
            return out

        return RKS_TDA_pure_MVP, self.hdiag

    #  TODO: UKS case

    def gen_vind(self):
        if self.RKS:
            self.build()
            if self.a_x != 0:
                TDA_MVP, hdiag = self.get_RKS_TDA_hybrid_MVP()

            elif self.a_x == 0:
                TDA_MVP, hdiag = self.get_RKS_TDA_pure_MVP()
        else:
            raise NotImplementedError('Does not support UKS method yet')
        return TDA_MVP, hdiag

    def kernel(self):

        '''for TDA, pure and hybrid share the same form of
                     AX = Xw
            always use the Davidson solver
            Unlike pure TDDFT, pure TDA is not using MZ=Zw^2 form
        '''
        log = self.log

        TDA_MVP, hdiag = self.gen_vind()

        converged, energies, X = _krylov_tools.krylov_solver(matrix_vector_product=TDA_MVP,hdiag=hdiag,
                                              n_states=self.nstates, problem_type='eigenvalue',
                                              conv_tol=self.conv_tol, max_iter=self.max_iter,
                                              extra_init=self.extra_init, restart_subspace=self.restart_subspace,
                                              gs_initial=False, gram_schmidt=self.gram_schmidt,
                                              single=self.single, in_ram=self._krylov_in_ram, verbose=log)

        self.converged = converged
        log.debug(f'check orthonormality of X: {cp.linalg.norm(cp.dot(X, X.T) - cp.eye(X.shape[0])):.2e}')

        cpu0 = log.init_timer()
        P = self.transition_dipole()
        log.timer('transition_dipole', *cpu0)
        cpu0 = log.init_timer()
        mdpol = self.transition_magnetic_dipole()
        log.timer('transition_magnetic_dipole', *cpu0)

        oscillator_strength, rotatory_strength = spectralib.get_spectra(energies=energies, X=X, Y=None,
                                                 P=P, mdpol=mdpol,
                                                 name=self.out_name+'_TDA_ris' if self.out_name else 'TDA_ris',
                                                 RKS=self.RKS, spectra=self.spectra,
                                                 print_threshold = self.print_threshold,
                                                 n_occ=self.n_occ, n_vir=self.n_vir, verbose=self.verbose)

        energies = energies*HARTREE2EV


        self.energies = energies
        self.xy = (X, None)
        self.oscillator_strength = oscillator_strength
        self.rotatory_strength = rotatory_strength


        if self._citation:
            log.info(CITATION_INFO)

        return energies, X, oscillator_strength, rotatory_strength

    def Gradients(self):
        if getattr(self._scf, 'with_df', None) is not None:
            from gpu4pyscf.df.grad import tdrks_ris
            return tdrks_ris.Gradients(self)
        else:
            from gpu4pyscf.grad import tdrks_ris
            return tdrks_ris.Gradients(self)

    def NAC(self):
        if getattr(self._scf, 'with_df', None) is not None:
            from gpu4pyscf.df.nac.tdrks_ris import NAC
            return NAC(self)
        else:
            from gpu4pyscf.nac.tdrks_ris import NAC
            return NAC(self)

class TDDFT(RisBase):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, **kwargs)
        log = self.log
        log.info('TDDFT-ris is initialized')

    ''' ===========  RKS hybrid =========== '''
    def gen_RKS_TDDFT_hybrid_MVP(self):
        '''hybrid RKS TDDFT'''
        log = self.log

        log.info(gpu_mem_info('before T_ia_J'))

        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)


        if self._store_Tpq_J:
            T_ia_J = self.get_T_J()
            iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J, log=log)
        else:
            iajb_MVP = gen_iajb_MVP_bdiv(mol=self.mol, auxmol=self.auxmol_J, lower_inv_eri2c=self.lower_inv_eri2c_J,
                                        C_p=self.C_occ_notrunc, C_q=self.C_vir_notrunc, log=log, single=self.single)


        T_ia_K, T_ij_K, T_ab_K = self.get_3T_K()
        ijab_MVP = gen_ijab_MVP_Tpq(T_ij=T_ij_K, T_ab=T_ab_K, log=log)
        ibja_MVP = gen_ibja_MVP_Tpq(T_ia=T_ia_K, log=log)

        a_x = self.a_x
        def RKS_TDDFT_hybrid_MVP(X, Y):
            '''
            RKS
            [A B][X] = [AX+BY] = [U1]
            [B A][Y]   [AY+BX]   [U2]
            we want AX+BY and AY+BX
            instead of directly computing AX+BY and AY+BX
            we compute (A+B)(X+Y) and (A-B)(X-Y)
            it can save one (ia|jb)V tensor contraction compared to directly computing AX+BY and AY+BX

            (A+B)V = hdiag_MVP(V) + 4*iajb_MVP(V) - a_x * [ ijab_MVP(V) + ibja_MVP(V) ]
            (A-B)V = hdiag_MVP(V) - a_x * [ ijab_MVP(V) - ibja_MVP(V) ]
            for RSH, a_x = 1, because the exchange component is defined by alpha+beta (alpha+beta not awlways == 1)

            # X Y in shape (m, n_occ*n_vir)
            '''
            nstates = X.shape[0]
            n_occ, rest_occ = self.n_occ, self.rest_occ
            n_vir, rest_vir= self.n_vir, self.rest_vir

            X = X.reshape(nstates, n_occ, n_vir)
            Y = Y.reshape(nstates, n_occ, n_vir)

            XpY = X + Y
            XmY = X - Y
            ApB_XpY = hdiag_MVP(XpY)

            # ApB_XpY += 4*iajb_MVP(XpY)

            # ApB_XpY[:,n_occ-rest_occ:,:rest_vir] -= self.a_x*ijab_MVP(XpY[:,n_occ-rest_occ:,:rest_vir])

            # ApB_XpY[:,n_occ-rest_occ:,:rest_vir] -= self.a_x*ibja_MVP(XpY[:,n_occ-rest_occ:,:rest_vir])

            # AmB_XmY = hdiag_MVP(XmY)
            # AmB_XmY[:,n_occ-rest_occ:,:rest_vir] -= self.a_x*ijab_MVP(XmY[:,n_occ-rest_occ:,:rest_vir])

            # AmB_XmY[:,n_occ-rest_occ:,:rest_vir] += self.a_x*ibja_MVP(XmY[:,n_occ-rest_occ:,:rest_vir])

            iajb_MVP(XpY, factor=4, out=ApB_XpY)

            ijab_MVP(XpY[:,n_occ-rest_occ:,:rest_vir], a_x=a_x, out=ApB_XpY[:,n_occ-rest_occ:,:rest_vir])

            ibja_MVP(XpY[:,n_occ-rest_occ:,:rest_vir], a_x=a_x, out=ApB_XpY[:,n_occ-rest_occ:,:rest_vir])

            AmB_XmY = hdiag_MVP(XmY)

            ijab_MVP(XmY[:,n_occ-rest_occ:,:rest_vir], a_x=a_x, out=AmB_XmY[:,n_occ-rest_occ:,:rest_vir])

            ibja_MVP(XmY[:,n_occ-rest_occ:,:rest_vir], a_x=-a_x, out=AmB_XmY[:,n_occ-rest_occ:,:rest_vir])


            ''' (A+B)(X+Y) = AX + BY + AY + BX   (1)
                (A-B)(X-Y) = AX + BY - AY - BX   (2)
                (1) + (1) /2 = AX + BY = U1
                (1) - (2) /2 = AY + BX = U2
            '''
            U1 = (ApB_XpY + AmB_XmY)/2
            U2 = (ApB_XpY - AmB_XmY)/2

            U1 = U1.reshape(nstates, n_occ*n_vir)
            U2 = U2.reshape(nstates, n_occ*n_vir)

            return U1, U2
        return RKS_TDDFT_hybrid_MVP, self.hdiag

    ''' ===========  RKS pure =========== '''
    def gen_RKS_TDDFT_pure_MVP(self):
        log = self.log
        log.info('==================== RIJ ====================')


        log.info(gpu_mem_info('before T_ia_J'))

        hdiag_sq = self.hdiag**2
        hdiag_sqrt_MVP = gen_hdiag_MVP(hdiag=self.hdiag**0.5, n_occ=self.n_occ, n_vir=self.n_vir)
        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)

        if self._store_Tpq_J:
            cpu0 = log.init_timer()
            T_ia_J = self.get_T_J()
            log.timer('T_ia_J', *cpu0)
            iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J)

        else:
            iajb_MVP = gen_iajb_MVP_bdiv(mol=self.mol, auxmol=self.auxmol_J, lower_inv_eri2c=self.lower_inv_eri2c_J,
                                        C_p=self.C_occ_notrunc, C_q=self.C_vir_notrunc, log=log, single=self.single)

        def RKS_TDDFT_pure_MVP(Z):
            '''(A-B)^1/2(A+B)(A-B)^1/2 Z = Z w^2
                                    MZ = Z w^2
                M = (A-B)^1/2 (A+B) (A-B)^1/2
                X+Y = (A-B)^1/2 Z

                (A+B)(V) = hdiag_MVP(V) + 4*iajb_MVP(V)
                (A-B)^1/2(V) = hdiag_sqrt_MVP(V)
            '''
            nstates = Z.shape[0]
            Z = Z.reshape(nstates, self.n_occ, self.n_vir)
            AmB_sqrt_V = hdiag_sqrt_MVP(Z)
            # ApB_AmB_sqrt_V = hdiag_MVP(AmB_sqrt_V) + 4*iajb_MVP(AmB_sqrt_V)
            ApB_AmB_sqrt_V = hdiag_MVP(AmB_sqrt_V)
            iajb_MVP(AmB_sqrt_V, factor=4, out=ApB_AmB_sqrt_V)

            MZ = hdiag_sqrt_MVP(ApB_AmB_sqrt_V)
            MZ = MZ.reshape(nstates, self.n_occ*self.n_vir)
            return MZ

        return RKS_TDDFT_pure_MVP, hdiag_sq

    def gen_vind(self):
        if self.RKS:
            self.build()
            if self.a_x != 0:
                TDDFT_MVP, hdiag = self.gen_RKS_TDDFT_hybrid_MVP()

            elif self.a_x == 0:
                TDDFT_MVP, hdiag = self.gen_RKS_TDDFT_pure_MVP()
        else:
            raise NotImplementedError('Does not support UKS method yet')
        return TDDFT_MVP, hdiag

    #  TODO: UKS
    def kernel(self):
        self.build()
        log = self.log
        TDDFT_MVP, hdiag = self.gen_vind()
        if self.a_x != 0:
            '''hybrid TDDFT'''
            converged, energies, X, Y = _krylov_tools.ABBA_krylov_solver(matrix_vector_product=TDDFT_MVP, hdiag=hdiag,
                                                    n_states=self.nstates, conv_tol=self.conv_tol,
                                                    max_iter=self.max_iter, gram_schmidt=self.gram_schmidt,
                                                    single=self.single, verbose=self.verbose)
            self.converged = converged
            if not all(self.converged):
                log.info('TD-SCF states %s not converged.',
                            [i for i, x in enumerate(self.converged) if not x])
        elif self.a_x == 0:
            '''pure TDDFT'''
            hdiag_sq = hdiag
            converged, energies_sq, Z = _krylov_tools.krylov_solver(matrix_vector_product=TDDFT_MVP, hdiag=hdiag_sq,
                                            n_states=self.nstates, conv_tol=self.conv_tol, max_iter=self.max_iter,
                                            gram_schmidt=self.gram_schmidt, single=self.single, verbose=self.verbose)
            self.converged = converged
            if not all(self.converged):
                log.info('TD-SCF states %s not converged.',
                            [i for i, x in enumerate(self.converged) if not x])

            energies = energies_sq**0.5
            Z = (energies**0.5).reshape(-1,1) * Z

            X, Y = math_helper.XmY_2_XY(Z=Z, AmB_sq=hdiag_sq, omega=energies)
        normality_error = cp.linalg.norm( (cp.dot(X, X.T) - cp.dot(Y, Y.T)) - cp.eye(self.nstates) )
        log.debug(f'check normality of X^TX - Y^YY - I = {normality_error:.2e}')

        cpu0 = log.init_timer()
        P = self.transition_dipole()
        log.timer('transition_dipole', *cpu0)
        cpu0 = log.init_timer()
        mdpol = self.transition_magnetic_dipole()
        log.timer('transition_magnetic_dipole', *cpu0)

        oscillator_strength, rotatory_strength = spectralib.get_spectra(energies=energies, X=X, Y=Y,
                                                    P=P, mdpol=mdpol,
                                                    name=self.out_name+'_TDDFT_ris' if self.out_name else 'TDDFT_ris',
                                                    spectra=self.spectra, RKS=self.RKS,
                                                    print_threshold = self.print_threshold,
                                                    n_occ=self.n_occ, n_vir=self.n_vir, verbose=self.verbose)
        energies = energies*HARTREE2EV
        if self._citation:
            log.info(CITATION_INFO)
        self.energies = energies
        self.xy = X, Y
        self.oscillator_strength = oscillator_strength
        self.rotatory_strength = rotatory_strength

        return energies, X, Y, oscillator_strength, rotatory_strength

    Gradients = TDA.Gradients
    NAC = TDA.NAC

class StaticPolarizability(RisBase):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, **kwargs)
        log = self.log
        log.info('Static Polarizability-ris initialized')

    ''' ===========  RKS hybrid =========== '''
    def get_ApB_hybrid_MVP(self):
        ''' RKS hybrid '''
        log = self.log

        T_ia_J = self.get_T_J()

        T_ia_K, T_ij_K, T_ab_K = self.get_3T_K()

        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)

        iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J)
        ijab_MVP = gen_ijab_MVP_Tpq(T_ij=T_ij_K, T_ab=T_ab_K)
        ibja_MVP = gen_ibja_MVP_Tpq(T_ia=T_ia_K)

        def RKS_ApB_hybrid_MVP(X):
            ''' hybrid or range-sparated hybrid, a_x > 0
                return AX
                (A+B)X = hdiag_MVP(X) + 4*iajb_MVP(X) - a_x*[ijab_MVP(X) + ibja_MVP(X)]
                for RSH, a_x = 1

                if not MO truncation, then n_occ-rest_occ=0 and rest_vir=n_vir
            '''
            nstates = X.shape[0]
            X = X.reshape(nstates, self.n_occ, self.n_vir)
            cpu0 = log.init_timer()
            ApBX = hdiag_MVP(X)
            ApBX += 4 * iajb_MVP(X)
            log.timer('--iajb_MVP', *cpu0)

            cpu1 = log.init_timer()
            exchange =  ijab_MVP(X[:,self.n_occ-self.rest_occ:,:self.rest_vir])
            exchange += ibja_MVP(X[:,self.n_occ-self.rest_occ:,:self.rest_vir])
            log.timer('--ijab_MVP & ibja_MVP', *cpu1)

            ApBX[:,self.n_occ-self.rest_occ:,:self.rest_vir] -= self.a_x * exchange
            ApBX = ApBX.reshape(nstates, self.n_occ*self.n_vir)

            return ApBX

        return RKS_ApB_hybrid_MVP, self.hdiag

    def gen_vind(self):
        self.build()
        if self.RKS:
            if self.a_x != 0:
                TDA_MVP, hdiag = self.get_ApB_hybrid_MVP()

            elif self.a_x == 0:
                TDA_MVP, hdiag = self.get_ApB_pure_MVP()
        else:
            raise NotImplementedError('Does not support UKS method yet')
        return TDA_MVP, hdiag


    def kernel(self):
        '''for static polarizability, the problem is to solve
            (A+B)(X+Y) = -(P+Q)
            Q=P
        '''

        log = self.log

        TDA_MVP, hdiag = self.gen_vind()
        transition_dipole = self.transition_dipole()

        _, XpY = _krylov_tools.krylov_solver(matrix_vector_product=TDA_MVP,hdiag=hdiag, problem_type='linear',
                                        rhs=-transition_dipole, conv_tol=self.conv_tol, max_iter=self.max_iter,
                                        gram_schmidt=self.gram_schmidt, single=self.single, verbose=log)

        alpha = cp.dot(XpY, transition_dipole.T)*4

        self.xy = XpY
        self.alpha = alpha

        if self._citation:
            log.info(CITATION_INFO)
        return XpY

