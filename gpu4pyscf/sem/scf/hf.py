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


import cupy as cp
import numpy as np
from gpu4pyscf.scf import hf as gpu_hf
from gpu4pyscf.scf import diis as gpu_diis
from gpu4pyscf.lib import logger
from gpu4pyscf.sem.integral import fock
from gpu4pyscf.sem.scf import diis
from gpu4pyscf import lib
from gpu4pyscf.lib.cupy_helper import tag_array, asarray

def get_hcore(mol):
    # TODO: in the calculation of integrals, the unit should be hartree.
    return mol.get_hcore() / mol.HARTREE2EV

def get_jk(mol, dm, direct=None, hermi=1):
    if hermi == 1:
        dm = (dm + dm.conj().T) * 0.5
    if direct is None:
        direct = mol.direct
    J, K = fock.get_jk(mol, dm, direct=direct)
    # TODO: in the calculation of integrals, the unit should be hartree.
    return J / mol.HARTREE2EV, K / mol.HARTREE2EV


def _kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    conv_tol = mf.conv_tol
    mol = mf.mol
    verbose = mf.verbose
    log = logger.new_logger(mf, verbose)
    t0 = t1 = log.init_timer()
    
    if conv_tol_grad is None:
        conv_tol_grad = conv_tol**.5
        log.info('Set gradient conv threshold to %g', conv_tol_grad)

    if dm0 is None:
        dm0 = mf.get_init_guess(mol, mf.init_guess)
        t1 = log.timer_debug1('generating initial guess', *t1)

    if hasattr(dm0, 'mo_coeff') and hasattr(dm0, 'mo_occ'):
        if dm0.ndim == 2:
            mo_coeff = cp.asarray(dm0.mo_coeff[:,dm0.mo_occ>0])
            mo_occ = cp.asarray(dm0.mo_occ[dm0.mo_occ>0])
            dm0 = tag_array(dm0, mo_occ=mo_occ, mo_coeff=mo_coeff)
        else:
            # Drop attributes like mo_coeff, mo_occ for UHF and other methods.
            dm0 = asarray(dm0, order='C')

    h1e = cp.asarray(mf.get_hcore())
    # ZDO Approximation: S matrix is Identity. We do not need to construct or store it.
    t1 = log.timer_debug1('hcore', *t1)

    dm = asarray(dm0, order='C')
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info('init E= %.15g', e_tot)
    
    # Linear dependency check and X_orth matrix are completely unnecessary for PM6
    t1 = log.timer('SCF initialization', *t0)
    scf_conv = False

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, None, vhf, dm)  # Pass None for S matrix
        mo_energy, mo_coeff = mf.eig(fock)      # Standard eigenvalue problem HC=CE
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback

    else:
        mf_diis = None

    dump_chk = dump_chk and mf.chkfile is not None
    if dump_chk:
        mf.chkfile.save_mol(mol, mf.chkfile)

    fock_last = None
    mf.cycles = 0
    for cycle in range(mf.max_cycle):
        t0 = log.init_timer()
        mo_coeff = mo_occ = mo_energy = fock = None
        dm_last = dm
        last_hf_e = e_tot

        fock = mf.get_fock(h1e, None, vhf, dm, cycle, mf_diis, fock_last=fock_last) # Pass None for S matrix
        t1 = log.timer_debug1('DIIS', *t0)
        
        # Pure diagonalization without overlap matrix operations
        mo_energy, mo_coeff = mf.eig(fock)
        if mf.damp is not None:
            fock_last = fock
        fock = None
        t1 = log.timer_debug1('eig', *t1)

        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        dm = asarray(dm) # Remove the attached attributes
        t1 = log.timer_debug1('veff', *t1)

        fock = mf.get_fock(h1e, None, vhf, dm)  # Pass None for S matrix
        e_tot = mf.energy_tot(dm, h1e, vhf)
        norm_gorb = cp.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))

        norm_ddm = cp.linalg.norm(dm-dm_last)
        t1 = log.timer(f'cycle={cycle+1}', *t0)

        log.info('cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                 cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        e_diff = abs(e_tot-last_hf_e)
        if(e_diff < conv_tol and norm_gorb < conv_tol_grad):
            scf_conv = True
            break
    else:
        log.warn("SCF failed to converge")

    mf.cycles = cycle + 1
    if scf_conv and mf.level_shift is not None:
        # An extra diagonalization, to remove level shift
        mo_energy, mo_coeff = mf.eig(fock)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, None, vhf, dm, level_shift_factor=0)
        norm_gorb = cp.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        norm_ddm = cp.linalg.norm(dm-dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        else:
            log.warn("Level-shifted SCF extra cycle failed to converge")
            scf_conv = False
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk:
            mf.dump_chk(locals())

    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


def scf(mf, dm0=None, **kwargs):
    cput0 = logger.init_timer(mf)

    mf.dump_flags()
    mf.build(mf.mol)

    if dm0 is None and mf.mo_coeff is not None and mf.mo_occ is not None:
        # Initial guess from existing wavefunction
        dm0 = mf.make_rdm1()

    if mf.max_cycle > 0 or mf.mo_coeff is None:
        mf.converged, mf.e_tot, \
                mf.mo_energy, mf.mo_coeff, mf.mo_occ = \
                _kernel(mf, mf.conv_tol, mf.conv_tol_grad,
                        dm0=dm0, callback=mf.callback,
                        conv_check=mf.conv_check, **kwargs)
    else:
        # Avoid to update SCF orbitals in the non-SCF initialization
        # (issue #495).  But run regular SCF for initial guess if SCF was
        # not initialized.
        mf.e_tot = _kernel(mf, mf.conv_tol, mf.conv_tol_grad,
                            dm0=dm0, callback=mf.callback,
                            conv_check=mf.conv_check, **kwargs)[1]

    logger.timer(mf, 'SCF', *cput0)
    mf._finalize()
    return mf.e_tot


class RHF(gpu_hf.RHF):
    def __init__(self, mol):
        super().__init__(mol)
        
        # Force the initial guess to use the MOPAC empirical method by default
        self.init_guess = 'mopac' 
        
        # Use gpu4pyscf's CDIIS for convergence acceleration
        self.DIIS = diis.PM6DIIS
        
        # PM6 integrals are evaluated fully in memory, so direct SCF is not applicable
        self.direct_scf = False 

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol)

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        if mol.direct:
            return 0.0
        else:
            return mol.get_ovlp()

    scf = scf
    kernel = scf

    def get_jk(self, mol=None, dm=None, hermi=1, *args, **kwargs):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        assert hermi == 1
        return get_jk(mol, dm, hermi)

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        
        vj, vk = self.get_jk(mol, dm, hermi)
        
        vhf = vj - 0.5 * vk
        return vhf

    def init_guess_by_mopac(self, mol=None):
        """
        Generate initial density matrix using the MOPAC empirical approach.
        Assigns electrons to atomic orbitals based on element blocks and empirical rules.
        """
        if mol is None: mol = self.mol
        logger.info(self, 'Generating initial guess from MOPAC empirical rules.')
        
        norbs = mol.nao
        charge = mol.charge
        yy = float(charge) / (norbs + 1e-10)
        
        pdiag = np.zeros(norbs, dtype=np.float64)
        
        Z_0based = mol.atom_ids_0based
        tore = mol.topology.core_charges.get() 
        aoslice = mol._aoslice
        
        for i in range(mol.natm):
            l0, b = aoslice[i]
            n_orb = b - l0
            if n_orb == 0:
                continue
                
            zi = Z_0based[i]
            te = float(tore[i]) 
            
            if n_orb == 1:
                # Hydrogen-like
                pdiag[l0] = te - yy

            elif n_orb == 4:
                # Normal heavy atom (s+p only)
                pdiag[l0:l0+4] = 0.25 * te - yy

            elif n_orb == 9:
                # d shell
                if (zi < 21) or (30 < zi < 39) or (48 < zi < 57):
                    pdiag[l0:l0+4] = 0.25 * te - yy
                    pdiag[l0+4:l0+9] = -yy
                elif zi < 99:
                    sum_e = te - 9.0 * yy

                    s_occ = max(0.0, min(sum_e, 2.0))
                    pdiag[l0] = s_occ
                    sum_e -= s_occ

                    if sum_e > 0.0:
                        d_occ = max(0.0, min(0.2 * sum_e, 2.0))
                        pdiag[l0+4:l0+9] = d_occ
                        sum_e -= 10.0  

                        if sum_e > 0.0:
                            p_occ = sum_e / 3.0
                            pdiag[l0+1:l0+4] = p_occ
        
        # Convert diagonal array into a dense GPU matrix
        return cp.diag(cp.asarray(pdiag))

    def init_guess_by_1e(self, mol=None):
        if mol is None: mol = self.mol
        logger.info(self, 'Generating initial guess from Core Hamiltonian.')
        
        h1e = self.get_hcore(mol)
        
        mo_energy, mo_coeff = cp.linalg.eigh(h1e)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        
        return self.make_rdm1(mo_coeff, mo_occ)

    def get_init_guess(self, mol=None, key='mopac'):
        if mol is None: mol = self.mol
        if key.lower() == 'mopac':
            return self.init_guess_by_mopac(mol)
        elif key.lower() == '1e':
            return self.init_guess_by_1e(mol)
        else:
            raise ValueError(f"Unknown initial_guess key: {key}")

    def eig(self, h, s=None, overwrite=False, x=None):
        """
        Solve standard eigenvalue problem.
        For PM6 (ZDO approximation), the overlap matrix S is the identity matrix.
        Therefore, we bypass the expensive generalized eigenvalue solver.
        """
        mo_energy, mo_coeff = cp.linalg.eigh(h)
        return mo_energy, mo_coeff

    _eigh = eig

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        """
        Compute total energy strictly in Hartree.
        """
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        
        e_elec, e_coul = self.energy_elec(dm, h1e, vhf)
        
        nuc_hartree = self.mol.enuc / self.mol.HARTREE2EV 
        
        e_tot = e_elec + nuc_hartree
        
        self.scf_summary['nuc'] = nuc_hartree
        self.e_tot = e_tot
        
        # TODO: heat of formation is needed.
            
        return e_tot