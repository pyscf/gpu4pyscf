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
from gpu4pyscf.sem.integral import fock  # Assuming your CUDA kernels are exposed here

def get_ovlp(mol):
    return cp.eye(mol.nao, dtype=cp.float64)

def get_hcore(mol):
    return mol.get_hcore()

def get_jk(mol, dm, hermi=1):
    if hermi == 1:
        dm = (dm + dm.conj().T) * 0.5
    J, K = fock.get_jk(mol, dm)
    return J, K

class RHF(gpu_hf.RHF):
    """
    Restricted Hartree-Fock tailored for the PM6 Semi-Empirical Method.
    Inherits the highly optimized SCF iteration engine (DIIS, damping, etc.) 
    from gpu4pyscf, but overrides integration and energy routines.
    """
    def __init__(self, mol):
        super().__init__(mol)
        
        # Force the initial guess to use the Core Hamiltonian
        self.init_guess = '1e' 
        
        # Use gpu4pyscf's CDIIS for convergence acceleration
        self.DIIS = gpu_diis.CDIIS
        
        # PM6 integrals are evaluated fully in memory, so direct SCF is not applicable
        self.direct_scf = False 

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol)

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        return get_ovlp(mol)

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

    def init_guess_by_1e(self, mol=None):
        if mol is None: mol = self.mol
        logger.info(self, 'Generating initial guess from Core Hamiltonian.')
        
        h1e = self.get_hcore(mol)
        
        mo_energy, mo_coeff = cp.linalg.eigh(h1e)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        
        return self.make_rdm1(mo_coeff, mo_occ)

    def get_init_guess(self, mol=None, key='1e'):
        """
        Route all initial guess requests to the 1e (Hcore) guess to prevent PySCF 
        from falling back to minao or other ab initio specific guesses.
        """
        return self.init_guess_by_1e(mol)

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        """
        Compute total energy and Heat of Formation.
        Overrides the PySCF standard to inject the PM6 empirical 'atheat' correction.
        """
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        
        # Electronic energy (in Hartree)
        # E_elec = 0.5 * Tr(P * (H + F)) = Tr(P * H) + 0.5 * Tr(P * Veff)
        e_elec, e_coul = self.energy_elec(dm, h1e, vhf)
        
        # Note: MOPAC typically calculates enuc in eV. We divide by HARTREE2EV.
        nuc_hartree = self.mol.enuc / self.mol.HARTREE2EV 
        
        e_tot = e_elec + nuc_hartree
        
        # Save components for the logger
        self.scf_summary['nuc'] = nuc_hartree
        self.e_tot = e_tot
        
        # TODO: heat of formation calculation
        # # Heat of Formation (Kcal/mol)
        # # Delta H_f = E_tot(eV) * 23.0605 + atheat
        # e_tot_ev = e_tot * self.mol.HARTREE2EV
        # EV2KCALMOL = 23.060547830619029
        
        # if hasattr(self.mol, 'atheat') and self.mol.atheat is not None:
        #     heat_of_formation = (e_tot_ev * EV2KCALMOL) + self.mol.atheat
        #     self.scf_summary['heat_of_formation'] = heat_of_formation
        
        return e_tot