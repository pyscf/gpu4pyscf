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
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.scf import hf as mol_hf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.dft.rkspu import reference_mol
from gpu4pyscf.dft.ucdft import _get_minao_basis_indices, CDFTBaseMixin
from gpu4pyscf.pbc.dft.krkspu import _make_minao_lo


class CDFT_KUKS(CDFTBaseMixin, dft.KUKS):
    '''
    Constrained DFT implementation for PBC Unrestricted Kohn-Sham with K-points.
    '''

    def __init__(self, cell, kpts, charge_constraints=None, spin_constraints=None, 
                 method='lagrange', penalty_weight=500.0,
                 projection_method='minao'):
        super().__init__(cell, kpts)
        self.init_cdft_params(charge_constraints, spin_constraints, 
                              method, penalty_weight, projection_method)

    def _build_minao_projectors(self):
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao_nr()
        
        logger.info(self, "CDFT: Building projectors using Orthogonalized MINAO reference (PBC)...")

        ovlp = int1e.int1e_ovlp(cell, kpts)
        C_minao = _make_minao_lo(cell, self.minao_ref, kpts)
        pcell = reference_mol(cell, self.minao_ref)
        SC = contract('kpq,kqr->kpr', ovlp, C_minao)
        
        projectors = []
        all_groups = self.charge_groups + self.spin_groups
        
        for group in all_groups:
            indices = []
            
            for identifier in group:
                idx_list = _get_minao_basis_indices(pcell, identifier)
                indices.extend(idx_list)
            
            if not indices:
                projectors.append(cp.zeros((nkpts, nao, nao)))
                continue
            
            indices = sorted(list(set(indices)))
            indices_gpu = cp.asarray(indices)
            sc_subset = SC[:, :, indices_gpu]
            
            w_matrix = contract('kpi,kqi->kpq', sc_subset, sc_subset.conj())
            
            projectors.append(w_matrix)
            
        self.constraint_projectors = projectors
        return self.constraint_projectors

    def _micro_objective_func(self, v_vec, f_std_a, f_std_b, s):
        vc_a, vc_b = self.get_constraint_potential(v_vec)
        f_tot_a = f_std_a + vc_a
        f_tot_b = f_std_b + vc_b
        
        f_tot = (f_tot_a, f_tot_b) 
        
        mo_e, mo_c = self.eig(f_tot, s)
        
        mo_occ = self.get_occ(mo_e, mo_c)
        
        nkpts = len(self.kpts)
        
        dm_a, dm_b = self.make_rdm1(mo_c, mo_occ)
        
        if hasattr(self.kpts, "weights_ibz"):
            k_weights = cp.asarray(self.kpts.weights_ibz)
        else:
            k_weights = cp.full(nkpts, 1.0/nkpts)

        def compute_pop(d, w):
            t = contract('kij,kji->k', d, w)
            return cp.sum(t * k_weights).real

        errors = []
        projector_idx = 0
        
        for target in self.charge_targets:
            w = self.constraint_projectors[projector_idx]
            val = compute_pop(dm_a, w) + compute_pop(dm_b, w)
            errors.append(float(val) - target)
            projector_idx += 1
            
        for target in self.spin_targets:
            w = self.constraint_projectors[projector_idx]
            val = compute_pop(dm_a, w) - compute_pop(dm_b, w)
            errors.append(float(val) - target)
            projector_idx += 1

        return np.array(errors)

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
                fock_last=None):
        if self.projection_method != 'minao':
            raise ValueError("CDFT_KUKS only supportsminao projection method.")
        
        h1e_kpts, s_kpts, vhf_kpts, dm_kpts = h1e, s1e, vhf, dm
        if h1e_kpts is None: h1e_kpts = self.get_hcore()
        if vhf_kpts is None: vhf_kpts = self.get_veff(self.cell, dm_kpts)
        f_kpts = h1e_kpts + vhf_kpts

        if s_kpts is None: s_kpts = self.get_ovlp()
        if dm_kpts is None: dm_kpts = self.make_rdm1()

        f_kpts = self.update_fock_with_constraints(f_kpts, s_kpts, dm_kpts, cycle)

        if cycle < 0 and diis is None:  # Not inside the SCF iteration
            return f_kpts

        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if damp_factor is None:
            damp_factor = self.damp
        if damp_factor is not None and 0 <= cycle < diis_start_cycle-1 and fock_last is not None:
            if isinstance(damp_factor, (tuple, list, np.ndarray)):
                dampa, dampb = damp_factor
            else:
                dampa = dampb = damp_factor
            f_a = []
            f_b = []
            for k in range(len(s_kpts)):
                f_a.append(mol_hf.damping(f_kpts[0][k], fock_last[0][k], dampa))
                f_b.append(mol_hf.damping(f_kpts[1][k], fock_last[1][k], dampb))
            f_kpts = cp.asarray([f_a, f_b])
        if diis and cycle >= diis_start_cycle:
            f_kpts = diis.update(s_kpts, dm_kpts, f_kpts, self, h1e_kpts, vhf_kpts, f_prev=fock_last)

        if level_shift_factor is None:
            level_shift_factor = self.level_shift
        if level_shift_factor is not None:
            if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
                shifta, shiftb = level_shift_factor
            else:
                shifta = shiftb = level_shift_factor
            f_kpts =([mol_hf.level_shift(s, dm_kpts[0,k], f_kpts[0,k], shifta)
                    for k, s in enumerate(s_kpts)],
                    [mol_hf.level_shift(s, dm_kpts[1,k], f_kpts[1,k], shiftb)
                    for k, s in enumerate(s_kpts)])
        return cp.asarray(f_kpts)

    def get_canonical_mo(self, dm=None):
        '''
        Diagonalize the standard Fock matrix (without constraint potentials) 
        using the converged density matrix to obtain canonical orbitals and energies.
        '''
        if dm is None:
            dm = self.make_rdm1()
        
        h1e_kpts = self.get_hcore()
        vhf_kpts = self.get_veff(self.cell, dm)
        f_kpts = h1e_kpts + vhf_kpts
        s_kpts = self.get_ovlp()
        mo_energy, mo_coeff = self.eig(f_kpts, s_kpts)

        return mo_energy.get(), mo_coeff.get()

    def get_all_atom_populations(self, dm=None):
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        if dm is None:
            dm = self.make_rdm1()
        
        ovlp = int1e.int1e_ovlp(cell, kpts)
        C_minao = _make_minao_lo(cell, self.minao_ref, kpts)
        pcell = reference_mol(cell, self.minao_ref)
        SC = contract('kpq,kqr->kpr', ovlp, C_minao)
        
        minao_slices = pcell.aoslice_by_atom()
        pops = []
        
        for ia in range(pcell.natm):
            p0, p1 = minao_slices[ia, 2], minao_slices[ia, 3]
            sc_subset = SC[:, :, p0:p1]
            W = contract('kpi,kqi->kpq', sc_subset, sc_subset.conj())
            pop_k = cp.trace(dm[0] @ W, axis1=1, axis2=2) + cp.trace(dm[1] @ W, axis1=1, axis2=2)
            pop = pop_k.sum()
                
            pops.append(float(pop.real) / nkpts)
            
        return pops

    def analyze(self, verbose=None, **kwargs):
        '''Analyze the given SCF object:  print orbital energies, occupancies;
        print orbital coefficients; Mulliken population analysis; Dipole moment
        '''
        from pyscf.pbc.scf.kuhf import mulliken_meta
        if verbose is None:
            verbose = self.verbose
        log = logger.new_logger(self, verbose)
        mo_energy, mo_coeff = self.get_canonical_mo()
        mo_occ = self.mo_occ.get()
        cell = self.cell
        kpts = self.kpts
        if log.verbose >= logger.NOTE:
            self.dump_scf_summary(log)
            log.note('**** MO energy ****')
            log.note('                           alpha                               | beta')
            log.note('k-point                    nocc    HOMO/AU         LUMO/AU     | nocc    HOMO/AU         LUMO/AU')
            for k, kpt in enumerate(cell.get_scaled_kpts(kpts)):
                nocca = np.count_nonzero(mo_occ[0,k])
                noccb = np.count_nonzero(mo_occ[1,k])
                homoa = mo_energy[0,k,nocca-1]
                homob = mo_energy[1,k,noccb-1]
                lumoa = mo_energy[0,k,nocca  ]
                lumob = mo_energy[1,k,noccb  ]
                log.note('%2d (%6.3f %6.3f %6.3f) %2d   %15.9f %15.9f |%2d   %15.9f %15.9f',
                         k, kpt[0], kpt[1], kpt[2], nocca, homoa, lumoa, noccb, homob, lumob)

        log.note('**** Population analysis for atoms in the reference cell ****')
        chg = self.get_all_atom_populations()
        for i, pop in enumerate(chg):
            log.note('Atom %d: %15.9f', i, pop)
        return (None, chg), None

    def Gradients(self):
        from gpu4pyscf.pbc.grad import kucdft
        return kucdft.Gradients(self)

    def nuc_grad_method(self):
        return self.Gradients()