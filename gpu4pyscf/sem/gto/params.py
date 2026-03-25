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

import os
import numpy as np
import cupy as cp
from dataclasses import dataclass
from pyscf.lib import logger
from gpu4pyscf.sem.data import atomic
from gpu4pyscf.sem.data import electron_repulsion
from gpu4pyscf.sem.data import corrections


@dataclass
class OneCenterIntegrals:
    """
    One-center one/two-electron integrals and associated empirical parameters.
    Replaces legacy loose variables: gss, gsp, gpp, gp2, hsp, f0_sd, g2_sd.
    """
    coulomb_ss: cp.ndarray       # Legacy 'gss': <ss|ss>
    coulomb_sp: cp.ndarray       # Legacy 'gsp': <ss|pp>
    exchange_sp: cp.ndarray      # Legacy 'hsp': <sp|sp>
    coulomb_pp: cp.ndarray       # Legacy 'gpp': <pp|pp> (same axis)
    coulomb_pp_diff: cp.ndarray  # Legacy 'gp2': <p_i p_i | p_j p_j> (different axes)
    f0_sd: cp.ndarray            # Legacy 'f0_sd'
    g2_sd: cp.ndarray            # Legacy 'g2_sd'

    # 1e2c derived parameters from get_eri1c2e
    repd: cp.ndarray = None       # 1c2e integral mapping array for d-orbitals (52, natm)
    
    # Parameters from params_dict
    f0_dd: cp.ndarray = None
    f2_dd: cp.ndarray = None
    f4_dd: cp.ndarray = None # TODO: The following is not used. delete
    f0_pd: cp.ndarray = None
    f2_pd: cp.ndarray = None
    g1_pd: cp.ndarray = None
    g3_pd: cp.ndarray = None

    # 1c1e (One-center one-electron)
    uspd: cp.ndarray = None       # Local core Hamiltonian matrix diagonal elements


@dataclass
class TwoCenterIntegrals:
    """
    Globally rotated two-center integrals.
    """
    w: cp.ndarray = None          # Globally transformed 2c2e repulsion integrals
    enuc: cp.ndarray = None       # Core-core repulsion energy terms for pairs
    e1b: cp.ndarray = None        # Core-electron attraction integrals (Atom A core on B electrons)
    e2a: cp.ndarray = None        # Core-electron attraction integrals (Atom B core on A electrons)
    aij_tensor: cp.ndarray = None # Multipole interaction distances tensor (s, p, d)


@dataclass
class TwoCenterIntegralParameters:
    """
    Klopman-Ohno and Multipole parameters for evaluating two-center interactions.
    """
    am: cp.ndarray = None         # Monopole scaling distance
    ad: cp.ndarray = None         # Dipole scaling distance
    aq: cp.ndarray = None         # Quadrupole scaling distance
    dd: cp.ndarray = None         # Dipole additive term (D)
    qq: cp.ndarray = None         # Quadrupole additive term (Q)
    core_rho: cp.ndarray = None   # Core-core interaction Klopman-Ohno parameter
    po_tensor: cp.ndarray = None  # Full Klopman-Ohno potential tensor for shell interactions
    ddp_tensor: cp.ndarray = None # Directional dipole lengths


@dataclass
class NuclearParameters:
    """
    Empirical parameters specifically governing core-core repulsion.
    """
    guess1: cp.ndarray = None     # Core repulsion amplitude 
    guess2: cp.ndarray = None     # Core repulsion exponent 
    guess3: cp.ndarray = None     # Core repulsion radius 
    xfac: cp.ndarray = None       # Pairwise scaling factor for core repulsion
    alpb: cp.ndarray = None       # Pairwise exponential factor for core repulsion
    v_par6: cp.ndarray = None     # Voigt or method-specific correction parameters


@dataclass
class AtomTopology:
    """
    Topological and electronic configuration data for atoms.
    Directly uses (N, 3) arrays to avoid redundant memory unpack.
    """
    principal_quantum_numbers: cp.ndarray       # Principal quantum numbers for s/p/d orbitals, shape (N, 3)
    principal_quantum_number_s: cp.ndarray      # Principal quantum number for s/p orbitals
    principal_quantum_number_d: cp.ndarray      # Principal quantum number for d orbitals
    eta_1e: cp.ndarray                          # Slater exponent for 1e integrals, shape (N, 3)
    eta_2e: cp.ndarray                          # Slater exponent for 2e integrals, shape (N, 3)
    is_main_group: cp.ndarray                   # Boolean mask for main group elements
    has_d_orbitals: cp.ndarray                  # Boolean mask for atoms with d orbitals
    core_charges: cp.ndarray                    # Core charges of atoms, shape (N,)
    norbitals_per_atom: cp.ndarray              # Number of orbitals per atom, shape (N,)

@dataclass
class HeatOfFormation:
    """
    Heat of formation parameters for atoms.
    """
    eisol_corr: cp.ndarray = None # Isolated atom energy correction term, shape (N,)

# TODO: we can calculate all the integrals!
def build_task_instructions():
    """
    Builds the complete set of index mappings and instructions for the 
    two-center two-electron (2c2e) integral evaluation.
    
    This function replaces the legacy 'fordd' routine. It first generates 
    the basic index and symmetry arrays, and then flattens them into a 
    1D instruction set (length 491) tailored for GPU execution.
    
    Returns:
        tuple of 8 np.ndarray (1D, length=491, dtype=np.int32):
        - task_action : 0 (copy from ri), 1 (compute), 2 (copy pos), 3 (copy neg)
        - task_target : Target index to copy from (if action != 1)
        - task_ij     : Orbital pair combination index for atom A (0-44)
        - task_kl     : Orbital pair combination index for atom B (0-44)
        - task_li     : Angular momentum of orbital i (0=s, 1=p, 2=d)
        - task_lj     : Angular momentum of orbital j
        - task_lk     : Angular momentum of orbital k
        - task_ll     : Angular momentum of orbital l
    """
    
    indx   = np.zeros((9, 9), dtype=np.int32)
    indexd = np.zeros((9, 9), dtype=np.int32)
    ind2   = np.zeros((45, 45), dtype=np.int32)
    isym   = np.zeros(492, dtype=np.int32)        

    def set2(a, i, j, v): a[i-1, j-1] = v
    def set1(a, i, v):    a[i] = v

    for i in range(1, 10):
        for j in range(1, i+1):
            val_indexd = (-(j*(j-1))//2) + i + 9*(j-1)   
            val_indx   = (i*(i-1))//2 + j 
            set2(indexd, i, j, val_indexd); set2(indexd, j, i, val_indexd)
            set2(indx,   i, j, val_indx);   set2(indx,   j, i, val_indx)

    s2 = set2
    # SP-SP
    s2(ind2,1,1,1);   s2(ind2,1,2,2);    s2(ind2,1,10,3);  s2(ind2,1,18,4);  s2(ind2,1,25,5)
    s2(ind2,2,1,6);   s2(ind2,2,2,7);    s2(ind2,2,10,8);  s2(ind2,2,18,9);  s2(ind2,2,25,10)
    s2(ind2,10,1,11); s2(ind2,10,2,12);  s2(ind2,10,10,13);s2(ind2,10,18,14);s2(ind2,10,25,15)
    s2(ind2,3,3,16);  s2(ind2,3,11,17);  s2(ind2,11,3,18); s2(ind2,11,11,19)
    s2(ind2,18,1,20); s2(ind2,18,2,21);  s2(ind2,18,10,22);s2(ind2,18,18,23);s2(ind2,18,25,24)
    s2(ind2,4,4,25);  s2(ind2,4,12,26);  s2(ind2,12,4,27); s2(ind2,12,12,28)
    s2(ind2,19,19,29)
    s2(ind2,25,1,30); s2(ind2,25,2,31);  s2(ind2,25,10,32);s2(ind2,25,18,33);s2(ind2,25,25,34)
    # SPD-SPD
    s2(ind2,1,5,35);   s2(ind2,1,13,36);  s2(ind2,1,31,37); s2(ind2,1,21,38); s2(ind2,1,36,39)
    s2(ind2,1,28,40);  s2(ind2,1,40,41);  s2(ind2,1,43,42); s2(ind2,1,45,43)
    s2(ind2,2,5,44);   s2(ind2,2,13,45);  s2(ind2,2,31,46); s2(ind2,2,21,47); s2(ind2,2,36,48)
    s2(ind2,2,28,49);  s2(ind2,2,40,50);  s2(ind2,2,43,51); s2(ind2,2,45,52)
    s2(ind2,10,5,53);  s2(ind2,10,13,54); s2(ind2,10,31,55);s2(ind2,10,21,56);s2(ind2,10,36,57)
    s2(ind2,10,28,58); s2(ind2,10,40,59); s2(ind2,10,43,60);s2(ind2,10,45,61)
    s2(ind2,3,20,62);  s2(ind2,3,6,63);   s2(ind2,3,14,64); s2(ind2,3,32,65); s2(ind2,3,23,66)
    s2(ind2,3,38,67);  s2(ind2,3,30,68);  s2(ind2,3,42,69)
    s2(ind2,11,20,70); s2(ind2,11,6,71);  s2(ind2,11,14,72);s2(ind2,11,32,73);s2(ind2,11,23,74)
    s2(ind2,11,38,75); s2(ind2,11,30,76); s2(ind2,11,42,77)
    s2(ind2,18,5,78);  s2(ind2,18,13,79); s2(ind2,18,31,80);s2(ind2,18,21,81);s2(ind2,18,36,82)
    s2(ind2,18,28,83); s2(ind2,18,40,84); s2(ind2,18,8,85); s2(ind2,18,16,86)
    s2(ind2,18,34,87); s2(ind2,18,43,88); s2(ind2,18,45,89)
    s2(ind2,4,26,90);  s2(ind2,4,7,91);   s2(ind2,4,15,92); s2(ind2,4,33,93); s2(ind2,4,29,94)
    s2(ind2,4,41,95);  s2(ind2,4,24,96);  s2(ind2,4,39,97)
    s2(ind2,12,26,98); s2(ind2,12,7,99);  s2(ind2,12,15,100);s2(ind2,12,33,101);s2(ind2,12,29,102)
    s2(ind2,12,41,103);s2(ind2,12,24,104);s2(ind2,12,39,105)
    s2(ind2,19,27,106);s2(ind2,19,22,107);s2(ind2,19,37,108);s2(ind2,19,9,109); s2(ind2,19,17,110)
    s2(ind2,19,35,111)
    s2(ind2,25,5,112); s2(ind2,25,13,113);s2(ind2,25,31,114);s2(ind2,25,21,115);s2(ind2,25,36,116)
    s2(ind2,25,28,117);s2(ind2,25,40,118);s2(ind2,25,8,119); s2(ind2,25,16,120);s2(ind2,25,34,121)
    s2(ind2,25,43,122);s2(ind2,25,45,123)
    s2(ind2,5,1,124);  s2(ind2,5,2,125);  s2(ind2,5,10,126);s2(ind2,5,18,127);s2(ind2,5,25,128)
    s2(ind2,5,5,129);  s2(ind2,5,13,130); s2(ind2,5,31,131);s2(ind2,5,21,132);s2(ind2,5,36,133)
    s2(ind2,5,28,134); s2(ind2,5,40,135); s2(ind2,5,43,136);s2(ind2,5,45,137)
    s2(ind2,13,1,138); s2(ind2,13,2,139); s2(ind2,13,10,140);s2(ind2,13,18,141);s2(ind2,13,25,142)
    s2(ind2,13,5,143); s2(ind2,13,13,144);s2(ind2,13,31,145);s2(ind2,13,21,146);s2(ind2,13,36,147)
    s2(ind2,13,28,148);s2(ind2,13,40,149);s2(ind2,13,43,150);s2(ind2,13,45,151)
    s2(ind2,20,3,152); s2(ind2,20,11,153);s2(ind2,20,20,154);s2(ind2,20,6,155); s2(ind2,20,14,156)
    s2(ind2,20,32,157);s2(ind2,20,23,158);s2(ind2,20,38,159);s2(ind2,20,30,160);s2(ind2,20,42,161)
    s2(ind2,26,4,162); s2(ind2,26,12,163);s2(ind2,26,26,164);s2(ind2,26,7,165); s2(ind2,26,15,166)
    s2(ind2,26,33,167);s2(ind2,26,29,168);s2(ind2,26,41,169);s2(ind2,26,24,170);s2(ind2,26,39,171)
    s2(ind2,31,1,172); s2(ind2,31,2,173); s2(ind2,31,10,174);s2(ind2,31,18,175);s2(ind2,31,25,176)
    s2(ind2,31,5,177); s2(ind2,31,13,178);s2(ind2,31,31,179);s2(ind2,31,21,180);s2(ind2,31,36,181)
    s2(ind2,31,28,182);s2(ind2,31,40,183);s2(ind2,31,43,184);s2(ind2,31,45,185)
    s2(ind2,6,3,186);  s2(ind2,6,11,187); s2(ind2,6,20,188); s2(ind2,6,6,189); s2(ind2,6,14,190)
    s2(ind2,6,32,191); s2(ind2,6,23,192); s2(ind2,6,38,193); s2(ind2,6,30,194); s2(ind2,6,42,195)
    s2(ind2,14,3,196); s2(ind2,14,11,197);s2(ind2,14,20,198);s2(ind2,14,6,199); s2(ind2,14,14,200)
    s2(ind2,14,32,201);s2(ind2,14,23,202);s2(ind2,14,38,203);s2(ind2,14,30,204);s2(ind2,14,42,205)
    s2(ind2,21,1,206); s2(ind2,21,2,207); s2(ind2,21,10,208);s2(ind2,21,18,209);s2(ind2,21,25,210)
    s2(ind2,21,5,211); s2(ind2,21,13,212);s2(ind2,21,31,213);s2(ind2,21,21,214);s2(ind2,21,36,215)
    s2(ind2,21,28,216);s2(ind2,21,40,217);s2(ind2,21,8,218); s2(ind2,21,16,219);s2(ind2,21,34,220)
    s2(ind2,21,43,221);s2(ind2,21,45,222)
    s2(ind2,27,19,223);s2(ind2,27,27,224);s2(ind2,27,22,225);s2(ind2,27,37,226);s2(ind2,27,9,227)
    s2(ind2,27,17,228);s2(ind2,27,35,229)
    s2(ind2,32,3,230); s2(ind2,32,11,231);s2(ind2,32,20,232);s2(ind2,32,6,233); s2(ind2,32,14,234)
    s2(ind2,32,32,235);s2(ind2,32,23,236);s2(ind2,32,38,237);s2(ind2,32,30,238);s2(ind2,32,42,239)
    s2(ind2,36,1,240); s2(ind2,36,2,241); s2(ind2,36,10,242);s2(ind2,36,18,243);s2(ind2,36,25,244)
    s2(ind2,36,5,245); s2(ind2,36,13,246);s2(ind2,36,31,247);s2(ind2,36,21,248);s2(ind2,36,36,249)
    s2(ind2,36,28,250);s2(ind2,36,40,251);s2(ind2,36,8,252); s2(ind2,36,16,253);s2(ind2,36,34,254)
    s2(ind2,36,43,255);s2(ind2,36,45,256)
    s2(ind2,7,4,257);  s2(ind2,7,12,258); s2(ind2,7,26,259); s2(ind2,7,7,260);  s2(ind2,7,15,261)
    s2(ind2,7,33,262); s2(ind2,7,29,263); s2(ind2,7,41,264); s2(ind2,7,24,265); s2(ind2,7,39,266)
    s2(ind2,15,4,267); s2(ind2,15,12,268);s2(ind2,15,26,269);s2(ind2,15,7,270); s2(ind2,15,15,271)
    s2(ind2,15,33,272);s2(ind2,15,29,273);s2(ind2,15,41,274);s2(ind2,15,24,275);s2(ind2,15,39,276)
    s2(ind2,22,19,277);s2(ind2,22,27,278);s2(ind2,22,22,279);s2(ind2,22,37,280);s2(ind2,22,9,281)
    s2(ind2,22,17,282);s2(ind2,22,35,283)
    s2(ind2,28,1,284); s2(ind2,28,2,285); s2(ind2,28,10,286);s2(ind2,28,18,287);s2(ind2,28,25,288)
    s2(ind2,28,5,289); s2(ind2,28,13,290);s2(ind2,28,31,291);s2(ind2,28,21,292);s2(ind2,28,36,293)
    s2(ind2,28,28,294);s2(ind2,28,40,295);s2(ind2,28,8,296); s2(ind2,28,16,297);s2(ind2,28,34,298)
    s2(ind2,28,43,299);s2(ind2,28,45,300)
    s2(ind2,33,4,301); s2(ind2,33,12,302);s2(ind2,33,26,303);s2(ind2,33,7,304); s2(ind2,33,15,305)
    s2(ind2,33,33,306);s2(ind2,33,29,307);s2(ind2,33,41,308);s2(ind2,33,24,309);s2(ind2,33,39,310)
    s2(ind2,37,19,311);s2(ind2,37,27,312);s2(ind2,37,22,313);s2(ind2,37,37,314);s2(ind2,37,9,315)
    s2(ind2,37,17,316);s2(ind2,37,35,317)
    s2(ind2,40,1,318); s2(ind2,40,2,319); s2(ind2,40,10,320);s2(ind2,40,18,321);s2(ind2,40,25,322)
    s2(ind2,40,5,323); s2(ind2,40,13,324);s2(ind2,40,31,325);s2(ind2,40,21,326);s2(ind2,40,36,327)
    s2(ind2,40,28,328);s2(ind2,40,40,329);s2(ind2,40,8,330); s2(ind2,40,16,331);s2(ind2,40,34,332)
    s2(ind2,40,43,333);s2(ind2,40,45,334)
    s2(ind2,8,18,335); s2(ind2,8,25,336); s2(ind2,8,21,337); s2(ind2,8,36,338); s2(ind2,8,28,339)
    s2(ind2,8,40,340); s2(ind2,8,8,341);  s2(ind2,8,16,342);  s2(ind2,8,34,343)
    s2(ind2,16,18,344);s2(ind2,16,25,345);s2(ind2,16,21,346);s2(ind2,16,36,347);s2(ind2,16,28,348)
    s2(ind2,16,40,349);s2(ind2,16,8,350); s2(ind2,16,16,351); s2(ind2,16,34,352)
    s2(ind2,23,3,353); s2(ind2,23,11,354);s2(ind2,23,20,355);s2(ind2,23,6,356); s2(ind2,23,14,357)
    s2(ind2,23,32,358);s2(ind2,23,23,359);s2(ind2,23,38,360);s2(ind2,23,30,361);s2(ind2,23,42,362)
    s2(ind2,29,4,363); s2(ind2,29,12,364);s2(ind2,29,26,365);s2(ind2,29,7,366); s2(ind2,29,15,367)
    s2(ind2,29,33,368);s2(ind2,29,29,369);s2(ind2,29,41,370);s2(ind2,29,24,371);s2(ind2,29,39,372)
    s2(ind2,34,18,373);s2(ind2,34,25,374);s2(ind2,34,21,375);s2(ind2,34,36,376);s2(ind2,34,28,377)
    s2(ind2,34,40,378);s2(ind2,34,8,379); s2(ind2,34,16,380); s2(ind2,34,34,381)
    s2(ind2,38,3,382); s2(ind2,38,11,383);s2(ind2,38,20,384);s2(ind2,38,6,385); s2(ind2,38,14,386)
    s2(ind2,38,32,387);s2(ind2,38,23,388);s2(ind2,38,38,389);s2(ind2,38,30,390);s2(ind2,38,42,391)
    s2(ind2,41,4,392); s2(ind2,41,12,393);s2(ind2,41,26,394);s2(ind2,41,7,395); s2(ind2,41,15,396)
    s2(ind2,41,33,397);s2(ind2,41,29,398);s2(ind2,41,41,399);s2(ind2,41,24,400);s2(ind2,41,39,401)
    s2(ind2,43,1,402); s2(ind2,43,2,403); s2(ind2,43,10,404);s2(ind2,43,18,405);s2(ind2,43,25,406)
    s2(ind2,43,5,407); s2(ind2,43,13,408);s2(ind2,43,31,409);s2(ind2,43,21,410);s2(ind2,43,36,411)
    s2(ind2,43,28,412);s2(ind2,43,40,413);s2(ind2,43,43,414);s2(ind2,43,45,415)
    s2(ind2,9,19,416); s2(ind2,9,27,417); s2(ind2,9,22,418); s2(ind2,9,37,419); s2(ind2,9,9,420)
    s2(ind2,9,17,421); s2(ind2,9,35,422)
    s2(ind2,17,19,423);s2(ind2,17,27,424);s2(ind2,17,22,425);s2(ind2,17,37,426);s2(ind2,17,9,427)
    s2(ind2,17,17,428);s2(ind2,17,35,429)
    s2(ind2,24,4,430); s2(ind2,24,12,431);s2(ind2,24,26,432);s2(ind2,24,7,433); s2(ind2,24,15,434)
    s2(ind2,24,33,435);s2(ind2,24,29,436);s2(ind2,24,41,437);s2(ind2,24,24,438);s2(ind2,24,39,439)
    s2(ind2,30,3,440); s2(ind2,30,11,441);s2(ind2,30,20,442);s2(ind2,30,6,443); s2(ind2,30,14,444)
    s2(ind2,30,32,445);s2(ind2,30,23,446);s2(ind2,30,38,447);s2(ind2,30,30,448);s2(ind2,30,42,449)
    s2(ind2,35,19,450);s2(ind2,35,27,451);s2(ind2,35,22,452);s2(ind2,35,37,453);s2(ind2,35,9,454)
    s2(ind2,35,17,455);s2(ind2,35,35,456)
    s2(ind2,39,4,457); s2(ind2,39,12,458);s2(ind2,39,26,459);s2(ind2,39,7,460); s2(ind2,39,15,461)
    s2(ind2,39,33,462);s2(ind2,39,29,463);s2(ind2,39,41,464);s2(ind2,39,24,465);s2(ind2,39,39,466)
    s2(ind2,42,3,467); s2(ind2,42,11,468);s2(ind2,42,20,469);s2(ind2,42,6,470); s2(ind2,42,14,471)
    s2(ind2,42,32,472);s2(ind2,42,23,473);s2(ind2,42,38,474);s2(ind2,42,30,475);s2(ind2,42,42,476)
    s2(ind2,44,44,477)
    s2(ind2,45,1,478); s2(ind2,45,2,479); s2(ind2,45,10,480);s2(ind2,45,18,481);s2(ind2,45,25,482)
    s2(ind2,45,5,483); s2(ind2,45,13,484);s2(ind2,45,31,485);s2(ind2,45,21,486);s2(ind2,45,36,487)
    s2(ind2,45,28,488);s2(ind2,45,40,489);s2(ind2,45,43,490);s2(ind2,45,45,491)

    # --- isym  ---
    s1 = set1
    s1(isym, 40, 38); s1(isym, 41, 39); s1(isym, 43, 42)
    s1(isym, 49, 47); s1(isym, 50, 48); s1(isym, 52, 51)
    s1(isym, 58, 56); s1(isym, 59, 57); s1(isym, 61, 60)
    s1(isym, 68, 66); s1(isym, 69, 67)
    s1(isym, 76, 74); s1(isym, 77, 75)
    s1(isym, 89, 88); s1(isym, 90, 62); s1(isym, 91, 63); s1(isym, 92, 64); s1(isym, 93, 65)
    s1(isym, 94, -66);s1(isym, 95, -67);s1(isym, 96, 66); s1(isym, 97, 67)
    s1(isym, 98, 70); s1(isym, 99, 71); s1(isym,100, 72); s1(isym,101, 73)
    s1(isym,102, -74);s1(isym,103, -75);s1(isym,104, 74); s1(isym,105, 75)
    s1(isym,106, 86); s1(isym,107, 86); s1(isym,109, 85); s1(isym,110, 86); s1(isym,111, 87)
    s1(isym,112, 78); s1(isym,113, 79); s1(isym,114, 80); s1(isym,115, 83); s1(isym,116, 84)
    s1(isym,117, 81); s1(isym,118, 82); s1(isym,119, -85);s1(isym,120, -86);s1(isym,121, -87)
    s1(isym,122, 88); s1(isym,123, 88)
    s1(isym,128,127); s1(isym,134,132); s1(isym,135,133); s1(isym,137,136)
    s1(isym,142,141); s1(isym,148,146); s1(isym,149,147); s1(isym,151,150)
    s1(isym,160,158); s1(isym,161,159)
    s1(isym,162,152); s1(isym,163,153); s1(isym,164,154); s1(isym,165,155); s1(isym,166,156); s1(isym,167,157)
    s1(isym,168,-158); s1(isym,169,-159); s1(isym,170,158); s1(isym,171,159)
    s1(isym,176,175)
    s1(isym,182,180); s1(isym,183,181); s1(isym,185,184)
    s1(isym,194,192); s1(isym,195,193)
    s1(isym,204,202); s1(isym,205,203)
    s1(isym,222,221)
    s1(isym,224,219); s1(isym,225,219); s1(isym,227,218); s1(isym,228,219); s1(isym,229,220)
    s1(isym,238,236); s1(isym,239,237)
    s1(isym,256,255)
    s1(isym,257,186); s1(isym,258,187); s1(isym,259,188); s1(isym,260,189); s1(isym,261,190); s1(isym,262,191)
    s1(isym,263,-192); s1(isym,264,-193); s1(isym,265,192); s1(isym,266,193)
    s1(isym,267,196); s1(isym,268,197); s1(isym,269,198); s1(isym,270,199); s1(isym,271,200); s1(isym,272,201)
    s1(isym,273,-202); s1(isym,274,-203); s1(isym,275,202); s1(isym,276,203)
    s1(isym,277,223); s1(isym,278,219); s1(isym,279,219); s1(isym,280,226); s1(isym,281,218)
    s1(isym,282,219); s1(isym,283,220)
    s1(isym,284,206); s1(isym,285,207); s1(isym,286,208); s1(isym,287,210); s1(isym,288,209)
    s1(isym,289,211); s1(isym,290,212); s1(isym,291,213); s1(isym,292,216); s1(isym,293,217)
    s1(isym,294,214); s1(isym,295,215)
    s1(isym,296,-218); s1(isym,297,-219); s1(isym,298,-220); s1(isym,299,221); s1(isym,300,221)
    s1(isym,301,230); s1(isym,302,231); s1(isym,303,232); s1(isym,304,233); s1(isym,305,234); s1(isym,306,235)
    s1(isym,307,-236); s1(isym,308,-237); s1(isym,309,236); s1(isym,310,237)
    s1(isym,312,253); s1(isym,313,253); s1(isym,315,252); s1(isym,316,253); s1(isym,317,254)
    s1(isym,318,240); s1(isym,319,241); s1(isym,320,242); s1(isym,321,244); s1(isym,322,243)
    s1(isym,323,245); s1(isym,324,246); s1(isym,325,247); s1(isym,326,250); s1(isym,327,251)
    s1(isym,328,248); s1(isym,329,249)
    s1(isym,330,-252); s1(isym,331,-253); s1(isym,332,-254); s1(isym,333,255); s1(isym,334,255)
    s1(isym,336,-335); s1(isym,339,-337); s1(isym,340,-338); s1(isym,342,337)
    s1(isym,344,223); s1(isym,345,-223); s1(isym,346,219); s1(isym,347,226); s1(isym,348,-219); s1(isym,349,-226)
    s1(isym,350,218); s1(isym,351,219); s1(isym,352,220)
    s1(isym,363,-353); s1(isym,364,-354); s1(isym,365,-355); s1(isym,366,-356); s1(isym,367,-357); s1(isym,368,-358)
    s1(isym,369,359); s1(isym,370,360); s1(isym,371,-361); s1(isym,372,-362)
    s1(isym,374,-373); s1(isym,377,-375); s1(isym,378,-376); s1(isym,380,375)
    s1(isym,392,-382); s1(isym,393,-383); s1(isym,394,-384); s1(isym,395,-385); s1(isym,396,-386); s1(isym,397,-387)
    s1(isym,398,388); s1(isym,399,389); s1(isym,400,-390); s1(isym,401,-391)
    s1(isym,406,405); s1(isym,412,410); s1(isym,413,411)
    s1(isym,416,335); s1(isym,417,337); s1(isym,418,337); s1(isym,419,338); s1(isym,420,341)
    s1(isym,421,337); s1(isym,422,343)
    s1(isym,423,223); s1(isym,424,219); s1(isym,425,219); s1(isym,426,226); s1(isym,427,218); s1(isym,428,219); s1(isym,429,220)
    s1(isym,430,353); s1(isym,431,354); s1(isym,432,355); s1(isym,433,356); s1(isym,434,357); s1(isym,435,358)
    s1(isym,436,-361); s1(isym,437,-362); s1(isym,438,359); s1(isym,439,360)
    s1(isym,440,353); s1(isym,441,354); s1(isym,442,355); s1(isym,443,356); s1(isym,444,357); s1(isym,445,358)
    s1(isym,446,361); s1(isym,447,362); s1(isym,448,359); s1(isym,449,360)
    s1(isym,450,373); s1(isym,451,375); s1(isym,452,375); s1(isym,453,376); s1(isym,454,379); s1(isym,455,375)
    s1(isym,456,381); s1(isym,457,382); s1(isym,458,383); s1(isym,459,384); s1(isym,460,385); s1(isym,461,386); s1(isym,462,387)
    s1(isym,463,-390); s1(isym,464,-391); s1(isym,465,388); s1(isym,466,389)
    s1(isym,467,382); s1(isym,468,383); s1(isym,469,384); s1(isym,470,385); s1(isym,471,386); s1(isym,472,387)
    s1(isym,473,390); s1(isym,474,391); s1(isym,475,388); s1(isym,476,389)
    s1(isym,478,402); s1(isym,479,403); s1(isym,480,404); s1(isym,481,405); s1(isym,482,405)
    s1(isym,483,407); s1(isym,484,408); s1(isym,485,409); s1(isym,486,410); s1(isym,487,411); s1(isym,488,410)
    s1(isym,489,411); s1(isym,490,415); s1(isym,491,414)
    indexd = indexd - 1
    ind2 = ind2 - 1 

    # the following lines are mapped from mndod.reppd2
    n_tasks = 491
    
    # Actions: 
    # 0 = Copy from ri array (the first 34 terms)
    # 1 = Independent calculation (call rijkl)
    # 2 = Copy internally from rep array (positive)
    # 3 = Copy internally from rep array and negate (negative)
    task_action = np.zeros(n_tasks, dtype=np.int32)
    task_target = np.zeros(n_tasks, dtype=np.int32)
    
    # Orbital parameters needed for calculation
    task_ij = np.zeros(n_tasks, dtype=np.int32)
    task_kl = np.zeros(n_tasks, dtype=np.int32)
    task_li = np.zeros(n_tasks, dtype=np.int32)
    task_lj = np.zeros(n_tasks, dtype=np.int32)
    task_lk = np.zeros(n_tasks, dtype=np.int32)
    task_ll = np.zeros(n_tasks, dtype=np.int32)

    ipos = np.array([
        1, 5,11,12,12, 2, 6,13,14,14, 3, 8,16,18,18, 7,15,10,20, 4, 9,17,19,21,
        7,15,10,20,22, 4, 9,17,21,19
    ], dtype=np.int32) - 1
    
    for i in range(34):
        task_action[i] = 0           # Action: Copy from RI
        task_target[i] = ipos[i]     # Target RI index

    # Angular momentum for each orbital type (s=0, 3*p=1, 5*d=2)
    lorb = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.int32)
    
    for i in range(9):
        li = lorb[i]
        for j in range(i + 1):
            lj = lorb[j]
            ij = int(indexd[i, j])
            
            for k in range(9):
                lk = lorb[k]
                for l in range(k + 1):
                    ll = lorb[l]
                    kl = int(indexd[k, l])
                    
                    idx = int(ind2[ij, kl])
                    
                    if idx <= 33:
                        continue # Skip, already handled by Action 0
                    
                    # Record angular momentum configuration for this position
                    task_ij[idx] = ij
                    task_kl[idx] = kl
                    task_li[idx] = li
                    task_lj[idx] = lj
                    task_lk[idx] = lk
                    task_ll[idx] = ll
                    
                    nold = int(isym[idx+1])
                    if nold >= 35:
                        task_action[idx] = 2       # Action: Positive Copy
                        task_target[idx] = nold - 1
                    elif nold <= -35:
                        task_action[idx] = 3       # Action: Negative Copy
                        task_target[idx] = -nold - 1
                    elif nold == 0:
                        task_action[idx] = 1       # Action: Compute via rijkl
                        task_target[idx] = -1      # Placeholder, not from copied data

    return (task_action, task_target, 
            task_ij, task_kl, task_li, task_lj, task_lk, task_ll, ind2, indexd)


class SEMParams:
    """
    Semi-Empirical Parameters Container.
    
    This class manages the loading, storage, and CPU-to-GPU transfer of 
    semi-empirical parameters. It also reconstructs derived parameters 
    (like core charges) that are typically calculated on-the-fly.
    
    Attributes:
        method (str): The method name (e.g., 'PM6').
        norbitals_per_atom (np.ndarray): Table of number of orbitals per atom (indexed by Z).
                             0 means the element is not supported. (formerly natorb)
        core_charges (np.ndarray): Table of core charges/valence electrons (indexed by Z).
    """
    def __init__(self, method='PM6'):
        self.method = method.upper()
        self._data = {}
        self._gpu_cache = {}
        self._check_method_supported()

        self._load_module_params(atomic)
        self._load_module_params(electron_repulsion)
        self._load_module_params(corrections)
        
        self._load_binary_matrices()
        self.norbitals_per_atom = self._compute_natorb()
        self._compute_core_charges()
        
        self._init_principal_quantum_numbers()
        self._init_electronic_configuration_metadata()
        self._init_reference_heats()
        self._compute_multipole_angular_factors()

        self.cutoff_radius = 15.0

        zd = self._data.get('exponent_d', np.zeros(107))
        self.has_d_orbitals = (self.principal_quantum_number_d > 0) & (zd > 1.0e-8)

    def _check_method_supported(self):
        supported_methods = ['PM6']
        if self.method not in supported_methods:
            raise ValueError(f"Method {self.method} is not supported. "
                             f"Supported methods are: {supported_methods}")

    def _load_module_params(self, module):
        for name in dir(module):
            if name.startswith("_"): 
                continue
            val = getattr(module, name)
            if isinstance(val, np.ndarray):
                self._data[name] = val

    def _load_binary_matrices(self):
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        prefix = self.method.lower()
        files_map = {
            "alpha_bond": f"{prefix}_alpha_bond.npy",
            "x_factor":   f"{prefix}_x_factor.npy"
        }
        
        for key, filename in files_map.items():
            path = os.path.join(data_dir, filename)
            if os.path.exists(path):
                self._data[key] = np.load(path)
            else:
                raise FileNotFoundError(f"Could not find file {filename} in {data_dir}")

    def _compute_natorb(self):
        """
        Infer the number of orbitals (0, 1, 4, or 9) for each element 
        based on the existence of s/p/d exponents.
        
        Returns:
            np.ndarray: Array of shape (107,) containing 0, 1, 4, or 9.
            0 indicates the element is not supported/has no basis.
        """
        natorb = np.zeros(107, dtype=np.int32)
        
        zs = self._data.get('exponent_s', np.zeros(107))
        zp = self._data.get('exponent_p', np.zeros(107))
        zd = self._data.get('exponent_d', np.zeros(107))
        
        threshold = 1e-6
        
        has_s = zs > threshold
        has_p = zp > threshold
        has_d = zd > threshold
        
        natorb[has_s] = 1
        natorb[has_s & has_p] = 4
        natorb[has_s & has_p & has_d] = 9
        
        return natorb

    def _compute_core_charges(self):
        """
        Compute core charges (tore) from occupation numbers (nocc_s, nocc_p, nocc_d).
        This reconstructs the logic from model_initialization.py.
        """
        if 'core_charge' in self._data:
            return self._data['core_charge']

        def rep(n, v): 
            return [v]*n

        # s-electrons (ios)
        nocc_s = []
        nocc_s += [1, 2]                              # 1..2 (H-He)
        nocc_s += [1, 2] + [2, 2, 2, 2, 2, 0]         # 3..10 (Li-Ne)
        nocc_s += [1, 2] + [2, 2, 2, 2, 2, 0]         # 11..18 (Na-Ar)
        nocc_s += ([1, 2, 2] +                        # 19..21 (K-Sc)
                [2, 2, 1, 2, 2, 2, 2, 1, 2] +         # 22..30 (Ti-Zn)
                [2, 2, 2, 2, 2, 0])                   # 31..36 (Ga-Kr)
        nocc_s += ([1, 2, 2] +                        # 37..39 (Rb-Y)
                [2, 1, 1, 2, 1, 1, 0, 1, 2] +         # 40..48 (Zr-Cd)
                [2, 2, 2, 2, 2, 0])                   # 49..54 (In-Xe)
        nocc_s += ([1, 2, 2] +                        # 55..57 (Cs-La)
                rep(5, 2) + rep(3, 2) + rep(6, 2)+    # 58..71 (Ce..Lu)
                [2, 2, 1, 2, 2, 2, 1, 1, 2] +         # 72..80 (Hf..Hg)
                [2, 2, 2, 2, 2, 0])                   # 81..86 (Tl..Rn)
        nocc_s += ([1, 1, 2, 4, 2, 2] +                  # 87..92
                [2, 2, 2, 2, 2, 1, 0, 3, -3] +        # 93..101
                [1, 2, 1, -2, -1, 0])                 # 102..107
        # ! it should be noted that there is negative occupancy for some elements

        # p-electrons (iop)
        nocc_p = []
        nocc_p += [0, 0]
        nocc_p += [0, 0] + [1, 2, 3, 4, 5, 6]
        nocc_p += [0, 0] + [1, 2, 3, 4, 5, 6]
        nocc_p += [0, 0, 0] + rep(9, 0) + [1,2,3,4,5,6]
        nocc_p += [0, 0, 0] + rep(9, 0) + [1,2,3,4,5,6]
        nocc_p += [0, 0, 0] + rep(14, 0) + [0]*9 + [1,2,3,4,5,6]
        nocc_p += rep(21, 0)

        # d-electrons (iod)
        nocc_d = []
        nocc_d += [0, 0]
        nocc_d += rep(8, 0)
        nocc_d += rep(8, 0)
        nocc_d += [0, 0, 1, 2, 3, 5, 5, 6, 7, 8, 10, 0] + rep(6, 0)
        nocc_d += [0, 0, 1, 2, 4, 5, 5, 7, 8, 10,10, 0] + rep(6, 0)
        nocc_d += [0, 0, 1] + rep(13, 1) + [1, 2, 3, 5, 5, 6, 7, 9, 10] + rep(7, 0)
        nocc_d += [0, 0, 1] + rep(9, 0) + rep(9, 0)

        nocc_s = np.array(nocc_s, dtype=int)
        nocc_p = np.array(nocc_p, dtype=int)
        nocc_d = np.array(nocc_d, dtype=int)

        core_charges = (nocc_s + nocc_p + nocc_d).astype(np.float64)
        
        self.nocc_s = nocc_s
        self.nocc_p = nocc_p
        self.nocc_d = nocc_d
        self.core_charges = core_charges

    def _init_principal_quantum_numbers(self):
        """
        Initialize Principal Quantum Numbers (PQN).
        Formerly: iii -> principal_quantum_number_s, i
        iiid -> principal_quantum_number_d, 
        npq -> principal_quantum_number_matrix
        """
        # PQN for s/p orbitals (107 elements)
        self.principal_quantum_number_s = np.array(
            [1]*2 + [2]*8 + [3]*8 + [4]*18 + [5]*18 + [6]*32 + [0]*21, 
            dtype=int
        )
        
        # PQN for d orbitals (107 elements)
        self.principal_quantum_number_d = np.array(
            [3]*30 + [4]*18 + [5]*32 + [6]*6 + [0]*21, 
            dtype=int
        )

        def rep(n, v): 
            return [v]*n
        
        npq_s = []
        npq_s += [1, 1]
        npq_s += [2, 2] + rep(5, 2) + [3]
        npq_s += [3, 3] + rep(5, 3) + [4]
        npq_s += rep(17, 4) + [5]
        npq_s += rep(17, 5) + [6]
        npq_s += rep(31, 6) + [7]
        npq_s = np.array(npq_s + [0]*(107-len(npq_s)), dtype=int)

        npq_p = []
        npq_p += [1, 2]
        npq_p += [2]*8
        npq_p += [3]*8
        npq_p += [4]*18
        npq_p += [5]*18
        npq_p += [6]*32
        npq_p += [0]*21
        npq_p = np.array(npq_p, dtype=int)

        npq_d = []
        npq_d += [0, 0] + rep(8, 0) # 1-10
        npq_d += [3, 3] + [3]*5 + [4] # 11-18
        npq_d += [3, 3] + rep(9, 3) + [4]*6 + [5] # 19-36
        npq_d += [4, 4] + rep(9, 4) + [5]*6 + [6] # 37-54
        npq_d += [5, 5] + rep(14, 5) + [5]*9 + [6]*6 + [7] # 55-86
        npq_d = np.array(npq_d + [0]*(107-len(npq_d)), dtype=int)

        self.principal_quantum_number_matrix = np.stack((npq_s, npq_p, npq_d), axis=-1)

    def _init_electronic_configuration_metadata(self):
        """
        Initialize metadata regarding electronic configuration.
        Formerly: ndelec, main_group
        """
        def rep(n, v): return [v]*n
        
        # d-shell occupation reference (ndelec)
        self.d_shell_occupation_ref = np.array(
            rep(20, 0) +
            [0, 0, 2, 2, 4, 4, 6, 8, 10, 10] +
            rep(8, 0) +
            [0, 0, 2, 2, 4, 4, 6, 8, 10, 10] +
            rep(22, 0) +
            [0, 0, 2, 2, 4, 4, 6, 8, 10, 10] +
            rep(27, 0),
            dtype=int
        )

        # Main group flag (main_group)
        self.is_main_group = np.array(
            [True]*2 +                    
            [True]*8 +                    
            [True]*8 +                    
            [True]*2 + [False]*9 + [True]*7 +   
            [True]*2 + [False]*9 + [True]*7 +   
            [True]*2 + [False]*23 + [True]*7 +  
            [True]*21,
            dtype=bool
        )

    def _init_reference_heats(self):
        """
        Initialize experimental Heat of Formation data.
        Formerly: eheat, eheat_sparkles
        """
        self.heat_formation_ref = np.zeros(107, dtype=np.float64)
        data_pairs = {
            1:  52.102,  3:  38.410,  4:  76.960,  5: 135.700, 6: 170.890,  7: 113.000,
            8:  59.559,  9:  18.890, 11:  25.650, 12:  35.000, 13:  79.490, 14: 108.390,
            15:  75.570, 16:  66.400, 17:  28.990, 19:  21.420, 20:  42.600, 21:  90.300,
            22: 112.300, 23: 122.900, 24:  95.000, 25:  67.700, 26:  99.300, 27: 102.400,
            28: 102.800, 29:  80.700, 30:  31.170, 31:  65.400, 32:  89.500, 33:  72.300,
            34:  54.300, 35:  26.740, 37:  19.600, 38:  39.100, 39: 101.500, 40: 145.500,
            41: 172.400, 42: 157.300, 43: 162.000, 44: 155.500, 45: 133.000, 46:  90.000,
            47:  68.100, 48:  26.720, 49:  58.000, 50:  72.200, 51:  63.200, 52:  47.000,
            53:  25.517, 55:  18.700, 56:  42.500, 57: 103.011, 58: 101.004, 59:  84.990,
            60:  78.298, 61:  83.174, 62:  49.402, 63:  41.898, 64:  95.007, 65:  92.902,
            66:  69.407, 67:  71.893, 68:  75.791, 69:  55.500, 70:  36.358, 71: 102.199,
            72: 148.000, 73: 186.900, 74: 203.100, 75: 185.000, 76: 188.000, 77: 160.000,
            78: 135.200, 79:  88.000, 80:  14.690, 81:  43.550, 82:  46.620, 83:  50.100,
            90: 1674.64, 102: 207.000
        }
        for n, val in data_pairs.items():
            self.heat_formation_ref[n-1] = val # formerly eheat

        self.heat_formation_sparkles_ref = np.zeros(107, dtype=np.float64)
        sparkles_pairs = {
            57:  928.90, 58:  944.70, 59:  952.90, 60:  962.80, 61:  976.90,
            62:  974.40, 63: 1006.60, 64:  991.37, 65:  999.00, 66: 1001.30,
            67: 1009.60, 68: 1016.15, 69: 1022.06, 70: 1039.03, 71: 1031.20,
        }
        for n, val in sparkles_pairs.items():
            self.heat_formation_sparkles_ref[n-1] = val

    def _compute_multipole_angular_factors(self):
        """
        Compute 'multipole_angular_factors' (formerly ch).
        This is used in the computation of multipole interaction to form 2c2e.
        
        Returns:
            np.ndarray: Shape (45, 3, 5). 
        """
        ch = np.zeros((45, 3, 5), dtype=np.float64)
        
        def set_ch(i_1b, l, m, v): 
            # i_1b is 1-based index from original code
            # ! i_1b is 0-based index in numpy in this code!
            ch[i_1b, l, m+2] = v

        set_ch(0,0,0, 1.0)
        set_ch(1,1,0, 1.0)
        set_ch(2,1,1, 1.0)
        set_ch(3,1,-1,1.0)
        set_ch(4,2,0, 1.15470054)
        set_ch(5,2,1, 1.0)
        set_ch(6,2,-1,1.0)
        set_ch(7,2,2, 1.0)
        set_ch(8,2,-2,1.0)
        set_ch(9,0,0,1.0); set_ch(9,2,0,1.33333333)
        set_ch(10,2,1,1.0)
        set_ch(11,2,-1,1.0)
        set_ch(12,1,0,1.15470054)
        set_ch(13,1,1,1.0)
        set_ch(14,1,-1,1.0)
        set_ch(17,0,0,1.0); set_ch(17,2,0,-0.66666667); set_ch(17,2,2,1.0)
        set_ch(18,2,-2,1.0)
        set_ch(19,1,1,-0.57735027)
        set_ch(20,1,0,1.0)
        set_ch(22,1,1,1.0)
        set_ch(23,1,-1,1.0)
        set_ch(24,0,0,1.0); set_ch(24,2,0,-0.66666667); set_ch(24,2,2,-1.0)
        set_ch(25,1,-1,-0.57735027)
        set_ch(27,1,0,1.0)
        set_ch(28,1,-1,-1.0)
        set_ch(29,1,1,1.0)
        set_ch(30,0,0,1.0); set_ch(30,2,0,1.33333333)
        set_ch(31,2,1,0.57735027)
        set_ch(32,2,-1,0.57735027)
        set_ch(33,2,2,-1.15470054)
        set_ch(34,2,-2,-1.15470054)
        set_ch(35,0,0,1.0); set_ch(35,2,0,0.66666667); set_ch(35,2,2,1.0)
        set_ch(36,2,-2,1.0)
        set_ch(37,2,1,1.0)
        set_ch(38,2,-1,1.0)
        set_ch(39,0,0,1.0); set_ch(39,2,0,0.66666667); set_ch(39,2,2,-1.0)
        set_ch(40,2,-1,-1.0)
        set_ch(41,2,1,1.0)
        set_ch(42,0,0,1.0); set_ch(42,2,0,-1.33333333)
        set_ch(44,0,0,1.0); set_ch(44,2,0,-1.33333333)
        
        self.multipole_angular_factors = ch

    def get_natorb_table(self):
        return self.norbitals_per_atom

    def get_core_charges(self):
        return self.core_charges

    def get_parameter(self, key, to_gpu=True):
        if (key not in self._data) and (key not in self.__dict__.keys()):
            raise KeyError(f"Parameter '{key}' not found in {self.method} library.")
        
        if key in self._data:
            data_cpu = self._data[key]
        else:
            data_cpu = self.__dict__[key]
        
        if not to_gpu:
            return data_cpu
            
        if key not in self._gpu_cache:
            arr_contiguous = np.ascontiguousarray(data_cpu, dtype=np.float64)
            self._gpu_cache[key] = cp.asarray(arr_contiguous)
            
        return self._gpu_cache[key]

# ===========================================================
# Cache Mechanism (Singleton Pattern)
# ===========================================================
_PARAM_CACHE = {}

def load_sem_params(method='PM6'):
    """
    Factory function to load and cache parameters.
    
    Args:
        method (str): Method name (e.g., 'PM6').
        
    Returns:
        SEMParams: The parameter object (singleton per method).
    """
    method = method.upper()
    if method in _PARAM_CACHE:
        return _PARAM_CACHE[method]
    
    try:
        params = SEMParams(method)
        _PARAM_CACHE[method] = params
        return params
    except Exception as e:
        if method in _PARAM_CACHE:
            del _PARAM_CACHE[method]
        raise e
