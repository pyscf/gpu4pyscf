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
    gss: cp.ndarray       # Legacy 'gss': <ss|ss>
    gsp: cp.ndarray       # Legacy 'gsp': <ss|pp>
    hsp: cp.ndarray      # Legacy 'hsp': <sp|sp>
    gpp: cp.ndarray       # Legacy 'gpp': <pp|pp> (same axis)
    gp2: cp.ndarray  # Legacy 'gp2': <p_i p_i | p_j p_j> (different axes)
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
    atom_ids_0based: cp.ndarray                 # Atom IDs (0-based), shape (N,)

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

    def set2(a, i, j, v):
        a[i-1, j-1] = v
    def set1(a, i, v):
        a[i] = v

    for i in range(1, 10):
        for j in range(1, i+1):
            val_indexd = (-(j*(j-1))//2) + i + 9*(j-1)   
            val_indx   = (i*(i-1))//2 + j 
            set2(indexd, i, j, val_indexd)
            set2(indexd, j, i, val_indexd)
            set2(indx,   i, j, val_indx)
            set2(indx,   j, i, val_indx)

    # SP-SP
    s2_data = [
        (1,1,1),   (1,2,2),    (1,10,3),  (1,18,4),  (1,25,5),
        (2,1,6),   (2,2,7),    (2,10,8),  (2,18,9),  (2,25,10),
        (10,1,11), (10,2,12),  (10,10,13),(10,18,14),(10,25,15),
        (3,3,16),  (3,11,17),  (11,3,18), (11,11,19),
        (18,1,20), (18,2,21),  (18,10,22),(18,18,23),(18,25,24),
        (4,4,25),  (4,12,26),  (12,4,27), (12,12,28),
        (19,19,29),
        (25,1,30), (25,2,31),  (25,10,32),(25,18,33),(25,25,34),
        # SPD-SPD
        (1,5,35),   (1,13,36),  (1,31,37), (1,21,38), (1,36,39),
        (1,28,40),  (1,40,41),  (1,43,42), (1,45,43),
        (2,5,44),   (2,13,45),  (2,31,46), (2,21,47), (2,36,48),
        (2,28,49),  (2,40,50),  (2,43,51), (2,45,52),
        (10,5,53),  (10,13,54), (10,31,55),(10,21,56),(10,36,57),
        (10,28,58), (10,40,59), (10,43,60),(10,45,61),
        (3,20,62),  (3,6,63),   (3,14,64), (3,32,65), (3,23,66),
        (3,38,67),  (3,30,68),  (3,42,69),
        (11,20,70), (11,6,71),  (11,14,72),(11,32,73),(11,23,74),
        (11,38,75), (11,30,76), (11,42,77),
        (18,5,78),  (18,13,79), (18,31,80),(18,21,81),(18,36,82),
        (18,28,83), (18,40,84), (18,8,85), (18,16,86),
        (18,34,87), (18,43,88), (18,45,89),
        (4,26,90),  (4,7,91),   (4,15,92), (4,33,93), (4,29,94),
        (4,41,95),  (4,24,96),  (4,39,97),
        (12,26,98), (12,7,99),  (12,15,100),(12,33,101),(12,29,102),
        (12,41,103),(12,24,104),(12,39,105),
        (19,27,106),(19,22,107),(19,37,108),(19,9,109), (19,17,110),
        (19,35,111),
        (25,5,112), (25,13,113),(25,31,114),(25,21,115),(25,36,116),
        (25,28,117),(25,40,118),(25,8,119), (25,16,120),(25,34,121),
        (25,43,122),(25,45,123),
        (5,1,124),  (5,2,125),  (5,10,126),(5,18,127),(5,25,128),
        (5,5,129),  (5,13,130), (5,31,131),(5,21,132),(5,36,133),
        (5,28,134), (5,40,135), (5,43,136),(5,45,137),
        (13,1,138), (13,2,139), (13,10,140),(13,18,141),(13,25,142),
        (13,5,143), (13,13,144),(13,31,145),(13,21,146),(13,36,147),
        (13,28,148),(13,40,149),(13,43,150),(13,45,151),
        (20,3,152), (20,11,153),(20,20,154),(20,6,155), (20,14,156),
        (20,32,157),(20,23,158),(20,38,159),(20,30,160),(20,42,161),
        (26,4,162), (26,12,163),(26,26,164),(26,7,165), (26,15,166),
        (26,33,167),(26,29,168),(26,41,169),(26,24,170),(26,39,171),
        (31,1,172), (31,2,173), (31,10,174),(31,18,175),(31,25,176),
        (31,5,177), (31,13,178),(31,31,179),(31,21,180),(31,36,181),
        (31,28,182),(31,40,183),(31,43,184),(31,45,185),
        (6,3,186),  (6,11,187), (6,20,188), (6,6,189), (6,14,190),
        (6,32,191), (6,23,192), (6,38,193), (6,30,194), (6,42,195),
        (14,3,196), (14,11,197),(14,20,198),(14,6,199), (14,14,200),
        (14,32,201),(14,23,202),(14,38,203),(14,30,204),(14,42,205),
        (21,1,206), (21,2,207), (21,10,208),(21,18,209),(21,25,210),
        (21,5,211), (21,13,212),(21,31,213),(21,21,214),(21,36,215),
        (21,28,216),(21,40,217),(21,8,218), (21,16,219),(21,34,220),
        (21,43,221),(21,45,222),
        (27,19,223),(27,27,224),(27,22,225),(27,37,226),(27,9,227),
        (27,17,228),(27,35,229),
        (32,3,230), (32,11,231),(32,20,232),(32,6,233), (32,14,234),
        (32,32,235),(32,23,236),(32,38,237),(32,30,238),(32,42,239),
        (36,1,240), (36,2,241), (36,10,242),(36,18,243),(36,25,244),
        (36,5,245), (36,13,246),(36,31,247),(36,21,248),(36,36,249),
        (36,28,250),(36,40,251),(36,8,252), (36,16,253),(36,34,254),
        (36,43,255),(36,45,256),
        (7,4,257),  (7,12,258), (7,26,259), (7,7,260),  (7,15,261),
        (7,33,262), (7,29,263), (7,41,264), (7,24,265), (7,39,266),
        (15,4,267), (15,12,268),(15,26,269),(15,7,270), (15,15,271),
        (15,33,272),(15,29,273),(15,41,274),(15,24,275),(15,39,276),
        (22,19,277),(22,27,278),(22,22,279),(22,37,280),(22,9,281),
        (22,17,282),(22,35,283),
        (28,1,284), (28,2,285), (28,10,286),(28,18,287),(28,25,288),
        (28,5,289), (28,13,290),(28,31,291),(28,21,292),(28,36,293),
        (28,28,294),(28,40,295),(28,8,296), (28,16,297),(28,34,298),
        (28,43,299),(28,45,300),
        (33,4,301), (33,12,302),(33,26,303),(33,7,304), (33,15,305),
        (33,33,306),(33,29,307),(33,41,308),(33,24,309),(33,39,310),
        (37,19,311),(37,27,312),(37,22,313),(37,37,314),(37,9,315),
        (37,17,316),(37,35,317),
        (40,1,318), (40,2,319), (40,10,320),(40,18,321),(40,25,322),
        (40,5,323), (40,13,324),(40,31,325),(40,21,326),(40,36,327),
        (40,28,328),(40,40,329),(40,8,330), (40,16,331),(40,34,332),
        (40,43,333),(40,45,334),
        (8,18,335), (8,25,336), (8,21,337), (8,36,338), (8,28,339),
        (8,40,340), (8,8,341),  (8,16,342), (8,34,343),
        (16,18,344),(16,25,345),(16,21,346),(16,36,347),(16,28,348),
        (16,40,349),(16,8,350), (16,16,351), (16,34,352),
        (23,3,353), (23,11,354),(23,20,355),(23,6,356), (23,14,357),
        (23,32,358),(23,23,359),(23,38,360),(23,30,361),(23,42,362),
        (29,4,363), (29,12,364),(29,26,365),(29,7,366), (29,15,367),
        (29,33,368),(29,29,369),(29,41,370),(29,24,371),(29,39,372),
        (34,18,373),(34,25,374),(34,21,375),(34,36,376),(34,28,377),
        (34,40,378),(34,8,379), (34,16,380), (34,34,381),
        (38,3,382), (38,11,383),(38,20,384),(38,6,385), (38,14,386),
        (38,32,387),(38,23,388),(38,38,389),(38,30,390),(38,42,391),
        (41,4,392), (41,12,393),(41,26,394),(41,7,395), (41,15,396),
        (41,33,397),(41,29,398),(41,41,399),(41,24,400),(41,39,401),
        (43,1,402), (43,2,403), (43,10,404),(43,18,405),(43,25,406),
        (43,5,407), (43,13,408),(43,31,409),(43,21,410),(43,36,411),
        (43,28,412),(43,40,413),(43,43,414),(43,45,415),
        (9,19,416), (9,27,417), (9,22,418), (9,37,419), (9,9,420),
        (9,17,421), (9,35,422),
        (17,19,423),(17,27,424),(17,22,425),(17,37,426),(17,9,427),
        (17,17,428),(17,35,429),
        (24,4,430), (24,12,431),(24,26,432),(24,7,433), (24,15,434),
        (24,33,435),(24,29,436),(24,41,437),(24,24,438),(24,39,439),
        (30,3,440), (30,11,441),(30,20,442),(30,6,443), (30,14,444),
        (30,32,445),(30,23,446),(30,38,447),(30,30,448),(30,42,449),
        (35,19,450),(35,27,451),(35,22,452),(35,37,453),(35,9,454),
        (35,17,455),(35,35,456),
        (39,4,457), (39,12,458),(39,26,459),(39,7,460), (39,15,461),
        (39,33,462),(39,29,463),(39,41,464),(39,24,465),(39,39,466),
        (42,3,467), (42,11,468),(42,20,469),(42,6,470), (42,14,471),
        (42,32,472),(42,23,473),(42,38,474),(42,30,475),(42,42,476),
        (44,44,477),
        (45,1,478), (45,2,479), (45,10,480),(45,18,481),(45,25,482),
        (45,5,483), (45,13,484),(45,31,485),(45,21,486),(45,36,487),
        (45,28,488),(45,40,489),(45,43,490),(45,45,491)
    ]

    for args in s2_data:
        set2(ind2, *args)

    # --- isym  ---
    s1_data = [
        (40, 38), (41, 39), (43, 42),
        (49, 47), (50, 48), (52, 51),
        (58, 56), (59, 57), (61, 60),
        (68, 66), (69, 67),
        (76, 74), (77, 75),
        (89, 88), (90, 62), (91, 63), (92, 64), (93, 65),
        (94, -66),(95, -67),(96, 66), (97, 67),
        (98, 70), (99, 71), (100, 72), (101, 73),
        (102, -74),(103, -75),(104, 74), (105, 75),
        (106, 86), (107, 86), (109, 85), (110, 86), (111, 87),
        (112, 78), (113, 79), (114, 80), (115, 83), (116, 84),
        (117, 81), (118, 82), (119, -85),(120, -86),(121, -87),
        (122, 88), (123, 88),
        (128,127), (134,132), (135,133), (137,136),
        (142,141), (148,146), (149,147), (151,150),
        (160,158), (161,159),
        (162,152), (163,153), (164,154), (165,155), (166,156), (167,157),
        (168,-158), (169,-159), (170,158), (171,159),
        (176,175),
        (182,180), (183,181), (185,184),
        (194,192), (195,193),
        (204,202), (205,203),
        (222,221),
        (224,219), (225,219), (227,218), (228,219), (229,220),
        (238,236), (239,237),
        (256,255),
        (257,186), (258,187), (259,188), (260,189), (261,190), (262,191),
        (263,-192), (264,-193), (265,192), (266,193),
        (267,196), (268,197), (269,198), (270,199), (271,200), (272,201),
        (273,-202), (274,-203), (275,202), (276,203),
        (277,223), (278,219), (279,219), (280,226), (281,218),
        (282,219), (283,220),
        (284,206), (285,207), (286,208), (287,210), (288,209),
        (289,211), (290,212), (291,213), (292,216), (293,217),
        (294,214), (295,215),
        (296,-218), (297,-219), (298,-220), (299,221), (300,221),
        (301,230), (302,231), (303,232), (304,233), (305,234), (306,235),
        (307,-236), (308,-237), (309,236), (310,237),
        (312,253), (313,253), (315,252), (316,253), (317,254),
        (318,240), (319,241), (320,242), (321,244), (322,243),
        (323,245), (324,246), (325,247), (326,250), (327,251),
        (328,248), (329,249),
        (330,-252), (331,-253), (332,-254), (333,255), (334,255),
        (336,-335), (339,-337), (340,-338), (342,337),
        (344,223), (345,-223), (346,219), (347,226), (348,-219), (349,-226),
        (350,218), (351,219), (352,220),
        (363,-353), (364,-354), (365,-355), (366,-356), (367,-357), (368,-358),
        (369,359), (370,360), (371,-361), (372,-362),
        (374,-373), (377,-375), (378,-376), (380,375),
        (392,-382), (393,-383), (394,-384), (395,-385), (396,-386), (397,-387),
        (398,388), (399,389), (400,-390), (401,-391),
        (406,405), (412,410), (413,411),
        (416,335), (417,337), (418,337), (419,338), (420,341),
        (421,337), (422,343),
        (423,223), (424,219), (425,219), (426,226), (427,218), (428,219), (429,220),
        (430,353), (431,354), (432,355), (433,356), (434,357), (435,358),
        (436,-361), (437,-362), (438,359), (439,360),
        (440,353), (441,354), (442,355), (443,356), (444,357), (445,358),
        (446,361), (447,362), (448,359), (449,360),
        (450,373), (451,375), (452,375), (453,376), (454,379), (455,375),
        (456,381), (457,382), (458,383), (459,384), (460,385), (461,386), (462,387),
        (463,-390), (464,-391), (465,388), (466,389),
        (467,382), (468,383), (469,384), (470,385), (471,386), (472,387),
        (473,390), (474,391), (475,388), (476,389),
        (478,402), (479,403), (480,404), (481,405), (482,405),
        (483,407), (484,408), (485,409), (486,410), (487,411), (488,410),
        (489,411), (490,415), (491,414)
    ]

    for args in s1_data:
        set1(isym, *args)
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
        def rep(n, v):
            return [v]*n
        
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
        set_ch(9,0,0,1.0)
        set_ch(9,2,0,1.33333333)
        set_ch(10,2,1,1.0)
        set_ch(11,2,-1,1.0)
        set_ch(12,1,0,1.15470054)
        set_ch(13,1,1,1.0)
        set_ch(14,1,-1,1.0)
        set_ch(17,0,0,1.0)
        set_ch(17,2,0,-0.66666667)
        set_ch(17,2,2,1.0)
        set_ch(18,2,-2,1.0)
        set_ch(19,1,1,-0.57735027)
        set_ch(20,1,0,1.0)
        set_ch(22,1,1,1.0)
        set_ch(23,1,-1,1.0)
        set_ch(24,0,0,1.0)
        set_ch(24,2,0,-0.66666667)
        set_ch(24,2,2,-1.0)
        set_ch(25,1,-1,-0.57735027)
        set_ch(27,1,0,1.0)
        set_ch(28,1,-1,-1.0)
        set_ch(29,1,1,1.0)
        set_ch(30,0,0,1.0)
        set_ch(30,2,0,1.33333333)
        set_ch(31,2,1,0.57735027)
        set_ch(32,2,-1,0.57735027)
        set_ch(33,2,2,-1.15470054)
        set_ch(34,2,-2,-1.15470054)
        set_ch(35,0,0,1.0)
        set_ch(35,2,0,0.66666667)
        set_ch(35,2,2,1.0)
        set_ch(36,2,-2,1.0)
        set_ch(37,2,1,1.0)
        set_ch(38,2,-1,1.0)
        set_ch(39,0,0,1.0)
        set_ch(39,2,0,0.66666667)
        set_ch(39,2,2,-1.0)
        set_ch(40,2,-1,-1.0)
        set_ch(41,2,1,1.0)
        set_ch(42,0,0,1.0)
        set_ch(42,2,0,-1.33333333)
        set_ch(44,0,0,1.0)
        set_ch(44,2,0,-1.33333333)
        
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
