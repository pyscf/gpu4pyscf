#!/usr/bin/env python
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

'''
Absolutely localized molecular orbital (ALMO) energy decomposition analysis (EDA) version 2
'''

import pyscf
from gpu4pyscf.properties import eda

basis = 'def2-TZVPD'

mol1 = pyscf.M(
    atom = '''
        O      0.000000   -0.000000    0.117400
        H     -0.757000   -0.000000   -0.469600
        H      0.757000    0.000000   -0.469600
    ''',
    basis = basis,
    charge = 0,
)

mol2 = pyscf.M(
    atom = '''
        Cl    -1.940555    0.888176   -2.558508
    ''',
    basis = basis,
    charge = -1,
)

mol_list = [mol1, mol2]

eda_result, dft_result = eda.eval_ALMO_EDA_2_energies(mol_list, xc = "wB97M-V")
print(f"EDA result in dict form: {eda_result}")
print(f"DFT energies of each fragment and the total system: {dft_result}")

### Reference output:
# Fragment 0 energy = -76.4334344665 Hartree
# Fragment 1 energy = -460.2597922428 Hartree
# Total system energy = -536.7128492630 Hartree
# EDA frozen energy = -0.0139122432 Hartree = -36.5265895348 kJ/mol
# EDA total = -0.0196225536 Hartree = -51.5190073780 kJ/mol
# EDA classical electrostatic = -0.0196654385 Hartree = -51.6316017069 kJ/mol
# EDA electrostatic = -0.0277539574 Hartree = -72.8680051124 kJ/mol
# EDA dispersion = -0.0029903292 Hartree = -7.8511082095 kJ/mol
# EDA Pauli (kinetic energy pressure + interfragment exchange) = -0.0119297257 Hartree = -31.3214903590 kJ/mol
# EDA Pauli (frozen - electrostatic - dispersion) = 0.0168320434 Hartree = 44.1925237871 kJ/mol
# EDA polarization = -0.0033884280 Hartree = -8.8963165666 kJ/mol
# EDA charge transfer = -0.0023218824 Hartree = -6.0961012766 kJ/mol
# EDA result in dict form: {'total': -51.519007378047505, 'frozen': -36.52658953482568,
#                           'electrostatic': -72.86800511241047, 'dispersion': -7.851108209545869,
#                           'pauli': 44.19252378713065, 'polarization': -8.896316566606751,
#                           'charge transfer': -6.096101276615072, 'unit': 'kJ/mol'}

### The result is consistent with the following Q-Chem input and output:
# $molecule
# -1 1
# --
# 0 1
#         O      0.000000   -0.000000    0.117400
#         H     -0.757000   -0.000000   -0.469600
#         H      0.757000    0.000000   -0.469600
# --
# -1 1
#         Cl    -1.940555    0.888176   -2.558508
# $end

# $rem
# JOBTYPE                     eda
# EDA2                        1
# METHOD                      wB97M-V
# BASIS                       def2-TZVPD
# XC_GRID                     000099000590
# NL_GRID                     000050000194
# MAX_SCF_CYCLES              100
# SCF_CONVERGENCE             10
# THRESH                      14
# MEM_STATIC                  8000
# MEM_TOTAL                   80000
# SYMMETRY                    FALSE
# SYM_IGNORE                  TRUE
# $end

# ================================
#         Results of EDA2         
# ================================
# Basic EDA Quantities
# --------------------
# Fragment Energies (Ha):
# 1  -76.4334344584
# 2 -460.2597934814
# --------------------
#   E_prp (kJ/mol) = -0.0000
#   E_frz (kJ/mol) = -36.5263
#   E_pol (kJ/mol) = -8.8958
#   E_vct (kJ/mol) = -6.0962
#   E_int (kJ/mol) = -51.5183
# --------------------


# Decomposition of frozen interaction energy
# --------------------
#   --------------------
#   Orthogonal Frozen Decomposition:
#   --------------------
#      E_elec   (ELEC)   (kJ/mol) = -72.8689
#      E_pauli  (PAULI)  (kJ/mol) = 44.1942
#      E_disp   (DISP)   (kJ/mol) = -7.8516
#   --------------------
#   Classical Frozen Decomposition:
#   --------------------
#      E_cls_elec  (CLS ELEC)  (kJ/mol) = -51.6322
#      E_mod_pauli (MOD PAULI) (kJ/mol) = 22.9576 (FRZ - CLS ELEC - DISP)
#      E_disp      (DISP)      (kJ/mol) = -7.8516
#   --------------------
# --------------------

# Simplified EDA Summary (kJ/mol)
# --------------------
#  PREPARATION      -0.0000
#  FROZEN           -36.5263 (ELEC + PAULI + DISP)
# [ELEC + PAULI = -28.6747, DISP = -7.8516]
#  POLARIZATION    -8.8958
#  CHARGE TRANSFER -6.0962
#  TOTAL           -51.5183     (PRP + FRZ + POL + CT)
# --------------------
