#!/usr/bin/env python
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

########################################
#  Example of DF-MP2
########################################

import pyscf
from gpu4pyscf.scf import RHF
from gpu4pyscf.mp import dfmp2

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='ccpvdz')
mf = RHF(mol).density_fit()
e_hf = mf.kernel()

ptobj = dfmp2.DFMP2(mf)
e_corr, t2 = ptobj.kernel()
e_mp2 = e_hf + e_corr

# It prints out MP2 energies, those energies are assessible in the PT object.
print('MP2 correlation energy:', ptobj.emp2)
print('SCS MP2 correlation energy:', ptobj.emp2_scs)
print('Total energy with SCS MP2:', ptobj.e_tot_scs)

print('----- frozen core --------')

# frozen core
ptobj.frozen = [0]
e_corr, t2 = ptobj.kernel()
e_mp2 = e_hf + e_corr

print('MP2 correlation energy:', ptobj.emp2)
print('SCS MP2 correlation energy:', ptobj.emp2_scs)
print('Total energy with SCS MP2:', ptobj.e_tot_scs)
