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

####################################################
#   Example of CCSD (experimental)
####################################################

import pyscf
from gpu4pyscf.cc import ccsd_incore

mol = pyscf.M(
    atom = 'Vitamin_C.xyz',
    basis = 'cc-pvdz',
    verbose=1)

mf = mol.RHF().run()
mf.with_df = None   # DF CCSD is not supported yet.
e_tot = ccsd_incore.CCSD(mf).kernel()
