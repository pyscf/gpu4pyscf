#!/usr/bin/env python
# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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

###########################################################
#  Example of PM6 calculations
###########################################################

from gpu4pyscf import sem

atoms = """
    O 0.0000 0.0000 0.0000
    H 0.7570 0.5860 0.0000
    H -0.7570 0.5860 0.0000
    """

# Create a PM6 molecule object
mol = sem.gto.mole.Mole(atoms, verbose=4)
mol.build()

# Create a PM6 meanfield object
mf = sem.scf.hf.RHF(mol)
mf.kernel()     # -11.7257185892348
