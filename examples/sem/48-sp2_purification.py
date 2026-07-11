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

"""
Example usage of SP2 density matrix purification.
This script demonstrates how to use SP2 purification in semi-empirical SCF calculations.
"""

from gpu4pyscf import sem

# Create a simple molecule
mol = sem.gto.mole.Mole(
    atom="""
        O 0.0000 0.0000 0.0000
        H 0.7570 0.5860 0.0000
        H -0.7570 0.5860 0.0000
        """,
    verbose=4
)
mol.build()

# Standard RHF calculation (for reference)
print("=" * 50)
print("Standard RHF Calculation")
print("=" * 50)
mf_ref = sem.scf.hf.RHF(mol)
e_ref = mf_ref.kernel()
dm_ref = mf_ref.make_rdm1()

# RHF with SP2 purification
print("\n" + "=" * 50)
print("RHF with SP2 Purification")
print("=" * 50)
mf_sp2 = sem.scf.hf.RHF(mol).purification(
    conv_tol=1e-8,
    max_cycle=50,
    eig_method='gershgorin',
)
e_sp2 = mf_sp2.kernel()

# Compare results
print("\n" + "=" * 50)
print("Comparison")
print("=" * 50)
print(f"Reference energy: {e_ref}")
print(f"SP2 energy: {e_sp2}")
print(f"Energy difference: {e_sp2 - e_ref}")


