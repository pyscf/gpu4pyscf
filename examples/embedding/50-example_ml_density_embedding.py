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

"""
Example: ML-Driven DFT Embedding (ONIOM-like scheme)

This example demonstrates how to use the `HarrisRKS` and `SingleFragmentEmbedding_ML` 
classes to perform a multi-scale quantum chemistry calculation (QM/QM). 
It uses a dummy ML density evaluator to simulate an ultra-fast global PBE calculation, 
and then performs a rigorous B3LYP high-level calculation only on the active fragment.
"""

import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.dft import rks
from gpu4pyscf.qmmm.embedding.embedding_dft_harris import HarrisRKS, SingleFragmentEmbedding_ML


def dummy_eval_density_func(mol, xc, grids):
    """
    A pure DFT surrogate that mimics the behavior of an ML density predictor.
    It performs a standard SCF to convergence and returns the exact potentials 
    and energies, acting as the "Ground Truth" ML model.
    """
    print("\n[ML Surrogate] Generating density and effective potentials...")
    mf = rks.RKS(mol)
    mf.xc = xc
    mf.grids = grids
    mf.verbose = 0
    mf.kernel()
    
    dm = cp.asarray(mf.make_rdm1())
    vj, vk = mf.get_jk(mol, dm)
    e_j = 0.5 * float(cp.sum(dm * vj))
    
    is_hybrid = mf._numint.libxc.is_hybrid_xc(xc)
    if is_hybrid:
        hyb = mf._numint.libxc.hybrid_coeff(xc, spin=mol.spin)
        vk = vk * hyb
        e_k = 0.5 * float(cp.sum(dm * vk))
    else:
        vk = None
        e_k = 0.0
        
    _, e_xc, vxc = mf._numint.nr_rks(mol, grids, xc, dm)
    int_rho_vxc = float(cp.sum(dm * vxc))
    
    print("[ML Surrogate] Potential generation completed.\n")
    return vj, vk, vxc, e_j, e_k, float(e_xc), int_rho_vxc


def main():
    # 1. Build a target molecule (e.g., Hexane)
    mol = gto.Mole()
    mol.atom = '''
        C   1.4522500000  -2.8230000000   0.0000000000
        C   1.4522500000  -1.2830000000   0.0000000000
        C   0.0002500000  -0.7700000000   0.0000000000
        C   0.0002500000   0.7700000000   0.0000000000
        C  -1.4517500000   1.2830000000   0.0000000000
        C  -1.4517500000   2.8230000000   0.0000000000
        H   2.4792500000  -3.1870000000   0.0000000000
        H   0.9382500000  -3.1870000000   0.8900000000
        H   0.9382500000  -3.1870000000  -0.8900000000
        H   1.9652500000  -0.9200000000   0.8900000000
        H   1.9652500000  -0.9200000000  -0.8900000000
        H  -0.5137500000  -1.1330000000  -0.8900000000
        H  -0.5137500000  -1.1330000000   0.8900000000
        H   0.5132500000   1.1330000000   0.8900000000
        H   0.5132500000   1.1330000000  -0.8900000000
        H  -1.9657500000   0.9200000000  -0.8900000000
        H  -1.9657500000   0.9200000000   0.8900000000
        H  -2.4797500000   3.1870000000   0.0000000000
        H  -0.9377500000   3.1870000000   0.8900000000
        H  -0.9377500000   3.1870000000  -0.8900000000
    '''
    mol.basis = 'sto3g' # Use a small basis set for quick demonstration
    mol.spin = 0
    mol.verbose = 4
    mol.build()

    # 2. Define the active region (e.g., the terminal methyl group: C + 3xH)
    methyl_fragment = [0, 6, 7, 8]
    
    print("==================================================")
    print("   Starting ML-Driven DFT Embedding Calculation   ")
    print("==================================================")

    # 3. Setup the Global Low-Level Solver (driven by ML)
    # This evaluates the full system using the Harris functional approach in 1 step.
    mf_outer = HarrisRKS(mol, dummy_eval_density_func, xc='PBE')
    
    # 4. Setup the Local High-Level Solver (Standard rigorous DFT)
    # This will only be executed within the embedded active space.
    mf_inner = rks.RKS(mol, xc='B3LYP')
    
    # 5. Initialize and execute the ML Embedding framework
    emb_obj = SingleFragmentEmbedding_ML(mf_outer, mf_inner, methyl_fragment)
    e_tot = emb_obj.kernel()
    
    print("\n==================================================")
    print("                 Summary of Results               ")
    print("==================================================")
    print(f"Global Low-Level E (ML-PBE) : {mf_outer.e_tot:.8f} Hartree")
    print(f"High-Level Local E (B3LYP)  : {emb_obj.e_inner[0]:.8f} Hartree")
    print(f"Low-Level Local E (PBE)     : {emb_obj.e_inner[0] - emb_obj.e_tot + mf_outer.e_tot:.8f} Hartree") # Reverse engineered for display
    print(f"--------------------------------------------------")
    print(f"FINAL ONIOM TOTAL ENERGY    : {e_tot:.8f} Hartree")
    print("==================================================")

if __name__ == '__main__':
    main()