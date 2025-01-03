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

import pyscf
import time
import argparse
from pyscf import lib
from gpu4pyscf import dft

parser = argparse.ArgumentParser(description='Run DFT with GPU4PySCF for molecules')
parser.add_argument("--input",        type=str,  default='benzene/coord')
parser.add_argument("--basis",        type=str,  default='def2-tzvpp')
parser.add_argument("--auxbasis",     type=str,  default='def2-tzvpp-jkfit')
parser.add_argument("--xc",           type=str,  default='B3LYP')
parser.add_argument("--solvent",      type=str,  default='')
args = parser.parse_args()

start_time = time.time()
bas = args.basis
mol = pyscf.M(
    atom=args.input,
    basis=bas,
    max_memory=32000)
# set verbose >= 6 for debugging timer
mol.verbose = 6

mf_df = dft.RKS(mol, xc=args.xc).density_fit(auxbasis=args.auxbasis)
mf_df.verbose = 6

if args.solvent:
    mf_df = mf_df.PCM()
    mf_df.with_solvent.lebedev_order = 29
    mf_df.with_solvent.method = args.solvent
    mf_df.with_solvent.eps = 78.3553

mf_df.grids.atom_grid = (99,590)
if mf_df._numint.libxc.is_nlc(mf_df.xc):
    mf_df.nlcgrids.atom_grid = (50,194)
mf_df.direct_scf_tol = 1e-14
mf_df.conv_tol = 1e-10
mf_df.chkfile = None
mf_df.conv_tol_cpscf = 1e-6
e_tot = mf_df.kernel()
scf_time = time.time() - start_time
print(f'compute time for energy: {scf_time:.3f} s')

start_time = time.time()
g = mf_df.nuc_grad_method()
g.auxbasis_response = True
f = g.kernel()
grad_time = time.time() - start_time
print(f'compute time for gradient: {grad_time:.3f} s')

start_time = time.time()
h = mf_df.Hessian()
h.auxbasis_response = 2
h_dft = h.kernel()
hess_time = time.time() - start_time
print(f'compute time for hessian: {hess_time:.3f} s')
