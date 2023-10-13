# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pyscf
import time
from pyscf import lib

from gpu4pyscf.dft import rks
lib.num_threads(8)

import argparse

parser = argparse.ArgumentParser(description='Run DFT with GPU4PySCF for molecules')
parser.add_argument("--input", type=str, default='benzene/coord')
parser.add_argument("--basis", type=str, default='def2-tzvpp')
parser.add_argument("--auxbasis", type=str, default='def2-tzvpp-jkfit')
args = parser.parse_args()

start_time = time.time()
bas = args.basis
mol = pyscf.M(
    atom=args.input, 
    basis=bas, 
    max_memory=32000)
# set verbose >= 6 for debugging timer
mol.verbose = 6
print(mol.nao)

mf_df = rks.RKS(mol, xc='HYB_GGA_XC_B3LYP').density_fit(auxbasis=args.auxbasis)
mf_df.grids.atom_grid = (99,590)
mf_df.kernel()
print('compute time for energy: {}s'.format((time.time() - start_time)))
exit()
start_time = time.time()
g = mf_df.nuc_grad_method()
g.auxbasis_response = True
f = g.kernel()
print('compute time for gradient: {}s'.format((time.time() - start_time)))

start_time = time.time()
h = mf_df.Hessian()
h.auxbasis_response = 2
h_dft = h.kernel()
print('compute time for hessian: {}s'.format((time.time() - start_time)))
