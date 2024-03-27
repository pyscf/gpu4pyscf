# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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

import os
import numpy as np
import pyscf
import cupy

from pyscf import lib
from pyscf.geomopt.geometric_solver import optimize
from gpu4pyscf.dft import uks, rks
from gpu4pyscf.scf import hf

"""
This script reproduces the result in JCTC, 16(3):2002–2012, 2020.
"""

# How to use
# 1. Download the dataset (1) from JCTC, 16(3):2002–2012, 2020. PMID: 32074450
# 2. Download the dataset (2) The Journal of Physical Chemistry A, 123(17):3761–3781, 2019.
# 2. Replace certain molecules in (1) with (2) 
#       11/ts.xyz with ts11.xyz, 
#       12/ts.xyz with ts12.xyz, 
#       14/ts.xyz with ts14.xyz 

path = 'ct9b01266_si_002/'
charges = np.zeros(35)
charges[1] = 1
charges[2] = 1
charges[32] = 1
charges[33] = 1
transition_metals = ['Mo','Nb','Ru','Pd','Rh','Ta','W','Re','Os','Ir','Pt']
ecp = {sym: 'def2-svp' for sym in transition_metals}
grids_for_heavy = {sym: (200, 1202) for sym in transition_metals}
molecule_list = [1,2,3,4,5,6,8,9,11,12,13,14,15,16,21,22,23,24,25,26,27,28,29,30,31,32,33,34]

def calc_hessian(pmol):
    mol = pmol.copy()
    mol.verbose = 4
    mf = hf.RHF(mol).density_fit()
    mf.disp = 'd3bj'
    mf.conv_tol = 1e-10
    mf.conv_tol_cpscf = 1e-3
    mf.kernel()
    h = mf.Hessian()
    h.auxbasis_response = 1
    h_dft = h.kernel()
    natm = h_dft.shape[0]
    h_dft = h_dft.transpose([0,2,1,3]).reshape([3*natm,3*natm])
    return h_dft

def _check_grad(mol, tol=1e-5):
    mol.basis = 'def2-tzvpp'
    mol.build()
    mf = rks.RKS(mol, xc='b3lyp').density_fit()
    mf.grids.level = 7
    mf.conv_tol = 1e-12
    mf.verbose = 1
    mf.screen_tol = 1e-14
    mf.disp = 'd3bj'
    mf.grids.prune = None
    mf.small_rho_cutoff = 1e-15
    e_tot = mf.kernel()
    
    g = mf.nuc_grad_method()
    g.auxbasis_response = True

    g_scanner = g.as_scanner()
    g_analy = g_scanner(mol)[1]
    g_disp = g_scanner.grad_disp
    
    eps = 1e-3
    f_scanner = mf.as_scanner()
    coords = mol.atom_coords()
    grad_fd = np.zeros_like(coords)
    for i in range(len(coords)):
        for j in range(3):
            #coords = mol.atom_coords()
            coords[i,j] += eps
            mol.set_geom_(coords, unit='Bohr')
            mol.build()
            e0 = f_scanner(mol)
            e0_disp = f_scanner.scf_summary['dispersion']
            coords[i,j] -= 2.0 * eps
            mol.set_geom_(coords, unit='Bohr')
            mol.build()
            e1 = f_scanner(mol)
            e1_disp = f_scanner.scf_summary['dispersion']
            coords[i,j] += eps
            mol.set_geom_(coords, unit='Bohr')
            grad_fd[i,j] = (e0-e1)/2.0/eps
            print(i,j,'difference between analytical and finite difference gradient:', abs(g_analy[i,j] - grad_fd[i,j]))
    grad_fd = np.array(grad_fd).reshape(-1,3)
    print('analytical gradient:')
    print(g_analy)
    print('finite difference gradient:')
    print(grad_fd)
    print('difference between analytical and finite difference gradient:', cupy.linalg.norm(g_analy - grad_fd))
    assert(cupy.linalg.norm(g_analy - grad_fd) < tol)

def run_optimize(atom, charge, transition):
    print(f'optimizing {atom}')
    charge = charge
    xc = 'tpss'
    bas = 'def2-svp'
    auxbasis = 'def2-tzvpp-jkfit'
    scf_tol = 1e-10
    max_scf_cycles = 100
    screen_tol = 1e-14

    mol = pyscf.M(
        atom=atom, 
        basis=bas, 
        ecp=ecp,
        max_memory=32000, 
        charge=charge, 
        spin=None)

    if transition:
        # hessian matrix for initial guess
        h_dft = calc_hessian(mol)
        from tempfile import NamedTemporaryFile
        outfile = NamedTemporaryFile()
        np.savetxt(outfile.name, h_dft)

    mol.output = atom[:-4] + '.log'
    mol.verbose = 1
    mf_GPU = rks.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf_GPU.disp = 'd3bj'
    mf_GPU.grids.level = 7
    mf_GPU.conv_tol = scf_tol
    mf_GPU.max_cycle = max_scf_cycles
    mf_GPU.screen_tol = screen_tol

    mf_GPU.grids.atom_grid = grids_for_heavy
    import time
    start_time = time.time()
    mf_GPU.kernel()
    print(mf_GPU.scf_summary)

    output_file = atom[:-4] + '_opt.xyz'
    conv_params = {
        'convergence_energy': 1e-4,  # Eh
        'convergence_grms': 3e-3,    # Eh/Bohr
        'convergence_gmax': 4.5e-3,  # Eh/Bohr
        'convergence_drms': 1.2e-2,  # Angstrom
        'convergence_dmax': 1.8e-2,  # Angstrom
    }
    if transition:
        mol_eq = optimize(mf_GPU, maxsteps=200, transition=transition, hessian='file:'+outfile.name, **conv_params)
    else:
        mol_eq = optimize(mf_GPU, maxsteps=200, **conv_params)
    print("Optimized coordinate:")
    print(mol_eq.atom_coords())
    print('Optimization takes', time.time() - start_time, 's')
    mol_eq.tofile(output_file, format='xyz')

# ===========================
# optimize start molecule
# ==========================
for i in molecule_list:
    charge = charges[i]
    if not os.path.isdir(path + str(i)):
        continue
    start_atom = path + str(i) + '/start.xyz'
    run_optimize(start_atom, charge, False)

# ===========================
# optimize end molecule
# ==========================
for i in molecule_list:
    charge = charges[i]
    if not os.path.isdir(path + str(i)):
        continue
    end_atom = path + str(i) + '/end.xyz'
    run_optimize(end_atom, charge, False)

# ===========================
# optimize transition state
# ===========================
for i in molecule_list:
    charge = charges[i]
    if not os.path.isdir(path + str(i)):
        continue
    ts_atom = path + str(i) + '/ts.xyz'
    run_optimize(ts_atom, charge, True)


def run_dft(atom, charge):
    xc = 'wb97m-v'
    bas = 'def2-tzvpp'
    auxbasis = 'def2-tzvpp-jkfit'
    scf_tol = 1e-11
    max_scf_cycles = 100
    screen_tol = 1e-14
    mol = pyscf.M(
        atom=atom, 
        basis=bas, 
        ecp=ecp,
        max_memory=32000, 
        charge=charge, 
        spin=None)
    mol.output = atom[:-4] + '.log'
    mol.verbose = 1
    mf_GPU = uks.UKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf_GPU.grids.atom_grid = (99,590)
    mf_GPU.nlcgrids.atom_grid = (50,194)
    mf_GPU.grids.atom_grid = grids_for_heavy
    mf_GPU.grids.prune = None
    mf_GPU.small_rho_cutoff = 1e-15
    mf_GPU.conv_tol = scf_tol
    mf_GPU.max_cycle = max_scf_cycles
    mf_GPU.screen_tol = screen_tol
    e_tot = mf_GPU.kernel()
    return e_tot

# ====================================
# single-point energy with wb97m-v
# ====================================
f = open("recompute.txt", "w+")
for i in molecule_list:
    charge = charges[i]
    if not os.path.isdir(path + str(i)):
        continue
    start_atom = path + str(i) + '/start.xyz'
    end_atom = path + str(i) + '/end.xyz'
    ts_atom = path + str(i) + '/ts.xyz'

    h2kcal = 627.509
    e_start = run_dft(start_atom, charge) * h2kcal
    e_end = run_dft(end_atom, charge) * h2kcal
    e_ts = run_dft(ts_atom, charge) * h2kcal
    print(f"{i}, {e_ts - e_start}, {e_end - e_start}, {e_start}, {e_ts}, {e_end}")
    f.write(f"{i}, {(e_ts - e_start):.3f}, {(e_end - e_start):.3f}, {e_start:.3f}, {e_ts:.3f}, {e_end:.3f}\n")
    f.flush()
f.close()

f = open("reoptimize.txt", "w+")
for i in molecule_list:
    charge = charges[i]
    if not os.path.isdir(path + str(i)):
        continue
    start_atom = path + str(i) + '/start_opt.xyz'
    end_atom = path + str(i) + '/end_opt.xyz'
    ts_atom = path + str(i) + '/ts_opt.xyz'

    h2kcal = 627.509
    e_start = run_dft(start_atom, charge) * h2kcal
    e_end = run_dft(end_atom, charge) * h2kcal
    e_ts = run_dft(ts_atom, charge) * h2kcal
    print(f"{i}, {e_ts - e_start}, {e_end - e_start}, {e_start}, {e_ts}, {e_end}")
    f.write(f"{i}, {e_ts - e_start}, {e_end - e_start}, {e_start}, {e_ts}, {e_end}\n")
    f.flush()
f.close()