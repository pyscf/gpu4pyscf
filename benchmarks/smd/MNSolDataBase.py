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

############################################
#  Benchmark Minnesota Solvation (MNSOL) Database with SMD solvent model
############################################

import pyscf
from gpu4pyscf import dft, scf

def calc_sfe(atom, charge, desc, method='b3lyp', basis='def2-tzvpp', water_solvent=False):
    """
    Calculate the solvation free energy of a molecule

    Parameters:
    atom (str): Atomic symbol of the molecule
    charge (int): Charge of the molecule
    desc (str): Description of the solvent molecule, will not be used for water solvent
    xc (str): Exchange-correlation functional
    basis (str): Basis set for the calculation
    water_solvent (bool): Whether to use water as the solvent
        if yes, desc is not effective

    Returns:
    float: The solvation free energy
    """
    mol = pyscf.M(atom=atom, basis=basis, verbose=1, charge=charge, spin=None)
    if method.lower() == 'hf':
        mf = scf.HF(mol).density_fit()
    else:
        mf = dft.KS(mol, xc=method).density_fit()
        mf.grids.atom_grid = (99,590)
    e_gas = mf.kernel()

    mf = mf.SMD()
    mf.with_solvent.method = 'SMD'

    if not water_solvent:
        mf.with_solvent.sol_desc = desc
    else:
        mf.with_solvent.solvent = 'water'
    e_sol = mf.kernel()
    return e_sol - e_gas

def read_xyz_charge(xyz_file):
    """
    Reads an xyz file and returns the atom structure and the charge.

    Parameters:
    xyz_file (str): The path to the xyz file.

    Returns:
    tuple: A tuple containing two elements:
        - atom (str): The atom structure.
        - charge (int): The charge.
    """
    with open(xyz_file) as f:
        lines = f.readlines()
    atom = lines[2].split()
    charge = int(atom[0])
    atom = ''.join(lines[3:])
    return atom, charge

if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    import argparse
    import csv

    parser = argparse.ArgumentParser(description='Calculate SFE with GPU4PySCF for molecules')
    parser.add_argument("--input_dir",    type=str,  default="./")
    parser.add_argument("--basis",        type=str,  default='def2-tzvpp')
    parser.add_argument("--method",       type=str,  default='B3LYP')
    parser.add_argument('--unrestricted', type=bool, default=False)
    args = parser.parse_args()

    f = open(args.method + '_' + args.basis + ".csv", "w+")
    writer = csv.writer(f)
    path = args.input_dir #'/mnt/bn/gpu4pyscf-lq3/MNSolDatabase_v2012'
    df = pd.read_excel(os.path.join(path, 'MNSol_alldata.xls'))
    for it, row in df.iterrows():
        # Skip the cases for testing
        if row['FileHandle'][:4] == 'test': 
            continue

        # Skip the cases for the diff of solvation free energy
        if row['Solvent'][-6:] == '-water':
            continue

        xyzfile = os.path.join(path, 'all_solutes/'+row['FileHandle']+'.xyz')
        atom, charge = read_xyz_charge(xyzfile)
        descriptor = [row['n'],
                      0, 
                      row['alpha'], 
                      row['beta'], 
                      row['gamma'], 
                      row['eps'], 
                      row['phi**2']**.5, 
                      row['psi**2']**.5]

        water_solvent = row['Solvent'] == 'water'
        e_sol = calc_sfe(atom,
                        charge, 
                        descriptor, 
                        method=args.method, 
                        basis=args.basis, 
                        water_solvent=water_solvent)
    
        writer.writerow([row['FileHandle'], 
                         row['Solvent'], 
                         charge, 
                         e_sol * 627.509, 
                         str(row['DeltaGsolv'])])

        print(it,
              row['FileHandle'], 
              row['Solvent'], 
              charge, 
              e_sol * 627.509, 
              row['DeltaGsolv'])
    f.close()
    
