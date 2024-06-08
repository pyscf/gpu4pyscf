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
from gpu4pyscf import dft

def calc_sfe(atom, charge, desc, xc='b3lyp', basis='def2-tzvpp'):
    mol = pyscf.M(atom=atom, basis=basis, verbose=1, charge=charge)
    mf = dft.rks.RKS(mol, xc=xc).density_fit()
    mf.grids.atom_grid = (99,590)
    e_gas = mf.kernel()

    mf = mf.SMD()
    mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
    mf.with_solvent.method = 'SMD'
    mf.with_solvent.sol_desc = desc
    e_sol = mf.kernel()
    return e_sol - e_gas

def read_xyz_charge(xyz_file):
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
    parser.add_argument("--basis",        type=str,  default='def2-tzvpp')
    parser.add_argument("--xc",           type=str,  default='B3LYP')
    parser.add_argument('--unrestricted', type=bool, default=False)
    args = parser.parse_args()

    f = open(args.xc + '_' + args.basis +".csv", "w+")
    writer = csv.writer(f)
    path = '/mnt/bn/gpu4pyscf-lq3/MNSolDatabase_v2012'
    df = pd.read_excel(os.path.join(path, 'MNSol_alldata.xls'))
    for it, row in df.iterrows():
        if row['FileHandle'][:4] == 'test': 
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
        try:
            e_sol = calc_sfe(atom, charge, descriptor, xc=args.xc, basis=args.basis)
        except:
            continue
        writer.writerow([row['FileHandle'], row['Solvent'], charge, e_sol * 627.509, str(row['DeltaGsolv'])])

        print(it, charge, e_sol * 627.509, row['DeltaGsolv'])
    f.close()