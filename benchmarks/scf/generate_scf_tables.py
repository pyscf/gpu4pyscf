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

import numpy as np
import h5py
import pandas as pd
import os
import argparse
import glob

import sys
sys.path.append('../lib')
import utils

if __name__ == '__main__':

    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default=None)
    parser.add_argument('--target_dir', type=str, default=None)
    parser.add_argument('--target_software', type=str, default=None)

    args = parser.parse_args()

    source_dir = args.source_dir
    target_dir = args.target_dir

    print(f'# {args.target_software} vs GPU4PySCF v1.0')

    # ============================ SCF ======================================
    molecules = [
            "002",
            "003",
            "004",
            "005",
            "006",
            "007",
            "008",
            "009",
            "010"]
    # basis
    print("")
    print("## Direct SCF Energy with B3LYP/*")
    print("")
    source_path = os.path.join(source_dir, 'water_clusters', 'basis')
    target_path = os.path.join(target_dir, 'water_clusters', 'basis')
    df_basis_diff = pd.DataFrame({'mol': molecules})
    df_basis_speedup = pd.DataFrame({'mol': molecules})
    for basis in ['sto-3g', '6-31g', 'def2-svp', 'def2-tzvpp', 'def2-tzvpd']:
        df_diff, df_speedup = utils.generate_scf_tables(
            f'{source_path}/{basis}/', 
            f'{target_path}/{basis}', 
            molecules)
        df_diff = df_diff.rename(columns={'scf diff': f'{basis}'})
        df_speedup = df_speedup.rename(columns={'scf speedup': f'{basis}'})
        df_basis_diff = df_basis_diff.merge(df_diff, how='outer', on='mol')
        df_basis_speedup = df_basis_speedup.merge(df_speedup, how='outer', on='mol')
    utils.print_markdown(df_basis_diff, df_basis_speedup, target_software=args.target_software)

    print("")
    print("## Direct SCF Gradient with B3LYP/*")
    print("")
    df_basis_diff = pd.DataFrame({'mol': molecules})
    df_basis_speedup = pd.DataFrame({'mol': molecules})
    for basis in ['sto-3g', '6-31g', 'def2-svp', 'def2-tzvpp', 'def2-tzvpd']:
        df_diff, df_speedup = utils.generate_grad_tables(
            f'{source_path}/{basis}/', 
            f'{target_path}/{basis}', 
            molecules)
        df_diff = df_diff.rename(columns={'grad diff': f'{basis}'})
        df_speedup = df_speedup.rename(columns={'grad speedup': f'{basis}'})
        df_basis_diff = df_basis_diff.merge(df_diff, how='outer', on='mol')
        df_basis_speedup = df_basis_speedup.merge(df_speedup, how='outer', on='mol')
    utils.print_markdown(df_basis_diff, df_basis_speedup, target_software=args.target_software)

    # XC
    print("")
    print('## Direct SCF Energy with */def2-tzvpp')
    print("")
    source_path = os.path.join(source_dir, 'water_clusters', 'xc')
    target_path = os.path.join(target_dir, 'water_clusters', 'xc')
    df_xc_diff = pd.DataFrame({'mol': molecules})
    df_xc_speedup = pd.DataFrame({'mol': molecules})
    for xc in ['HF', 'LDA', 'PBE', 'M06', 'B3LYP', 'wB97m-v']:
        df_diff, df_speedup = utils.generate_scf_tables(
            f'{source_path}/{xc}/', 
            f'{target_path}/{xc}', 
            molecules)
        df_diff = df_diff.rename(columns={'scf diff': f'{xc}'})
        df_speedup = df_speedup.rename(columns={'scf speedup': f'{xc}'})
        df_xc_diff = df_xc_diff.merge(df_diff, how='outer', on='mol')
        df_xc_speedup = df_xc_speedup.merge(df_speedup, how='outer', on='mol')
    utils.print_markdown(df_xc_diff, df_xc_speedup, target_software=args.target_software)

    print("")
    print("## Direct SCF Gradient with */def2-tzvpp")
    print("")
    df_xc_diff = pd.DataFrame({'mol': molecules})
    df_xc_speedup = pd.DataFrame({'mol': molecules})
    for xc in ['HF', 'LDA', 'PBE', 'M06', 'B3LYP', 'wB97m-v']:
        df_diff, df_speedup = utils.generate_grad_tables(
            f'{source_path}/{xc}/', 
            f'{target_path}/{xc}', 
            molecules)
        df_diff = df_diff.rename(columns={'grad diff': f'{xc}'})
        df_speedup = df_speedup.rename(columns={'grad speedup': f'{xc}'})
        df_xc_diff = df_xc_diff.merge(df_diff, how='outer', on='mol')
        df_xc_speedup = df_xc_speedup.merge(df_speedup, how='outer', on='mol')
    utils.print_markdown(df_xc_diff, df_xc_speedup, target_software=args.target_software)
