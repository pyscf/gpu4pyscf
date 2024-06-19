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
import utils

def read_solvent_gradient(path, mol):
    ''' special treatment, solvent gradient is stored in .out '''
    keywords = ' -- total gradient after adding PCM contribution --'
    l = len(keywords)
    qchem_out = f'{path}/{mol}_qchem.out'
    with open(qchem_out, 'r') as out_file:
        lines = out_file.readlines()
        for i, line in enumerate(lines):
            if lines[i][:l] == keywords:
                break
        i += 4
        g = []
        while(lines[i][:8] != '--------'):
            g.append(lines[i].split('     ')[1:])
            i += 1
    return np.asarray(g, dtype=np.float64)

def generate_scf_tables(source_path, target_path, molecules):
    scf_diff = []
    scf_speedup = []
    for mol in molecules:
        scf_speedup_tmp = -1
        scf_diff_tmp = -1
        try:
            source_file = glob.glob(f'{source_path}/{mol}_*.h5')[-1]
            target_file = glob.glob(f'{target_path}/{mol}_*.h5')[-1]
            fp_source_h5 = h5py.File(source_file, 'r')
            fp_target_h5 = h5py.File(target_file, 'r')

            if ('scf_time' in fp_target_h5 and 'scf_time' in fp_source_h5 and
            fp_target_h5['scf_time'][()] != -1 and
            fp_source_h5['scf_time'][()] != -1):
                scf_speedup_tmp = fp_target_h5['scf_time'][()]/fp_source_h5['scf_time'][()]
            
            if ('e_tot' in fp_target_h5 and 'e_tot' in fp_source_h5 and
            fp_target_h5['e_tot'][()] != -1 and
            fp_source_h5['e_tot'][()] != -1):
                scf_diff_tmp = np.linalg.norm(fp_target_h5['e_tot'][()] - fp_source_h5['e_tot'][()])
        except Exception:
            pass
        
        scf_speedup.append(scf_speedup_tmp)
        scf_diff.append(scf_diff_tmp)
    df_diff = pd.DataFrame({'mol': molecules, 'scf diff': scf_diff})
    df_speedup = pd.DataFrame({'mol': molecules, 'scf speedup': scf_speedup})
    return df_diff, df_speedup

def generate_grad_tables(source_path, target_path, molecules, solvent=False):
    grad_speedup = []
    grad_diff = []
    for mol in molecules:
        grad_speedup_tmp = -1
        grad_diff_tmp = -1
        try:
            source_file = glob.glob(f'{source_path}/{mol}_*.h5')[-1]
            target_file = glob.glob(f'{target_path}/{mol}_*.h5')[-1]
            fp_source_h5 = h5py.File(source_file, 'r')
            fp_target_h5 = h5py.File(target_file, 'r')

            if ('grad_time' in fp_target_h5 and 'grad_time' in fp_source_h5 and
            fp_target_h5['grad_time'][()] != -1 and
            fp_source_h5['grad_time'][()] != -1):
                grad_speedup_tmp = fp_target_h5['grad_time'][()]/fp_source_h5['grad_time'][()]

            # special treatment for solvent model
            if not solvent:
                qchem_grad = fp_target_h5['grad'][()]
            else:
                qchem_grad = read_solvent_gradient(target_path, mol)

            if ('grad' in fp_target_h5 and 'grad' in fp_source_h5 and
            isinstance(fp_target_h5['grad'][()], np.ndarray) and
            isinstance(fp_source_h5['grad'][()], np.ndarray)):
                grad_diff_tmp = np.linalg.norm(qchem_grad - fp_source_h5['grad'][()])

        except Exception:
            pass

        grad_speedup.append(grad_speedup_tmp)
        grad_diff.append(grad_diff_tmp)

    df_diff = pd.DataFrame({'mol': molecules, 'grad diff': grad_diff})
    df_speedup = pd.DataFrame({'mol': molecules, 'grad speedup': grad_speedup})
    return df_diff, df_speedup

def generate_hess_tables(source_path, target_path, molecules):
    hess_speedup = []
    hess_diff = []
    for mol in molecules:
        hess_speedup_tmp = -1
        hess_diff_tmp = -1
        try:
            source_file = glob.glob(f'{source_path}/{mol}_*.h5')[-1]
            target_file = glob.glob(f'{target_path}/{mol}_*.h5')[-1]
            fp_source_h5 = h5py.File(source_file, 'r')
            fp_target_h5 = h5py.File(target_file, 'r')

            if ('total_time' in fp_target_h5 and 'hess_time' in fp_source_h5 and
            fp_target_h5['total_time'][()] != -1 and
            fp_source_h5['hess_time'][()] != -1):
                hess_speedup_tmp = fp_target_h5['total_time'][()]/fp_source_h5['hess_time'][()]

            if ('hess' in fp_target_h5 and 'hess' in fp_source_h5 and
            isinstance(fp_target_h5['hess'][()], np.ndarray) and
            isinstance(fp_source_h5['hess'][()], np.ndarray)):
                pyscf_hess = fp_source_h5['hess'][()]
                hess_diag = pyscf_hess.sum(axis=0)
                pyscf_hess -= np.diag(hess_diag)
                hess_diff_tmp = np.linalg.norm(fp_target_h5['hess'][()] - pyscf_hess)

        except Exception:
            pass

        hess_speedup.append(hess_speedup_tmp)
        hess_diff.append(hess_diff_tmp)

    df_diff = pd.DataFrame({'mol': molecules, 'hess diff': hess_diff})
    df_speedup = pd.DataFrame({'mol': molecules,'hess speedup': hess_speedup})
    return df_diff, df_speedup

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

    diff_empty    = {'mol': [], 'scf': [], 'grad': [], 'hess': []}
    speedup_empty = {'mol': [], 'scf': [], 'grad': [], 'hess': []}

    # ================== DF ==========================
    molecules = [
            "020_Vitamin_C",
            "031_Inosine",
            "033_Bisphenol_A",
            "037_Mg_Porphin",
            "042_Penicillin_V",
            "045_Ochratoxin_A",
            "052_Cetirizine",
            "057_Tamoxifen",
            "066_Raffinose",
            "084_Sphingomyelin",
            "095_Azadirachtin",
            "113_Taxol",
            "168_Valinomycin"]

    # solvent
    print("")
    print("DF with solvent methods")
    print("")
    source_path = os.path.join(source_dir, 'df', 'organic', 'solvent')
    target_path = os.path.join(target_dir, 'scf', 'organic', 'solvent')
    df_sol_diff = pd.DataFrame({'mol': molecules})
    df_sol_speedup = pd.DataFrame({'mol': molecules})
    for sol in ['CPCM', 'IEFPCM']:
        df_diff, df_speedup = generate_scf_tables(f'{source_path}/{sol}/', f'{target_path}/{sol}', molecules)
        df_diff = df_diff.rename(columns={'scf diff': f'{sol}\n scf'})
        df_speedup = df_speedup.rename(columns={'scf speedup': f'{sol}\n scf'})
        df_sol_diff = df_sol_diff.merge(df_diff, how='outer', on='mol')
        df_sol_speedup = df_sol_speedup.merge(df_speedup, how='outer', on='mol')
        
        df_diff, df_speedup = generate_grad_tables(f'{source_path}/{sol}/', f'{target_path}/{sol}', molecules, solvent=True)
        df_diff = df_diff.rename(columns={'grad diff': f'{sol}\n grad'})
        df_speedup = df_speedup.rename(columns={'grad speedup': f'{sol}\n grad'})
        df_sol_diff = df_sol_diff.merge(df_diff, how='outer', on='mol')
        df_sol_speedup = df_sol_speedup.merge(df_speedup, how='outer', on='mol')

        df_diff, df_speedup = generate_hess_tables(f'{source_path}/{sol}/', f'{target_path}/{sol}', molecules)
        df_diff = df_diff.rename(columns={'hess diff': f'{sol}\n hess'})
        df_speedup = df_speedup.rename(columns={'hess speedup': f'{sol}\n hess'})
        df_sol_diff = df_sol_diff.merge(df_diff, how='outer', on='mol')
        df_sol_speedup = df_sol_speedup.merge(df_speedup, how='outer', on='mol')
    utils.print_markdown(df_sol_diff, df_sol_speedup, target_software=args.target_software)
