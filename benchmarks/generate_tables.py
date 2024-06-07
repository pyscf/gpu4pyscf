import numpy as np
import h5py
import pandas as pd
import os
import argparse

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark_dir', type=str, default='tmp/2023-11-23-14-46-22-7cee/output/benchmarks')
parser.add_argument('--pyscf_dir', type=str, default=None)
parser.add_argument('--qchem_dir', type=str, default=None)
args = parser.parse_args()

benchmark_dir = args.benchmark_dir
pyscf_dir = args.pyscf_dir
qchem_dir = args.qchem_dir
if pyscf_dir is None:
    pyscf_dir = benchmark_dir
if qchem_dir is None:
    qchem_dir = benchmark_dir

diff_empty    = {'mol': [], 'scf': [], 'grad': [], 'hess': []}
speedup_empty = {'mol': [], 'scf': [], 'grad': [], 'hess': []}

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

def gen_table(pyscf_path, qchem_path, molecules, solvent=False):
    df_diff = pd.DataFrame(diff_empty)
    df_speedup = pd.DataFrame()
    scf_speedup = []
    grad_speedup = []
    hess_speedup = []
    scf_diff = []
    grad_diff = []
    hess_diff = []
    for mol in molecules:

        scf_speedup_tmp = -1
        grad_speedup_tmp = -1
        hess_speedup_tmp = -1
        scf_diff_tmp = -1
        grad_diff_tmp = -1
        hess_diff_tmp = -1
        try:
            fp_pyscf_h5 = h5py.File(f'{pyscf_path}/{mol}_pyscf.h5', 'r')
            fp_qchem_h5 = h5py.File(f'{qchem_path}/{mol}_qchem.h5', 'r')

            if ('scf_time' in fp_qchem_h5 and 'scf_time' in fp_pyscf_h5 and
            fp_qchem_h5['scf_time'][()] != -1 and
            fp_pyscf_h5['scf_time'][()] != -1):
                scf_speedup_tmp = fp_qchem_h5['scf_time'][()]/fp_pyscf_h5['scf_time'][()]

            if ('grad_time' in fp_qchem_h5 and 'grad_time' in fp_pyscf_h5 and
            fp_qchem_h5['grad_time'][()] != -1 and
            fp_pyscf_h5['grad_time'][()] != -1):
                grad_speedup_tmp = fp_qchem_h5['grad_time'][()]/fp_pyscf_h5['grad_time'][()]

            if ('total_time' in fp_qchem_h5 and 'hess_time' in fp_pyscf_h5 and
            fp_qchem_h5['total_time'][()] != -1 and
            fp_pyscf_h5['hess_time'][()] != -1):
                hess_speedup_tmp = fp_qchem_h5['total_time'][()]/fp_pyscf_h5['hess_time'][()]

            # special treatment for solvent model
            if not solvent:
                qchem_grad = fp_qchem_h5['grad'][()]
            else:
                qchem_grad = read_solvent_gradient(qchem_path, mol)

            if ('e_tot' in fp_qchem_h5 and 'e_tot' in fp_pyscf_h5 and
            fp_qchem_h5['e_tot'][()] != -1 and
            fp_pyscf_h5['e_tot'][()] != -1):
                scf_diff_tmp = np.linalg.norm(fp_qchem_h5['e_tot'][()] - fp_pyscf_h5['e_tot'][()])

            if ('grad' in fp_qchem_h5 and 'grad' in fp_pyscf_h5 and
            isinstance(fp_qchem_h5['grad'][()], np.ndarray) and
            isinstance(fp_pyscf_h5['grad'][()], np.ndarray)):
                grad_diff_tmp = np.linalg.norm(qchem_grad - fp_pyscf_h5['grad'][()])

            if ('hess' in fp_qchem_h5 and 'hess' in fp_pyscf_h5 and
            isinstance(fp_qchem_h5['hess'][()], np.ndarray) and
            isinstance(fp_pyscf_h5['hess'][()], np.ndarray)):
                pyscf_hess = fp_pyscf_h5['hess'][()]
                hess_diag = pyscf_hess.sum(axis=0)
                pyscf_hess -= np.diag(hess_diag)
                hess_diff_tmp = np.linalg.norm(fp_qchem_h5['hess'][()] - pyscf_hess)

        except Exception:
            pass

        scf_speedup.append(scf_speedup_tmp)
        grad_speedup.append(grad_speedup_tmp)
        hess_speedup.append(hess_speedup_tmp)
        scf_diff.append(scf_diff_tmp)
        grad_diff.append(grad_diff_tmp)
        hess_diff.append(hess_diff_tmp)

    df_diff = pd.DataFrame(
        {
            'mol': molecules,
            'scf diff': scf_diff,
            'grad diff': grad_diff,
            'hess diff': hess_diff
        })

    df_speedup = pd.DataFrame(
        {
            'mol': molecules,
            'scf speedup': scf_speedup,
            'grad speedup': grad_speedup,
            'hess speedup': hess_speedup
        }
    )
    return df_diff, df_speedup

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

# Basis
pyscf_path = os.path.join(pyscf_dir, 'df', 'organic', 'basis')
qchem_path = os.path.join(qchem_dir, 'df', 'organic', 'basis')
df_basis_diff = pd.DataFrame({'mol': molecules})
df_basis_speedup = pd.DataFrame({'mol': molecules})
for basis in ['sto-3g', '6-31g', 'def2-svp', 'def2-tzvpp', 'def2-tzvpd']:
    df_diff, df_speedup = gen_table(f'{pyscf_path}/{basis}/', f'{qchem_path}/{basis}', molecules)
    df_diff = df_diff.rename(columns={
        'scf diff': f'{basis}\n scf',
        'grad diff': f'{basis}\n grad',
        'hess diff': f'{basis}\n hess'})
    df_speedup = df_speedup.rename(columns={
        'scf speedup': f'{basis}\n scf',
        'grad speedup': f'{basis}\n grad',
        'hess speedup': f'{basis}\n hess'})
    df_basis_diff = df_basis_diff.merge(df_diff, how='outer', on='mol')
    df_basis_speedup = df_basis_speedup.merge(df_speedup, how='outer', on='mol')
print("=====================================================================")
print("=                         DF basis table                            =")
print("=====================================================================")
print('---------------------  accuracy --------------------------------------')
print(df_basis_diff.to_markdown(index=False, floatfmt='.2e'))
print('----------------------- speedup --------------------------------------')
print(df_basis_speedup.to_markdown(index=False, floatfmt='.3f'))
df_basis_diff.to_csv('basis_diff.csv', index=False)

# XC
pyscf_path = os.path.join(pyscf_dir, 'df', 'organic', 'xc')
qchem_path = os.path.join(qchem_dir, 'df', 'organic', 'xc')
df_xc_diff = pd.DataFrame({'mol': molecules})
df_xc_speedup = pd.DataFrame({'mol': molecules})
for xc in ['LDA', 'PBE', 'M06', 'B3LYP', 'wB97m-v']:
    df_diff, df_speedup = gen_table(f'{pyscf_path}/{xc}/', f'{qchem_path}/{xc}', molecules)
    df_diff = df_diff.rename(columns={
        'scf diff': f'{xc}\n scf',
        'grad diff': f'{xc}\n grad',
        'hess diff': f'{xc}\n hess'})
    df_speedup = df_speedup.rename(columns={
        'scf speedup': f'{xc}\n scf',
        'grad speedup': f'{xc}\n grad',
        'hess speedup': f'{xc}\n hess'})
    df_xc_diff = df_xc_diff.merge(df_diff, how='outer', on='mol')
    df_xc_speedup = df_xc_speedup.merge(df_speedup, how='outer', on='mol')

print("======================================================================")
print("=                        DF xc table                                 =")
print("======================================================================")
print('---------------------  accuracy --------------------------------------')
print(df_xc_diff.to_markdown(index=False, floatfmt='.2e'))
print('----------------------- speedup --------------------------------------')
print(df_xc_speedup.to_markdown(index=False, floatfmt='.3f'))
df_xc_diff.to_csv('xc_diff.csv', index=False)

# solvent
pyscf_path = os.path.join(pyscf_dir, 'df', 'organic', 'solvent')
qchem_path = os.path.join(qchem_dir, 'scf', 'organic', 'solvent')
df_sol_diff = pd.DataFrame({'mol': molecules})
df_sol_speedup = pd.DataFrame({'mol': molecules})
for sol in ['CPCM', 'IEFPCM']:
    df_diff, df_speedup = gen_table(f'{pyscf_path}/{sol}/', f'{qchem_path}/{sol}', molecules, solvent=True)
    df_diff = df_diff.rename(columns={
        'scf diff': f'{sol}\n scf',
        'grad diff': f'{sol}\n grad',
        'hess diff': f'{sol}\n hess'})
    df_speedup = df_speedup.rename(columns={
        'scf speedup': f'{sol}\n scf',
        'grad speedup': f'{sol}\n grad',
        'hess speedup': f'{sol}\n hess'})
    df_sol_diff = df_sol_diff.merge(df_diff, how='outer', on='mol')
    df_sol_speedup = df_sol_speedup.merge(df_speedup, how='outer', on='mol')

print("======================================================================")
print("=                       DF solvent table                             =")
print("======================================================================")
print('---------------------  accuracy --------------------------------------')
print(df_sol_diff.to_markdown(index=False, floatfmt='.2e'))
print('----------------------- speedup --------------------------------------')
print(df_sol_speedup.to_markdown(index=False, floatfmt='.3f'))
df_sol_diff.to_csv('solvent_diff.csv', index=False)
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
pyscf_path = os.path.join(pyscf_dir, 'scf', 'water_clusters', 'basis')
qchem_path = os.path.join(qchem_dir, 'scf', 'water_clusters', 'basis')
df_basis_diff = pd.DataFrame({'mol': molecules})
df_basis_speedup = pd.DataFrame({'mol': molecules})
for basis in ['sto-3g', '6-31g', 'def2-svp', 'def2-tzvpp', 'def2-tzvpd']:
    df_diff, df_speedup = gen_table(f'{pyscf_path}/{basis}/', f'{qchem_path}/{basis}', molecules)
    df_diff = df_diff.drop(columns=['hess diff'])
    df_speedup = df_speedup.drop(columns=['hess speedup'])
    df_diff = df_diff.rename(columns={
        'scf diff': f'{basis}\n scf',
        'grad diff': f'{basis}\n grad'})
    df_speedup = df_speedup.rename(columns={
        'scf speedup': f'{basis}\n scf',
        'grad speedup': f'{basis}\n grad'})
    df_basis_diff = df_basis_diff.merge(df_diff, how='outer', on='mol')
    df_basis_speedup = df_basis_speedup.merge(df_speedup, how='outer', on='mol')

print("======================================================================")
print("=                       SCF basis table                              =")
print("======================================================================")
print('---------------------  accuracy --------------------------------------')
print(df_basis_diff.to_markdown(index=False, floatfmt='.2e'))
print('----------------------- speedup --------------------------------------')
print(df_basis_speedup.to_markdown(index=False, floatfmt='.3f'))

# XC
pyscf_path = os.path.join(pyscf_dir, 'scf', 'water_clusters', 'xc')
qchem_path = os.path.join(qchem_dir, 'scf', 'water_clusters', 'xc')
df_xc_diff = pd.DataFrame({'mol': molecules})
df_xc_speedup = pd.DataFrame({'mol': molecules})
for xc in ['LDA', 'PBE', 'M06', 'B3LYP', 'wB97m-v']:
    df_diff, df_speedup = gen_table(f'{pyscf_path}/{xc}/', f'{qchem_path}/{xc}', molecules)
    df_diff = df_diff.drop(columns={'hess diff'})
    df_speedup = df_speedup.drop(columns={'hess speedup'})
    df_diff = df_diff.rename(columns={
        'scf diff': f'{xc}\n scf',
        'grad diff': f'{xc}\n grad'})
    df_speedup = df_speedup.rename(columns={
        'scf speedup': f'{xc}\n scf',
        'grad speedup': f'{xc}\n grad'})
    df_xc_diff = df_xc_diff.merge(df_diff, how='outer', on='mol')
    df_xc_speedup = df_xc_speedup.merge(df_speedup, how='outer', on='mol')

print("======================================================================")
print('=                      SCF xc table                                  =')
print("======================================================================")
print('---------------------  accuracy --------------------------------------')
print(df_xc_diff.to_markdown(index=False, floatfmt='.2e'))
print('----------------------- speedup --------------------------------------')
print(df_xc_speedup.to_markdown(index=False, floatfmt='.3f'))
