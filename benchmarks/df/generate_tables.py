import pandas as pd
import numpy as np

# -------------------------------------------
# |   Density fitting with different basis  |
# -------------------------------------------

A100_file = 'NVIDIA A100-SXM4-80GB.csv'
V100_file = 'Tesla V100-SXM2-32GB.csv'
qchem_file = 'qchem-32-cores-cpu.csv'

keys = ['mol', 'natm']
empty = {'mol':[], 'natm':[]}
df_A100_scf = pd.DataFrame(empty)
df_V100_scf = pd.DataFrame(empty)
df_A100_grad = pd.DataFrame(empty)
df_V100_grad = pd.DataFrame(empty)
path = 'organic/basis/'

for basis in ['sto-3g', '6-31g', 'def2-svp', 'def2-tzvpp', 'def2-tzvpd']:
    df_qchem = pd.read_csv(path + basis + '/' + qchem_file)
    df_qchem = df_qchem.rename(columns={'t_scf':'scf_qchem', 't_gradient':'grad_qchem'})

    df_A100 = pd.read_csv(path + basis + '/' + A100_file)
    df_A100 = df_A100.rename(columns={'t_scf':'scf_A100', 't_gradient':'grad_A100'})
    df_A100 = df_A100.merge(df_qchem, how='outer', on='mol')

    df_A100['scf_'+basis] = df_A100['scf_qchem']/df_A100['scf_A100']
    df_A100['grad_'+basis] = df_A100['grad_qchem']/df_A100['grad_A100']
    df_A100 = df_A100[keys+['scf_'+basis, 'grad_'+basis]]

    df_A100_scf = df_A100_scf.merge(df_A100[keys+['scf_'+basis]], how='outer', on=keys)
    df_A100_grad= df_A100_grad.merge(df_A100[keys+['grad_'+basis]], how='outer', on=keys)
    df_A100_scf = df_A100_scf.rename(columns={'scf_'+basis:basis})
    df_A100_grad = df_A100_grad.rename(columns={'grad_'+basis:basis})
    df_A100_scf[basis] = df_A100_scf[basis].apply(lambda x: round(x,2))
    df_A100_grad[basis] = df_A100_grad[basis].apply(lambda x: round(x,2))

    df_V100 = pd.read_csv(path + basis + '/' + V100_file)
    df_V100 = df_V100.rename(columns={'t_scf':'scf_V100', 't_gradient':'grad_V100'})
    df_V100 = df_V100.merge(df_qchem, how='outer', on='mol')
    df_V100['scf_'+basis] = df_V100['scf_qchem']/df_V100['scf_V100']
    df_V100['grad_'+basis] = df_V100['grad_qchem']/df_V100['grad_V100']

    df_V100_scf = df_V100_scf.merge(df_V100[keys+['scf_'+basis,]], how='outer', on=keys)
    df_V100_grad= df_V100_grad.merge(df_V100[keys+['grad_'+basis]], how='outer', on=keys)
    df_V100_scf = df_V100_scf.rename(columns={'scf_'+basis:basis})

    df_V100_grad = df_V100_grad.rename(columns={'grad_'+basis:basis})
    df_V100_scf[basis] = df_V100_scf[basis].apply(lambda x: round(x,2))
    df_V100_grad[basis] = df_V100_grad[basis].apply(lambda x: round(x,2))

print("\n============SCF speedup with A100-80G============\n")
print(df_A100_scf.to_markdown(index=False))
print("\n============SCF speedup with V100-32G============\n")
print(df_V100_scf.to_markdown(index=False))
print("\n============Gradient speedup with A100-80G=======\n")
print(df_A100_grad.to_markdown(index=False))
print("\n============Gradient speedup with V100-32G=======\n")
print(df_V100_grad.to_markdown(index=False))

# -----------------------------------------
# |   Density fitting with different xc   |
# -----------------------------------------

keys = ['mol', 'natm']
empty = {'mol':[], 'natm':[]}
df_A100_scf = pd.DataFrame(empty)
df_V100_scf = pd.DataFrame(empty)
df_A100_grad = pd.DataFrame(empty)
df_V100_grad = pd.DataFrame(empty)
path = 'organic/xc/'
for xc in ['LDA', 'PBE', 'B3LYP', 'M06', 'wB97m-v']:
    df_qchem = pd.read_csv(path + xc + '/' + qchem_file)
    df_qchem = df_qchem.rename(columns={'t_scf':'scf_qchem', 't_gradient':'grad_qchem'})

    df_A100 = pd.read_csv(path + xc + '/' + A100_file)
    df_A100 = df_A100.rename(columns={'t_scf':'scf_A100', 't_gradient':'grad_A100'})
    df_A100 = df_A100.merge(df_qchem, how='outer', on='mol')

    df_A100['scf_'+xc] = df_A100['scf_qchem']/df_A100['scf_A100']
    df_A100['grad_'+xc] = df_A100['grad_qchem']/df_A100['grad_A100']
    df_A100 = df_A100[keys+['scf_'+xc, 'grad_'+xc]]

    df_A100_scf = df_A100_scf.merge(df_A100[keys+['scf_'+xc]], how='outer', on=keys)
    df_A100_grad= df_A100_grad.merge(df_A100[keys+['grad_'+xc]], how='outer', on=keys)
    df_A100_scf = df_A100_scf.rename(columns={'scf_'+xc:xc})
    df_A100_grad = df_A100_grad.rename(columns={'grad_'+xc:xc})
    df_A100_scf[xc] = df_A100_scf[xc].apply(lambda x: round(x,2))
    df_A100_grad[xc] = df_A100_grad[xc].apply(lambda x: round(x,2))

    df_V100 = pd.read_csv(path + xc + '/' + V100_file)
    df_V100 = df_V100.rename(columns={'t_scf':'scf_V100', 't_gradient':'grad_V100'})
    df_V100 = df_V100.merge(df_qchem, how='outer', on='mol')
    df_V100['scf_'+xc] = df_V100['scf_qchem']/df_V100['scf_V100']
    df_V100['grad_'+xc] = df_V100['grad_qchem']/df_V100['grad_V100']

    df_V100_scf = df_V100_scf.merge(df_V100[keys+['scf_'+xc,]], how='outer', on=keys)
    df_V100_grad= df_V100_grad.merge(df_V100[keys+['grad_'+xc]], how='outer', on=keys)
    df_V100_scf = df_V100_scf.rename(columns={'scf_'+xc:xc})

    df_V100_grad = df_V100_grad.rename(columns={'grad_'+xc:xc})
    df_V100_scf[xc] = df_V100_scf[xc].apply(lambda x: round(x,2))
    df_V100_grad[xc] = df_V100_grad[xc].apply(lambda x: round(x,2))

print("\n============SCF speedup with A100-80G============\n")
print(df_A100_scf.to_markdown(index=False))
print("\n============SCF speedup with V100-32G============\n")
print(df_V100_scf.to_markdown(index=False))
print("\n============Gradient speedup with A100-80G=======\n")
print(df_A100_grad.to_markdown(index=False))
print("\n============Gradient speedup with V100-32G=======\n")
print(df_V100_grad.to_markdown(index=False))
