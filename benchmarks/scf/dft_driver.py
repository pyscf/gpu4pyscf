import os
import csv
import pyscf
import time
import argparse
from pyscf import lib
from pyscf.dft import rks

lib.num_threads(8)

parser = argparse.ArgumentParser(description='Run SCF, grad, and Hessian in GPU4PySCF for molecules')
parser.add_argument('--basis',        type=str, default='def2-tzvpp')
parser.add_argument('--verbose',      type=int, default=1)
parser.add_argument('--xc',           type=str, default='B3LYP')
parser.add_argument('--device',       type=str, default='GPU')
parser.add_argument('--input_path',   type=str, default='./')
parser.add_argument('--output_path',  type=str, default='./')
parser.add_argument('--with_hessian', type=bool, default=False)
args = parser.parse_args()
bas = args.basis
verbose = args.verbose
xc = args.xc

if xc == 'LDA':
    xc = 'LDA,VWN5'

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

if args.device == 'GPU':
    import cupy
    import gpu4pyscf
    from gpu4pyscf.dft import rks
    props = cupy.cuda.runtime.getDeviceProperties(0)
    device = props['name'].decode('ascii')
    output_file = device+'.csv'
else:
    from pyscf.dft import rks
    output_file = 'PySCF-16-cores-CPU.csv'
output_file = args.output_path + output_file

def run_dft(filename):  
    mol = pyscf.M(atom=filename, basis=bas, max_memory=64000)
    start_time = time.time()  
    # set verbose >= 6 for debugging timer
    mol.verbose = 4 #verbose
    mol.max_memory = 40000
    mf = rks.RKS(mol, xc=xc)
    mf.grids.atom_grid = (99,590)
    mf.chkfile = None
    prep_time = time.time() - start_time
    mf.conv_tol = 1e-9
    mf.nlcgrids.atom_grid = (50,194)
    mf.max_cycle = 100
    try:
        e_dft = mf.kernel()
        scf_time = time.time() - start_time
    except:
        scf_time = -1
        e_dft = 0

    # calculate gradient
    if args.device == 'GPU':
        cupy.get_default_memory_pool().free_all_blocks()
    try:
        start_time = time.time()
        g = mf.nuc_grad_method()
        g.max_memory = 40000
        f = g.kernel()
        grad_time = time.time() - start_time
    except:
        grad_time = -1

    # calculate hessian
    if args.device == 'GPU':
        cupy.get_default_memory_pool().free_all_blocks()
    
    hess_time = -1
    if args.with_hessian:
        try:
            start_time = time.time()
            h = mf.Hessian()
            h.max_memory = 40000
            hess = h.kernel()
            hess_time = time.time() - start_time
        except:
            hess_time = -1

    return mol.natm, mol.nao, scf_time, grad_time, hess_time, e_dft

fields = ['mol','natm', 'nao', 't_scf', 't_gradient', 't_hessian', 'e_tot']
csvfile = open(output_file, 'w')
csvwriter = csv.writer(csvfile)
csvwriter.writerow(fields)

for filename in sorted(os.listdir(args.input_path)):
    if filename.endswith(".xyz"):
        print(f'running DFT {filename}')
        info = run_dft(args.input_path+filename)
        row = [filename[:-4]]+list(info)
        csvwriter.writerow(row)
        csvfile.flush()
csvfile.close()
