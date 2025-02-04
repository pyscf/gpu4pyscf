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

###################################################################
# This is a customized driver for three composite methods only
# It only works for b97-3c, r2scan-3c, and wb97x-3c
###################################################################

import os
import json
import pyscf
import argparse
import tempfile
import shutil
import h5py
from types import MethodType
from pyscf import lib, gto
from pyscf import dft, scf
from pyscf.geomopt.geometric_solver import kernel

from gpu4pyscf.drivers.dft_3c_driver import (
    parse_3c, gen_disp_fun, gen_disp_grad_fun)

def opt_mol(mol_name, config, constraints, charge=None, spin=0):
    xc              = config.get('xc',              'b3lyp')
    verbose         = config.get('verbose',         4)
    scf_conv_tol    = config.get('scf_conv_tol',    1e-10)
    with_df         = config.get('with_df',         True)
    auxbasis        = config.get('auxbasis',       None)
    with_gpu        = config.get('with_gpu',        True)
    with_solvent    = config.get('with_solvent',    False)
    maxsteps        = config.get('maxsteps',        50)
    convergence_set = config.get('convergence_set', 'GAU')

    default_solvent = {'method': 'iefpcm', 'eps': 78.3553, 'solvent': 'water'}
    with_solvent   = config.get('with_solvent',   False)
    solvent        = config.get('solvent',        default_solvent)
    
    # I/O
    fp = tempfile.TemporaryDirectory()
    local_dir = f'{fp.name}/'
    logfile = f'{mol_name[:-4]}_pyscf.log'

    shutil.copyfile(config['input_dir']+mol_name, local_dir+mol_name)
    if constraints is not None:
        shutil.copyfile(config['input_dir']+constraints, local_dir+constraints)

    pyscf_xc, nlc, basis, ecp, (xc_disp, disp), xc_gcp = parse_3c(xc)

    lib.num_threads(8)
    mol = pyscf.M(
        atom=local_dir+mol_name,
        basis=basis,
        ecp=ecp,
        max_memory=32000,
        verbose=verbose,
        charge=charge,
        spin=spin,
        output=f'{local_dir}/{logfile}')
    mol.build()

    mf = dft.KS(mol, xc=pyscf_xc)
    mf.grids.atom_grid = (99,590)
    if mf._numint.libxc.is_nlc(mf.xc):
        mf.nlcgrids.atom_grid = (50,194)
    mf.disp = disp
    if with_df:
        pyscf_auxbasis = auxbasis
        if auxbasis == "RIJK-def2-tzvp":
            pyscf_auxbasis = 'def2-tzvp-jkfit'
        mf = mf.density_fit(auxbasis=pyscf_auxbasis)
    if with_gpu:
        mf = mf.to_gpu()

    #### Changes for 3C methods #####
    # Setup dispersion correction and GCP
    mf.nlc = nlc
    mf.get_dispersion = MethodType(gen_disp_fun(xc_disp, xc_gcp), mf)
    mf.do_disp = lambda: True
    #################################

    mf.chkfile = None

    if with_solvent:
        if solvent['method'].endswith(('PCM', 'pcm')):
            mf = mf.PCM()
            mf.with_solvent.lebedev_order = 29
            mf.with_solvent.method = solvent['method'].replace('PCM','-PCM')
            mf.with_solvent.eps = solvent['eps']
        elif with_solvent and solvent['method'].endswith(('smd', 'SMD')):
            mf = mf.SMD()
            mf.with_solvent.lebedev_order = 29
            mf.with_solvent.method = 'SMD'
            mf.with_solvent.solvent = solvent['solvent']
        else:
            raise NotImplementedError

    mf.direct_scf_tol = 1e-14
    mf.chkfile = None
    mf.conv_tol = scf_conv_tol

    history = []
    def callback(envs):
        result = {
            'energy':    envs['energy'],
            'gradients': envs['gradients'],
            'coords':    envs['coords'].tolist(),
            'e1':        mf.scf_summary.get('e1',         0.0),
            'e_coul':    mf.scf_summary.get('coul',       0.0),
            'e_xc':      mf.scf_summary.get('exc',        0.0),
            'e_disp':    mf.scf_summary.get('dispersion', 0.0)
        }
        history.append(result)

    grad_scanner = mf.nuc_grad_method().as_scanner()
    get_disp = gen_disp_grad_fun(xc_disp, xc_gcp)
    grad_scanner.get_dispersion = MethodType(get_disp, grad_scanner)

    geometric_log = f'{mol_name[:-4]}_geometric.log'
    import sys
    # PySCF forwards geometric log to sys.stderr
    with open(f'{local_dir}/{geometric_log}', 'w') as log_file:
        sys.stderr = log_file
        conv, mol_eq = kernel(
            grad_scanner,
            maxsteps=maxsteps,
            callback=callback,
            convergence_set=convergence_set,
            constraints=constraints)
    sys.stderr = sys.__stderr__

    # copy the files to destination folder
    output_dir = config['output_dir']
    isExist = os.path.exists(output_dir)
    if not isExist:
        os.makedirs(output_dir)
    optimized_xyz = f'{mol_name[:-4]}_opt.xyz'
    hist_file = f'{mol_name[:-4]}_hist.h5'
    mol_eq.tofile(f'{local_dir}/{optimized_xyz}', format='xyz')

    with h5py.File(f'{local_dir}/{hist_file}', 'w') as h5f:
        #json.dump(history, f)
        for step, info in enumerate(history):
            group = h5f.create_group(f'step_{step}')
            for key, array in info.items():
                group.create_dataset(key, data=array)

    shutil.copyfile(f'{local_dir}/{optimized_xyz}', f'{output_dir}/{optimized_xyz}')
    shutil.copyfile(f'{local_dir}/{hist_file}', f'{output_dir}/{hist_file}')
    shutil.copyfile(f'{local_dir}/{logfile}', f'{output_dir}/{logfile}')
    shutil.copyfile(f'{local_dir}/{geometric_log}', f'{output_dir}/{geometric_log}')
    if conv:
        with open(f'{output_dir}/{mol_name[:-4]}_success.txt', 'w') as file:
            file.write("Geometry optimization converged\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DFT with GPU4PySCF for molecules')
    parser.add_argument("--config", type=str, default='example.json')
    parser.add_argument("--charge", type=int, default=None)
    parser.add_argument("--spin",   type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
        if isinstance(config, list):
            config = config[0]
    for i, mol_name in enumerate(config['molecules']):
        constraints = None
        if 'constraints' in config and config['constraints']:
            assert len(config['constraints']) == len(config['molecules'])
            constraints = config['constraints'][i]
        opt_mol(mol_name, config, constraints, charge=args.charge, spin=args.spin)
