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

import os
import sys
import time
import json
import pyscf
import argparse
import cupy
import h5py
import logging

from pyscf import lib, gto
from pyscf import dft, scf
from pyscf.geomopt.geometric_solver import kernel

default_config = {
    'input_dir': './',
    'output_dir': './',
    'molecule': 'molecule.xyz',
    'threads': 8,
    'max_memory': 32000,

    'charge': 0,
    'spin': None,
    'xc': 'b3lyp',
    'disp': None,
    'basis': 'def2-tzvpp',
    'verbose': 4,
    'scf_conv_tol': 1e-10,
    'with_df': True,
    'auxbasis': None,
    'with_gpu': True,
    'maxsteps': 50,
    'convergence_set': 'GAU',
    'constraints': None,

    'with_solvent': False,
    'solvent': {'method': 'iefpcm', 'eps': 78.3553, 'solvent': 'water'},
}

def opt_mol(config):
    config = {**default_config, **config}

    mol_name = config['molecule']
    assert isinstance(mol_name, str)
    assert mol_name.endswith('.xyz')
    input_dir = config['input_dir']
    output_dir = config['output_dir']
    if not os.path.exists(f'{input_dir}/{mol_name}'):
        raise RuntimeError(f'Input file {input_dir}/{mol_name} does not exist.')

    # I/O
    logfile = mol_name[:-4] + '_pyscf.log'
    os.makedirs(output_dir, exist_ok=True)

    lib.num_threads(config['threads'])
    mol = pyscf.M(
        atom=f'{input_dir}/{mol_name}',
        basis=config['basis'],
        max_memory=float(config['max_memory']),
        verbose=config['verbose'],
        charge=config['charge'],
        spin=config['spin'],
        output=f'{output_dir}/{logfile}')

    # To match default LDA in Q-Chem
    xc = config['xc']
    if xc == 'LDA':
        xc = 'LDA,VWN5'

    if xc.lower() == 'hf':
        mf = scf.HF(mol)
    else:
        mf = dft.KS(mol, xc=xc)
        mf.grids.atom_grid = (99,590)
        if mf._numint.libxc.is_nlc(mf.xc):
            mf.nlcgrids.atom_grid = (50,194)
    mf.disp = config['disp']
    if config['with_df']:
        auxbasis = config['auxbasis']
        if auxbasis == "RIJK-def2-tzvp":
            auxbasis = 'def2-tzvp-jkfit'
        mf = mf.density_fit(auxbasis=auxbasis)

    if config['with_gpu']:
        cupy.get_default_memory_pool().free_all_blocks()
        mf = mf.to_gpu()

    mf.chkfile = None

    if config['with_solvent']:
        solvent = config['solvent']
        if solvent['method'].endswith(('PCM', 'pcm')):
            mf = mf.PCM()
            mf.with_solvent.lebedev_order = 29
            mf.with_solvent.method = solvent['method'].replace('PCM','-PCM')
            mf.with_solvent.eps = solvent['eps']
        elif solvent['method'].endswith(('smd', 'SMD')):
            mf = mf.SMD()
            mf.with_solvent.lebedev_order = 29
            mf.with_solvent.method = 'SMD'
            mf.with_solvent.solvent = solvent['solvent']
        else:
            raise NotImplementedError

    mf.direct_scf_tol = 1e-14
    mf.chkfile = None
    mf.conv_tol = float(config['scf_conv_tol'])

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

    # constraints for goemeTRIC are saved in a text file. The input might be the
    # contents of this file.
    constraints = config['constraints']
    if constraints is not None:
        if os.path.exists(constraints):
            constraints = f"{input_dir}/{constraints}"
        else:
            with open(f"{input_dir}/constraints.txt", 'w') as f:
                f.write(constraints)
            constraints = f"{input_dir}/constraints.txt"

    geometric_log = f'{mol_name[:-4]}_geometric.log'
    # PySCF forwards geometric log to sys.stderr
    with open(f'{output_dir}/{geometric_log}', 'w') as log_file:
        try:
            sys.stderr = log_file
            conv, mol_eq = kernel(
                mf,
                maxsteps=config['maxsteps'],
                callback=callback,
                convergence_set=config['convergence_set'],
                constraints=constraints)
        finally:
            sys.stderr = sys.__stderr__

    optimized_xyz = f'{mol_name[:-4]}_opt.xyz'
    hist_file = f'{mol_name[:-4]}_hist.h5'
    mol_eq.tofile(f'{output_dir}/{optimized_xyz}', format='xyz')

    with h5py.File(f'{output_dir}/{hist_file}', 'w') as h5f:
        #json.dump(history, f)
        for step, info in enumerate(history):
            group = h5f.create_group(f'step_{step}')
            for key, array in info.items():
                group.create_dataset(key, data=array)

    if conv:
        with open(f'{output_dir}/{mol_name[:-4]}_success.txt', 'w') as file:
            file.write("Geometry optimization converged\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DFT with GPU4PySCF for molecules')
    parser.add_argument(
        "config",
        type=str,
        help="Path to the configuration file (e.g., example.json)"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
        if isinstance(config, list):
            config = config[0]
    opt_mol(config)
