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
from gpu4pyscf.tools import get_default_config, method_from_config

# Config for geometric
optimizer_config = {
    'maxsteps': 50,
    'convergence_set': 'GAU',
    'constraints': None,
}

def opt_mol(config):
    """
    Operform geometry optimization based on the configuration file.
    Saving the final xyz file, geomeTric log, historic geometries and pyscf log.  
    """
    pyscf_default_config = get_default_config()
    config = {**pyscf_default_config, **optimizer_config, **config}

    mol_name = config['molecule']
    assert isinstance(mol_name, str)
    assert mol_name.endswith('.xyz')
    input_dir = config['input_dir']
    output_dir = config['output_dir']
    if not os.path.exists(f'{input_dir}/{mol_name}'):
        raise RuntimeError(f'Input file {input_dir}/{mol_name} does not exist.')
    
    # I/O
    os.makedirs(output_dir, exist_ok=True)
    
    # Build PySCF object
    config['logfile'] = mol_name[:-4] + '_pyscf.log'
    config['atom'] = f'{input_dir}/{mol_name}'
    mf = method_from_config(config)

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
