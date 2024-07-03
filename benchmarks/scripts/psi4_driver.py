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

import os
import shutil
import argparse
import time
import json
import tempfile
import h5py
import numpy as np
import psi4
from psi4 import *

psi4_io = psi4.core.IOManager.shared_object()
psi4_io.set_default_path('./')

psi4.set_options({"SCF_TYPE": "DIRECT",
                "dft_radial_points": 99,
                "dft_spherical_points": 590,
                "dft_density_tolerance": 1e-7,
                "dft_pruning_scheme": "robust",
                "INTS_TOLERANCE": 1e-14,
                "E_CONVERGENCE": 1e-10,
                "DFT_VV10_SPHERICAL_POINTS": 194,
                "DFT_VV10_RADIAL_POINTS": 50,
                "DF_SCF_GUESS": False,
                "S_TOLERANCE": 1e-12,
                "INCFOCK": True,
                "screening": "density",
                "INCFOCK_CONVERGENCE": 0.0,
                "INCFOCK_FULL_FOCK_EVERY": 100,
                "PUREAM": True
                })

psi4.set_memory('521 Gb')
psi4.core.set_num_threads(32)

def run_dft_with_psi4(mol_name, config):
    xc = config['xc']
    basis = config['basis']
    with_df = 'with_df' in config and config['with_df']
    with_solvent = 'with_solvent' in config and config['with_solvent']
    with_grad = 'with_grad' in config and config['with_grad']
    with_hess = 'with_hess' in config and config['with_hess']
    solvent_method = config['solvent']['method']
    solvent_eps = config['solvent']['eps']
    
    # Setup I/O
    fp = tempfile.TemporaryDirectory()
    local_dir = f'{fp.name}/'
    logfile = f'{mol_name[:-4]}_psi4'
    shutil.copyfile(config['input_dir']+mol_name, local_dir+mol_name)
    data_file = mol_name[:-4] + '_psi4.h5'
    h5f = h5py.File(f'{local_dir}/{data_file}', 'w')

    with open(config['input_dir']+mol_name, 'r') as xyz_file:
        coords = xyz_file.readlines()[2:]
    natm = len(coords)

    # Setup Psi4 based on config file
    psi4.core.set_output_file(f'{local_dir}/{logfile}')
    psi4.set_output_file(f'{local_dir}/{logfile}')
    molecule = """
    0 1\n""" + "\n".join(coords)
    mol = psi4.geometry(molecule)

    if with_solvent:
        raise NotImplementedError
    
    if with_df:
        psi4.set_options({
            'scf_type': 'df',  # Use density fitting
            'guess': 'sad',
            'INCFOCK': False,
            "screening": "SCHWARZ"
            })

    # Calculate the total energy
    start_time = time.time()
    e_tot = psi4.energy(f'{xc}/{basis}')
    scf_time = time.time() - start_time

    h5f.create_dataset('e_tot', data=e_tot)
    h5f.create_dataset('scf_time', data=scf_time)

    # Calculate Gradient
    if with_grad:
        try:
            start_time = time.time()
            g_psi4 = psi4.gradient(f'{xc}/{basis}')
            grad_time = time.time() - start_time
            g = np.array(g_psi4.to_array())
        except:
            g = -1
            grad_time = -1
        h5f.create_dataset('grad', data=g)
        h5f.create_dataset('grad_time', data=grad_time)

    # Calculate Hessian matrix
    if with_hess:
        try:
            start_time = time.time()
            h_psi4 = psi4.hessian(f'{xc}/{basis}')
            hess_time = time.time() - start_time
            h = np.array(h_psi4.to_array())
        except:
            hess_time = -1
            h = -1
        h5f.create_dataset('hess', data=h)
        h5f.create_dataset('hess_time', data=hess_time)

    h5f.close()

    # copy the files to destination folder
    output_dir = config['output_dir']
    isExist = os.path.exists(output_dir)
    if not isExist:
        os.makedirs(output_dir)

    shutil.copyfile(f'{local_dir}/{data_file}', f'{output_dir}/{data_file}')
    shutil.copyfile(f'{local_dir}/{logfile}.log', f'{output_dir}/{logfile}.log')
    shutil.copyfile(f'{local_dir}/{logfile}', f'{output_dir}/{logfile}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DFT with Psi4 for molecules')
    parser.add_argument("--config",    type=str,  default='benchmark.json')
    args = parser.parse_args()

    with open(args.config) as f:
        config_template = json.load(f)[0]

    # Generate benchmark data for different xc
    for xc in ['HF', 'PBE', 'B3LYP', 'M06']:
        config = config_template.copy()
        config['basis'] = 'def2-tzvpp'
        config['xc'] = xc
        config['output_dir'] = config['output_dir'] + '/xc/' + xc
        for mol_name in config['molecules']:
            print(f'running {mol_name} with {xc}/def2-tzvpp')
            run_dft_with_psi4(mol_name, config)
    
    # Generate benchmark data for different basis
    for bas in ['sto-3g', '6-31g', 'def2-svp', 'def2-tzvpp', 'def2-tzvpd']:
        config['xc'] = 'b3lyp'
        config['basis'] = bas
        config['output_dir'] = config['output_dir'] + '/basis/' + bas
        for mol_name in config['molecules']:
            print(f'running {mol_name} with b3lyp/{bas}')
            run_dft_with_psi4(mol_name, config)
    
