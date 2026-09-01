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

import os
import json
import argparse
import cupy
from pyscf import lib
from gpu4pyscf.drivers.dft_driver import run_dft, warmup

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DFT with GPU4PySCF for molecules')
    parser.add_argument("--config",    type=str,  default='benchmark_df.json')
    args = parser.parse_args()

    with open(args.config) as f:
        config_template = json.load(f)[0]

    isExist = os.path.exists(config_template['output_dir'])
    if not isExist:
        os.makedirs(config_template['output_dir'])

    config_template['input_dir'] = '../molecules/organic/'

    # Warmup
    for i in range(3):
        warmup(atom='../molecules/organic/020_Vitamin_C.xyz')

    # Generate benchmark data for different xc
    config = config_template.copy()
    for xc in ['LDA', 'PBE', 'B3LYP', 'M06']:
        config['xc'] = xc
        config['output_dir'] = './organic/xc/' + xc
        config['basis'] = 'def2-tzvpp'
        config['verbose'] = 4
        for mol_name in config['molecules']:
            if mol_name in ["095_Azadirachtin.xyz","113_Taxol.xyz","168_Valinomycin.xyz"]:
                continue
            run_dft(mol_name, config)

    # vv10 Hessian is not supported yet
    xc = 'wB97m-v'
    config = config_template.copy()
    config['xc'] = xc
    config['output_dir'] = './organic/xc/' + xc
    config['with_hess'] = False
    config['basis'] = 'def2-tzvpp'
    for mol_name in config['molecules']:
        if mol_name in ["095_Azadirachtin.xyz","113_Taxol.xyz","168_Valinomycin.xyz"]:
            continue
        run_dft(mol_name, config)

    # Generate benchmark data for different basis
    config = config_template.copy()
    for bas in ['sto-3g', '6-31g', 'def2-svp', 'def2-tzvpp', 'def2-tzvpd']:
        config['xc'] = 'b3lyp'
        config['basis'] = bas
        config['output_dir'] = './organic/basis/' + bas
        for mol_name in config['molecules']:
            if mol_name in ["095_Azadirachtin.xyz", "113_Taxol.xyz","168_Valinomycin.xyz"]:
                continue
            run_dft(mol_name, config)

    # Generate benchmark data for different solvent
    config = config_template.copy()
    for mol_name in config['molecules']:
        if mol_name in ["095_Azadirachtin.xyz", "113_Taxol.xyz","168_Valinomycin.xyz"]:
            continue
        config['xc'] = 'b3lyp'
        config['basis'] = 'def2-tzvpp'
        config['with_solvent'] = True

        solvent_method = "CPCM"
        config['solvent']['method'] = solvent_method
        config['output_dir'] = './organic/solvent/' + solvent_method
        run_dft(mol_name, config)

        solvent_method = "IEFPCM"
        config['solvent']['method'] = solvent_method
        config['output_dir'] = './organic/solvent/' + solvent_method
        run_dft(mol_name, config)

