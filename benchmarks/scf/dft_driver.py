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

from gpu4pyscf.drivers.dft_driver import run_dft, warmup

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DFT with GPU4PySCF for molecules')
    parser.add_argument("--config",    type=str,  default='benchmark_scf.json')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)[0]

    isExist = os.path.exists(config['output_dir'])
    if not isExist:
        os.makedirs(config['output_dir'])

    config['input_dir'] = '../molecules/water_clusters/'

    # Warmup
    warmup()
    
    # Generate benchmark data for different xc
    config['basis'] = 'def2-tzvpp'
    for xc in ['LDA', 'PBE', 'B3LYP', 'M06', 'wB97m-v']:
        config['xc'] = xc
        config['output_dir'] = './water_clusters/xc/' + xc 
        for mol_name in config['molecules']:
            run_dft(mol_name, config)

    # Generate benchmark data for different basis
    config['xc'] = 'b3lyp'
    for bas in ['sto-3g', '6-31g', 'def2-svp', 'def2-tzvpp', 'def2-tzvpd']:
        config['basis'] = bas
        config['output_dir'] = './water_clusters/basis/' + bas
        for mol_name in config['molecules']:
            run_dft(mol_name, config)
            