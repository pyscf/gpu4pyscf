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
import json
import numpy
import h5py
import argparse
import subprocess
import tempfile
import shutil

def read_fd_hess(logfile, natm):
    import re
    with open(logfile, 'r') as output_file:
        lines = output_file.readlines()
        i = 0
        for line in lines:
            if line[:15] == " Final Hessian.":
                break
            i += 1

        i += 1
        array_list = [[] for _ in range(3*natm)]

        while(lines[i][:8] != " -------"):
            i += 1
            for j in range(3*natm):
                data = re.split('\s+', lines[i+j])
                array_list[j] += data[2:-1]
            i += 3*natm
    hess = numpy.asarray(array_list, dtype=numpy.float64)
    return hess

def read_qchem_timing(logfile):
    scf_time = grad_time = total_time = None
    with open(logfile, 'r') as output_file:
        lines = output_file.readlines()
        for line in lines:
            if line[:16] == " SCF time:   CPU":
                if scf_time is not None:
                    continue
                info = line[16:].split(' ')[4]
                scf_time = float(info[:-1])
            if line[:20] == " Gradient time:  CPU":
                if grad_time is not None:
                    continue
                info = line[20:].split(' ')[5]
                grad_time = float(info)
            if line[:16] == " Total job time:":
                info = line[16:].split(',')[0]
                total_time = float(info.replace('s(wall)', ''))
    return scf_time, grad_time, total_time

def run_dft(mol_name):
    '''run DFT routine, and save *.out, *_qchem.h5, *.h5'''
    with open(config['input_dir']+mol_name, 'r') as xyz_file:
        coords = xyz_file.readlines()[2:]
    natm = len(coords)

    # create qchem input file
    qchem_in = mol_name[:-4]+'.in'
    with open(f'{qchem_in}', "w") as input:
        input.write("$molecule\n")
        input.write("0 1\n")
        for line in coords:
            input.write(line)
        input.write("\n$end")
        input.write("\n")

        input.write("$rem\n")
        if 'with_hess' in config and config['with_hess']:
            input.write("JOBTYPE freq\n")
        elif 'with_grad' in config and config['with_grad']:
            input.write("JOBTYPE force\n")
        else:
            input.write("JOBTYPE sp\n")

        input.write("METHOD " + config['xc'] + "\n")
        input.write("BASIS " + config['basis'] + "\n")
        input.write("SYMMETRY      FALSE\n")
        input.write("SYM_IGNORE    TRUE\n")
        input.write("XC_GRID       000099000590\n")
        input.write("NL_GRID       000050000194\n")
        input.write("MAX_SCF_CYCLES 100\n")
        input.write("PURECART 1111\n")
        input.write("MEM_STATIC 4096\n")

        # density fitting
        if 'with_df' in config and config['with_df']:
            input.write("ri_j        True\n")
            input.write("ri_k        True\n")
            if 'auxbasis' in config:
                auxbasis = config['auxbasis']
            else:
                auxbasis = 'rijk-' + config['basis']
            input.write(f"aux_basis     {auxbasis}\n")
        input.write("SCF_CONVERGENCE 10\n")
        input.write("THRESH        14\n")
        input.write("BASIS_LIN_DEP_THRESH 12\n")

        # direct SCF scheme
        input.write("INCDFT_DENDIFF_THRESH 14\n")
        input.write("INCDFT_GRIDDIFF_THRESH 14\n")

        # solvent models
        if 'with_solvent' in config and config['with_solvent']:
            input.write("SOLVENT_METHOD PCM\n")

        # qchem does fully support df hessian, and solvent hessian
        if 'with_hess' in config and config["with_hess"]:
            if 'with_df' in config and config['with_df']:
                input.write("IDERIV 1\n")
            if 'with_solvent' in config and config['with_solvent']:
                input.write("IDERIV 1\n")

        input.write("$end\n")
        input.write("\n")
        input.write("$archive\n")
        input.write("enable_archive = True !Turns on generation of Archive\n")
        input.write("$end\n")

        # solvent information
        if 'with_solvent' in config and config['with_solvent']:
            input.write("$PCM\n")
            input.write(f"Theory {config['solvent']['method']}\n")
            input.write("HeavyPoints 302\n")
            input.write("HPoints 302\n")
            input.write("$end\n")
            input.write("\n")
            input.write("$solvent\n")
            input.write(f"dielectric {config['solvent']['eps']}\n")
            input.write("$end\n")

    # create a temp folder, and run qchem
    fp = tempfile.TemporaryDirectory()
    local_dir = f'/tmp/{fp.name}/'
    logfile = f'{mol_name[:-4]}_qchem.out'

    print(f'creating a temp folder named {local_dir}')
    cmd = f'qchem -save -nt 32 {qchem_in} {local_dir}/{logfile} {fp.name}'
    print(f'runing cmd: {cmd}')

    env = os.environ.copy()
    result = subprocess.run(cmd.split(' '), capture_output=True, env=env)
    print(result.stdout)
    print(result.stderr)

    output_dir = config['output_dir']
    isExist = os.path.exists(output_dir)
    if not isExist:
        os.makedirs(output_dir)
    shutil.copyfile(f'{qchem_in}', f'{output_dir}/{qchem_in}')
    shutil.copyfile(f'{local_dir}/{logfile}', f'{output_dir}/{logfile}')

    qchem_h5 = f'/tmp/{fp.name}/qarchive.h5'

    # search SCF time, Gradient time, Hessian time
    scf_time, grad_time, total_time = read_qchem_timing(local_dir+'/'+logfile)

    # Read qchem data
    fp_qchem_h5 = h5py.File(qchem_h5)
    sp_data = fp_qchem_h5['job']['1']['sp']['energy_function']['1']
    e_tot = sp_data['energy'][()]
    if 'with_grad' in config and config['with_grad']:
        grad = sp_data['gradient'][()]
    if 'with_hess' in config and config['with_hess'] and 'hessian' in sp_data:
        hess = sp_data['hessian'][()]

    # Read finite difference Hessian if with solvent, or  and hessian
    if 'with_hess' in config and config['with_hess']:
        if 'with_df' in config and config['with_df']:
            hess = read_fd_hess(local_dir+'/'+logfile, natm)
        if 'with_solvent' in config and config['with_solvent']:
            hess = read_fd_hess(local_dir+'/'+logfile, natm)

    h5file = f'{mol_name[:-4]}_qchem.h5'
    h5f = h5py.File(f'{local_dir}/{h5file}', 'w')
    h5f.create_dataset('e_tot', data=e_tot)
    h5f.create_dataset('scf_time', data=scf_time)
    h5f.create_dataset('total_time', data=total_time)

    if 'with_grad' in config and config['with_grad']:
        h5f.create_dataset('grad', data=grad)
        h5f.create_dataset('grad_time', data=grad_time)

    if 'with_hess' in config and config['with_hess']:
        h5f.create_dataset('hess', data=hess)

    h5f.close()

    # copy the files to destination folder
    shutil.copyfile(f'{local_dir}/{h5file}', f'{output_dir}/{h5file}')
    shutil.copyfile(f'{qchem_h5}', f'{output_dir}/{mol_name[:-4]}_qarchive.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DFT with Q-Chem for molecules')
    parser.add_argument("--config",    type=str,  default='example_config.json')
    parser.add_argument("--molecule",  type=str,  default='')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)[0]

    bas = config['basis'] if 'basis' in config else 'def2-tzvpp'
    xc = config['xc'] if 'xc' in config else 'B3LYP'
    output_dir = config['output_dir'] if 'output_dir' in config else './'
    verbose = config['verbose'] if 'verbose' in config else 0

    for mol in config['molecules']:
        run_dft(mol)
