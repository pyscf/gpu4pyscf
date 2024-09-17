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
import time
import json
import pyscf
import argparse
import tempfile
import shutil
import cupy
import h5py
import logging

from pyscf import lib, gto
from pyscf import dft, scf
from pyscf.geomopt.geometric_solver import kernel

def opt_mol(mol_name, config, constraints, charge=None, spin=0):
    xc              = config.get('xc',              'b3lyp')
    disp            = config.get('disp',            None)
    bas             = config.get('basis',           'def2-tzvpp')
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

    lib.num_threads(8)
    mol = pyscf.M(
        atom=local_dir+mol_name,
        basis=bas,
        max_memory=32000,
        verbose=verbose,
        charge=charge,
        spin=spin,
        output=f'{local_dir}/{logfile}')
    mol.build()

    # To match default LDA in Q-Chem
    if xc == 'LDA':
        pyscf_xc = 'LDA,VWN5'
    else:
        pyscf_xc = xc

    if xc.lower() == 'hf':
        mf = scf.HF(mol)
    else:
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

    geometric_log = f'{mol_name[:-4]}_geometric.log'
    import sys
    # PySCF forwards geometric log to sys.stderr
    with open(f'{local_dir}/{geometric_log}', 'w') as log_file:
        sys.stderr = log_file
        conv, mol_eq = kernel(
            mf,
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
