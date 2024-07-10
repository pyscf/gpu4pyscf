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
import traceback

from pyscf import lib, gto
from pyscf import dft, scf
from pyscf.lib import logger
from pyscf.geomopt.geometric_solver import optimize

def opt_mol(mol_name, config, constraints, charge=None, spin=0):
    xc           = config.get('xc',           'b3lyp')
    disp         = config.get('disp',          None)
    bas          = config.get('basis',        'def2-tzvpp')
    verbose      = config.get('verbose',      4)
    scf_conv_tol = config.get('scf_conv_tol', 1e-10)
    with_df      = config.get('with_df',      True)
    with_gpu     = config.get('with_gpu',     True)
    with_solvent = config.get('with_solvent', False)

    # I/O
    fp = tempfile.TemporaryDirectory()
    local_dir = f'{fp.name}/'
    logfile = f'{mol_name[:-4]}_pyscf.log'
    shutil.copyfile(config['input_dir']+mol_name, local_dir+mol_name)
    if constraints is not None:
        shutil.copyfile(config['input_dir']+constraints, local_dir+constraints)
    cupy.get_default_memory_pool().free_all_blocks()
    lib.num_threads(8)
    start_time = time.time()
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
        if 'auxbasis' in config and config['auxbasis'] == "RIJK-def2-tzvp":
            auxbasis = 'def2-tzvp-jkfit'
        else:
            auxbasis = None
        mf = mf.density_fit(auxbasis=auxbasis)

    if with_gpu:
        mf = mf.to_gpu()

    mf.chkfile = None
    if with_solvent:
        mf = mf.PCM()
        mf.with_solvent.lebedev_order = 29
        mf.with_solvent.method = config['solvent']['method'].replace('PCM','-PCM')
        mf.with_solvent.eps = config['solvent']['eps']

    mf.direct_scf_tol = 1e-14
    mf.chkfile = None
    mf.conv_tol = scf_conv_tol

    opt_params = {
        "convergence_energy": config.get("conv_e", 1e-6),  # Eh
        "convergence_grms": config.get("grms", 3.0e-4),  # Eh/Bohr
        "convergence_gmax": config.get("gmax", 4.5e-4),  # Eh/Bohr
        "convergence_drms": config.get("drms", 1.2e-3),  # Angstrom
        "convergence_dmax": config.get("dmax", 1.8e-3),  # Angstrom
        "prefix": config.get("prefix", "test"),
    }
    maxsteps = config.get('maxsteps', 20)
    gradients = []
    def callback(envs):
        gradients.append(envs['gradients'])
    mol_eq = optimize(mf,
        maxsteps=maxsteps,
        callback=callback,
        constraints=constraints,
        **opt_params
        )

    # copy the files to destination folder
    output_dir = config['output_dir']
    isExist = os.path.exists(output_dir)
    if not isExist:
        os.makedirs(output_dir)
    optimized_xyz = mol_name[:-4] + '_opt.xyz'
    mol_eq.tofile(f'{local_dir}/{optimized_xyz}', format='xyz')
    shutil.copyfile(f'{local_dir}/{optimized_xyz}', f'{output_dir}/{optimized_xyz}')
    shutil.copyfile(f'{local_dir}/{logfile}', f'{output_dir}/{logfile}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DFT with GPU4PySCF for molecules')
    parser.add_argument("--config", type=str, default='example.json')
    parser.add_argument("--charge", type=int, default=None)
    parser.add_argument("--spin",   type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)[0]
    for i, mol_name in enumerate(config['molecule']):
        constraints = None
        if 'constraints' in config:
            assert len(config['constraints']) == len(config['molecule'])
            constraints = config['constraints'][i]
        opt_mol(mol_name, config, constraints, charge=args.charge, spin=args.spin)
