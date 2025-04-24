# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

'''
Construct a PySCF/GPU4PySCF object with config file
'''

import cupy
import pyscf
from pyscf import lib, scf, dft

def get_default_config():
    """
    Generate a default configuration to build a PySCF object
    The default config is designed for calculating small molecules
    """
    default_pyscf_config = {
        'atom': 'mol.xyz',
        'unit': 'angstrom',
        'logfile': 'pyscf.log',
        'threads': 8,
        'max_memory': 32000,
        'charge': 0,
        'spin': None,
        'xc': 'b3lyp',
        'disp': None,
        'grids': {'atom_grid': (99,590)},
        'nlcgrids': {'atom_grid': (50,194)},
        'basis': 'def2-tzvpp',
        'verbose': 4,
        'scf_conv_tol': 1e-10,
        'max_cycle': 50,
        'direct_scf_tol': 1e-14,
        'with_df': True,
        'auxbasis': None,
        'with_gpu': True,
        'with_solvent': False,
        'solvent': {'method': 'iefpcm', 'eps': 78.3553, 'solvent': 'water'},
    }
    return default_pyscf_config

def method_from_config(config):
    """
    Construct a SCF/DFT object from dict
    """
    lib.num_threads(config['threads'])
    
    mol = pyscf.M(
        atom=config['atom'],
        basis=config['basis'],
        max_memory=float(config['max_memory']),
        verbose=config['verbose'],
        charge=config['charge'],
        spin=config['spin'],
        output=config['logfile'],
        unit=config['unit'])

    # To match default LDA in Q-Chem
    xc = config['xc']
    if xc == 'LDA':
        xc = 'LDA,VWN5'

    if xc.lower() == 'hf':
        mf = scf.HF(mol)
    else:
        mf = dft.KS(mol, xc=xc)
        grids = config['grids']
        nlcgrids = config['nlcgrids']
        if 'atom_grid' in grids: mf.grids.atom_grid = grids['atom_grid']
        if 'level' in grids:     mf.grids.level     = grids['level']
        if mf._numint.libxc.is_nlc(mf.xc):
            if 'atom_grid' in nlcgrids: mf.nlcgrids.atom_grid = nlcgrids['atom_grid']
            if 'level' in nlcgrids:     mf.nlcgrids.level     = nlcgrids['level']
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

    mf.direct_scf_tol = config['direct_scf_tol']
    mf.chkfile = None
    mf.conv_tol = float(config['scf_conv_tol'])
    mf.max_cycle = config['max_cycle']

    return mf
