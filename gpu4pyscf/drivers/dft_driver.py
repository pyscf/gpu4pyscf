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
import time
import json
import pyscf
import argparse
import tempfile
import shutil
import cupy
import traceback
import h5py

from pyscf import lib, gto
from pyscf import dft, scf
from pyscf.hessian import thermo
from pyscf.lib import logger

def warmup(atom=None):
    """
    Perform a warm-up calculation to initialize the GPU.

    Returns:
        None
    """
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = '/dev/null'
    if atom is None:
        atom = '''
C                 -0.07551087    1.68127663   -0.10745193
O                  1.33621755    1.87147409   -0.39326987
C                  1.67074668    2.95729545    0.49387976
C                  0.41740763    3.77281969    0.78495878
C                 -0.60481480    3.07572636    0.28906224
H                 -0.19316298    1.01922455    0.72486113
O                  0.35092043    5.03413298    1.45545728
H                  0.42961487    5.74279041    0.81264173
O                 -1.95331750    3.53349874    0.15912025
H                 -2.55333895    2.78846397    0.23972698
O                  2.81976302    3.20110148    0.94542226
C                 -0.81772499    1.09230218   -1.32146482
H                 -0.70955636    1.74951833   -2.15888136
C                 -2.31163857    0.93420736   -0.98260166
H                 -2.72575463    1.89080093   -0.74107186
H                 -2.41980721    0.27699120   -0.14518512
O                 -0.26428017   -0.18613595   -1.64425697
H                 -0.72695910   -0.55328886   -2.40104423
O                 -3.00083741    0.38730252   -2.10989934
H                 -3.93210821    0.28874990   -1.89865997
'''
    mol.atom = atom
    mol.basis = 'def2-tzvpp'
    mol.spin = 1
    mol.charge = 1
    mol.build()
    mf = dft.rks.RKS(mol, xc='b3lyp').density_fit().to_gpu()
    mf.kernel()

    return

def run_dft(mol_name, config, charge=None, spin=0):
    ''' Perform DFT calculations based on the configuration file.
    Saving the results, timing, and log to a HDF5 file.
    '''
    xc             = config.get('xc',             'b3lyp')
    disp           = config.get('disp',            None)
    grids          = config.get('grids',          {'atom_grid': (99,590)})
    nlcgrids       = config.get('nlcgrids',       {'atom_grid': (50,194)})
    bas            = config.get('basis',          'def2-tzvpp')
    verbose        = config.get('verbose',        4)
    scf_conv_tol   = config.get('scf_conv_tol',   1e-10)
    direct_scf_tol = config.get('direct_scf_tol', 1e-14)
    with_df        = config.get('with_df',        True)
    auxbasis       = config.get('auxbasis',       None)
    with_gpu       = config.get('with_gpu',       True)

    with_grad      = config.get('with_grad',      True)
    with_hess      = config.get('with_hess',      True)
    with_thermo    = config.get('with_thermo',    False) 
    save_density   = config.get('save_density',   False)
    input_dir      = config.get('input_dir',      './')
    
    default_solvent = {'method': 'iefpcm', 'eps': 78.3553, 'solvent': 'water'}
    with_solvent   = config.get('with_solvent',   False)
    solvent        = config.get('solvent',        default_solvent)

    # I/O
    fp = tempfile.TemporaryDirectory()
    local_dir = f'{fp.name}/'
    logfile = f'{mol_name[:-4]}_pyscf.log'
    shutil.copyfile(f'{input_dir}/{mol_name}', local_dir+mol_name)
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
        if 'atom_grid' in grids: mf.grids.atom_grid = grids['atom_grid']
        if 'level' in grids:     mf.grids.level     = grids['level']
        if mf._numint.libxc.is_nlc(mf.xc):
            if 'atom_grid' in nlcgrids: mf.nlcgrids.atom_grid = nlcgrids['atom_grid']
            if 'level' in nlcgrids:     mf.nlcgrids.level     = nlcgrids['level']
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
        elif solvent['method'].endswith(('smd', 'SMD')):
            mf = mf.SMD()
            mf.with_solvent.lebedev_order = 29
            mf.with_solvent.method = 'SMD'
            mf.with_solvent.solvent = solvent['solvent']
        else:
            raise NotImplementedError

    mf.direct_scf_tol = direct_scf_tol
    mf.chkfile = None
    mf.conv_tol = scf_conv_tol
    e_tot = mf.kernel()

    if not mf.converged:
        logger.warn(mf, 'SCF failed to converge')

    scf_time = time.time() - start_time
    print(f'compute time for energy: {scf_time:.3f} s')

    e1        = mf.scf_summary.get('e1',         0.0)
    e_coul    = mf.scf_summary.get('coul',       0.0)
    e_xc      = mf.scf_summary.get('exc',        0.0)
    e_disp    = mf.scf_summary.get('dispersion', 0.0)
    e_solvent = mf.scf_summary.get('e_solvent',  0.0)

    data_file = mol_name[:-4] + '_pyscf.h5'

    with h5py.File(f'{local_dir}/{data_file}', 'w') as h5f:
        h5f.create_dataset('e_tot',     data=e_tot)
        h5f.create_dataset('e1',        data=e1)
        h5f.create_dataset('e_coul',    data=e_coul)
        h5f.create_dataset('e_xc',      data=e_xc)
        h5f.create_dataset('e_disp',    data=e_disp)
        h5f.create_dataset('e_solvent', data=e_solvent)
        h5f.create_dataset('scf_time',  data=scf_time)
    
        dm = mf.make_rdm1()
        if isinstance(dm, cupy.ndarray): dm = dm.get()
        h5f.create_dataset('dm',       data=dm)

        if save_density and xc.lower() != 'hf':
            weights = mf.grids.weights
            coords = mf.grids.coords
            dm0 = dm[0] + dm[1] if dm.ndim == 3 else dm
            rho = mf._numint.get_rho(mf.mol, dm0, mf.grids)

            if isinstance(weights, cupy.ndarray): weights = weights.get()
            if isinstance(coords, cupy.ndarray):  coords  = coords.get()
            if isinstance(rho, cupy.ndarray):     rho     = rho.get()

            h5f.create_dataset('grids_weights',      data=weights)
            h5f.create_dataset('grids_coords',       data=coords)
            h5f.create_dataset('grids_rho',          data=rho)

        if dm.ndim == 3:
            # open-shell case
            mo_energy = mf.mo_energy
            if isinstance(mo_energy, cupy.ndarray): mo_energy = mo_energy.get()
            mo_energy[0].sort()
            mo_energy[1].sort()
            na, nb = mf.nelec
            h5f.create_dataset('e_lumo_alpha',   data=mo_energy[0][na])
            h5f.create_dataset('e_lumo_beta',    data=mo_energy[1][nb])
            h5f.create_dataset('e_homo_alpha',   data=mo_energy[0][na-1])
            h5f.create_dataset('e_homo_beta',    data=mo_energy[1][nb-1])
        else:
            # closed-shell case
            mo_energy = mf.mo_energy
            if isinstance(mo_energy, cupy.ndarray): mo_energy = mo_energy.get()
            mo_energy.sort()
            nocc = mf.mol.nelectron // 2
            h5f.create_dataset('e_lumo',     data=mo_energy[nocc])
            h5f.create_dataset('e_homo',     data=mo_energy[nocc-1])
    
    ##################### Gradient Calculation ###############################
    g = None
    if with_grad:
        try:
            start_time = time.time()
            g = mf.nuc_grad_method()
            if with_df:
                g.auxbasis_response = True
            f = g.kernel()
            g = None
            grad_time = time.time() - start_time
            print(f'compute time for gradient: {grad_time:.3f} s')
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
            f = -1
            grad_time = -1
        
        with h5py.File(f'{local_dir}/{data_file}', 'a') as h5f:
            h5f.create_dataset('grad', data=f)
            h5f.create_dataset('grad_time', data=grad_time)

    #################### Hessian Calculation ###############################
    h = None
    if with_hess:
        try:
            natm = mol.natm
            start_time = time.time()
            h = mf.Hessian()
            h.auxbasis_response = 2
            _h_dft = h.kernel()
            h_dft = _h_dft.transpose([0,2,1,3]).reshape([3*natm, 3*natm])
            hess_time = time.time() - start_time
            print(f'compute time for hessian: {hess_time:.3f} s')

            if with_thermo:
                # harmonic analysis
                start_time = time.time()
                normal_mode = thermo.harmonic_analysis(mol, _h_dft)

                thermo_dat = thermo.thermo(
                    mf,                            # GPU4PySCF object
                    normal_mode['freq_au'],
                    298.15,                            # room temperature
                    101325)                            # standard atmosphere
                thermo_time = time.time() - start_time
                print(f'compute time for harmonic analysis: {thermo_time:.3f} s')

        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
            h_dft = -1
            hess_time = -1

        with h5py.File(f'{local_dir}/{data_file}', 'a') as h5f:
            h5f.create_dataset('hess', data=h_dft)
            h5f.create_dataset('hess_time', data=hess_time)

            if with_thermo: 
                h5f.create_dataset('freq_au',         data=normal_mode['freq_au'])
                h5f.create_dataset('freq_wavenumber', data=normal_mode['freq_wavenumber'])
                h5f.create_dataset('E_tot',           data=thermo_dat['E_tot'][0])
                h5f.create_dataset('H_tot',           data=thermo_dat['H_tot'][0])
                h5f.create_dataset('G_tot',           data=thermo_dat['G_tot'][0])
                h5f.create_dataset('E_elec',          data=thermo_dat['E_elec'][0])
                h5f.create_dataset('E_trans',         data=thermo_dat['E_trans'][0])
                h5f.create_dataset('E_rot',           data=thermo_dat['E_rot'][0])
                h5f.create_dataset('E_vib',           data=thermo_dat['E_vib'][0])
                h5f.create_dataset('E_0K',            data=thermo_dat['E_0K'][0])
                h5f.create_dataset('H_elec',          data=thermo_dat['H_elec'][0])
                h5f.create_dataset('H_trans',         data=thermo_dat['H_trans'][0])
                h5f.create_dataset('H_rot',           data=thermo_dat['H_rot'][0])
                h5f.create_dataset('H_vib',           data=thermo_dat['H_vib'][0])
                h5f.create_dataset('G_elec',          data=thermo_dat['G_elec'][0])
                h5f.create_dataset('G_trans',         data=thermo_dat['G_trans'][0])
                h5f.create_dataset('G_rot',           data=thermo_dat['G_rot'][0])
                h5f.create_dataset('G_vib',           data=thermo_dat['G_vib'][0])

    # copy the files to destination folder
    output_dir = config['output_dir']
    isExist = os.path.exists(output_dir)
    if not isExist:
        os.makedirs(output_dir)

    shutil.copyfile(f'{local_dir}/{data_file}', f'{output_dir}/{data_file}')
    shutil.copyfile(f'{local_dir}/{logfile}', f'{output_dir}/{logfile}')

    return mf

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
    for mol_name in config['molecules']:
        run_dft(mol_name, config, charge=args.charge, spin=args.spin)
