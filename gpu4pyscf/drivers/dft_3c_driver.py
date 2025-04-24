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

###################################################################
# This is a customized driver for three composite methods only
# It only works for b97-3c, r2scan-3c, and wb97x-3c
###################################################################

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
import numpy as np
from types import MethodType
from pyscf import lib
from pyscf import dft
from pyscf.hessian import thermo
from pyscf.lib import logger
from pyscf.dispersion import dftd3, dftd4, gcp

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

import importlib.metadata
required_version = "1.3.0"
installed_version = importlib.metadata.version('pyscf-dispersion')
assert installed_version >= required_version

def parse_3c(xc_name):
    """
    return xc, nlc, basis, ecp, (xc_disp, disp), xc_gcp
    """
    if xc_name == 'b973c':
        return 'GGA_XC_B97_3C', 0, 'def2-mtzvp', None, ('b97-3c', 'D3BJ'), 'b973c'
    elif xc_name == 'r2scan3c':
        return 'r2scan', 0, 'def2-mtzvpp', None, ('r2scan-3c', 'D4'), 'r2scan3c'
    elif xc_name == 'wb97x3c':
        # 'Grimme vDZP' is available is BSE, but pyscf 2.8 is not able to parse ECP properly
        basis = 'Grimme vDZP'
        # ecp = 'Grimme vDZP'
        # basis = os.path.join(CURRENT_DIR, 'basis_vDZP_NWCHEM.dat')
        ecp = os.path.join(CURRENT_DIR, 'ecp_vDZP_NWCHEM.dat')
        return 'wb97x-v', 0, basis, ecp, ('wb97x-3c', 'D4'), None
    else:
        raise RuntimeError('Unknow xc functionals for parsing 3c')

def get_dispersion(mol, xc, grad=True):
    if xc == 'b97-3c':
        d3_model = dftd3.DFTD3Dispersion(mol, xc=xc, atm=True)
        res = d3_model.get_dispersion(grad=grad)
    elif xc == 'r2scan-3c':
        # r2scan-3c use customized parameters
        # https://github.com/psi4/psi4/blob/0e54962d629494f4ed142d0499d7faeaf36effdd/psi4/driver/procrouting/dft/mgga_functionals.py#L250
        d4_model = dftd4.DFTD4Dispersion(mol, xc=xc, atm=True, ga=2.0, gc=1.0)
        d4_model.set_param(0.0, 0.42, 5.65, s9=2.0)
        res = d4_model.get_dispersion(grad=grad)
    elif xc == 'wb97x-3c':
        d4_model = dftd4.DFTD4Dispersion(mol, xc=xc, atm=True)
        res = d4_model.get_dispersion(grad=grad)
    else:
        raise NotImplementedError
    return res

def gen_disp_fun(xc_disp, xc_gcp):
    """
    Generate a function to calculate the sum of dispersion and gcp contributions
    """
    def get_disp(mf, disp=None, with_3body=None, verbose=None):
        mol = mf.mol
        energy = 0.0
        if xc_disp is not None:
            res = get_dispersion(mol, xc_disp, grad=False)
            energy += res.get('energy')
            mf.scf_summary['dispersion'] = energy
        if xc_gcp is not None:
            gcp_model = gcp.GCP(mol, method=xc_gcp)
            res = gcp_model.get_counterpoise()
            energy += res['energy']
        return energy
    return get_disp

def gen_disp_grad_fun(xc_disp, xc_gcp):
    """
    Generate a function to calculate gradient of dispersion + gcp
    """
    def get_disp_grad(mf_grad, disp=None, with_3body=None, verbose=None):
        mf = mf_grad.base
        mol = mf.mol
        gradient = 0.0
        if xc_disp is not None:
            res = get_dispersion(mol, xc_disp, grad=True)
            gradient += res.get('gradient')
            
        if xc_gcp is not None:
            gcp_model = gcp.GCP(mol, method=xc_gcp)
            res = gcp_model.get_counterpoise(grad=True)
            gradient += res['gradient']
        return gradient
    return get_disp_grad

def gen_disp_hess_fun(xc_disp, xc_gcp):
    """
    Generate a function to calculate Hessian of dispersion + gcp
    """
    def get_disp_hess(mf_hess, disp=None, with_3body=None):
        mf = mf_hess.base
        mol = mf.mol
        natm = mol.natm
        h_disp = np.empty([natm,natm,3,3])

        coords = mf_hess.mol.atom_coords()
        mol = mol.copy()
        eps = 1e-5
        for i in range(natm):
            for j in range(3):
                coords[i,j] += eps
                mol.set_geom_(coords, unit='Bohr')
                g1 = 0.0
                if xc_disp is not None:
                    res = get_dispersion(mol, xc_disp, grad=True)
                    g1 += res.get('gradient')
                if xc_gcp is not None:
                    gcp_model = gcp.GCP(mol, method=xc_gcp)
                    res = gcp_model.get_counterpoise(grad=True)
                    g1 += res['gradient']

                coords[i,j] -= 2.0*eps
                mol.set_geom_(coords, unit='Bohr')
                g2 = 0.0
                if xc_disp is not None:
                    res = get_dispersion(mol, xc_disp, grad=True)
                    g2 += res.get('gradient')
                if xc_gcp is not None:
                    gcp_model = gcp.GCP(mol, method=xc_gcp)
                    res = gcp_model.get_counterpoise(grad=True)
                    g2 += res['gradient']

                coords[i,j] += eps
                h_disp[i,:,j,:] = (g1 - g2)/(2.0*eps)
        return h_disp
    return get_disp_hess

def run_dft(mol_name, config, charge=None, spin=0):
    ''' Perform DFT calculations based on the configuration file.
    Saving the results, timing, and log to a HDF5 file.
    '''
    xc             = config.get('xc',             'b3lyp')
    grids          = config.get('grids',          {'atom_grid': (99,590)})
    nlcgrids       = config.get('nlcgrids',       {'atom_grid': (50,194)})
    verbose        = config.get('verbose',        4)
    scf_conv_tol   = config.get('scf_conv_tol',   1e-10)
    direct_scf_tol = config.get('direct_scf_tol', 1e-14)
    with_df        = config.get('with_df',        True)
    auxbasis       = config.get('auxbasis',       'def2-universal-jkfit')
    with_gpu       = config.get('with_gpu',       True)

    with_grad      = config.get('with_grad',      True)
    with_hess      = config.get('with_hess',      True)
    with_thermo    = config.get('with_thermo',    False) 
    save_density   = config.get('save_density',   False)
    input_dir      = config.get('input_dir',      './')
    
    default_solvent = {'method': 'iefpcm', 'eps': 78.3553, 'solvent': 'water'}
    with_solvent   = config.get('with_solvent',   False)
    solvent        = config.get('solvent',        default_solvent)

    pyscf_xc, nlc, basis, ecp, (xc_disp, disp), xc_gcp = parse_3c(xc)

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
        basis=basis,
        ecp=ecp,
        max_memory=32000,
        verbose=verbose,
        charge=charge,
        spin=spin,
        output=f'{local_dir}/{logfile}')
    mol.build()

    mf = dft.KS(mol, xc=pyscf_xc)
    if 'atom_grid' in grids: mf.grids.atom_grid = grids['atom_grid']
    if 'level' in grids:     mf.grids.level     = grids['level']
    if mf._numint.libxc.is_nlc(mf.xc):
        if 'atom_grid' in nlcgrids: mf.nlcgrids.atom_grid = nlcgrids['atom_grid']
        if 'level' in nlcgrids:     mf.nlcgrids.level     = nlcgrids['level']

    if with_df:
        mf = mf.density_fit(auxbasis=auxbasis)
    if with_gpu:
        mf = mf.to_gpu()

    #### Changes for 3C methods #####
    # Setup dispersion correction and GCP
    # To developers: Right now the 3c method is supported via modifications of mf class methods.
    #                These modifications are not copied over during to_gpu() or to_cpu() calls.
    #                As a result, to_gpu() or to_cpu() has to be called before the following lines,
    #                otherwise you'll get a wrong dispersion.
    #                If you need to reproduce 3c method in other packages, please pay attention.
    mf.nlc = nlc
    mf.get_dispersion = MethodType(gen_disp_fun(xc_disp, xc_gcp), mf)
    mf.do_disp = lambda: True
    #################################

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
            # Overwrite the method for 3C method
            g.get_dispersion = MethodType(gen_disp_grad_fun(xc_disp, xc_gcp), g)
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
            # Overwrite the method for 3C method
            h.get_dispersion = MethodType(gen_disp_hess_fun(xc_disp, xc_gcp), h)
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
