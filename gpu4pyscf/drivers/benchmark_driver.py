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

from pyscf import lib, gto
from pyscf import dft, scf
from pyscf.lib import logger

def warmup():
    """
    Perform a warm-up calculation to initialize the GPU.

    Returns:
        None
    """
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = 'def2-tzvpp'
    mol.spin = 1
    mol.charge = 1
    mol.build()
    mf = dft.rks.RKS(mol, xc='b3lyp').density_fit().to_gpu()
    mf.kernel()

    g = mf.nuc_grad_method()
    g = g.kernel()

    h = mf.Hessian()
    h.kernel()
    return

def run_dft(mol_name, config):
    xc           = config.get('xc',           'b3lyp')
    bas          = config.get('basis',        'def2-tzvpp')
    verbose      = config.get('verbose',      4)
    with_df      = config.get('with_df',      True)
    with_gpu     = config.get('with_gpu',     True)
    with_solvent = config.get('with_solvent', False)
    with_grad    = config.get('with_grad',    True)
    with_hess    = config.get('with_hess',    True)

    # I/O
    fp = tempfile.TemporaryDirectory()
    local_dir = f'{fp.name}/'
    logfile = f'{mol_name[:-4]}_pyscf.log'
    shutil.copyfile(config['input_dir']+mol_name, local_dir+mol_name)
    cupy.get_default_memory_pool().free_all_blocks()
    lib.num_threads(8)
    start_time = time.time()
    mol = pyscf.M(
        atom=local_dir+mol_name,
        basis=bas, max_memory=32000,
        verbose=verbose,
        output=f'{local_dir}/{logfile}')

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
    mf.conv_tol = 1e-10
    e_tot = mf.kernel()

    if not mf.converged:
        logger.warn(mf, 'SCF failed to converge')

    scf_time = time.time() - start_time
    print(f'compute time for energy: {scf_time:.3f} s')

    data_file = mol_name[:-4] + '_pyscf.h5'
    import h5py
    h5f = h5py.File(f'{local_dir}/{data_file}', 'w')
    h5f.create_dataset('e_tot', data=e_tot)
    h5f.create_dataset('scf_time', data=scf_time)

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

        h5f.create_dataset('grad', data=f)
        h5f.create_dataset('grad_time', data=grad_time)

    h = None
    if with_hess:
        try:
            natm = mol.natm
            start_time = time.time()
            h = mf.Hessian()
            h.auxbasis_response = 2
            h_dft = h.kernel().transpose([0,2,1,3]).reshape([3*natm, 3*natm])
            hess_time = time.time() - start_time
            print(f'compute time for hessian: {hess_time:.3f} s')
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
            h_dft = -1
            hess_time = -1

        h5f.create_dataset('hess', data=h_dft)
        h5f.create_dataset('hess_time', data=hess_time)

    h5f.close()

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
    run_dft()
