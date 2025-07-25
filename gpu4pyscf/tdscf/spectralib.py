# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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


import numpy as np
import cupy as cp
import sys

from pyscf.data.nist import HARTREE2EV, HARTREE2WAVENUMBER
from gpu4pyscf.lib import logger

'''
    This file prints spectral data in Multiwfn format
    also prints transition coefficient in Gaussian16 format

    standard TDDFT can also use this module.

    actually, pyscf already has a similar function to print transition coefficient
    pyscf/pyscf/tdscf/rhf.py
        def analyze(tdobj, verbose=None):

    ECD rotatory strength is in length representation
    unit cgs (10**-40 erg-esu-cm/Gauss)
    ECD_SCALING_FACTOR is to match Gaussian16 results
'''
ECD_SCALING_FACTOR = 500

def get_g16style_trasn_coeff(state, coeff_vec, sybmol, n_occ, n_vir, print_threshold):

    abs_coeff = cp.abs(coeff_vec[state, :, :])
    mask = abs_coeff >= print_threshold

    occ_indices, vir_indices = cp.where(mask)

    if len(occ_indices) == 0:
        return []

    coeff_values = coeff_vec[state, occ_indices, vir_indices]

    occ_indices += 1  # Convert to 1-based index
    vir_indices += 1 + n_occ  # Convert to 1-based index and offset for vir_indices

    format_str = np.vectorize(lambda occ, vir, coeff: f"{occ:>15d} {sybmol} {vir:<8d} {coeff:>15.5f}")
    trasn_coeff = format_str(occ_indices.get(), vir_indices.get(), coeff_values.get()).tolist()

    return trasn_coeff

def get_spectra(energies, P, X, Y, name, RKS, n_occ, n_vir, spectra=True, print_threshold=0.001, mdpol=None, verbose=logger.NOTE):
    '''
    E = hν
    c = λ·ν
    E = hc/λ = hck   k in cm-1

    energy in unit eV
    1240.7011/ xyz eV = xyz nm

    oscilator strength f = 2/3 E u


    in general transition dipole moment u =  [Pα]^T [Xα] = Pα^T Xα + Pβ^TXβ + Qα^TYα + Qβ^T Yβ
                                             [Pβ]   [Xβ]
                                             [Qα]   [Yα]
                                             [Qβ]   [Yβ]
    P = Q

    TDA:
        u =  [Pα]^T [Xα]
             [Pβ]   [Xβ]

        RKS: u = 2 P^T X
        UKS: u = Pα^T Xα + Pβ^TXβ = P^T X (P = [Pα]  X = [Xα])
                                               [Pβ]      [Xβ]
    TDDFT:
        RKS: u = 2 P^T X + 2 P^T Y = 2 P^T(X+Y)
        UKS: u = Pα^T Xα + Pβ^TXβ + Qα^TYα + Qβ^T Yβ =  P^T(X+Y)  (P = [Pα]  X = [Xα] )
                                                                       [Pβ]      [Xβ]
                                                                       [Qα]      [Yα]
                                                                       [Qβ]      [Yβ]

    for TDA,   f = 2/3 E 2*|<P|X>|**2
    for TDDFT, f = 2/3 E 2*|<P|X+Y>|**2
    P is transition dipole
    TDA:   transition_vector is X*2**0.5
    TDDFT: transition_vector is (X*2**0.5 + Y*2**0.5)

    energies are in Hartree
    '''

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)


    energies = energies.reshape(-1,)

    eV = energies * HARTREE2EV

    cm_1 = energies * HARTREE2WAVENUMBER
    nm = 1e7/cm_1


    if isinstance(Y, cp.ndarray):
        trans_dipole_moment = -cp.dot(X*2**0.5 + Y*2**0.5, P.T)
    else:
        trans_dipole_moment = -cp.dot(X*2**0.5, P.T)

    if RKS:

        '''
        2* because alpha and beta spin
        '''
        fosc = 2/3 * energies * cp.sum(2 * trans_dipole_moment**2, axis=1)

    if isinstance(Y, cp.ndarray):
        trans_magnetic_moment = -cp.dot((X*2**0.5 - Y*2**0.5), mdpol.T )
    else:
        trans_magnetic_moment = -cp.dot(X*2**0.5, mdpol.T)

    rotatory_strength = ECD_SCALING_FACTOR * cp.sum(2*trans_dipole_moment * trans_magnetic_moment, axis=1)/2

    if spectra:
        entry = [eV, nm, cm_1, fosc, rotatory_strength]
        data = cp.zeros((eV.shape[0],len(entry)))
        for i in range(len(entry)):
            data[:,i] = entry[i]
        log.info('================================================')
        log.info('#eV       nm      cm^-1    fosc            R')
        for row in range(data.shape[0]):
            log.info(f'{data[row,0]:<8.3f} {data[row,1]:<8.0f} {data[row,2]:<8.0f} {data[row,3]:<15.8f} {data[row,4]:8.8f}')


        filename = name + '_eV_os_Multiwfn.txt'
        with open(filename, 'w') as f:
            cp.savetxt(f, data[:,(0,3)], fmt='%.5f', header=f'{len(energies)} 1', comments='')
        log.info(f'eV Oscillator strength spectra data written to {filename}')

        filename = name + '_eV_rs_Multiwfn.txt'
        with open(filename, 'w') as f:
            new_rs_data = cp.hstack((data[:,0].reshape(-1,1), rotatory_strength.reshape(-1,1)))
            cp.savetxt(f, new_rs_data, fmt='%.5f', header=f'{len(energies)} 1', comments='')
        log.info(f'eV Rotatory strength spectra data written to {filename}')


        if RKS:
            log.info(f"print RKS transition coefficients larger than {print_threshold:.2e}")
            log.info(f'index of HOMO: {n_occ}')
            log.info(f'index of LUMO: {n_occ+1}')
            n_state = X.shape[0]
            X = X.reshape(n_state, n_occ, n_vir)
            if isinstance(Y, cp.ndarray):
                Y = Y.reshape(n_state, n_occ, n_vir)

            filename = name + '_coeff_Multiwfn.txt'

            with open(filename, 'w') as f:

                for state in range(n_state):
                    log.info(f" Excited State{state+1:4d}:      Singlet-A      {eV[state]:>.4f} eV  {nm[state]:>.2f} nm  f={fosc[state]:>.4f}   <S**2>=0.000")
                    f.write(f" Excited State{state+1:4d}   1    {eV[state]:>.4f} \n")
                    trasn_coeff = get_g16style_trasn_coeff(state, X, '->', n_occ=n_occ, n_vir=n_vir, print_threshold=print_threshold)

                    if isinstance(Y, cp.ndarray):
                        trasn_coeff += get_g16style_trasn_coeff(state, Y, '<-', n_occ=n_occ, n_vir=n_vir, print_threshold=print_threshold)

                    results = '\n'.join(trasn_coeff) + '\n\n'
                    log.info(results)
                    f.write(results)
            log.info(f'transition coefficient data written to {filename}')
        else:
            log.warn('printing UKS transition coefficient not implemented yet')



    return fosc, rotatory_strength


