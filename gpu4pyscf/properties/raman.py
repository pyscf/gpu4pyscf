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

import numpy as np
import cupy as cp
from pyscf.hessian import thermo
from gpu4pyscf.properties import polarizability
from gpu4pyscf.lib.cupy_helper import contract
from pyscf.data import nist
from gpu4pyscf.scf.hf import RHF

def polarizability_derivative_numerical_dx(mf, dx = 1e-3):
    # Return in (   natm, 3,           3, 3      )
    #            < derivative > < polarizability >
    #
    # This function destroys the content of mf object, please call mf.kernel() afterward.
    mol = mf.mol

    dpdx = np.empty([mol.natm, 3, 3, 3])
    mol_copy = mol.copy()
    mol_copy.verbose = 0
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            mol_copy.build()
            mf.reset(mol_copy)
            mf.kernel()
            assert mf.converged
            p_p = polarizability.eval_polarizability(mf)

            xyz_m = mol.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            mf.reset(mol_copy)
            mf.kernel()
            assert mf.converged
            p_m = polarizability.eval_polarizability(mf)

            dpdx[i_atom, i_xyz, :, :] = (p_p - p_m) / (2 * dx)

    mf.reset(mol)

    return dpdx

def polarizability_derivative_numerical_dEdE(mf, dE = 2.5e-3):
    # Return in (   natm, 3,           3, 3      )
    #            < derivative > < polarizability >
    #
    # This function makes the mf object unusable, please make a new one after calling this function.
    mol = mf.mol

    with mol.with_common_orig((0, 0, 0)):
        dipole_integral = mol.intor('int1e_r')
        dipole_integral = cp.asarray(dipole_integral)
        dipole_integral_derivative = -mol.intor('int1e_irp').reshape(3, 3, mol.nao, mol.nao)
        dipole_integral_derivative = cp.asarray(dipole_integral_derivative)
    Hcore = mf.get_hcore()
    Hcore = cp.asarray(Hcore)

    if hasattr(mf, "with_solvent"):
        if not mf.with_solvent.equilibrium_solvation:
            dm0 = mf.make_rdm1()

            mf.with_solvent.equilibrium_solvation = True
            hess_obj = mf.Hessian()
            mo_coeff = mf.mo_coeff
            mo_occ = mf.mo_occ
            mo_energy = mf.mo_energy
            mocc = mo_coeff[:,mo_occ>0]
            atmlst = range(mol.natm)
            h1ao = hess_obj.make_h1(mo_coeff, mo_occ, None, atmlst)
            fx = hess_obj.gen_vind(mo_coeff, mo_occ)
            mo1, _ = hess_obj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao, fx, atmlst)
            mo1 = cp.asarray(mo1)
            # dm1 = 2 * contract('pu,Aduq->Adpq', mo_coeff, mo1 @ mocc.T)
            # dm1 += dm1.transpose(0,1,3,2)
            mf.with_solvent.equilibrium_solvation = False

    def get_gradient_at_E(mf, E):
        mf.get_hcore = lambda *args: Hcore + cp.einsum('d,dij->ij', E, dipole_integral)
        if hasattr(mf, "with_solvent"):
            if not mf.with_solvent.equilibrium_solvation:
                mf.with_solvent.frozen_dm0_for_finite_difference_without_response = dm0

        mf.kernel()
        assert mf.converged
        dm = mf.make_rdm1()
        gradient = mf.nuc_grad_method().kernel()
        gradient = cp.asarray(gradient)

        mol = mf.mol
        aoslices = mol.aoslice_by_atom()
        for i_atom in range(mol.natm):
            p0, p1 = aoslices[i_atom][2:]

            d_dipoleintegral_dA = cp.zeros([3, 3, mol.nao, mol.nao])
            d_dipoleintegral_dA[:, :, :, p0:p1] += dipole_integral_derivative[:, :, :, p0:p1]
            d_dipoleintegral_dA[:, :, p0:p1, :] += dipole_integral_derivative[:, :, :, p0:p1].transpose(0, 1, 3, 2)
            d_dipoleintegral_dA = d_dipoleintegral_dA.transpose(1,0,2,3) # Place derivative into leading dimension

            gradient[i_atom, :] += contract('dEij,ij->dE', d_dipoleintegral_dA, dm) @ E

        if hasattr(mf, "with_solvent"):
            if not mf.with_solvent.equilibrium_solvation:
                v_grids = mf.with_solvent._get_vgrids(dm, with_nuc = True)[0]
                for i_atom in range(mol.natm):
                    for i_xyz in range(3):
                        dm1_A = 2 * contract('pu,uq->pq', mo_coeff, mo1[i_atom, i_xyz, :, :] @ mocc.T)
                        dm1_A += dm1_A.T
                        dq_sym_dA, _ = mf.with_solvent._get_qsym(dm1_A, with_nuc = False)
                        gradient[i_atom, i_xyz] += v_grids @ dq_sym_dA

                mf.with_solvent.frozen_dm0_for_finite_difference_without_response = None

        return gradient

    dpdx = cp.empty([mol.natm, 3, 3, 3])

    E_0 = cp.zeros(3)
    gradient_0 = get_gradient_at_E(mf, E_0)

    for i_xyz in range(3):
        for j_xyz in range(i_xyz + 1, 3):
            E_pp = cp.zeros(3)
            E_pp[i_xyz] += dE
            E_pp[j_xyz] += dE
            gradient_pp = get_gradient_at_E(mf, E_pp)

            E_pm = cp.zeros(3)
            E_pm[i_xyz] += dE
            E_pm[j_xyz] -= dE
            gradient_pm = get_gradient_at_E(mf, E_pm)

            E_mp = cp.zeros(3)
            E_mp[i_xyz] -= dE
            E_mp[j_xyz] += dE
            gradient_mp = get_gradient_at_E(mf, E_mp)

            E_mm = cp.zeros(3)
            E_mm[i_xyz] -= dE
            E_mm[j_xyz] -= dE
            gradient_mm = get_gradient_at_E(mf, E_mm)

            dpdx_ij = (gradient_pp + gradient_mm - gradient_pm - gradient_mp) / (4 * dE**2)
            dpdx[:, :, i_xyz, j_xyz] = dpdx_ij
            dpdx[:, :, j_xyz, i_xyz] = dpdx_ij

        E_p = cp.zeros(3)
        E_p[i_xyz] += dE
        gradient_p = get_gradient_at_E(mf, E_p)

        E_m = cp.zeros(3)
        E_m[i_xyz] -= dE
        gradient_m = get_gradient_at_E(mf, E_m)

        dpdx[:, :, i_xyz, i_xyz] = (gradient_p + gradient_m - 2 * gradient_0) / (dE**2)

    mf.get_hcore = lambda *args: Hcore

    dpdx *= -1
    return dpdx

def eval_raman_intensity(mf, hessian = None):
    '''
    Main driver of Raman spectra intensity

    Args:
        mf: mean field object
        hessian: the hessian matrix in shape (natm, natm, 3, 3), if available

    Returns:
        node frequency: in cm^-1
        Raman scattering activity: in Angstrom**4 / AMU (consistent with Q-Chem)
        Depolarization ratio: dimensionless

    Computation cost:
        19 * time of single point SCF
        + 1 * time of single point Hessian, if hessian matrix not provided

    Reference:
        - Implementation detail:
        Porezag, D.; Pederson, M. R. Infrared intensities and Raman-scattering activities within density-functional theory.
        Physical Review B 1996, 54, 7830.
        doi: https://doi.org/10.1103/PhysRevB.54.7830

        - Clear definition:
        olavarapu, P. L. Ab initio vibrational Raman and Raman optical activity spectra.
        Journal of Physical Chemistry 1990, 94, 8106-8112.
        doi: https://doi.org/10.1021/j100384a024

        - Analytical polarizability derivative, if anyone wants an attempt:
        Amos, R. Calculation of polarizability derivatives using analytic gradient methods.
        Chemical physics letters 1986, 124, 376-381.
        doi: https://doi.org/10.1016/0009-2614(86)85037-0
    '''
    assert isinstance(mf, RHF)
    mol = mf.mol

    if hasattr(mf, "with_solvent"):
        if not mf.with_solvent.equilibrium_solvation:
            print("Warning: The PCM response for polarizability is turned off, "
                  "because we believe the solvent doesn't response instantaneously under an electric field perturbation. "
                  "This might not be consistent with other program, for example the Q-Chem default implementation includes PCM response. "
                  "If you want to reproduce that behavior, set \"mf.with_solvent.equilibrium_solvation = True\"")

    if hessian is None:
        hess_obj = mf.Hessian()
        hess_obj.auxbasis_response = 2
        hessian = hess_obj.kernel()
    assert hessian.shape == (mol.natm, mol.natm, 3, 3)

    freq_info = thermo.harmonic_analysis(mol, hessian)

    norm_mode = freq_info['norm_mode']
    freq_wavenumber = freq_info['freq_wavenumber']

    mf_copy = mf.copy() # Preserve the original mf, since the method of mf is replaced in finite difference
    dalpha_dR = polarizability_derivative_numerical_dEdE(mf_copy)
    dalpha_dQ = contract('AdEe,iAd->iEe', dalpha_dR, norm_mode)

    n_mode = len(freq_wavenumber)
    raman_intensities = np.zeros(n_mode)
    depolarization_ratio = np.zeros(n_mode)

    for i_mode in range(n_mode):
        dalpha_dQi = dalpha_dQ[i_mode]
        alpha_prime = 1.0/3.0 * (dalpha_dQi[0,0] + dalpha_dQi[1,1] + dalpha_dQi[2,2])
        alpha_prime_square = alpha_prime**2
        beta_prime_square = 0.5 * (
            + (dalpha_dQi[0,0] - dalpha_dQi[1,1])**2
            + (dalpha_dQi[0,0] - dalpha_dQi[2,2])**2
            + (dalpha_dQi[1,1] - dalpha_dQi[2,2])**2
            + 6 * (dalpha_dQi[0,1]**2 + dalpha_dQi[0,2]**2 + dalpha_dQi[1,2]**2)
        )

        raman_intensities[i_mode] = 45 * alpha_prime_square + 7 * beta_prime_square
        depolarization_ratio[i_mode] = 3 * beta_prime_square / (45 * alpha_prime_square + 4 * beta_prime_square)

    # You might wonder where does the following unit conversion factor come from, and how does it yields the final unit Angstrom^4 / AMU.
    # The raman intensity has the same unit as (polarizability / length * normal mode eigenvector)^2
    # So there're two parts of the story: What's the unit of polarizability? And, what's the unit of normal mode eigenvector?
    #
    # (1) What's the unit of normal mode eigenvector?
    # The normal mode eigenvector comes from solving the generalized eigenvalue equation H X = lambda M X,
    # where H is the Hessian matrix, with unit (energy / length)^2, all in au,
    # lambda is the eigenvalue, and has the unit of (frequency)^2, in au,
    # and M is a diagonal matrix with atomic masses on diagonal, with unit of mass, in AMU (NOT au!!!)
    # And because of the constraint X^T M X = I, each element of X has unit of (mass)^-1/2, and differs from au by AMU^-1/2.
    #
    # (2) What's the unit of polarizability?
    # In SI, polarizability has unit (C * m^2 / V).
    # In SI, vacuum permittivity has unit (C / V / m).
    # Why does vacuum permittivity matter here? Because we can define polarizability volume as: alpha_V = alpha / (4 * pi * epsilon_0)
    # In SI, polarizability volume has unit (m^3).
    # In au, factor (4 * pi * epsilon_0) is defined as one. As a result, in au, polarizability and polarizability volume are the same.
    # So, in au, we can treat polarizability as having unit of (length)^3.
    # And, not surprisingly, we got polarizability in Bohr^3.
    #
    # Combining the two statements above, the unit of raman intensity is the same as (length^2 / mass^-1/2)^2 = length^4 / mass
    # And from the computation above, we got it in unit Bohr^4 / AMU
    # So, the following unit conversion factor convert it into Angstrom^4 / AMU.
    raman_intensities *= nist.BOHR**4

    return freq_wavenumber, raman_intensities, depolarization_ratio
