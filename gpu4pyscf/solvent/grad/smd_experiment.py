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

'''
Gradient of SMD solvent model (for experiment and education)
'''

import numpy as np
import cupy
from pyscf.data import radii
from gpu4pyscf.solvent import pcm
from gpu4pyscf.solvent import smd_experiment as smd
from gpu4pyscf.solvent.grad import pcm as pcm_grad

def grad_switch_function(R, r, dr):
    if R < r + dr:
        return -np.exp(dr/(R-dr-r)) * dr / (R-dr-r)**2
    else:
        return 0.0

from gpu4pyscf.solvent.smd import (
    sigma_water, sigma_n, sigma_alpha, sigma_beta, r_zz, switch_function,
    hartree2kcal)

def atomic_surface_tension(symbols, coords, n, alpha, beta, water=True):
    '''
    - list of atomic symbols
    - atomic coordinates in Anstrong
    - solvent descriptors: n, alpha, beta
    '''

    def get_bond_tension(bond):
        if water:
            return sigma_water.get(bond, 0.0)
        t = 0.0
        t += sigma_n.get(bond, 0.0) * n
        t += sigma_alpha.get(bond, 0.0) * alpha
        t += sigma_beta.get(bond, 0.0) * beta
        return t

    def get_atom_tension(sym_i):
        if water:
            return sigma_water.get(sym_i, 0.0)
        t = 0.0
        t += sigma_n.get(sym_i, 0.0) * n
        t += sigma_alpha.get(sym_i, 0.0) * alpha
        t += sigma_beta.get(sym_i, 0.0) * beta
        return t
    natm = coords.shape[0]
    ri_rj = coords[:,None,:] - coords[None,:,:]
    rij = np.sum(ri_rj**2, axis=2)**0.5
    np.fill_diagonal(rij, 1)
    drij = ri_rj / np.expand_dims(rij, axis=-1)
    tensions = []
    for i, sym_i in enumerate(symbols):
        if sym_i not in ['H', 'C', 'N', 'O', 'F', 'Si', 'S', 'Cl', 'Br']:
            tensions.append(np.zeros([natm,3]))
            continue

        tension = np.zeros([natm,3])
        if sym_i in ['F', 'Si', 'S', 'Cl', 'Br']:
            tensions.append(tension)
            continue

        if sym_i == 'H':
            dt_HC = np.zeros([natm,3])
            dt_HO = np.zeros([natm,3])
            for j, sym_j in enumerate(symbols):
                if sym_j == 'C':
                    r, dr = r_zz.get(('H','C'), (0.0, 0.0))
                    dt_drij = grad_switch_function(rij[i,j], r, dr) * drij[i,j]
                    dt_HC[i] += dt_drij
                    dt_HC[j] -= dt_drij
                if sym_j == 'O':
                    r, dr = r_zz.get(('H','O'), (0.0, 0.0))
                    dt_drij = grad_switch_function(rij[i,j], r, dr) * drij[i,j]
                    dt_HO[i] += dt_drij
                    dt_HO[j] -= dt_drij
            sig_HC = get_bond_tension(('H','C'))
            sig_HO = get_bond_tension(('H','O'))
            tension += sig_HC * dt_HC + sig_HO * dt_HO
            tensions.append(tension)
            continue

        if sym_i == 'C':
            dt_CC = np.zeros([natm,3])
            dt_CN = np.zeros([natm,3])
            t_CN = 0.0
            for j, sym_j in enumerate(symbols):
                if sym_j == 'C' and i != j:
                    r, dr = r_zz.get(('C', 'C'), (0.0, 0.0))
                    dt_drij = grad_switch_function(rij[i,j], r, dr) * drij[i,j]
                    dt_CC[i] += dt_drij
                    dt_CC[j] -= dt_drij
                if sym_j == 'N':
                    r, dr = r_zz.get(('C', 'N'), (0.0, 0.0))
                    t_CN += switch_function(rij[i,j], r, dr)
                    dt_drij = grad_switch_function(rij[i,j], r, dr) * drij[i,j]
                    dt_CN[i] += dt_drij
                    dt_CN[j] -= dt_drij
            sig_CC = get_bond_tension(('C','C'))
            sig_CN = get_bond_tension(('C','N'))
            tension += sig_CC * dt_CC + sig_CN * (2 * t_CN * dt_CN)
            tensions.append(tension)
            continue

        if sym_i == 'N':
            t_NC = 0.0
            dt_NC = np.zeros([natm,3])
            dt_NC3 = np.zeros([natm,3])
            for j, sym_j in enumerate(symbols):
                if sym_j == 'C':
                    r, dr = r_zz.get(('N','C'), (0.0, 0.0))
                    tk = 0.0
                    dtk = np.zeros([natm,3])
                    for k, sym_k in enumerate(symbols):
                        if k != i and k != j:
                            rjk, drjk = r_zz.get(('C', sym_k), (0.0, 0.0))
                            tk += switch_function(rij[j,k], rjk, drjk)
                            dtk_rjk = grad_switch_function(rij[j,k], rjk, drjk) * drij[j,k]
                            dtk[j] += dtk_rjk
                            dtk[k] -= dtk_rjk

                    dt_drij = grad_switch_function(rij[i,j], r, dr) * drij[i,j] * tk**2
                    dt_NC[i] += dt_drij
                    dt_NC[j] -= dt_drij

                    t = switch_function(rij[i,j], r, dr)
                    dt_NC += t * (2 * tk * dtk)
                    t_NC += t * tk**2

                    r, dr = r_zz.get(('N','C3'), (0.0, 0.0))
                    dt_drij = grad_switch_function(rij[i,j], r, dr) * drij[i,j]
                    dt_NC3[i] += dt_drij
                    dt_NC3[j] -= dt_drij
            sig_NC = get_bond_tension(('N','C'))
            sig_NC3= get_bond_tension(('N','C3'))
            tension += sig_NC * (1.3 * t_NC**0.3 * dt_NC) + sig_NC3 * dt_NC3
            tensions.append(tension)
            continue

        if sym_i == 'O':
            dt_OC = np.zeros([natm,3])
            dt_ON = np.zeros([natm,3])
            dt_OO = np.zeros([natm,3])
            dt_OP = np.zeros([natm,3])
            for j, sym_j in enumerate(symbols):
                if sym_j == 'C':
                    r, dr = r_zz.get(('O','C'), (0.0, 0.0))
                    dt_drij = grad_switch_function(rij[i,j], r, dr) * drij[i,j]
                    dt_OC[i] += dt_drij
                    dt_OC[j] -= dt_drij
                if sym_j == 'N':
                    r, dr = r_zz.get(('O','N'), (0.0, 0.0))
                    dt_drij = grad_switch_function(rij[i,j], r, dr) * drij[i,j]
                    dt_ON[i] += dt_drij
                    dt_ON[j] -= dt_drij
                if sym_j == 'O' and j != i:
                    r, dr = r_zz.get(('O','O'), (0.0, 0.0))
                    dt_drij = grad_switch_function(rij[i,j], r, dr) * drij[i,j]
                    dt_OO[i] += dt_drij
                    dt_OO[j] -= dt_drij
                if sym_j == 'P':
                    r, dr = r_zz.get(('O','P'), (0.0, 0.0))
                    dt_drij = grad_switch_function(rij[i,j], r, dr) * drij[i,j]
                    dt_OP[i] += dt_drij
                    dt_OP[j] -= dt_drij
            sig_OC = get_bond_tension(('O','C'))
            sig_ON = get_bond_tension(('O','N'))
            sig_OO = get_bond_tension(('O','O'))
            sig_OP = get_bond_tension(('O','P'))
            tension += sig_OC * dt_OC + sig_ON * dt_ON + sig_OO * dt_OO + sig_OP * dt_OP
            tensions.append(tension)
            continue

    return cupy.asarray(tensions)

def get_cds(smdobj):
    mol = smdobj.mol
    n, _, alpha, beta, gamma, _, phi, psi = smdobj.solvent_descriptors
    symbols = [mol.atom_symbol(ia) for ia in range(mol.natm)]
    coords = mol.atom_coords(unit='A')
    if smdobj._solvent.lower() != 'water':
        grad_tension = atomic_surface_tension(symbols, coords, n, alpha, beta, water=False)
        atm_tension = smd.atomic_surface_tension(symbols, coords, n, alpha, beta, water=False)
        mol_tension = smd.molecular_surface_tension(beta, gamma, phi, psi)
    else:
        grad_tension = atomic_surface_tension(symbols, coords, n, alpha, beta, water=True)
        atm_tension = smd.atomic_surface_tension(symbols, coords, n, alpha, beta, water=True)
        mol_tension = 0.0

    # generate surface for calculating SASA
    rad = radii.VDW + 0.4/radii.BOHR
    surface = pcm.gen_surface(mol, ng=smdobj.sasa_ng, rad=rad)
    _, grad_area = pcm_grad.get_dF_dA(surface)
    area = surface['area']
    gridslice = surface['gslice_by_atom']
    SASA = cupy.asarray([cupy.sum(area[p0:p1], axis=0) for p0,p1, in gridslice]).get()
    grad_SASA = cupy.asarray([cupy.sum(grad_area[p0:p1], axis=0) for p0,p1, in gridslice]).get()
    SASA *= radii.BOHR**2
    grad_SASA *= radii.BOHR**2
    mol_cds = mol_tension * np.sum(grad_SASA, axis=0) / 1000
    grad_tension *= radii.BOHR
    atm_cds = np.einsum('i,ijx->jx', SASA, grad_tension.get()) / 1000
    atm_cds+= np.einsum('i,ijx->jx', atm_tension.get(), grad_SASA) / 1000
    return (mol_cds + atm_cds)/hartree2kcal # hartree
