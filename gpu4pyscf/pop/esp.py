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
references:
https://onlinelibrary.wiley.com/doi/abs/10.1002/qua.26035
'''

import numpy as np
from scipy.spatial import distance_matrix
import cupy
from pyscf import gto
from pyscf.data import radii
from gpu4pyscf.gto.int3c1e import int1e_grids
from gpu4pyscf.lib.cupy_helper import dist_matrix

#modified_Bondi = radii.VDW.copy()
#modified_Bondi[1] = 1.1/radii.BOHR      # modified version

# Van der Waals radii (in angstrom) are taken from GAMESS.
R_VDW = 1.0/radii.BOHR * np.asarray([
    -1,
    1.20, # H
    1.20, # He
    1.37, # Li
    1.45, # Be
    1.45, # B
    1.50, # C
    1.50, # N,
    1.40, # O
    1.35, # F,
    1.30, # Ne,
    1.57, # Na,
    1.36, # Mg
    1.24, # Al,
    1.17, # Si,
    1.80, # P,
    1.75, # S,
    1.70]) # Cl

def unit_surface(n):
    '''
    Generate spherical harmonics grid points on unit sphere
    The number of generated points is less than n in general.
    '''
    ux = []
    uy = []
    uz = []

    eps = 1e-10
    nequat = int(np.sqrt(np.pi*n))
    nvert = int(nequat/2)
    for i in range(nvert+1):
        fi = np.pi*i/nvert
        z = np.cos(fi)
        xy = np.sin(fi)
        nhor = int(nequat*xy+eps)
        if nhor < 1:
            nhor = 1
        
        fj = 2.0 * np.pi * np.arange(nhor) / nhor
        x = np.cos(fj) * xy
        y = np.sin(fj) * xy

        ux.append(x)
        uy.append(y)
        uz.append(z*np.ones_like(x))
    
    ux = np.concatenate(ux)
    uy = np.concatenate(uy)
    uz = np.concatenate(uz)

    return np.array([ux[:n], uy[:n], uz[:n]]).T

def vdw_surface(mol, scales=[1.0], density=1.0*radii.BOHR**2, rad=R_VDW):
    '''
    Generate vdw surface of molecules, in Bohr
    '''
    coords = mol.atom_coords(unit='B')
    charges = [gto.charge(sym) for sym in mol.elements]
    atom_radii = rad[charges]

    surface_points = []
    for scale in scales:
        scaled_radii = atom_radii * scale
        for i, coord in enumerate(coords):
            r = scaled_radii[i]
            # nd is an indicator of density, not exactly the same as number of points
            nd = int(density * 4.0 * np.pi * r**2)
            points = coord + r * unit_surface(nd)
            dist = distance_matrix(points, coords) + 1e-10
            included = np.all(dist >= scaled_radii, axis=1)
            surface_points.append(points[included])
    return np.concatenate(surface_points)

def build_ab(mol, dm, 
             grid_density=1.0*radii.BOHR**2, rad=R_VDW, 
             sum_constraints=[], equal_constraints=[]):
    dm = cupy.asarray(dm)
    natm = mol.natm

    # Total constraints = 
    # total charge constraints 
    # + subtotal charge constraints 
    # + equal charge constraints
    nconstraints = 1
    nconstraints += len(sum_constraints)
    nconstraints += sum([len(group)-1 for group in equal_constraints])
    
    surface_points = vdw_surface(
        mol, 
        scales=[1.4,1.6,1.8,2.0], 
        density=grid_density,
        rad=rad)
    charges = mol.atom_charges()
    charges = cupy.asarray(charges)

    # For nxn matrix A
    dim = natm + nconstraints
    coords = mol.atom_coords(unit='B')
    coords = cupy.asarray(coords)
    r = dist_matrix(coords, cupy.asarray(surface_points))
    rinv = 1.0/r
    A = cupy.zeros((dim, dim))
    A[:natm, :natm] = rinv.dot(rinv.T)
    
    # For right hand side B
    v_grids_e = int1e_grids(mol, surface_points, dm=dm, direct_scf_tol=1e-14)
    v_grids_n = cupy.dot(charges, rinv)
    
    B = cupy.empty([dim])
    B[:natm] = rinv.dot(v_grids_n - v_grids_e)
    
    dim_offset = natm

    # Add total charge constraints
    A[:natm, dim_offset] = 1.0
    A[dim_offset, :natm] = 1.0
    B[dim_offset] = mol.charge
    
    dim_offset += 1
    if len(sum_constraints) > 0:
        # Add subtotal charge constraints
        for i, group in enumerate(sum_constraints):
            A[i+dim_offset, group[1]] = 1
            A[group[1], i+dim_offset] = 1
            B[i+dim_offset] = group[0]
        dim_offset += len(sum_constraints)

    if len(equal_constraints) > 0:
        # Add charge equal constraints
        for group in equal_constraints:
            i = group[0]
            for ic, j in enumerate(group[1:]):
                A[dim_offset+ic, i] = 1
                A[dim_offset+ic, j] = -1
                A[i, dim_offset+ic] = 1
                A[j, dim_offset+ic] = -1
                B[dim_offset + ic] = 0
            dim_offset += len(group) - 1
    return A, B

def esp_solve(mol, dm, grid_density=1.0*radii.BOHR**2, rad=R_VDW):
    natm = mol.natm
    A, B = build_ab(mol, dm, grid_density=grid_density, rad=rad)
    q = cupy.linalg.solve(A, B)[:natm]
    return q.get()

def resp_solve(mol, dm, grid_density=1.0*radii.BOHR**2, 
               rad=R_VDW, maxit=25, resp_a=5e-4, resp_b=0.1, hfree=True, tol=1e-5, 
               sum_constraints=[], equal_constraints=[]):
    ''' 
    sum_constraints = [
    [c0, [i,j,k,l]],
    [c1, [i,j,l]]]
    --> 
    c0 = q[i] + q[j] + q[k] + q[l]
    c1 = q[i] + q[j] + q[l]

    equal_contrains = [
    [i,j,k],
    [u,v,w]]
    -->
    q[i] = q[j] = q[k] = q[l]
    q[u] = q[v] = q[w]
    '''

    charges = np.asarray([gto.charge(sym) for sym in mol.elements])
    natm = mol.natm
    is_restraint = charges > 1
    is_restraint[charges == 1] = not hfree

    A0, B = build_ab(mol, dm, 
                     grid_density=grid_density, rad=rad,
                     sum_constraints=sum_constraints, 
                     equal_constraints=equal_constraints)
    q = cupy.linalg.solve(A0, B)[:natm]

    q_prev = q.copy()
    for it in range(maxit):
        penalty = resp_a/(q**2 + resp_b**2)**0.5
        A = A0.copy()
        A[np.diag_indices(natm)] += cupy.asarray(is_restraint) * penalty
        q = cupy.linalg.solve(A, B)[:natm]
        q_diff = cupy.linalg.norm(q_prev - q)
        if q_diff < tol:
            break
        q_prev = q
    return q.get()
