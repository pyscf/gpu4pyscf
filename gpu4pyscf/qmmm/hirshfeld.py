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

import cupy as cp
import numpy as np

import pyscf
from pyscf.data.elements import charge as charge_of_element
from pyscf.gto.mole import ANG_OF
from pyscf.dft.LebedevGrid import LEBEDEV_NGRID, MakeAngularGrid
from pyscf.data.elements import NEUTRAL_ATOM_SPIN
from gpu4pyscf.lib import logger
from gpu4pyscf.dft.numint import NumInt
from gpu4pyscf.solvent.pcm import natm_without_ghost
from gpu4pyscf.dft.gen_grid import Grids
from gpu4pyscf.dft.uks import UKS
from gpu4pyscf.scf.uhf import UHF as UHF_GPU
from pyscf.scf.uhf import UHF as UHF_CPU
from gpu4pyscf.qmmm.mbis import properties_from_partition

HIRSHFELD_REMOVE_ZERO_RHO_GRID_THRESHOLD = 1e-10

def hirshfeld_kernel(mol, grids, dm, make_mf):
    log = logger.new_logger(mol)

    dm = cp.asarray(dm)
    if dm.ndim == 3:
        assert dm.shape[0] == 2
        dm = dm[0] + dm[1]
    assert dm.shape == (mol.nao, mol.nao)

    natm = natm_without_ghost(mol)
    if natm != mol.natm:
        raise NotImplementedError("Ghost atoms are not supported in Hirshfeld yet")
    if len(mol._ecpbas) > 0:
        log.warn("It is not recommended to use ECP for Hirshfeld multipole calculation.")
    if mol.pseudo:
        raise NotImplementedError("GTH pseudopotential is not supported in Hirshfeld yet")

    assert callable(make_mf)

    elements = mol.elements
    unique_elements = list(set(elements))

    mf_per_element = {}
    for element in unique_elements:
        Z = charge_of_element(element)
        assert 0 < Z and Z < len(NEUTRAL_ATOM_SPIN)
        spin = NEUTRAL_ATOM_SPIN[Z - 1]
        charge = 0 # Only support neutral atoms for now

        mol_atom = pyscf.M(
            atom = f"{element} 0 0 0",
            basis = mol.basis,
            ecp = mol.ecp,
            charge = charge,
            spin = spin,
            verbose = mol.verbose,
        )

        mf_atom = make_mf(mol_atom)

        if not (isinstance(mf_atom, UHF_GPU) or isinstance(mf_atom, UHF_CPU)):
            log.warn("We highly recommend using unrestricted method to perform atomic calculation.")

        mf_atom.kernel()
        assert mf_atom.converged

        mf_per_element[element] = mf_atom

    if grids.coords is None:
        grids.build()

    grid_coords = cp.asarray(grids.coords)
    grid_weights = cp.asarray(grids.weights)
    ni = NumInt()
    grid_rho = ni.get_rho(mol, dm, grids)

    rho_nonzero_mask = cp.logical_and(
        grid_rho >= HIRSHFELD_REMOVE_ZERO_RHO_GRID_THRESHOLD,
        cp.abs(grid_weights) > 1e-14,
    )

    grid_rho = grid_rho[rho_nonzero_mask]
    grid_coords = cp.ascontiguousarray(grid_coords[rho_nonzero_mask, :])
    grid_weights = grid_weights[rho_nonzero_mask]
    grid_w_rho = grid_weights * grid_rho
    ngrids = grid_coords.shape[0]

    nelec = float(cp.sum(grid_w_rho))
    log.info(f"Total number of electrons integrated by Hirshfeld grid = {nelec}")

    atom_coords = cp.asarray(mol.atom_coords())

    atomic_rho = []
    for i_atom in range(natm):
        element = elements[i_atom]
        mf_atom = mf_per_element[element]

        atom_grid_rij = cp.linalg.norm(grid_coords - atom_coords[i_atom], axis = 1)

        mol_atom = mf_atom.mol
        L_max = np.max(mol_atom._bas[:, ANG_OF])
        n_angular_grid = LEBEDEV_NGRID[L_max]
        if n_angular_grid > 1:
            angular_grid = cp.asarray(MakeAngularGrid(n_angular_grid))
        else:
            angular_grid = cp.array([[0, 0, 1.0, 1.0]])
        spherical_average_grid_coords = angular_grid[None, :, :3] * atom_grid_rij[:, None, None] # Centered at the atom
        spherical_average_grid_coords = spherical_average_grid_coords.reshape(ngrids * n_angular_grid, 3)
        del atom_grid_rij

        fake_grids = Grids(pyscf.M())
        fake_grids.coords = spherical_average_grid_coords
        fake_grids.weights = spherical_average_grid_coords[:, 0] # Just need a correct shape
        def just_raise(*args, **kwargs):
            raise RuntimeError("Should never be called")
        fake_grids.build = just_raise

        dm_atom = mf_atom.make_rdm1()
        if dm_atom.ndim == 3:
            assert dm_atom.shape[0] == 2
            dm_atom = dm_atom[0] + dm_atom[1]
        assert dm_atom.shape == (mol_atom.nao, mol_atom.nao)

        ni = NumInt()
        rho_atom = ni.get_rho(mol_atom, dm_atom, fake_grids)
        rho_atom = rho_atom.reshape(ngrids, n_angular_grid)

        del spherical_average_grid_coords, dm_atom

        rho_atom = cp.sum(rho_atom * angular_grid[:, 3], axis = 1) # Spherical averaged
        del angular_grid
        nelec_atom = float(cp.sum(rho_atom * grid_weights))
        log.info(f"Hirshfeld grid intergation of atom {element} = {nelec_atom}")
        atomic_rho.append(rho_atom)

    atomic_rho = cp.stack(atomic_rho, axis = 0)
    sum_atomic_rho = cp.sum(atomic_rho, axis = 0)

    sum_atomic_rho_nonzero_mask = sum_atomic_rho >= HIRSHFELD_REMOVE_ZERO_RHO_GRID_THRESHOLD
    atom_partition = cp.zeros_like(atomic_rho)
    atom_partition[:, sum_atomic_rho_nonzero_mask] = atomic_rho[:, sum_atomic_rho_nonzero_mask] / sum_atomic_rho[sum_atomic_rho_nonzero_mask]
    del sum_atomic_rho, sum_atomic_rho_nonzero_mask, atomic_rho

    partitioned_w_rho = atom_partition * grid_w_rho[None, :]

    atom_grid_vecrij = grid_coords[None, :, :] - atom_coords[:, None, :]

    return properties_from_partition(mol, log, partitioned_w_rho, atom_grid_vecrij, "Hirshfeld")

def hirshfeld(mol, grids, dm, xc = "wB97X-V", xc_grid = (99,590), nlc_grid = (50,194), auxbasis = None,
              conv_tol = 1e-10, max_cycle = 100, nlc = None, disp = None):
    """
        Original Hirshfeld algorithm following:
        Hirshfeld, F. L. (1977). Bonded-atom fragments for describing molecular charge densities. Theoretica chimica acta, 44(2), 129-138.

        i.e. Use atomic density of neutral atoms regardless of total molecular charge, and perform spherical averaging for atomic density.
    """

    def _make_mf(mol):
        if xc is None or xc.upper() == "HF":
            mf = UHF_GPU(mol)
        else:
            mf = UKS(mol, xc = xc)
            mf.grids.atom_grid = xc_grid
            mf.nlcgrids.atom_grid = nlc_grid
        if nlc is not None:
            mf.nlc = nlc
        if disp is not None:
            mf.disp = disp
        mf.conv_tol = conv_tol
        mf.max_cycle = max_cycle
        if auxbasis is not None:
            mf = mf.density_fit(auxbasis = auxbasis)
        return mf

    return hirshfeld_kernel(mol, grids, dm, make_mf = _make_mf)
