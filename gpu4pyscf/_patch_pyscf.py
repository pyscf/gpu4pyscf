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

import numpy as np
import cupy
import pyscf

if int(pyscf.__version__.split('.')[1]) <= 10:
    def _fftdf_to_gpu(self):
        from gpu4pyscf.pbc.df.fft import FFTDF
        return FFTDF(self.cell, self.kpts)
    from pyscf.pbc.df.fft import FFTDF
    FFTDF.to_gpu = _fftdf_to_gpu

    def _aftdf_to_gpu(self):
        from gpu4pyscf.pbc.df.aft import AFTDF
        return AFTDF(self.cell, self.kpts)
    from pyscf.pbc.df.aft import AFTDF
    AFTDF.to_gpu = _aftdf_to_gpu

    def _gdf_to_gpu(self):
        from gpu4pyscf.pbc.df.df import GDF
        return GDF(self.cell, self.kpts)
    from pyscf.pbc.df.df import GDF
    GDF.to_gpu = _gdf_to_gpu

    # patch PySCF Cell class, updating lattice parameters is not avail in pyscf 2.10
    from pyscf import lib
    from pyscf.lib import logger
    from pyscf.gto import mole
    from pyscf.pbc.gto.cell import Cell
    def set_geom_(self, atoms_or_coords=None, unit=None, symmetry=None,
                  a=None, inplace=True):
        '''Update geometry and lattice parameters

        Kwargs:
            atoms_or_coords : list, str, or numpy.ndarray
                When specified in list or str, it is processed as the Mole.atom
                attribute. If inputing a (N, 3) numpy array, this array
                represents the coordinates of the atoms in the molecule.
            a : list, str, or numpy.ndarray
                If specified, it is assigned to the cell.a attribute. Its data
                format should be the same to cell.a
            unit : str
                The unit for the input `atoms_or_coords` and `a`. If specified,
                cell.unit will be updated to this value. If not provided, the
                current cell.unit will be used for the two inputs.
            symmetry : bool
                Whether to enable space_group_symmetry. It is a reserved input
                argument. This functionality is not supported yet.
            inplace : bool
                Whether to overwrite the existing Mole object.
        '''
        if inplace:
            cell = self
        else:
            cell = self.copy(deep=False)
            cell._env = cell._env.copy()

        if unit is not None and cell.unit != unit:
            if isinstance(unit, str):
                if mole.is_au(unit):
                    _unit = 1.
                else:
                    _unit = lib.param.BOHR
            else:
                _unit = unit
            if a is None:
                a = self.lattice_vectors() * _unit
            if atoms_or_coords is None:
                atoms_or_coords = self.atom_coords() * _unit

        if a is not None:
            logger.info(cell, 'Set new lattice vectors')
            logger.info(cell, '%s', a)
            cell.a = a
            if cell._mesh_from_build:
                cell.mesh = None
            if cell._rcut_from_build:
                cell.rcut = None
            cell._built = False
        cell.enuc = None

        if atoms_or_coords is not None:
            cell = mole.MoleBase.set_geom_(cell, atoms_or_coords, unit, symmetry)
        if not cell._built:
            cell.build(False, False)
        return cell
    Cell.set_geom_ = set_geom_

    def get_lattice_Ls(cell, nimgs=None, rcut=None, dimension=None, discard=True):
        '''This version employs more strict criteria when discarding images in lattice sum.
        It can be replaced by the built-in version available in PySCF 2.10.
        '''
        if dimension is None:
            # For atoms near the boundary of the cell, it is necessary (even in low-
            # dimensional systems) to include lattice translations in all 3 dimensions.
            if cell.dimension < 2 or cell.low_dim_ft_type == 'inf_vacuum':
                dimension = cell.dimension
            else:
                dimension = 3
        if rcut is None:
            rcut = cell.rcut

        if dimension == 0 or rcut <= 0 or cell.natm == 0:
            return np.zeros((1, 3))

        a = cell.lattice_vectors()

        scaled_atom_coords = cell.get_scaled_atom_coords()
        atom_boundary_max = scaled_atom_coords[:,:dimension].max(axis=0)
        atom_boundary_min = scaled_atom_coords[:,:dimension].min(axis=0)
        if (np.any(atom_boundary_max > 1) or np.any(atom_boundary_min < -1)):
            atom_boundary_max[atom_boundary_max > 1] = 1
            atom_boundary_min[atom_boundary_min <-1] = -1
        ovlp_penalty = atom_boundary_max - atom_boundary_min
        dR = ovlp_penalty.dot(a[:dimension])
        dR_basis = np.diag(dR)

        # Search the minimal x,y,z requiring |x*a[0]+y*a[1]+z*a[2]+dR|^2 > rcut^2
        # Ls boundary should be derived by decomposing (a, Rij) for each atom-pair.
        # For reasons unclear, the so-obtained Ls boundary seems not large enough.
        # The upper-bound of the Ls boundary is generated by find_boundary function.
        def find_boundary(a):
            aR = np.vstack([a, dR_basis])
            r = np.linalg.qr(aR.T)[1]
            ub = (rcut + abs(r[2,3:]).sum()) / abs(r[2,2])
            return ub

        xb = find_boundary(a[[1,2,0]])
        if dimension > 1:
            yb = find_boundary(a[[2,0,1]])
        else:
            yb = 0
        if dimension > 2:
            zb = find_boundary(a)
        else:
            zb = 0
        bounds = np.ceil([xb, yb, zb]).astype(int)
        Ts = lib.cartesian_prod((np.arange(-bounds[0], bounds[0]+1),
                                 np.arange(-bounds[1], bounds[1]+1),
                                 np.arange(-bounds[2], bounds[2]+1)))
        Ls = np.dot(Ts[:,:dimension], a[:dimension])

        if discard and len(Ls) > 1:
            r = cell.atom_coords()
            rr = r[:,None] - r
            dist_max = np.linalg.norm(rr, axis=2).max()
            Ls_mask = np.linalg.norm(Ls, axis=1) < rcut + dist_max
            Ls = Ls[Ls_mask]
        return np.asarray(Ls, order='C')
    # Patch the get_lattice_Ls for pyscf-2.9 or older
    Cell.get_lattice_Ls = get_lattice_Ls
