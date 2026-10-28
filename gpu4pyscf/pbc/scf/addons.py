# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
Add-ons for periodic SCF calculations.
'''

import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array


def slab_dipole_correction(mf, dir_idx=2, z0=None, cut=None, rho_tol=1e-5):
    r'''Apply self-consistent slab dipole corrections in place.

    Implements Bengtsson correction (doi:10.1103/PhysRevB.59.12301).
    For the TOTAL charge dipole p, E_dip = 2*pi*p**2/Omega and the
    electronic potential is -4*pi*p*(z_wrapped - offset)/Omega.

    Supports RKS, UKS, KRKS and KUKS for neutral, three-dimensionally
    periodic slabs. The selected lattice vector must be parallel to its
    Cartesian axis and perpendicular to the other two lattice vectors.
    Analytical nuclear gradients are supported. Stress, response functions
    and band k-points are not implemented.

    Args:
        mf: GPU periodic Kohn-Sham object.
        dir_idx: Surface normal, 0, 1 or 2 (default: z).
        z0: Potential zero in Bohr. If None, subtract the volume-weighted
            mean of the sawtooth on the integration grid, so the applied
            potential has zero average. Otherwise, set v_dip(z0) = 0.
            This changes only the potential gauge, not the discontinuity.
        cut: Sawtooth discontinuity in Bohr; it must lie in vacuum.
            If None, find the largest periodic gap between real atoms, then
            choose the midpoint of its longest low-electron-density window
            on a uniform grid.
        rho_tol: Maximum plane-averaged electron density (electrons/Bohr**3)
            for automatic vacuum selection. At least three consecutive grid
            planes must lie below this threshold.

    The automatic cut is chosen from the first density matrix and held
    fixed until reset(). All length inputs and reported positions use Bohr,
    regardless of cell.unit. The selected discontinuity is available as
    dipole_cut; dipole_z0 reports the resulting potential zero modulo the
    cell length.

    The correction uses mf.grids, or the MultiGrid mesh when MultiGrid is
    active. The dipole and energy use the branch centered half a cell away
    from cut.
    Gauge shifts are applied with the analytic AO overlap matrix,
    so changing z0 does not change the dipole, energy or orbitals for neutral slabs,
    but it does change the orbital energies.
    The potential is the energy derivative at fixed electron number.
    To reproduce PySCF's slab_center=c convention, set z0=c and cut=c+L/2,
    with c and L expressed in Bohr.
    As in PySCF, the energy correction is included in veff.exc, and is also
    reported separately as veff.edip.

    For geometry optimization, set cut explicitly to keep the discontinuity
    fixed across resets. Becke grid response is not implemented; requesting
    it through gradient.grid_response raises NotImplementedError.
    '''
    from gpu4pyscf.pbc.dft import rks, uks, krks, kuks

    if not isinstance(mf, (rks.RKS, uks.UKS, krks.KRKS, kuks.KUKS)):
        raise NotImplementedError('Slab dipole correction requires RKS/UKS/KRKS/KUKS')
    _slab_length(mf.cell, dir_idx)
    if rho_tol <= 0:
        raise ValueError('rho_tol must be positive')
    if not isinstance(mf, SlabDipoleCorrection):
        lib.set_class(mf, (SlabDipoleCorrection, mf.__class__))
    mf.dipole_dir = dir_idx
    mf.z0 = z0
    mf.cut = cut                 # Explicit cut is in Bohr and persists on reset.
    mf.dipole_rho_tol = rho_tol
    mf.dipole_cut = None         # changed in reset()
    mf.dipole_z0 = None          # changed in reset()   
    mf.dipole_offset = None
    return mf


def _slab_length(cell, dir_idx):
    if dir_idx not in (0, 1, 2):
        raise ValueError('dir_idx must be 0, 1 or 2')
    if cell.dimension != 3 or cell.charge != 0:
        # TODO: in the future, support non-neutral cell slabs.
        raise NotImplementedError('Slab dipole correction requires a neutral 3D cell')
    a = cell.lattice_vectors()
    other = [i for i in range(3) if i != dir_idx]
    if (a[dir_idx, dir_idx] <= 0 or
            np.any(abs(a[dir_idx, other]) > 1e-10) or
            np.any(abs(a[other, dir_idx]) > 1e-10)):
        raise NotImplementedError('Slab normal must be parallel to a Cartesian axis')
    return a[dir_idx, dir_idx] # (0, 0), (1, 1) or (2, 2)


def _sawtooth(z, cut, length):
    return (z - cut) % length - length/2


def _dipole_grids(mf, cell):
    from gpu4pyscf.pbc.dft import gen_grid, multigrid

    if isinstance(mf._numint, multigrid.MultiGridNumIntBase):
        grids = gen_grid.UniformGrids(cell)
        grids.mesh = mf._numint.mesh
    else:
        grids = mf.grids
        if grids.coords is None:
            grids.build()
    return grids


def _automatic_cut(cell, dm, kpts, mesh, dir_idx, rho_tol):
    from gpu4pyscf.pbc.dft import numint, gen_grid

    length = _slab_length(cell, dir_idx)
    z = cell.atom_coords()[cell.atom_charges() > 0, dir_idx] / length % 1 # fractional coordinates
    z = np.sort(z)
    gaps = np.diff(np.append(z, z[0] + 1))
    i = gaps.argmax()
    start, end = z[i], z[i] + gaps[i]

    grids = gen_grid.UniformGrids(cell)
    grids.mesh = mesh
    nplane = mesh[dir_idx]
    profile = cp.zeros(nplane)
    ni = numint.KNumInt()
    for ao, weight, coords in ni.block_loop(cell, grids, kpts=kpts):
        rho = ni.eval_rho(cell, ao, dm, xctype='LDA', hermi=1)
        layers = cp.rint(coords[:, dir_idx] / length * nplane).astype(int) % nplane
        profile += cp.bincount(layers, weights=rho, minlength=nplane)
    # 1/A int n(x,y,z) dx dy, A is the surface area of the slab.
    profile = profile.get() / (np.prod(mesh) / nplane)

    # Unwrap the candidate gap, including one that crosses the cell boundary.
    layers = np.arange(int(np.floor(start*nplane)) + 1, int(np.ceil(end*nplane))) # vacuum layers
    low = profile[layers % nplane] < rho_tol
    edges = np.flatnonzero(np.diff(np.r_[False, low, False]))
    windows = edges.reshape(-1, 2)
    if len(windows) == 0 or np.max(windows[:, 1] - windows[:, 0]) < 3:
        raise ValueError('No low-density vacuum window for dipole correction; '
                         'increase the vacuum or refine the mesh')
    first, last = windows[np.argmax(windows[:, 1] - windows[:, 0])]
    # place the cut in the middle of the vacuum window.
    cut = (layers[first] + layers[last - 1]) / (2*nplane)
    return (cut % 1) * length # in Bohr


class SlabDipoleCorrection:
    _keys = {
        'dipole_dir', 'z0', 'cut', 'dipole_rho_tol',
        'dipole_cut', 'dipole_z0', 'dipole_offset',
    }

    def reset(self, cell=None):
        # Reset geometry-dependent values; an explicit cut in Bohr persists (.cut).
        self.dipole_cut = None
        self.dipole_z0 = None
        self.dipole_offset = None
        return super().reset(cell)

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None,
                 hermi=1, kpts=None, kpts_band=None, *, kpt=None):
        from gpu4pyscf.pbc.scf import khf
        from gpu4pyscf.pbc.dft import numint, gen_grid

        if kpts_band is not None:
            raise NotImplementedError('Slab dipole correction for band k-points')
        if cell is None:
            cell = self.cell
        if dm is None:
            dm = self.make_rdm1()
        length = _slab_length(cell, self.dipole_dir)
        is_kpts = isinstance(self, khf.KSCF)
        if kpts is None:
            kpts = self.kpts if is_kpts else (self.kpt if kpt is None else kpt)
        veff = super().get_veff(cell, dm, dm_last, vhf_last, hermi, kpts, kpts_band)
        kpts = np.asarray(kpts).reshape(-1, 3)
        dm_charge = cp.asarray(dm)
        if self.istype('UHF') or self.istype('KUHF'):
            dm_charge = dm_charge[0] + dm_charge[1]
        dm_charge = dm_charge.reshape(len(kpts), cell.nao, cell.nao)

        grids = _dipole_grids(self, cell)
        mesh = grids.mesh if isinstance(grids, gen_grid.UniformGrids) else cell.mesh
        cut = self.cut
        if cut is None:
            if self.dipole_cut is None:
                self.dipole_cut = _automatic_cut(
                    cell, dm_charge, kpts, mesh, self.dipole_dir, self.dipole_rho_tol)
            cut = self.dipole_cut
        else:
            cut = self.dipole_cut = cut % length

        # Only a (nkpts, nao, nao) position matrix is retained. Using its trace
        # for the dipole makes the zero-offset potential the density derivative
        # of the discretized correction energy, including the k-point weight.
        ni = numint.KNumInt()
        position = None
        z_sum = cp.asarray(0.)
        weight_sum = cp.asarray(0.)
        for ao_kpts, weight, coords in ni.block_loop(cell, grids, kpts=kpts):
            zw = _sawtooth(coords[:, self.dipole_dir], cut, length) * weight
            z_sum += zw.sum()
            weight_sum += weight.sum()
            if position is None:
                position = cp.zeros(dm_charge.shape, dtype=ao_kpts.dtype)
            for k, ao in enumerate(ao_kpts):
                position[k] += ao.conj().T @ (zw[:, None] * ao)
        p_elec = cp.einsum('kij,kji->', dm_charge, position).real / len(kpts)
        p_ion = np.dot(cell.atom_charges(),
                       _sawtooth(cell.atom_coords()[:, self.dipole_dir], cut, length))
        dipole = p_ion - float(p_elec)
        field = 4 * np.pi * dipole / cell.vol
        edip = .5 * field * dipole
        self.dipole_offset = (
            float(z_sum / weight_sum) if self.z0 is None else
            _sawtooth(self.z0, cut, length))
        offset = self.dipole_offset
        self.dipole_z0 = (cut + length/2 + offset) % length
        v_dip = -field * position
        if not is_kpts:
            v_dip = v_dip[0]
        # A constant real-space potential is offset * S, not offset * I.
        overlap = self.get_ovlp(cell, kpts if is_kpts else kpts[0])
        v_dip += field * offset * overlap
        out = tag_array(veff + v_dip, **veff.__dict__)
        out.exc = veff.exc + edip
        out.edip = edip
        out.dipole = dipole
        logger.info(self, 'Dipole correction: E = %.10g, p = %.6g, '
                    'cut = %.6g, z0 = %.6g Bohr',
                    edip, dipole, cut, self.dipole_z0)
        return out

    def Gradients(self):
        from gpu4pyscf.pbc.grad.slabdipole import _SlabDipoleGradients
        grad = super().Gradients()
        return lib.set_class(grad, (_SlabDipoleGradients, grad.__class__))

    def Hessian(self):
        raise NotImplementedError('Slab dipole correction Hessian')

    def gen_response(self, *args, **kwargs):
        raise NotImplementedError('Slab dipole correction response kernel')

    def to_cpu(self):
        raise NotImplementedError('Apply the PySCF slab dipole addon to a CPU object')
