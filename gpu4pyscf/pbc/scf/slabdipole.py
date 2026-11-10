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
Self-consistent slab dipole correction for periodic SCF calculations.
'''

import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array


def slab_dipole_correction(mf, z0=None, cut=None, rho_tol=1e-5, width=0.):
    r'''Remove the artificial electric field between periodically repeated slabs.
    Apply self-consistent slab dipole corrections to the given mf object in place.

    A neutral slab can have a dipole perpendicular to its surface, for example
    when its two surfaces have different chemical compositions. Repeating such
    a slab periodically introduces an artificial electric field. This function
    adds a correction potential and energy to remove that field.

    The correction does not force the slab's own dipole to zero. Its strength
    is calculated from the ionic charges and electronic density and is updated
    during SCF.

    The added potential is a sawtooth: it varies linearly along the slab normal,
    jumps at one position in the vacuum, and repeats in each cell. This jump
    represents an ideal dipole sheet.

    Args:
        mf: GPU RKS, UKS, KRKS or KUKS object.
            Modified in place. The cell must be neutral or compensated by
            gate_charge, and periodic in all three dimensions, with a vacuum
            region separating slab images.
            The slab normal is always z. The third lattice vector must lie
            along the positive z-axis, and the other two must lie in the xy plane.

        z0: Position along z where the correction potential is zero, in Bohr.
            Changing z0 shifts the potential by a constant; it does not move
            the jump or change the correction field. It shifts orbital
            energies but leaves the self-consistent density and total energy
            unchanged.
            If z0=None, the potential zero is automatically chosen so that the
            correction potential averages to zero over the integration grid.
            gate_charge sets z0=(cut+L/2) % L to reproduce the analytic
            zero-cell-integral potential in Brumme Eq. (7).

        cut: Position of the potential jump along z, in Bohr.
            This is a position within the cell.
            It must lie in the vacuum, where the electron density is small.
            The jump represents an artificial dipole sheet. Non-vanishing
            electron density near cut would interact with the jump.

            If cut=None, choose a low-density position in the largest periodic
            gap between atoms, using the first density matrix during an SCF
            iteration. This position should be kept fixed during SCF.
            For geometry optimization, it's importatnt to explicitly set cut to
            keep the jump fixed between geometries.

        rho_tol: Electron-density threshold for automatic cut selection.
            At least three consecutive grid planes must have a plane-averaged
            density below this threshold. Ignored when cut is supplied.

        width: Width in Bohr of the linear return region centered on cut.
            Zero retains the discontinuous Bengtsson sawtooth. A positive
            width gives the continuous Brumme sawtooth; the entire return
            region must lie in vacuum. Keep cut explicit for this mode.

    Returns:
        The same mf object with the dipole correction enabled.

    Notes:
        After calling veff = get_veff(), mf.dipole_cut gives the selected jump position
        and mf.dipole_z0 gives the position where the correction potential is zero.
        veff.dipole stores the charge dipole and veff.edip stores the correction energy.

        The dipole and correction energy use wrapped coordinates centered
        half a cell away from cut. Changing z0 only changes the potential
        reference; it does not change these coordinates or the calculated
        dipole.

        Setting gradient.grid_response=True with Becke grids raises NotImplementedError.

    Reference:
        Bengtsson, Phys. Rev. B 59, 12301 (1999).
        doi:10.1103/PhysRevB.59.12301
    '''
    from gpu4pyscf.pbc.dft import rks, uks, krks, kuks

    if not isinstance(mf, (rks.RKS, uks.UKS, krks.KRKS, kuks.KUKS)):
        raise NotImplementedError('Slab dipole correction requires RKS/UKS/KRKS/KUKS')
    length = _slab_length(mf.cell)
    _check_dipole_charge(mf, mf.cell)
    if not 0 <= width < length:
        raise ValueError('Dipole width must satisfy 0 <= width < cell length')
    if width > 0 and cut is None:
        raise ValueError('Set cut explicitly for a finite-width dipole return region')
    if rho_tol <= 0:
        raise ValueError('rho_tol must be positive')
    if not isinstance(mf, SlabDipoleCorrection):
        lib.set_class(mf, (SlabDipoleCorrection, mf.__class__))
    mf.z0 = z0
    mf.cut = cut                 # Explicit cut is in Bohr and persists on reset.
    mf.dipole_width = width
    mf.dipole_rho_tol = rho_tol
    mf.dipole_cut = None         # changed in reset()
    mf.dipole_z0 = None          # changed in reset()
    mf.dipole_offset = None
    return mf


def _slab_length(cell):
    if cell.dimension != 3:
        raise NotImplementedError('Slab corrections require a 3D cell')
    a = cell.lattice_vectors()
    if (a[2, 2] <= 0 or
            np.any(abs(a[2, :2]) > 1e-10) or
            np.any(abs(a[:2, 2]) > 1e-10)):
        raise NotImplementedError(
            'Slab correction requires the third lattice vector along positive z '
            'and the other two in the xy plane')
    return a[2, 2]


def _slab_charge(mf, cell):
    nkpts = len(mf.kpts) if mf.istype('KSCF') else 1
    return cell.charge / nkpts


def _check_dipole_charge(mf, cell):
    from gpu4pyscf.pbc.scf.slabgate import SlabGateCharge
    if _slab_charge(mf, cell) != 0 and not isinstance(mf, SlabGateCharge):
        raise NotImplementedError('Charged slab dipole correction requires gate_charge')


def _sawtooth(z, cut, length, width=0.):
    """Periodic dipole sawtooth: L=length, d_dip=width, z_dip=cut.

    t is z-z_dip wrapped into [-d_dip/2, L-d_dip/2).
    f(t) = t - L/2                  if t >= d_dip/2
         = -t*(L-d_dip)/d_dip       if t < d_dip/2
    The linear return region [-d_dip/2, d_dip/2) has width d_dip.
    width=0 recovers the discontinuous sawtooth.
    """
    z = cp.asarray(z)
    if width == 0:
        return (z - cut) % length - length/2
    t = (z - cut + width/2) % length - width/2
    return cp.where(t < width/2, -t*(length-width)/width, t-length/2)


def _sawtooth_deriv(z, cut, length, width=0.):
    z = cp.asarray(z)
    if width == 0:
        return cp.ones_like(z)
    t = (z - cut + width/2) % length - width/2
    return cp.where(t < width/2, -(length-width)/width, 1.)


def _dipole_grids(mf, cell):
    from gpu4pyscf.pbc.dft import gen_grid, multigrid

    if isinstance(mf._numint, multigrid.MultiGridNumIntBase):
        grids = gen_grid.UniformGrids(cell)
        grids.mesh = mf._numint.mesh
    else:
        grids = mf.grids
        if not isinstance(grids, gen_grid.UniformGrids) and grids.coords is None:
            grids.build()
    return grids


def _automatic_cut(cell, dm, kpts, mesh, rho_tol):
    from gpu4pyscf.pbc.dft import numint, gen_grid

    length = _slab_length(cell)
    z = cell.atom_coords()[cell.atom_charges() > 0, 2] / length % 1 # fractional coordinates
    z = np.sort(z)
    gaps = np.diff(np.append(z, z[0] + 1))
    i = gaps.argmax()
    start, end = z[i], z[i] + gaps[i]

    grids = gen_grid.UniformGrids(cell)
    grids.mesh = mesh
    nplane = mesh[2]
    profile = cp.zeros(nplane)
    ni = numint.KNumInt()
    for ao, weight, coords in ni.block_loop(cell, grids, kpts=kpts):
        rho = ni.eval_rho(cell, ao, dm, xctype='LDA', hermi=1)
        layers = cp.rint(coords[:, 2] / length * nplane).astype(int) % nplane
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
    __name_mixin__ = 'Dipole'

    _keys = {
        'z0', 'cut', 'dipole_rho_tol', 'dipole_width',
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
        length = _slab_length(cell)
        _check_dipole_charge(self, cell)
        width = self.dipole_width
        if not 0 <= width < length:
            raise ValueError('Dipole width must satisfy 0 <= width < cell length')
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
                    cell, dm_charge, kpts, mesh, self.dipole_rho_tol)
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
            zw = _sawtooth(coords[:, 2], cut, length, width) * weight
            z_sum += zw.sum()
            weight_sum += weight.sum()
            if position is None:
                position = cp.zeros(dm_charge.shape, dtype=ao_kpts.dtype)
            for k, ao in enumerate(ao_kpts):
                position[k] += ao.conj().T @ (zw[:, None] * ao)
        p_elec = cp.einsum('kij,kji->', dm_charge, position).real / len(kpts)
        p_ion = np.dot(cell.atom_charges(),
                       _sawtooth(cell.atom_coords()[:, 2], cut, length, width).get())
        charge = _slab_charge(self, cell)
        # Add the gate charge contribution to the dipole.
        if charge != 0:
            p_ion -= charge * float(_sawtooth(self._gate_parameters(cell)[1],
                                             cut, length, width))
        dipole = p_ion - float(p_elec)
        field = 4 * np.pi * dipole / cell.vol
        edip = .5 * field * dipole
        v_dip = -field * position
        if not is_kpts:
            v_dip = v_dip[0]
        self.dipole_offset = (
            float(z_sum / weight_sum) if self.z0 is None else
            float(_sawtooth(self.z0, cut, length, width)))
        offset = self.dipole_offset
        self.dipole_z0 = ((cut + length/2 + offset) % length if self.z0 is None
                          else self.z0 % length)
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
