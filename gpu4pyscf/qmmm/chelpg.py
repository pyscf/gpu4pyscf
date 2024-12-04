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

import cupy
import numpy as np
import scipy
from gpu4pyscf.gto.moleintor import intor, VHFOpt
from gpu4pyscf.lib import logger

from pyscf.data import radii
modified_Bondi = radii.VDW.copy()
modified_Bondi[1] = 1.1/radii.BOHR      # modified version

def eval_chelpg_layer_gpu(mf, deltaR=0.3, Rhead=2.8, ifqchem=True, Rvdw=modified_Bondi, verbose=None):
    """Cal chelpg charge

    Args:
        mf: mean field object in pyscf
        deltaR (float, optional): the intervel in the cube. Defaults to 0.3.
        Rhead (float, optional): the head length. Defaults to 3.0.
        ifqchem (bool, optional): whether use the modification in qchem. Defaults to True.
        Rvdw (dict, optional): vdw radius. Defaults to modified Bondi radii.
    Returns:
        numpy.array: charges
    """
    log = logger.new_logger(mf, verbose)
    t1 = log.init_timer()

    atomcoords = mf.mol.atom_coords(unit='B')
    dm = cupy.array(mf.make_rdm1())

    Roff = Rhead/radii.BOHR
    Deltar = 0.1

    # smoothing function
    def tau_f(R, Rcut, Roff):
        return (R - Rcut)**2 * (3*Roff - Rcut - 2*R) / (Roff - Rcut)**3

    Rshort = np.array([Rvdw[iatom] for iatom in mf.mol._atm[:, 0]])
    idxxmin = np.argmin(atomcoords[:, 0] - Rshort)
    idxxmax = np.argmax(atomcoords[:, 0] + Rshort)
    idxymin = np.argmin(atomcoords[:, 1] - Rshort)
    idxymax = np.argmax(atomcoords[:, 1] + Rshort)
    idxzmin = np.argmin(atomcoords[:, 2] - Rshort)
    idxzmax = np.argmax(atomcoords[:, 2] + Rshort)
    atomtypes = np.array(mf.mol._atm[:, 0])
    # Generate the grids in the cube
    xmin = atomcoords[:, 0].min() - Rhead/radii.BOHR - Rvdw[atomtypes[idxxmin]]
    xmax = atomcoords[:, 0].max() + Rhead/radii.BOHR + Rvdw[atomtypes[idxxmax]]
    ymin = atomcoords[:, 1].min() - Rhead/radii.BOHR - Rvdw[atomtypes[idxymin]]
    ymax = atomcoords[:, 1].max() + Rhead/radii.BOHR + Rvdw[atomtypes[idxymax]]
    zmin = atomcoords[:, 2].min() - Rhead/radii.BOHR - Rvdw[atomtypes[idxzmin]]
    zmax = atomcoords[:, 2].max() + Rhead/radii.BOHR + Rvdw[atomtypes[idxzmax]]
    x = np.arange(xmin, xmax, deltaR/radii.BOHR)
    y = np.arange(ymin, ymax, deltaR/radii.BOHR)
    z = np.arange(zmin, zmax, deltaR/radii.BOHR)
    gridcoords = np.meshgrid(x, y, z)
    gridcoords = np.vstack(list(map(np.ravel, gridcoords))).T

    # [natom, ngrids] distance between an atom and a grid
    r_pX = scipy.spatial.distance.cdist(atomcoords, gridcoords)
    # delete the grids in the vdw surface and out the Rhead surface.
    # the minimum distance to any atom
    Rkmin = (r_pX - np.expand_dims(Rshort, axis=1)).min(axis=0)
    Ron = Rshort + Deltar
    Rlong = Roff - Deltar
    AJk = np.ones(r_pX.shape)  # the short-range weight
    idx = r_pX < np.expand_dims(Rshort, axis=1)
    AJk[idx] = 0
    if ifqchem:
        idx2 = (r_pX < np.expand_dims(Ron, axis=1)) * \
            (r_pX >= np.expand_dims(Rshort, axis=1))
        AJk[idx2] = tau_f(r_pX, np.expand_dims(Rshort, axis=1),
                          np.expand_dims(Ron, axis=1))[idx2]
        wLR = 1 - tau_f(Rkmin, Rlong, Roff)  # the long-range weight
        idx1 = Rkmin < Rlong
        idx2 = Rkmin > Roff
        wLR[idx1] = 1
        wLR[idx2] = 0
    else:
        wLR = np.ones(r_pX.shape[-1])  # the long-range weight
        idx = Rkmin > Roff
        wLR[idx] = 0
    w = wLR*np.prod(AJk, axis=0)  # weight for a specific poing
    idx = w <= 1.0E-14
    w = np.delete(w, idx)
    r_pX = np.delete(r_pX, idx, axis=1)
    gridcoords = np.delete(gridcoords, idx, axis=0)

    ngrids = gridcoords.shape[0]
    r_pX = cupy.array(r_pX)
    r_pX_potential = 1/r_pX
    potential_real = cupy.dot(cupy.array(
        mf.mol.atom_charges()), r_pX_potential)
    nbatch = 256*256

    # assert nbatch < ngrids
    intopt = VHFOpt(mf.mol)
    intopt.build(cutoff=1e-14)
    for ibatch in range(0, ngrids, nbatch):
        max_grid = min(ibatch+nbatch, ngrids)
        potential_real[ibatch:max_grid] -= \
            intor(mf.mol, 'int1e_grids', gridcoords[ibatch:max_grid], dm=dm, intopt=intopt)

    w = cupy.array(w)
    r_pX_potential_omega = r_pX_potential*w
    GXA = r_pX_potential_omega@r_pX_potential.T
    eX = r_pX_potential_omega@potential_real
    GXA_inv = cupy.linalg.inv(GXA)
    g = GXA_inv@eX
    alpha = (g.sum() - mf.mol.charge)/(GXA_inv.sum())
    q = g - alpha*GXA_inv@cupy.ones((mf.mol.natm))
    t1 = log.timer_debug1('compute ChElPG charge', *t1)
    return q

