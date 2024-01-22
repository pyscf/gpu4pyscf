# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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
import numpy
from pyscf import gto
from gpu4pyscf.solvent import smd
from gpu4pyscf.solvent.grad import smd as smd_grad

def _check_grad(mol, solvent='water'):
    natm = mol.natm
    fd_cds = numpy.zeros([natm,3])
    eps = 1e-4
    for ia in range(mol.natm):
        for j in range(3):
            coords = mol.atom_coords(unit='B')
            coords[ia,j] += eps
            mol.set_geom_(coords, unit='B')
            mol.build()

            smdobj = smd.SMD(mol)
            smdobj.solvent = solvent
            e0_cds = smdobj.get_cds()

            coords[ia,j] -= 2.0*eps
            mol.set_geom_(coords, unit='B')
            mol.build()

            smdobj = smd.SMD(mol)
            smdobj.solvent = solvent
            e1_cds = smdobj.get_cds()

            coords[ia,j] += eps
            mol.set_geom_(coords, unit='B')
            fd_cds[ia,j] = (e0_cds - e1_cds) / (2.0 * eps)

    smdobj = smd.SMD(mol)
    smdobj.solvent = solvent
    grad_cds = smd_grad.get_cds(smdobj)
    print(numpy.linalg.norm(fd_cds - grad_cds.get()))
    assert numpy.linalg.norm(fd_cds - grad_cds.get()) < 1e-8

if __name__ == "__main__":
    path = '../molecules/organic/'
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        mol = gto.Mole(atom=f)
        mol.build()
        print(f'benchmarking {f} in water')
        _check_grad(mol, solvent='water')
        print(f'benchmarking {f} in ethanol')
        _check_grad(mol, solvent='ethanol')