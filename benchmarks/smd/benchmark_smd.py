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

'''
Benchmark CDS, gradient, hessian in the SMD model
'''

path = '../molecules/organic/'

# calculated with qchem 6.1, in kcal/mol
e_cds_qchem = {}
e_cds_qchem['water'] = {
    '020_Vitamin_C.xyz': 5.0737,
    '031_Inosine.xyz': 2.7129,
    '033_Bisphenol_A.xyz': 6.2620,
    '037_Mg_Porphin.xyz': 6.0393,
    '042_Penicillin_V.xyz': 6.4349,
    '045_Ochratoxin_A.xyz': 8.8526,
    '052_Cetirizine.xyz': 4.6430,
    '057_Tamoxifen.xyz': 5.4743,
    '066_Raffinose.xyz': 10.2543,
    '084_Sphingomyelin.xyz': 15.0308,
    '095_Azadirachtin.xyz': 16.9321,
    '113_Taxol.xyz': 17.2585,
    '168_Valinomycin.xyz': 27.3499,
}

e_cds_qchem['ethanol'] = {
    '020_Vitamin_C.xyz': 4.2119,
    '031_Inosine.xyz': 1.0175,
    '033_Bisphenol_A.xyz': -0.2454,
    '037_Mg_Porphin.xyz': -2.2391,
    '042_Penicillin_V.xyz': 1.8338,
    '045_Ochratoxin_A.xyz': 1.0592,
    '052_Cetirizine.xyz': -2.5099,
    '057_Tamoxifen.xyz': -3.9320,
    '066_Raffinose.xyz': 3.1120,
    '084_Sphingomyelin.xyz': -3.1963,
    '095_Azadirachtin.xyz': 6.5286,
    '113_Taxol.xyz': 2.7271,
    '168_Valinomycin.xyz': 4.0013,
}

def _check_energy_grad(filename, solvent='water'):
    xyz = os.path.join(path, filename)
    mol = gto.Mole(atom=xyz)
    mol.build()
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
    e_cds = smd.get_cds(smdobj) * smd.hartree2kcal
    grad_cds = smd_grad.get_cds(smdobj)
    print(f'e_cds by GPU4PySCF: {e_cds}')
    print(f'e_cds by Q-Chem: {e_cds_qchem[solvent][filename]}')
    print(f'e_cds(Q-Chem) - e_cds(GPU4PySCF): {e_cds - e_cds_qchem[solvent][filename]}')
    print(f'norm (fd gradient - analy gradient: {numpy.linalg.norm(fd_cds - grad_cds.get())}')
    assert numpy.linalg.norm(fd_cds - grad_cds.get()) < 1e-8

if __name__ == "__main__":
    for filename in os.listdir(path):
        print(f'---- benchmarking {filename} ----------')
        print('in water')
        _check_energy_grad(filename, solvent='water')
        print('in ethanol')
        _check_energy_grad(filename, solvent='ethanol')