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


'''
Hessian of dispersion correction for HF and DFT
'''

import numpy
from gpu4pyscf import dft

def get_dispersion(hessobj, disp_version=None):
    if disp_version is None:
        disp_version = hessobj.base.disp
    mol = hessobj.base.mol
    natm = mol.natm
    mf = hessobj.base
    h_disp = numpy.zeros([natm,natm,3,3])
    if disp_version is None:
        return h_disp
    if isinstance(hessobj.base, dft.rks.KohnShamDFT):
        method = hessobj.base.xc
    else:
        method = 'hf'

    if mf.disp[:2].upper() == 'D3':
        from gpu4pyscf.lib import dftd3
        coords = mol.atom_coords()
        natm = mol.natm
        h_d3 = numpy.zeros([mol.natm, mol.natm, 3,3])
        pmol = mol.copy()
        pmol.verbose = 0
        eps = 1e-5
        for i in range(natm):
            for j in range(3):
                coords[i,j] += eps
                pmol.set_geom_(coords, unit='Bohr')
                pmol.build()
                dftd3_model = dftd3.DFTD3Dispersion(pmol, xc=method, version=disp_version)
                res = dftd3_model.get_dispersion(grad=True)
                g1 = res['gradient']

                coords[i,j] -= 2.0*eps
                pmol.set_geom_(coords, unit='Bohr')
                pmol.build()
                dftd3_model = dftd3.DFTD3Dispersion(pmol, xc=method, version=disp_version)
                res = dftd3_model.get_dispersion(grad=True)
                g2 = res['gradient']

                coords[i,j] += eps
                h_d3[i,:,j,:] = (g1 - g2)/(2.0*eps)
        return h_d3

    elif mf.disp[:2].upper() == 'D4':
        from gpu4pyscf.lib import dftd4
        coords = mol.atom_coords()
        natm = mol.natm
        pmol = mol.copy()
        pmol.verbose = 0
        h_d4 = numpy.zeros([mol.natm, mol.natm, 3,3])
        eps = 1e-5
        for i in range(natm):
            for j in range(3):
                coords[i,j] += eps
                pmol.set_geom_(coords, unit='Bohr')
                pmol.build()
                dftd4_model = dftd4.DFTD4Dispersion(pmol, xc=method)
                res = dftd4_model.get_dispersion(grad=True)
                g1 = res.get("gradient")

                coords[i,j] -= 2.0*eps
                pmol.set_geom_(coords, unit='Bohr')
                pmol.build()
                dftd4_model = dftd4.DFTD4Dispersion(pmol, xc=method)
                res = dftd4_model.get_dispersion(grad=True)
                g2 = res.get("gradient")

                coords[i,j] += eps
                h_d4[i,:,j,:] = (g1 - g2)/(2.0*eps)

        return h_d4
    else:
        raise RuntimeError(f'dispersion correction: {disp_version} is not supported.')

# Inject to SCF class
from gpu4pyscf.hessian import rhf, uhf, rks, uks
rhf.Hessian.get_dispersion = get_dispersion
uhf.Hessian.get_dispersion = get_dispersion
rks.Hessian.get_dispersion = get_dispersion
uks.Hessian.get_dispersion = get_dispersion