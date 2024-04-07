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
gradient of dispersion correction for HF and DFT
'''

import numpy
from gpu4pyscf import dft

def get_dispersion(mf_grad, disp_version=None):
    '''gradient of dispersion correction for RHF/RKS'''
    if disp_version is None:
        disp_version = mf_grad.base.disp
    mol = mf_grad.base.mol
    disp_version = mf_grad.base.disp
    if disp_version is None:
        return numpy.zeros([mol.natm,3])

    if isinstance(mf_grad.base, dft.rks.KohnShamDFT):
        method = mf_grad.base.xc
    else:
        method = 'hf'

    if disp_version[:2].upper() == 'D3':
        # raised error in SCF module, assuming dftd3 installed
        from gpu4pyscf.lib import dftd3
        dftd3_model = dftd3.DFTD3Dispersion(mol, xc=method, version=disp_version)
        res = dftd3_model.get_dispersion(grad=True)
        return res['gradient']

    elif disp_version[:2].upper() == 'D4':
        from gpu4pyscf.lib import dftd4
        dftd4_model = dftd4.DFTD4Dispersion(mol, xc=method)
        res = dftd4_model.get_dispersion(grad=True)
        print(method, disp_version)
        print(res.get("gradient"))
        return res.get("gradient")
    else:
        raise RuntimeError(f'dispersion correction: {disp_version} is not supported.')

# Inject to Gradient
from gpu4pyscf.grad import rhf, uhf, rks, uks
rhf.Gradients.get_dispersion = get_dispersion
uhf.Gradients.get_dispersion = get_dispersion
rks.Gradients.get_dispersion = get_dispersion
uks.Gradients.get_dispersion = get_dispersion