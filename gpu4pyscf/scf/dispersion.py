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
dispersion correction for HF and DFT
'''


from gpu4pyscf.scf import hf, uhf
from gpu4pyscf.dft import rks, uks

def get_dispersion(mf, disp_version=None):
    if disp_version is None:
        disp_version = mf.disp
    mol = mf.mol
    if disp_version is None:
        return 0.0
    if isinstance(mf, rks.KohnShamDFT):
        method = mf.xc
    else:
        method = 'hf'

    # for dftd3
    if disp_version[:2].upper() == 'D3':
        from gpu4pyscf.lib import dftd3
        dftd3_model = dftd3.DFTD3Dispersion(mol, xc=method, version=disp_version)
        res = dftd3_model.get_dispersion()
        return res['energy']

    # for dftd4
    elif disp_version[:2].upper() == 'D4':
        from gpu4pyscf.lib import dftd4
        dftd4_model = dftd4.DFTD4Dispersion(mol, xc=method)
        res = dftd4_model.get_dispersion()
        return res.get("energy")
    else:
        raise RuntimeError(f'dipersion correction: {disp_version} is not supported.')

# Inject to SCF class
hf.RHF.get_dispersion = get_dispersion
uhf.UHF.get_dispersion = get_dispersion
rks.RKS.get_dispersion = get_dispersion
uks.UKS.get_dispersion = get_dispersion