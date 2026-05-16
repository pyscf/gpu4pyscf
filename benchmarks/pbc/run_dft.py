# Copyright 2026 The GPU4PySCF Authors. All Rights Reserved.
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
import numpy as np
import pyscf
import yaml
from pyscf import lib
from pyscf.dft import libxc
from pyscf.pbc import tools as pbctools
from pyscf.pbc import gto as pbcgto
from gpu4pyscf.pbc.scf import rsjk

def parse_cif(filename):
    with open(filename) as f:
        string = f.read()
    return pbcgto.cell._parse_cif(string)

basis_tpl = {
    'C': r'TZVP-MOLOPT-{}-GTH-q4',
    'O': r'TZVP-MOLOPT-{}-GTH-q6',
    'Mg': r'TZVP-MOLOPT-{}-GTH-q10',
    'Al': r'TZVP-MOLOPT-{}-GTH-q3',
    'Si': r'TZVP-MOLOPT-{}-GTH-q4',
    'S': r'TZVP-MOLOPT-{}-GTH-q6',
    'Ca': r'TZVP-MOLOPT-{}-GTH-q10',
    'Ti': r'TZVP-MOLOPT-{}-GTH-q12',
    'Ga': r'TZVP-MOLOPT-{}-GTH-q13',
    'As': r'TZVP-MOLOPT-{}-GTH-q5',
    'Mo': r'TZVP-MOLOPT-{}-GTH-q14',
    'Zr': r'TZVP-MOLOPT-{}-GTH-q12',
}

def xc_type(xc):
    if libxc.is_hybrid_xc(xc):
        return 'HYB'
    return libxc.xc_type(xc)

configs = yaml.safe_load(
    '''
- filename:
  - Si_primitive.cif
  - MgO_primitive.cif
  - Al2O3_ICSD_31548.cif
  - CaTiO3.cif
  - ZrO2_14.cif
  method:
  - xc: PBE
    kmesh:
    - [3,3,3]
    - [4,4,4]
    - [5,5,5]
    - [6,6,6]
    - [7,7,7]
  - xc: R2SCAN
    kmesh:
    - [3,3,3]
    - [4,4,4]
    - [5,5,5]
    - [6,6,6]
    - [7,7,7]
  - xc: PBE0
    density_fit: null
    kmesh:
    - [3,3,3]
    - [4,4,4]
    - [5,5,5]
    - [6,6,6]
    - [7,7,7]
  - xc: HSE06
    density_fit: null
    kmesh:
    - [3,3,3]
    - [4,4,4]
    - [5,5,5]
    - [6,6,6]
    - [7,7,7]
  - xc: PBE0
    rsjk: null
    kmesh:
    - [5,5,5]
    - [6,6,6]
    - [7,7,7]
    - [8,8,8]
    - [9,9,9]
  - xc: HSE06
    rsjk: null
    kmesh:
    - [5,5,5]
    - [6,6,6]
    - [7,7,7]
    - [8,8,8]
    - [9,9,9]
- filename:
  - Mg4O4_cubic.cif
  method:
  - xc: PBE
  - xc: PBE
    supercell: [2,2,2]
  - xc: PBE
    supercell: [3,3,3]
  - xc: PBE
    supercell: [4,4,4]
  - xc: PBE0
    rsjk: null
  - xc: PBE0
    rsjk: null
    supercell: [2,2,2]
  - xc: PBE0
    rsjk: null
    supercell: [3,3,3]
  - xc: PBE0
    rsjk: null
    supercell: [4,4,4]
- filename:
  - GaAs.cif
  method:
  - xc: PBE
    smearing:
      sigma: 0.005
    kmesh:
    - [3,3,3]
    - [4,4,4]
    - [5,5,5]
    - [6,6,6]
    - [7,7,7]
  - xc: R2SCAN
    smearing:
      sigma: 0.005
    kmesh:
    - [3,3,3]
    - [4,4,4]
    - [5,5,5]
    - [6,6,6]
    - [7,7,7]
  - xc: PBE0
    smearing:
      sigma: 0.005
    density_fit: null
    kmesh:
    - [3,3,3]
    - [4,4,4]
    - [5,5,5]
    - [6,6,6]
    - [7,7,7]
  - xc: HSE06
    smearing:
      sigma: 0.005
    density_fit: null
    kmesh:
    - [3,3,3]
    - [4,4,4]
    - [5,5,5]
    - [6,6,6]
    - [7,7,7]
- filename:
  - Al.cif
  method:
  - xc: PBE
    smearing:
      sigma: 0.005
    kmesh:
    - [6,6,6]
    - [8,8,8]
    - [12,12,12]
  - xc: R2SCAN
    smearing:
      sigma: 0.005
    kmesh:
    - [6,6,6]
    - [8,8,8]
    - [12,12,12]
  - xc: HSE06
    rsjk: null
    smearing:
      sigma: 0.005
    kmesh:
    - [6,6,6]
    - [8,8,8]
    - [12,12,12]
- filename:
  - Al2Mg3O12Si3_ICSD_80847.cif
  method:
  - xc: PBE
  - xc: PBE
    supercell: [2,1,1]
  - xc: PBE
    supercell: [2,2,1]
  - xc: PBE
    supercell: [2,2,2]
  - xc: PBE0
    rsjk: null
    supercell: [2,1,1]
  - xc: PBE0
    rsjk: null
    supercell: [2,2,1]
  - xc: PBE0
    rsjk: null
    supercell: [2,2,2]
- filename:
  - MoS2.cif
  method:
  - xc: PBE
    smearing:
      sigma: 0.005
    kmesh:
    - [6,6,1]
    - [7,7,1]
    - [8,8,1]
    - [12,12,1]
  method:
  - xc: HSE06
    density_fit: null
    smearing:
      sigma: 0.005
    kmesh:
    - [6,6,1]
    - [7,7,1]
    - [8,8,1]
    - [12,12,1]
''')

crystal_dir = '../crystals'
output_dir = '.'

for conf in configs:
    for filename in conf['filename']:
        a, elements, coords, fractional = parse_cif(f'{crystal_dir}/{filename}')
        if fractional:
            coords = coords.dot(a)
        atoms = list(zip(elements, coords))
        for method in conf['method']:
            xc = method['xc']
            xc_family = xc_type(xc)
            basis = {k: v.format(xc_family) for k, v in basis_tpl.items()}
            pseudo = {k: f"GTH-{xc_family}-{v.split('-')[-1]}" for k, v in basis_tpl.items()}

            for kmesh in method.get('kmesh', [[1,1,1]])
                kstring = 'x'.join([str(x) for x in kmesh])
                confstr = f'{xc}-k{kstring}'
                if 'density_fit' in method:
                    confstr = confstr + '-GDF'
                if 'rsjk' in method:
                    confstr = confstr + '-RSJK'
                if 'supercell' in method:
                    ncopy = method['supercell']
                    scstring = 'x'.join([str(x) for x in ncopy])
                    confstr = f'{scstring}-{confstr}'
                    cell = pyscf.M(atom=atoms, a=a, basis=basis, pseudo=pseudo,
                                   output=f'{output_dir}/{filename}-{confstr}.out')
                    cell = pbctools.super_cell(cell, ncopy)
                else:
                    cell = pyscf.M(atom=atoms, a=a, basis=basis, pseudo=pseudo,
                                   output=f'{output_dir}/{filename}-{confstr}.out')
                cell.verbose = 5

                Nk = np.prod(kmesh)
                if Nk == 1:
                    mf = cell.RKS(xc=xc).to_gpu()
                else:
                    kpts = cell.make_kpts(kmesh)
                    mf = cell.KRKS(xc=xc, kpts=kpts).to_gpu()
                mf = mf.multigrid_numint()
                if 'density_fit' in method:
                    mf = mf.density_fit()
                mf.max_cycle = 20
                mf.conv_tol = 1e-6
                if 'rsjk' in method:
                    mf.rsjk = rsjk.PBCJKMatrixOpt(cell)
                if 'smearing' in method:
                    mf = mf.smearing(**method['smearing'])
                try:
                    mf.run()
                except Exception as e:
                    import traceback
                    traceback.print_stack()
                    traceback.print_exception(e)
