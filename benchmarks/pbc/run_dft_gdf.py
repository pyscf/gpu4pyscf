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
import cupy
import pyscf
import yaml
from pyscf.dft import libxc
from pyscf.pbc import tools as pbctools
from pyscf.pbc import gto as pbcgto

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
  - GaAs.cif
  method:
  - xc:
    - PBE0
    - HSE06
    kmesh:
    - [3,3,3]
    - [4,4,4]
    - [5,5,5]
    - [6,6,6]
    - [7,7,7]
- filename:
  - CaTiO3.cif
  - ZrO2_14.cif
  - Al2O3_ICSD_31548.cif
  method:
  - xc:
    - PBE0
    - HSE06
    kmesh:
    - [3,3,3]
    - [4,4,4]
- filename:
  - Al_primitive.cif
  method:
  - xc:
    - PBE0
    - HSE06
    smearing:
      sigma: 0.005
    kmesh:
    - [6,6,6]
    - [8,8,8]
    - [10,10,10]
- filename:
  - MoS2.cif
  method:
  - xc:
    - PBE0
    kmesh:
    - [6,6,1]
    - [7,7,1]
    - [8,8,1]
    - [10,10,1]
    smearing:
      sigma: 0.005
- filename:
  - MgO.cif
  method:
  - xc: PBE0
    supercell: [2,2,2]
  - xc: PBE0
    supercell: [3,3,3]
  - xc: PBE0
    supercell: [4,4,4]
- filename:
  - Al2Mg3O12Si3_ICSD_80847.cif
  method:
  - xc: PBE0
    supercell: [1,1,1]
  - xc: PBE0
    supercell: [2,1,1]
''')

crystal_dir = os.path.abspath(f'{__file__}/../../crystals')
for conf in configs:
    for filename in conf['filename']:
        a, elements, coords, fractional = parse_cif(f'{crystal_dir}/{filename}')
        if fractional:
            coords = coords.dot(a)
        atoms = list(zip(elements, coords))

        for method in conf['method']:
            xc_list = method['xc']
            if isinstance(xc_list, str):
                xc_list = [xc_list]
            for xc in xc_list:
                xct = xc_type(xc)
                basis = {k: v.format(xct) for k, v in basis_tpl.items()}
                pseudo = {k: f"GTH-{xct}-{v.split('-')[-1]}" for k, v in basis_tpl.items()}
                kmesh_list = method.get('kmesh', [1,1,1])
                if isinstance(kmesh_list[0], int):
                    kmesh_list = [kmesh_list]
                for kmesh in kmesh_list:
                    Nk = np.prod(kmesh)
                    kstring = 'x'.join([str(x) for x in kmesh])
                    confstr = f'{xc}-k{kstring}'
                    if 'supercell' in method:
                        ncopy = method['supercell']
                        scstring = 'x'.join([str(x) for x in ncopy])
                        confstr = f'{scstring}-{confstr}'
                        cell = pyscf.M(atom=atoms, a=a, basis=basis, pseudo=pseudo,
                                       precision=1e-7,
                                       output=f'{filename}-{confstr}-GDF.out')
                        cell = pbctools.super_cell(cell, ncopy)
                    else:
                        cell = pyscf.M(atom=atoms, a=a, basis=basis, pseudo=pseudo,
                                       precision=1e-7,
                                       output=f'{filename}-{confstr}-GDF.out')
                    cell.verbose = 5

                    if Nk == 1:
                        mf = cell.RKS(xc=xc).to_gpu()
                    else:
                        kpts = cell.make_kpts(kmesh)
                        mf = cell.KRKS(xc=xc, kpts=kpts).to_gpu()
                    mf = mf.multigrid_numint()
                    mf = mf.density_fit()
                    mf.max_cycle = 20
                    mf.conv_tol = 1e-6
                    if 'smearing' in method:
                        mf = mf.smearing(**method['smearing'])
                    try:
                        mf.run()
                    except Exception as e:
                        import traceback
                        traceback.print_stack()
                        traceback.print_exception(e)
                    cupy.get_default_memory_pool().free_all_blocks()
