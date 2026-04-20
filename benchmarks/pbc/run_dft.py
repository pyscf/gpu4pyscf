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
    'Mo': r'TZVP-MOLOPT-{}-GTH-q14',
}

def xc_type(xc):
    if libxc.is_hybrid_xc(xc):
        return 'HYB'
    return libxc.xc_type(xc)

configs = yaml.safe_load(
    '''
- filename:
    - Si_primitive.cif
  method:
    - xc: PBE
      kmesh: [5,5,5]
    - xc: PBEsol
      kmesh: [5,5,5]
    - xc: PBE0
      kmesh: [5,5,5]
      rsjk: True
    - xc: R2SCAN
      kmesh: [5,5,5]
    - xc: PBE0
      density_fit: null
      kmesh: [5,5,5]
    - xc: HSE06
      density_fit: null
      kmesh: [5,5,5]
- filename:
    - Al_primitive.cif
  method:
    - xc: PBE
      smearing:
        sigma: 0.005
      kmesh: [12,12,12]
    - xc: R2SCAN
      smearing:
        sigma: 0.005
      kmesh: [12,12,12]
- filename:
    - Al2Mg3O12Si3_ICSD_80847.cif
  method:
    - xc: PBE
- filename:
    - Al2O3_ICSD_31548.cif
    - CaTiO3.cif
    - GaAs.cif
    - MgO_primitive.cif
  method:
    - xc: PBE
      kmesh: [5,5,5]
    - xc: R2SCAN
      kmesh: [5,5,5]
    - xc: HSE06
      density_fit: null
      kmesh: [3,3,3]
- filename:
    - MgO_primitive.cif
  method:
    - xc: PBE0
      density_fit: null
      kmesh: [4,4,4]
- filename:
    - MoS2.cif
  method:
    - xc: PBE
      kmesh: [6,6,1]
      smearing:
        sigma: 0.005
    - xc: R2SCAN
      kmesh: [6,6,1]
      smearing:
        sigma: 0.005
    - xc: PBE
      kmesh: [8,8,1]
      smearing:
        sigma: 0.005
    - xc: R2SCAN
      kmesh: [8,8,1]
      smearing:
        sigma: 0.005
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
            kmesh = method.get('kmesh', [1,1,1])

            kstring = 'x'.join([str(x) for x in kmesh])
            confstr = f'{xc}-k{kstring}'
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
            if 'rsjk' in method:
                mf.rsjk = rsjk.PBCJKMatrixOpt(cell)
            if 'smearing' in method:
                mf = mf.smearing(**method['smearing'])
            try:
                mf.run()
            except Exception:
                pass
