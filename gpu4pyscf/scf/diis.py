# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# modified by Xiaojie Wu <wxj6000@gmail.com>; Zhichen Pu <hoshishin@163.com>

"""
DIIS
"""

import numpy
import cupy
import scipy.linalg
import scipy.optimize
import pyscf.scf.diis as cpu_diis
import gpu4pyscf.lib as lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract

# J. Mol. Struct. 114, 31-34 (1984); DOI:10.1016/S0022-2860(84)87198-7
# PCCP, 4, 11 (2002); DOI:10.1039/B108658H
# GEDIIS, JCTC, 2, 835 (2006); DOI:10.1021/ct050275a
# C2DIIS, IJQC, 45, 31 (1993); DOI:10.1002/qua.560450106
# SCF-EDIIS, JCP 116, 8255 (2002); DOI:10.1063/1.1470195

# error vector = SDF-FDS
# error vector = F_ai ~ (S-SDS)*S^{-1}FDS = FDS - SDFDS ~ FDS-SDF in converge
class CDIIS(lib.diis.DIIS):
    def __init__(self, mf=None, filename=None):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = False
        self.space = 8

    def update(self, s, d, f, *args, **kwargs):
        errvec = get_err_vec(s, d, f)
        xnew = lib.diis.DIIS.update(self, f, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return xnew

    def get_num_vec(self):
        if self.rollback:
            return self._head
        else:
            return len(self._bookkeep)

SCFDIIS = SCF_DIIS = DIIS = CDIIS

def get_err_vec(s, d, f):
    '''error vector = SDF - FDS'''
    if f.ndim == s.ndim+1: # UHF
        assert len(f) == 2
        if s.ndim == 2: # molecular SCF or single k-point
            sdf = cupy.stack([s.dot(d[0]).dot(f[0]),
                              s.dot(d[1]).dot(f[1])])
            errvec = sdf - sdf.conj().transpose(0,2,1)
        else: # k-points
            nkpts = len(f)
            sdf = cupy.empty_like(f)
            for k in range(nkpts):
                sdf[0,k] = s[k].dot(d[0,k]).dot(f[0,k])
                sdf[1,k] = s[k].dot(d[1,k]).dot(f[1,k])
            sdf = sdf - sdf.conj().transpose(0,1,3,2)
            df0 = contract('Kij,Kjk->Kik', d[0], f[0])
            df1 = contract('Kij,Kjk->Kik', d[1], f[1])
            sdf = cupy.stack([contract('Kij,Kjk->Kik', s, df0),
                              contract('Kij,Kjk->Kik', s, df1)])
            errvec = sdf - sdf.conj().transpose(0,1,3,2)
    else: # RHF
        assert f.ndim == s.ndim
        if f.ndim == 2: # molecular SCF or single k-point
            sdf = s.dot(d).dot(f)
            errvec = sdf - sdf.conj().T
        else: # k-points
            nkpts = len(f)
            sd = contract('Kij,Kjk->Kik', s, d)
            sdf = contract('Kij,Kjk->Kik', sd, f)
            errvec = sdf - sdf.conj().transpose(0,2,1)
    return errvec.ravel()
