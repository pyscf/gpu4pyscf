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

import cupy as cp
import scipy.linalg
import scipy.optimize
import pyscf.scf.diis as cpu_diis
import gpu4pyscf.lib as lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, eigh, sandwich_dot, pack_tril, unpack_tril, get_avail_mem,
    asarray)

# J. Mol. Struct. 114, 31-34 (1984); DOI:10.1016/S0022-2860(84)87198-7
# PCCP, 4, 11 (2002); DOI:10.1039/B108658H
# GEDIIS, JCTC, 2, 835 (2006); DOI:10.1021/ct050275a
# C2DIIS, IJQC, 45, 31 (1993); DOI:10.1002/qua.560450106
# SCF-EDIIS, JCP 116, 8255 (2002); DOI:10.1063/1.1470195

# error vector = SDF-FDS
# error vector = F_ai ~ (S-SDS)*S^{-1}FDS = FDS - SDFDS ~ FDS-SDF in converge
class CDIIS(lib.diis.DIIS):
    incore = None

    def __init__(self, mf=None, filename=None):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = False
        self.Corth = None
        self.space = 8

    def update(self, s, d, f, *args, **kwargs):
        errvec = self._sdf_err_vec(s, d, f)
        if self.incore is None:
            mem_avail = get_avail_mem()
            self.incore = errvec.nbytes*2 * (20+self.space) < mem_avail
            if not self.incore:
                logger.debug(self, 'Large system detected. DIIS intermediates '
                             'are saved in the host memory')
        nao = self.Corth.shape[1]
        errvec = pack_tril(errvec.reshape(-1,nao,nao))
        f_tril = pack_tril(f.reshape(-1,nao,nao))
        xnew = lib.diis.DIIS.update(self, f_tril, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return unpack_tril(xnew).reshape(f.shape)

    def get_num_vec(self):
        if self.rollback:
            return self._head
        else:
            return len(self._bookkeep)

    def _sdf_err_vec(self, s, d, f):
        '''error vector = SDF - FDS'''
        if f.ndim == s.ndim+1: # UHF
            assert len(f) == 2
            if s.ndim == 2: # molecular SCF or single k-point
                if self.Corth is None:
                    self.Corth = eigh(f[0], s)[1]
                sdf = cp.empty_like(f)
                s.dot(d[0]).dot(f[0], out=sdf[0])
                s.dot(d[1]).dot(f[1], out=sdf[1])
                sdf = sandwich_dot(sdf, self.Corth)
                errvec = sdf - sdf.conj().transpose(0,2,1)
            else: # k-points
                if self.Corth is None:
                    self.Corth = cp.empty_like(s)
                    for k, (fk, sk) in enumerate(zip(f[0], s)):
                        self.Corth[k] = eigh(fk, sk)[1]
                Corth = asarray(self.Corth)
                sdf = cp.empty_like(f)
                tmp = None
                tmp = contract('Kij,Kjk->Kik', d[0], f[0], out=tmp)
                contract('Kij,Kjk->Kik', s, tmp, out=sdf[0])
                tmp = contract('Kpq,Kqj->Kpj', sdf[0], Corth, out=tmp)
                contract('Kpj,Kpi->Kij', tmp, Corth.conj(), out=sdf[0])

                tmp = contract('Kij,Kjk->Kik', d[1], f[1], out=tmp)
                contract('Kij,Kjk->Kik', s, tmp, out=sdf[1])
                tmp = contract('Kpq,Kqj->Kpj', sdf[1], Corth, out=tmp)
                contract('Kpj,Kpi->Kij', tmp, Corth.conj(), out=sdf[1])
                errvec = sdf - sdf.conj().transpose(0,1,3,2)
        else: # RHF
            assert f.ndim == s.ndim
            if f.ndim == 2: # molecular SCF or single k-point
                if self.Corth is None:
                    self.Corth = eigh(f, s)[1]
                sdf = s.dot(d).dot(f)
                sdf = sandwich_dot(sdf, self.Corth)
                errvec = sdf - sdf.conj().T
            else: # k-points
                if self.Corth is None:
                    self.Corth = cp.empty_like(s)
                    for k, (fk, sk) in enumerate(zip(f, s)):
                        self.Corth[k] = eigh(fk, sk)[1]
                sd = contract('Kij,Kjk->Kik', s, d)
                sdf = contract('Kij,Kjk->Kik', sd, f)
                Corth = asarray(self.Corth)
                sdf = contract('Kpq,Kqj->Kpj', sdf, Corth)
                sdf = contract('Kpj,Kpi->Kij', sdf, Corth.conj())
                errvec = sdf - sdf.conj().transpose(0,2,1)
        return errvec.ravel()

SCFDIIS = SCF_DIIS = DIIS = CDIIS
