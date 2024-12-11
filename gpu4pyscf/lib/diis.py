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
# modified by Xiaojie Wu <wxj6000@gmail.com>

"""
DIIS
"""

import sys
import cupy
from pyscf.lib import logger
from pyscf.lib import misc
from pyscf import __config__

# TODO: should be different for GPU?
INCORE_SIZE = getattr(__config__, 'lib_diis_incore_size', 1000000000)  # 8000 MB
BLOCK_SIZE  = getattr(__config__, 'lib_diis_block_size', 20000000)  # ~ 160/320 MB

# PCCP, 4, 11 (2002); DOI:10.1039/B108658H
# GEDIIS, JCTC, 2, 835 (2006); DOI:10.1021/ct050275a
# C2DIIS, IJQC, 45, 31 (1993); DOI:10.1002/qua.560450106
# SCF-EDIIS, JCP 116, 8255 (2002); DOI:10.1063/1.1470195

class DIIS(object):
    '''Direct inversion in the iterative subspace method.

    Attributes:
        space : int
            DIIS subspace size. The maximum number of the vectors to be stored.
        min_space
            The minimal size of subspace before DIIS extrapolation.

    Functions:
        update(x, xerr=None) :
            If xerr the error vector is given, this function will push the target
            vector and error vector in the DIIS subspace, and use the error vector
            to extrapolate the vector and return the extrapolated vector.
            If xerr is None, this function will take the difference between
            the current given vector and the last given vector as the error
            vector to extrapolate the vector.

    Examples:

    >>> from pyscf import gto, scf, lib
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='ccpvdz')
    >>> mf = scf.RHF(mol)
    >>> h = mf.get_hcore()
    >>> s = mf.get_ovlp()
    >>> e, c = mf.eig(h, s)
    >>> occ = mf.get_occ(e, c)
    >>> # DIIS without error vector
    >>> adiis = lib.diis.DIIS()
    >>> for i in range(7):
    ...     dm = mf.make_rdm1(c, occ)
    ...     f = h + mf.get_veff(mol, dm)
    ...     if i > 1:
    ...         f = adiis.update(f)
    ...     e, c = mf.eig(f, s)
    ...     print('E_%d = %.12f' % (i, mf.energy_tot(dm, h, mf.get_veff(mol, dm))))
    E_0 = -1.050329433306
    E_1 = -1.098566175145
    E_2 = -1.100103795287
    E_3 = -1.100152104615
    E_4 = -1.100153706922
    E_5 = -1.100153764848
    E_6 = -1.100153764878

    >>> # Take Hartree-Fock gradients as the error vector
    >>> adiis = lib.diis.DIIS()
    >>> for i in range(7):
    ...     dm = mf.make_rdm1(c, occ)
    ...     f = h + mf.get_veff(mol, dm)
    ...     if i > 1:
    ...         f = adiis.update(f, mf.get_grad(c, occ, f))
    ...     e, c = mf.eig(f, s)
    ...     print('E_%d = %.12f' % (i, mf.energy_tot(dm, h, mf.get_veff(mol, dm))))
    E_0 = -1.050329433306
    E_1 = -1.098566175145
    E_2 = -1.100103795287
    E_3 = -1.100152104615
    E_4 = -1.100153763813
    E_5 = -1.100153764878
    E_6 = -1.100153764878
    '''
    def __init__(self, dev=None, filename=None,
                 incore=getattr(__config__, 'lib_diis_DIIS_incore', True)):
        '''
        use incore by default
        '''

        if dev is not None:
            self.verbose = dev.verbose
            self.stdout = dev.stdout
        else:
            self.verbose = logger.INFO
            self.stdout = sys.stdout
        self.space = 6
        self.min_space = 1
        self.incore = incore

##################################################
# don't modify the following private variables, they are not input options
        self.filename = filename
        self._diisfile = None
        self._buffer = {}
        self._bookkeep = [] # keep the ordering of input vectors
        self._head = 0
        self._H = None
        self._xprev = None
        self._err_vec_touched = False

    def _store(self, key, value):
        incore = value.size < INCORE_SIZE or self.incore
        if incore:
            self._buffer[key] = value

        # save the error vector if filename is given, this file can be used to
        # restore the DIIS state
        if (not incore) or isinstance(self.filename, str):
            if self._diisfile is None:
                self._diisfile = misc.H5TmpFile(self.filename, 'w')
            if key in self._diisfile:
                self._diisfile[key][:] = value
            else:
                self._diisfile[key] = value
# to avoid "Unable to find a valid file signature" error when reload the hdf5
# file from a crashed calculation
            self._diisfile.flush()

    def push_err_vec(self, xerr):
        self._err_vec_touched = True
        if self._head >= self.space:
            self._head = 0
        key = 'e%d' % self._head
        self._store(key, xerr.ravel())

    def push_vec(self, x):
        x = x.ravel()

        if len(self._bookkeep) >= self.space:
            self._bookkeep = self._bookkeep[1-self.space:]

        if self._err_vec_touched:
            self._bookkeep.append(self._head)
            key = 'x%d' % (self._head)
            self._store(key, x)
            self._head += 1

        elif self._xprev is None:
            # If push_err_vec is not called in advance, the error vector is generated
            # as the diff of the current vec and previous returned vec (._xprev)
            # So store the first trial vec as the previous returned vec
            self._xprev = x
            self._store('xprev', x)
            if 'xprev' not in self._buffer:  # not incore
                self._xprev = self._diisfile['xprev']

        else:
            if self._head >= self.space:
                self._head = 0
            self._bookkeep.append(self._head)
            ekey = 'e%d'%self._head
            xkey = 'x%d'%self._head
            self._store(xkey, x)
            if x.size < INCORE_SIZE or self.incore:
                self._store(ekey, x - cupy.asarray(self._xprev))
            else:  # not call _store to reduce memory footprint
                if ekey not in self._diisfile:
                    self._diisfile.create_dataset(ekey, (x.size,), x.dtype)
                edat = self._diisfile[ekey]
                for p0, p1 in misc.prange(0, x.size, BLOCK_SIZE):
                    edat[p0:p1] = x[p0:p1] - self._xprev[p0:p1]
                self._diisfile.flush()
            self._head += 1

    def get_err_vec(self, idx):
        if self._buffer:
            return self._buffer['e%d'%idx]
        else:
            return self._diisfile['e%d'%idx]

    def get_vec(self, idx):
        if self._buffer:
            return self._buffer['x%d'%idx]
        else:
            return self._diisfile['x%d'%idx]

    def get_num_vec(self):
        return len(self._bookkeep)

    def update(self, x, xerr=None):
        '''Extrapolate vector

        * If xerr the error vector is given, this function will push the target
        vector and error vector in the DIIS subspace, and use the error vector
        to extrapolate the vector and return the extrapolated vector.
        * If xerr is None, this function will take the difference between
        the current given vector and the last given vector as the error
        vector to extrapolate the vector.
        '''
        if xerr is not None:
            self.push_err_vec(xerr)
        self.push_vec(x)

        nd = self.get_num_vec()
        if nd < self.min_space:
            return x

        dt = cupy.array(self.get_err_vec(self._head-1), copy=False)
        if self._H is None:
            self._H = cupy.zeros((self.space+1,self.space+1), dt.dtype)
            self._H[0,1:] = self._H[1:,0] = 1
        for i in range(nd):
            tmp = 0
            dti = self.get_err_vec(i)
            for p0, p1 in misc.prange(0, dt.size, BLOCK_SIZE):
                tmp += cupy.dot(dt[p0:p1].conj(), dti[p0:p1])
            self._H[self._head,i+1] = tmp
            self._H[i+1,self._head] = tmp.conjugate()
        dt = None

        if self._xprev is None:
            xnew = self.extrapolate(nd)
        else:
            self._xprev = None # release memory first
            self._xprev = xnew = self.extrapolate(nd)

            self._store('xprev', xnew)
            if 'xprev' not in self._buffer:  # not incore
                self._xprev = self._diisfile['xprev']
        return xnew.reshape(x.shape)

    def extrapolate(self, nd=None):
        if nd is None:
            nd = self.get_num_vec()
        if nd == 0:
            raise RuntimeError('No vector found in DIIS object.')

        h = self._H[:nd+1,:nd+1]
        g = cupy.zeros(nd+1, h.dtype)
        g[0] = 1

        w, v = cupy.linalg.eigh(h)
        if cupy.any(abs(w)<1e-14):
            logger.debug(self, 'Linear dependence found in DIIS error vectors.')
            idx = abs(w)>1e-14
            c = cupy.dot(v[:,idx]*(1./w[idx]), cupy.dot(v[:,idx].T.conj(), g))
        else:
            try:
                c = cupy.linalg.solve(h, g)
            except cupy.linalg.linalg.LinAlgError as e:
                logger.warn(self, ' diis singular, eigh(h) %s', w)
                raise e
        logger.debug1(self, 'diis-c %s', c)

        xnew = None
        for i, ci in enumerate(c[1:]):
            xi = self.get_vec(i)
            if xnew is None:
                xnew = cupy.zeros(xi.size, c.dtype)
            for p0, p1 in misc.prange(0, xi.size, BLOCK_SIZE):
                xnew[p0:p1] += xi[p0:p1] * ci
        return xnew


def restore(filename):
    '''Restore/construct diis object based on a diis file'''
    return DIIS().restore(filename)
