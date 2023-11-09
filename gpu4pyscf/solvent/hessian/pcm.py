# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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
Gradient of PCM family solvent model
'''
# pylint: disable=C0103

import numpy
import cupy
from cupyx import scipy
from pyscf import lib
from pyscf import gto, df
from pyscf.grad import rhf as rhf_grad
from gpu4pyscf.solvent.pcm import PI, switch_h
from gpu4pyscf.solvent.grad.pcm import grad_switch_h, get_dF_dA, get_dD_dS, grad_kernel
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib import logger

libdft = lib.load_library('libdft')

def hess_kernel(pcmobj, dm, verbose=None):
    '''
    slow version with finite difference
    '''
    log = logger.new_logger(pcmobj, verbose)
    t1 = log.init_timer()
    mol = pcmobj.mol.copy()
    coords = mol.atom_coords(unit='Bohr')
    def pcm_grad_scanner(mol):
        pcmobj.reset(mol)
        e, v = pcmobj._get_vind(dm)
        return grad_kernel(pcmobj, dm)

    de = numpy.zeros([mol.natm, mol.natm, 3, 3])
    eps = 1e-3
    for ia in range(mol.natm):
        for ix in range(3):
            dv = numpy.zeros_like(coords)
            dv[ia,ix] = eps
            mol.set_geom_(coords + dv, unit='Bohr')
            mol.build()
            g0 = pcm_grad_scanner(mol)

            mol.set_geom_(coords - dv, unit='Bohr')
            mol.build()
            g1 = pcm_grad_scanner(mol)
            de[ia,:,ix] = (g0 - g1)/2.0/eps
    t1 = log.timer_debug1('solvent energy', *t1)
    return de

def grad_qv(pcmobj, mo_coeff, mo_occ, atmlst=None, verbose=None):
    '''
    slow version with finite difference
    '''
    log = logger.new_logger(pcmobj, verbose)
    t1 = log.init_timer()
    mol = pcmobj.mol.copy()
    if atmlst is None:
        atmlst = range(mol.natm)
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    dm = cupy.dot(mocc, mocc.T) * 2
    coords = mol.atom_coords(unit='Bohr')
    def pcm_vmat_scanner(mol):
        pcmobj.reset(mol)
        e, v = pcmobj._get_vind(dm)
        return v

    vmat = cupy.zeros([len(atmlst), 3, nao, nocc])
    eps = 1e-4
    for i0, ia in enumerate(atmlst):
        for ix in range(3):
            dv = numpy.zeros_like(coords)
            dv[ia,ix] = eps
            mol.set_geom_(coords + dv, unit='Bohr')
            mol.build()
            vmat0 = pcm_vmat_scanner(mol)

            mol.set_geom_(coords - dv, unit='Bohr')
            mol.build()
            vmat1 = pcm_vmat_scanner(mol)
            grad_vmat = (vmat0 - vmat1)/2.0/eps
            grad_vmat = contract("ij,jq->iq", grad_vmat, mocc)
            grad_vmat = contract("iq,ip->pq", grad_vmat, mo_coeff)
            vmat[i0,ix] = grad_vmat
    t1 = log.timer_debug1('computing solvent grad veff', *t1)
    return vmat

def make_hess_object(hess_method):
    '''
    return solvent hessian object
    '''
    hess_method_class = hess_method.__class__
    class WithSolventHess(hess_method_class):
        def __init__(self, hess_method):
            self.__dict__.update(hess_method.__dict__)
            self.de_solvent = None
            self.de_solute = None
            self._keys = self._keys.union(['de_solvent', 'de_solute'])

        def kernel(self, *args, dm=None, atmlst=None, **kwargs):
            dm = kwargs.pop('dm', None)
            if dm is None:
                dm = self.base.make_rdm1(ao_repr=True)
            self.de_solvent = hess_kernel(self.base.with_solvent, dm, verbose=self.verbose)
            self.de_solute = hess_method_class.kernel(self, *args, **kwargs)
            self.de = self.de_solute + self.de_solvent
            return self.de

        def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
            if atmlst is None:
                atmlst = range(self.mol.natm)
            h1ao = hess_method_class.make_h1(self, mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
            dv = grad_qv(self.base.with_solvent, mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
            for i0, ia in enumerate(atmlst):
                h1ao[i0] += dv[i0]
            return h1ao

        def _finalize(self):
            # disable _finalize. It is called in grad_method.kernel method
            # where self.de was not yet initialized.
            pass

    return WithSolventHess(hess_method)

