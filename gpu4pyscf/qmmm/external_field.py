#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

import numpy as np
import cupy as cp

import pyscf
from pyscf import lib

import gpu4pyscf
from gpu4pyscf.lib import logger

def add_external_field(scf_method, electric_field=None, origin=None):
    ''' Field and origin in AU '''
    # mol = scf_method.mol
    # if unit is None:
    #     unit = mol.unit
    return external_field_for_scf(scf_method, electric_field, origin)

def external_field_for_scf(method, electric_field=None, origin=None):
    assert (isinstance(method, gpu4pyscf.scf.hf.SCF))

    if isinstance(method, EXTF):
        method.electric_field = cp.asarray(electric_field)
        method.origin = cp.asarray(origin).get()
        return method

    cls = EXTFSCF

    return lib.set_class(cls(method, electric_field, origin), (cls, method.__class__))

class EXTF:
    __name_mixin__ = 'EXTF'

class EXTFSCF(EXTF):
    _keys = {'electric_field', 'origin'}

    def __init__(self, method, electric_field=None, origin=None):
        self.__dict__.update(method.__dict__)

        if electric_field is not None:
            electric_field = cp.asarray(electric_field)
            assert type(electric_field) is cp.ndarray
            assert electric_field.shape == (3,)
        self.electric_field = electric_field

        if origin is None:
            origin = np.zeros(3)
        else:
            origin = cp.asarray(origin).get()
            assert type(origin) is np.ndarray
            assert origin.shape == (3,)
        self.origin = origin

    def undo_external_field(self):
        obj = lib.view(self, lib.drop_class(self.__class__, EXTF))
        del obj.electric_field
        del obj.origin
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, '** Add field for %s **',
                    self.__class__.__name__)
        if self.verbose >= logger.DEBUG:
            if self.electric_field is not None:
                logger.debug(self, f'Electric field (in AU) = {self.electric_field}')
                logger.debug(self, f'Origin (in Bohr) = {self.origin}')
            else:
                logger.debug(self, 'No electric field is applied')
        return self

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        h1e = super().get_hcore(mol)

        if self.electric_field is not None:
            with mol.with_common_orig(self.origin):
                dipole_integral = cp.asarray(mol.intor('int1e_r'))
            h1e -= cp.einsum('d,dij->ij', self.electric_field, dipole_integral)

        return h1e

    def energy_nuc(self):
        nuc = super().energy_nuc()

        nuclear_charges = self.mol.atom_charges()
        nuclear_coords = self.mol.atom_coords()
        if self.electric_field is not None:
            nuclear_dipole = nuclear_charges @ (nuclear_coords - self.origin[None, :])
            nuc += float(nuclear_dipole @ self.electric_field.get())
        return nuc

    def Gradients(self):
        scf_grad = super().Gradients()
        return external_field_grad_for_scf(scf_grad)


def add_external_field_grad(scf_grad, electric_field=None, origin=None):
    assert (isinstance(scf_grad, gpu4pyscf.grad.rhf.Gradients))
    # mol = scf_grad.mol
    # if unit is None:
    #     unit = mol.unit
    gobj = external_field_grad_for_scf(scf_grad)
    gobj.base.electric_field = cp.asarray(electric_field)
    gobj.base.origin = cp.asarray(origin).get()
    return gobj

def external_field_grad_for_scf(scf_grad):
    if getattr(scf_grad.base, 'with_x2c', None):
        raise NotImplementedError('X2C with external field')

    # Avoid to initialize EXTFGrad twice
    if isinstance(scf_grad, EXTFGrad):
        return scf_grad

    assert (isinstance(scf_grad.base, gpu4pyscf.scf.hf.SCF) and
           isinstance(scf_grad.base, EXTF))

    return scf_grad.view(lib.make_class((EXTFGrad, scf_grad.__class__)))

class EXTFGrad:
    __name_mixin__ = 'EXTF'

    def __init__(self, scf_grad):
        self.__dict__.update(scf_grad.__dict__)

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, '** Add field for %s **',
                    self.__class__.__name__)
        if self.verbose >= logger.DEBUG:
            if self.base.electric_field is not None:
                logger.debug(self, f'Electric field (in AU) = {self.base.electric_field}')
                logger.debug(self, f'Origin (in Bohr) = {self.base.origin}')
            else:
                logger.debug(self, 'No electric field is applied')
        return self

    def get_hcore(self, mol=None, exclude_ecp=False):
        if mol is None:
            mol = self.mol
        dhcore = super().get_hcore(mol, exclude_ecp)

        if self.base.electric_field is not None:
            with mol.with_common_orig(self.base.origin):
                # The original order is (3 dimension of r, 3 dimension of derivative, ao (not differentiated), ao (differentiated))
                dipole_integral_derivative = mol.intor('int1e_irp').reshape(3, 3, mol.nao, mol.nao).transpose(0,1,3,2)
                dipole_integral_derivative = cp.asarray(dipole_integral_derivative)

            dhcore += cp.einsum('Edij,E->dij', dipole_integral_derivative, self.base.electric_field)
        return dhcore

    def grad_nuc(self, mol=None, atmlst=None):
        if mol is None: mol = self.mol
        g_nuc = super().grad_nuc(mol, atmlst)

        nuclear_charges = mol.atom_charges()
        # nuclear_coords = mol.atom_coords()
        if self.base.electric_field is not None:
            g_nuc += np.einsum('q,E->qE', nuclear_charges, self.base.electric_field.get())
        return g_nuc

# Inject EXTF interface wrapper to other modules
gpu4pyscf.scf.hf.SCF.EXTF = add_external_field
gpu4pyscf.grad.rhf.Gradients.EXTF = add_external_field_grad
