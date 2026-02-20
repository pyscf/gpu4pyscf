# Copyright 2024 The PySCF Developers. All Rights Reserved.
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
import ctypes
from pyscf import lib, gto
from gpu4pyscf.dispersion.dftd3 import libdftd3, _d3_p, error_check

libdftd3.dftd3_new_error.restype                   = _d3_p
libdftd3.dftd3_new_structure.restype               = _d3_p
libdftd3.dftd3_load_gcp_param.restype              = _d3_p
libdftd3.dftd3_check_error.restype                 = ctypes.c_int

# Somehow, GCP does not check the error for default method or basis
_white_list = {
    'b973c',
    'r2scan3c',
    'wb97x3c'
}

class GCP(lib.StreamObject):
    def __init__(self, mol, method=None, basis=None):
        self._gcp = None
        self._mol = None
        if method is not None and method.lower() not in _white_list:
            raise RuntimeError()

        coords = np.asarray(mol.atom_coords(), dtype=np.double, order='C')
        nuc_types = [gto.charge(mol.atom_symbol(ia))
                     for ia in range(mol.natm)]
        nuc_types = np.asarray(nuc_types, dtype=np.int32)
        self.natm = mol.natm
        if isinstance(mol, gto.Mole):
            lattice = lib.c_null_ptr()
            periodic = lib.c_null_ptr()
        else: # pbc.gto.Cell
            a = mol.lattice_vectors()
            lattice = a.ctypes
            periodic = ctypes.byref(ctypes.c_bool(True))

        err = libdftd3.dftd3_new_error()
        self._mol = libdftd3.dftd3_new_structure(
            err,
            ctypes.c_int(mol.natm),
            nuc_types.ctypes.data_as(ctypes.c_void_p),
            coords.ctypes.data_as(ctypes.c_void_p),
            lattice, periodic,
        )
        error_check(err)

        method_ptr = lib.c_null_ptr()
        basis_ptr = lib.c_null_ptr()
        if method is not None:
            method_ptr = method.encode()
        if basis is not None:
            basis_ptr = basis.encode()
        
        self._gcp = libdftd3.dftd3_load_gcp_param(
            err, 
            self._mol, 
            method_ptr, 
            basis_ptr)
        error_check(err)
        libdftd3.dftd3_delete_error(ctypes.byref(err))
        
    def __del__(self):
        err = libdftd3.dftd3_new_error()
        if self._gcp:
            libdftd3.dftd3_delete_gcp(err, ctypes.byref(self._gcp))
        if self._mol:
            libdftd3.dftd3_delete_structure(err, ctypes.byref(self._mol))
        libdftd3.dftd3_delete_error(ctypes.byref(err))

    def get_counterpoise(self, grad=False):
        res = {}
        _energy = np.array(0.0, dtype=np.double)
        _energy_str = _energy.ctypes.data_as(ctypes.c_void_p)
        if grad:
            _gradient = np.zeros((self.natm,3))
            _sigma = np.zeros((3,3))
            _gradient_str = _gradient.ctypes.data_as(ctypes.c_void_p)
            _sigma_str = _sigma.ctypes.data_as(ctypes.c_void_p)
        else:
            _gradient = None
            _sigma = None
            _gradient_str = lib.c_null_ptr()
            _sigma_str = lib.c_null_ptr()

        err = libdftd3.dftd3_new_error()
        libdftd3.dftd3_get_counterpoise(
            err,
            self._mol,
            self._gcp,
            _energy_str,
            _gradient_str,
            _sigma_str)
        error_check(err)

        res = dict(energy=_energy)
        if _gradient is not None:
            res.update(gradient=_gradient)
        if _sigma is not None:
            res.update(virial=_sigma)

        libdftd3.dftd3_delete_error(ctypes.byref(err))

        return res
