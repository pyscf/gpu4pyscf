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

libdftd4 = lib.load_library('libdftd4')

class _d4_restype(ctypes.Structure):
    pass

_d4_p = ctypes.POINTER(_d4_restype)

libdftd4.dftd4_new_error.restype             = _d4_p
libdftd4.dftd4_new_structure.restype         = _d4_p
libdftd4.dftd4_new_d4_model.restype          = _d4_p
libdftd4.dftd4_new_d4s_model.restype         = _d4_p
libdftd4.dftd4_custom_d4_model.restype       = _d4_p
libdftd4.dftd4_custom_d4s_model.restype       = _d4_p
libdftd4.dftd4_load_rational_damping.restype = _d4_p
libdftd4.dftd4_new_rational_damping.restype  = _d4_p

def error_check(err):
    if libdftd4.dftd4_check_error(err):
        size = ctypes.c_int(2048)
        message = ctypes.create_string_buffer(2048)
        libdftd4.dftd4_get_error(err, message, ctypes.byref(size))
        raise RuntimeError(message.value.decode())

class DFTD4Dispersion(lib.StreamObject):
    def __init__(self, mol, xc, version='d4', ga=None, gc=None, wf=None, atm=False):
        xc_lc = xc.lower().encode()
        self._disp = None
        self._mol = None
        self._param = None

        log = lib.logger.new_logger(mol)
        # https://github.com/dftd4/dftd4/pull/276
        if xc_lc == 'wb97x':
            log.warn('The previous wb97x is renamed as wb97x-2008. \
                     Since pyscf-dispersion v1.4, D4 dispersion for wb97x is \
                     the replacement of vv10 in wb97x-v.  \
                     See https://github.com/dftd4/dftd4/blob/main/README.md')
        
        coords = np.asarray(mol.atom_coords(), dtype=np.double, order='C')
        charge = np.array([mol.charge], dtype=np.double)
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

        err = libdftd4.dftd4_new_error()
        self._mol = libdftd4.dftd4_new_structure(
            err,
            ctypes.c_int(mol.natm),
            nuc_types.ctypes.data_as(ctypes.c_void_p),
            coords.ctypes.data_as(ctypes.c_void_p),
            charge.ctypes.data_as(ctypes.c_void_p),
            lattice, periodic,
        )
        error_check(err)
        if version.lower() == 'd4':
            if ga is None and gc is None and wf is None:
                self._disp = libdftd4.dftd4_new_d4_model(err, self._mol)
            else:
                # Default from DFTD4 repo, https://github.com/dftd4/dftd4/blob/main/python/dftd4/interface.py#L290
                if ga is None: ga = 3.0
                if gc is None: gc = 2.0
                if wf is None: wf = 6.0
                self._disp = libdftd4.dftd4_custom_d4_model(err, self._mol,
                                                            ctypes.c_double(ga),
                                                            ctypes.c_double(gc),
                                                            ctypes.c_double(wf))
        elif version.lower() == 'd4s':
            if ga is None and gc is None:
                self._disp = libdftd4.dftd4_new_d4s_model(err, self._mol)
            else:
                if ga is None: ga = 3.0
                if gc is None: gc = 2.0
                self._disp = libdftd4.dftd4_custom_d4s_model(err, self._mol,
                                                ctypes.c_double(ga),
                                                ctypes.c_double(gc))
        else:
            raise ValueError('version must be d4 or d4s')
        error_check(err)
        self._param = libdftd4.dftd4_load_rational_damping(
            err,
            xc_lc,
            ctypes.c_bool(atm))
        error_check(err)

        libdftd4.dftd4_delete_error(ctypes.byref(err))

    def __del__(self):
        err = libdftd4.dftd4_new_error()
        if self._param:
            libdftd4.dftd4_delete_param(ctypes.byref(self._param))
        if self._mol:
            libdftd4.dftd4_delete_structure(err, ctypes.byref(self._mol))
        if self._disp:
            libdftd4.dftd4_delete_model(err, ctypes.byref(self._disp))
        libdftd4.dftd4_delete_error(ctypes.byref(err))

    def set_param(self, s8, a1, a2, s6=1.0, s9=1.0, alp=16.0):
        self._param = libdftd4.dftd4_new_rational_damping(
            ctypes.c_double(s6),
            ctypes.c_double(s8),
            ctypes.c_double(s9),
            ctypes.c_double(a1),
            ctypes.c_double(a2),
            ctypes.c_double(alp))
        return

    def get_dispersion(self, grad=False):
        res = {}
        _energy = np.array(0.0, dtype=np.double)
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

        err = libdftd4.dftd4_new_error()
        libdftd4.dftd4_get_dispersion(
            err,
            self._mol,
            self._disp,
            self._param,
            _energy.ctypes.data_as(ctypes.c_void_p),
            _gradient_str,
            _sigma_str)
        error_check(err)

        res = dict(energy=_energy)
        if _gradient is not None:
            res.update(gradient=_gradient)
        if _sigma is not None:
            res.update(virial=_sigma)

        libdftd4.dftd4_delete_error(ctypes.byref(err))

        return res
