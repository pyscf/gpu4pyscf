# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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

import ctypes
import cupy as cp
import numpy as np

from gpu4pyscf.gto.int3c1e import VHFOpt, get_int3c1e, get_int3c1e_density_contracted, get_int3c1e_charge_contracted
from gpu4pyscf.gto.int3c1e_ip import get_int3c1e_ip, get_int3c1e_ip_contracted, get_int3c1e_ip1_charge_contracted

def intor(mol, intor, grids, charge_exponents=None, dm=None, charges=None, direct_scf_tol=1e-13, intopt=None):
    assert grids is not None

    if intopt is None:
        intopt = VHFOpt(mol)
        aosym = False if 'ip' in intor else True
        intopt.build(direct_scf_tol, aosym=aosym)
    else:
        assert isinstance(intopt, VHFOpt), \
            f"Please make sure intopt is a {VHFOpt.__module__}.{VHFOpt.__name__} object."
        assert hasattr(intopt, "density_offset"), "Please call build() function for VHFOpt object first."

    if intor == 'int1e_grids':
        assert dm is None or charges is None, \
            "Are you sure you want to contract the one electron integrals with both charge and density? " + \
            "If so, pass in density, obtain the result with n_charge and contract with the charges yourself."
        assert intopt.aosym

        if dm is None and charges is None:
            return get_int3c1e(mol, grids, charge_exponents, intopt)
        elif dm is not None:
            return get_int3c1e_density_contracted(mol, grids, charge_exponents, dm, intopt)
        elif charges is not None:
            return get_int3c1e_charge_contracted(mol, grids, charge_exponents, charges, intopt)
        else:
            raise ValueError(f"Logic error in {__file__} {__name__}")
    elif intor == 'int1e_grids_ip':
        assert not intopt.aosym

        if dm is None and charges is None:
            return get_int3c1e_ip(mol, grids, charge_exponents, intopt)
        else:
            assert dm is not None
            assert charges is not None
            return get_int3c1e_ip_contracted(mol, grids, charge_exponents, dm, charges, intopt, True, True)
    elif intor == 'int1e_grids_ip1':
        assert not intopt.aosym
        assert charges is not None
        if dm is not None:
            return get_int3c1e_ip_contracted(mol, grids, charge_exponents, dm, charges, intopt, True, False)
        else:
            return get_int3c1e_ip1_charge_contracted(mol, grids, charge_exponents, charges, intopt)
    elif intor == 'int1e_grids_ip2':
        assert not intopt.aosym
        assert dm is not None
        assert charges is not None
        return get_int3c1e_ip_contracted(mol, grids, charge_exponents, dm, charges, intopt, False, True)
    else:
        raise NotImplementedError(f"GPU intor {intor} is not implemented.")
