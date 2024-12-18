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

import ctypes
import cupy as cp
import numpy as np

from gpu4pyscf.gto.int3c1e import VHFOpt, get_int3c1e, get_int3c1e_density_contracted, get_int3c1e_charge_contracted
from gpu4pyscf.gto.int3c1e_ip import get_int3c1e_ip, get_int3c1e_ip_contracted

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
            return get_int3c1e_ip_contracted(mol, grids, charge_exponents, dm, charges, intopt)
    else:
        raise NotImplementedError(f"GPU intor {intor} is not implemented.")
