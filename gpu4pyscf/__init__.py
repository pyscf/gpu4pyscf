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

__version__ = '1.4.0'

import sys
from gpu4pyscf.lib import dpnp_helper
# Inject alias before any other submodules are imported
sys.modules['gpu4pyscf.lib.cupy_helper'] = dpnp_helper

#from . import cupy, lib, grad, hessian, solvent, scf, dft, tdscf
from . import cupy

