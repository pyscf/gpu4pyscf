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

import os
import numpy
from gpu4pyscf.lib import diis

from importlib.util import find_spec
has_dpctl = find_spec("dpctl")
if not has_dpctl:
    from gpu4pyscf.lib import cupy_helper
    from gpu4pyscf.lib import cutensor
else:
    from importlib.util import find_spec as _find_spec
    import sys as _sys, importlib as _importlib

    _mod = _importlib.import_module(".dpnp_helper", __name__)
    _sys.modules[__name__ + ".cupy_helper"] = _mod
    setattr(_sys.modules[__name__], "cupy_helper", _mod)

from gpu4pyscf.lib import utils

from pyscf import lib
lib.misc.format_sys_info = utils.format_sys_info
