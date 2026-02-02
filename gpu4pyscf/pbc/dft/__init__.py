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

'''Kohn-Sham DFT for periodic systems
'''

from .gen_grid import UniformGrids, BeckeGrids
from . import rks
from . import uks
from . import krks
from . import kuks
from . import krkspu
from . import kukspu
from .rks import KohnShamDFT

KRKS = krks.KRKS
KUKS = kuks.KUKS
KRKSpU = krkspu.KRKSpU
KUKSpU = kukspu.KUKSpU

def RKS(cell, *args, **kwargs):
    if 'kpts' in kwargs:
        return KRKS(cell, *args, **kwargs)
    if cell.spin == 0:
        return rks.RKS(cell, *args, **kwargs)
    else:
        raise NotImplementedError

def UKS(cell, *args, **kwargs):
    if 'kpts' in kwargs:
        return KUKS(cell, *args, **kwargs)
    return uks.UKS(cell, *args, **kwargs)

def KS(cell, *args, **kwargs):
    if cell.spin == 0:
        return RKS(cell, *args, **kwargs)
    else:
        return UKS(cell, *args, **kwargs)
