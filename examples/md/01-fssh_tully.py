# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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


'''
This example demonstrates the FSSH simualtion with Tully model
'''

import numpy as np
import pyscf
from gpu4pyscf.md.fssh_tully import FSSH_Tully

fssh = FSSH_Tully(model='sac',mass=2000)
fssh.cur_state = 0
fssh.decoherence = False
fssh.nsteps = 300
fssh.kernel(np.array([[-10]]), np.array([[20/2000]]) , np.array([1.0,0.0]))
