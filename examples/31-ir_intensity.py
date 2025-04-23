'''
Author: puzhichen.996 puzhichen.996@bytedance.com
Date: 2025-04-23 00:43:36
LastEditors: puzhichen.996 puzhichen.996@bytedance.com
LastEditTime: 2025-04-23 00:43:07
FilePath: /playground/root/gpu4pysc/gpu4pyscf/examples/31-ir_intensity.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/env python
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

###################################
#  Example of IR intensity
###################################

import pyscf
from gpu4pyscf.dft import rks
from gpu4pyscf.properties import ir

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='631g'

mol = pyscf.M(atom=atom, basis=bas, max_memory=32000)
mol.build()

mf = rks.RKS(mol, xc='b3lyp')
e_gpu = mf.kernel() # -76.3849465432042

h = mf.Hessian()
freq, intensity = ir.eval_ir_freq_intensity(mf, h)
print('------------------- IR frequncy and intensity -----------------------------')
for i in range(freq.shape[0]):
    print(f"IR frequency|intensity for {i}-th mode is {freq[i]:.4f}|{intensity[i]:.4f}")
"""
IR frequency|intensity for 0-th mode is 1613.0866|62.3982
IR frequency|intensity for 1-th mode is 3874.9540|3.7823
IR frequency|intensity for 2-th mode is 4006.0173|5.0603
"""