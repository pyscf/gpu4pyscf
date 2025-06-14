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

import gpu4pyscf.tdscf.ris as ris
import pyscf

from gpu4pyscf import dft
from gpu4pyscf.tdscf import _krylov_family
from pyscf.data.nist import HARTREE2EV


mol = pyscf.M(atom='037_Fluorescein.xyz', basis='def2-svp', verbose=3)

mf = dft.RKS(mol, xc='pbe0')

mf=mf.to_gpu()
mf.kernel()

 

mftda = mf.TDA()
vind, hdiag = mftda.gen_vind(mftda._scf)

tda_ris = ris.TDA(mf)  
ris_mvp, _hdiag = tda_ris.gen_vind()

energies, X = _krylov_family.nested_krylov_solver(matrix_vector_product=vind,hdiag=hdiag, 
                                                problem_type='eigenvalue', n_states=5,
                                                init_mvp=ris_mvp, precond_mvp=ris_mvp)

print('energies', energies*HARTREE2EV) 