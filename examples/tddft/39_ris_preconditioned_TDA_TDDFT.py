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
import cupy as cp
import pyscf

from gpu4pyscf import dft
from gpu4pyscf.tdscf import _krylov_tools
from pyscf.data.nist import HARTREE2EV

# this example shows how to use the TDA-rid (up to d orbitals) preconditioned TDA solver
# converged the TDA calculation in 5 iterations

mol = pyscf.M(atom='Vitamin_C.xyz', basis='def2-tzvp', verbose=3)

mf = dft.RKS(mol, xc='pbe0')

mf=mf.to_gpu()
mf.kernel()

def precoditioned_TDA(mf):
    ''' TDA-TDDFT '''
    td = mf.TDA()
    # construct the TDA matrix vector product function and the hdiag
    vind, hdiag = td.gen_vind(td._scf)

    #construct the preconditioner,  the TDA-ris matrix vector product function (and the hdiag)gi
    tda_ris = ris.TDA(mf, J_fit='spd')  
    ris_mvp, _hdiag = tda_ris.gen_vind()


    # use the nested krylov solver, instead of the default krylov solver
    _converged, energies, X = _krylov_tools.nested_krylov_solver(matrix_vector_product=vind,hdiag=hdiag, 
                                                    problem_type='eigenvalue', n_states=5,
                                                    init_mvp=ris_mvp, precond_mvp=ris_mvp)

    print('TDA energies', energies*HARTREE2EV) 

def precoditioned_TDDFT(mf):
    ''' full TDDFT '''
    td = mf.TDDFT()
    # construct the TDA matrix vector product function and the hdiag
    _vind, _hdiag = td.gen_vind(td._scf)
    def vind(X, Y):
        U = _vind(cp.hstack((X, Y)))
        A_size = X.shape[1]
        U1 = U[:, :A_size]
        U2 = -U[:, A_size:]
        return U1, U2

    #construct the preconditioner,  the TDA-ris matrix vector product function (and the hdiag)gi
    tddft_ris = ris.TDDFT(mf, J_fit='spd', verbose=4)  
    
    ris_mvp, hdiag = tddft_ris.gen_vind()


    # use the nested krylov solver, instead of the default krylov solver
    _converged, energies, X, Y = _krylov_tools.nested_ABBA_krylov_solver(matrix_vector_product=vind,hdiag=hdiag, 
                                                    problem_type='eigenvalue', n_states=5,
                                                    init_mvp=ris_mvp, precond_mvp=ris_mvp)

    print('TDDFT energies', energies*HARTREE2EV) 


precoditioned_TDA(mf)
precoditioned_TDDFT(mf)