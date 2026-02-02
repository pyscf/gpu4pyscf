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


from gpu4pyscf.df import df
from gpu4pyscf.tdscf import uhf as tduhf
from gpu4pyscf.grad import tduhf as tduhf_grad
from gpu4pyscf.df.grad.rhf import Int3c2eOpt
from gpu4pyscf.df.grad.uhf import _jk_energy_per_atom

class Gradients(tduhf_grad.Gradients):

    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base._scf, df.df_jk._DFHF)
        assert isinstance(self.base, tduhf.TDHF) or isinstance(self.base, tduhf.TDA)

    def get_veff(self, mol=None, dm=None, j_factor=1, k_factor=1, omega=0,
                 hermi=0, verbose=None):
        if mol is None: mol = self.mol
        mf = self.base._scf
        if dm is None: dm = mf.make_rdm1()
        auxmol = mf.with_df.auxmol
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
            return _jk_energy_per_atom(
                int3c2e_opt, dm, j_factor, k_factor, hermi,
                auxbasis_response=self.auxbasis_response, verbose=verbose) * .5

Grad = Gradients
