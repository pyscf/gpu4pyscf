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

from gpu4pyscf.solvent import pcm, smd

def PCM(method_or_mol, solvent_obj=None, dm=None):
    '''Initialize PCM model.

    Examples:

    >>> mf = PCM(scf.RHF(mol))
    >>> mf.kernel()
    >>> sol = PCM(mol)
    >>> mc = PCM(CASCI(mf, 6, 6), sol)
    >>> mc.kernel()
    '''
    from pyscf import gto
    from gpu4pyscf import scf

    if isinstance(method_or_mol, gto.mole.Mole):
        return pcm.PCM(method_or_mol)
    elif isinstance(method_or_mol, scf.hf.SCF):
        return pcm.pcm_for_scf(method_or_mol, solvent_obj, dm)
    else:
        raise NotImplementedError(f'PCM model does not support {method_or_mol}')

def SMD(method_or_mol, solvent_obj=None, dm=None):
    '''Initialize SMD model.

    Examples:

    >>> mf = PCM(scf.RHF(mol))
    >>> mf.kernel()
    >>> sol = PCM(mol)
    >>> mc = PCM(CASCI(mf, 6, 6), sol)
    >>> mc.kernel()
    '''
    from pyscf import gto
    from gpu4pyscf import scf

    if isinstance(method_or_mol, gto.mole.Mole):
        return smd.SMD(method_or_mol)
    elif isinstance(method_or_mol, scf.hf.SCF):
        return smd.smd_for_scf(method_or_mol, solvent_obj, dm)
    else:
        raise NotImplementedError(f'SMD model does not support {method_or_mol}')
