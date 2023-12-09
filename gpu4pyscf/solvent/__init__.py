# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
    from pyscf import scf

    if isinstance(method_or_mol, gto.mole.Mole):
        return pcm.PCM(method_or_mol)
    elif isinstance(method_or_mol, scf.hf.SCF):
        return pcm.pcm_for_scf(method_or_mol, solvent_obj, dm)
    else:
        raise NotImplementedError('PCM model only support SCF')

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
    from pyscf import scf

    if isinstance(method_or_mol, gto.mole.Mole):
        return smd.SMD(method_or_mol)
    elif isinstance(method_or_mol, scf.hf.SCF):
        return pcm.pcm_for_scf(method_or_mol, solvent_obj, dm)
    else:
        raise NotImplementedError('SMD model only support SCF')