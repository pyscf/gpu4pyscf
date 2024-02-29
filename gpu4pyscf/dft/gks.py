# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
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

from pyscf.dft import gks
from gpu4pyscf.dft import numint
from gpu4pyscf.dft import rks
from gpu4pyscf.scf.ghf import GHF

class GKS(gks.GKS, GHF):
    from gpu4pyscf.lib.utils import to_cpu, to_gpu, device

    def __init__(self, mol, xc='LDA,VWN'):
        raise NotImplementedError

    energy_elec = rks.RKS.energy_elec
    get_veff = NotImplemented
    nuc_grad_method = NotImplemented
    to_hf = NotImplemented
