# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

import cupy
import numpy
from pyscf import mcscf as cpu_mcscf
from pyscf.mcscf import mc1step as cpu_mc1step

from gpu4pyscf.df import df as gpu_df
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import utils
from gpu4pyscf.mcscf.casci import _CASCI


def _get_with_df(mc, auxbasis=None, with_df=None):
    if with_df is None:
        scf_df = getattr(mc._scf, 'with_df', None)
        if (scf_df is not None and
                (auxbasis is None or auxbasis == scf_df.auxbasis)):
            return scf_df

        if auxbasis is None and isinstance(mc.mol.basis, str):
            from pyscf.df.addons import predefined_auxbasis
            auxbasis = predefined_auxbasis(mc.mol, mc.mol.basis, xc='HF')
        with_df = gpu_df.DF(mc.mol, auxbasis)
        with_df.max_memory = mc.max_memory
        with_df.stdout = mc.stdout
        with_df.verbose = mc.verbose
    return with_df


class _DFCAS:
    _keys = {'with_df'}

    def reset(self, mol=None):
        if self.with_df is not getattr(self._scf, 'with_df', None):
            self.with_df.reset(mol)
        return super().reset(mol)

    def get_jk(self, mol, dm, hermi=1, with_j=True, with_k=True, omega=None):
        return self.with_df.get_jk(
            dm, hermi, with_j=with_j, with_k=with_k, omega=omega)

    def get_h2eff(self, mo_coeff=None):
        wall0 = logger.perf_counter()
        ncore = self.ncore
        nocc = ncore + self.ncas
        if mo_coeff is None:
            mo_coeff = self.mo_coeff[:, ncore:nocc]
        elif mo_coeff.shape[1] != self.ncas:
            mo_coeff = mo_coeff[:, ncore:nocc]
        eri = self.with_df.ao2mo(mo_coeff)
        out = eri.get() if isinstance(eri, cupy.ndarray) else eri
        timing = getattr(self, 'timing', None)
        if isinstance(timing, dict):
            timing['ao2mo_wall'] = (timing.get('ao2mo_wall', 0.) +
                                    logger.perf_counter() - wall0)
        return out

    def to_cpu(self):
        out = cpu_mcscf.DFCASCI(
            self._scf.to_cpu(), self.ncas, self.nelecas,
            auxbasis=self.with_df.auxbasis, ncore=self.ncore)
        return utils.to_cpu(self, out=out)


class DFCASCI(_DFCAS, _CASCI):
    def __init__(self, mf_or_mol, ncas, nelecas, auxbasis=None, ncore=None,
                 with_df=None):
        _CASCI.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.with_df = _get_with_df(self, auxbasis, with_df)


def from_cpu(mc):
    if isinstance(mc, cpu_mc1step.CASSCF):
        raise NotImplementedError('GPU DF-CASSCF is not implemented')
    out = DFCASCI(mc._scf, mc.ncas, mc.nelecas,
                  auxbasis=mc.with_df.auxbasis, ncore=mc.ncore)
    for key, value in mc.__dict__.items():
        if key not in ('_scf', 'with_df', 'fcisolver'):
            if isinstance(value, numpy.ndarray):
                value = cupy.asarray(value)
            out.__dict__[key] = value
    out.fcisolver.__dict__.update(mc.fcisolver.__dict__)
    return out
