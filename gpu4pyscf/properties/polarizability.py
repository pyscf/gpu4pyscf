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

import numpy as np
import cupy
from gpu4pyscf.scf import hf, cphf, _response_functions
from gpu4pyscf.lib.cupy_helper import contract

def gen_vind(mf, mo_coeff, mo_occ, with_nlc=True):
    """get the induced potential. This is the same as contract the mo1 with the kernel.

    Args:
        mf: mean field object
        mo_coeff (numpy.array): mo coefficients
        mo_occ (numpy.array): mo_coefficients

    Returns:
        fx (function): a function to calculate the induced potential with the input as the mo1.
    """
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:, mo_occ > 0]
    mvir = mo_coeff[:, mo_occ == 0]
    nocc = mocc.shape[1]
    nvir = nmo - nocc
    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1, with_nlc=with_nlc)

    def fx(mo1):
        mo1 = mo1.reshape(-1, nvir, nocc)  # * the saving pattern
        mo1_mo_real = contract('nai,ua->nui', mo1, mvir)
        dm1 = 2*contract('nui,vi->nuv', mo1_mo_real, mocc.conj())
        dm1+= dm1.transpose(0,2,1)

        v1 = vresp(dm1)  # (nset, nao, nao)
        tmp = contract('nuv,vi->nui', v1, mocc)
        v1vo = contract('nui,ua->nai', tmp, mvir.conj())

        return v1vo
    return fx


def eval_polarizability(mf, max_cycle=100, tol=1e-7, with_nlc=True):
    """main function to calculate the polarizability

    Args:
        mf: mean field object

    Returns:
        polarizability (numpy.array): polarizability in au
    """
    assert isinstance(mf, hf.RHF), "Unrestricted mf object is not supported."

    polarizability = np.empty((3, 3))

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    mo_coeff = cupy.array(mo_coeff)
    mo_occ = cupy.array(mo_occ)
    mo_energy = cupy.array(mo_energy)
    fx = gen_vind(mf, mo_coeff, mo_occ, with_nlc=with_nlc)
    mocc = mo_coeff[:, mo_occ > 0]
    mvir = mo_coeff[:, mo_occ == 0]

    with mf.mol.with_common_orig((0, 0, 0)):
        h1 = mf.mol.intor('int1e_r')
        h1 = cupy.array(h1)
    h1ai = -contract('ap,dpj->daj', mvir.T.conj(), h1 @ mocc)
    mo1 = cphf.solve(fx, mo_energy, mo_occ, h1ai, max_cycle=max_cycle, tol=tol)[0]
    for idirect in range(3):
        for jdirect in range(idirect, 3):
            p10 = np.trace(mo1[idirect].conj().T @ mvir.conj().T @ h1[jdirect] @ mocc) * 2
            p01 = np.trace(mocc.conj().T @ h1[jdirect] @ mvir @ mo1[idirect]) * 2
            polarizability[idirect, jdirect] = p10+p01
    polarizability[1, 0] = polarizability[0, 1]
    polarizability[2, 0] = polarizability[0, 2]
    polarizability[2, 1] = polarizability[1, 2]

    return polarizability

