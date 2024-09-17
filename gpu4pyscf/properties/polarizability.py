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

import numpy as np
from gpu4pyscf.scf import cphf
import cupy
from gpu4pyscf.lib.cupy_helper import contract


def gen_vind(mf, mo_coeff, mo_occ):
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
    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)

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


def eval_polarizability(mf, unit='au'):
    """main function to calculate the polarizability

    Args:
        mf: mean field object
        unit (str, optional): the unit of the polarizability. Defaults to 'au'.

    Returns:
        polarizability (numpy.array): polarizability
    """

    polarizability = np.empty((3, 3))

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    mo_coeff = cupy.array(mo_coeff)
    mo_occ = cupy.array(mo_occ)
    mo_energy = cupy.array(mo_energy)
    fx = gen_vind(mf, mo_coeff, mo_occ)
    mocc = mo_coeff[:, mo_occ > 0]
    mvir = mo_coeff[:, mo_occ == 0]

    with mf.mol.with_common_orig((0, 0, 0)):
        h1 = mf.mol.intor('int1e_r')
        h1 = cupy.array(h1)
    for idirect in range(3):
        h1ai = -mvir.T.conj()@h1[idirect]@mocc
        mo1 = cphf.solve(fx, mo_energy, mo_occ, h1ai,  max_cycle=20, tol=1e-10)[0]
        for jdirect in range(idirect, 3):
            p10 = np.trace(mo1.conj().T@mvir.conj().T@h1[jdirect]@mocc)*2
            p01 = np.trace(mocc.conj().T@h1[jdirect]@mvir@mo1)*2
            polarizability[idirect, jdirect] = p10+p01
    polarizability[1, 0] = polarizability[0, 1]
    polarizability[2, 0] = polarizability[0, 2]
    polarizability[2, 1] = polarizability[1, 2]

    return polarizability

