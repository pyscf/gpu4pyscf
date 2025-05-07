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

from functools import reduce
import cupy as cp
import numpy as np
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import cphf
from pyscf import __config__
from gpu4pyscf.lib import utils
from gpu4pyscf import tdscf
from pyscf.scf import _vhf


def get_nacv(td_nac, x_yI, x_yJ, EI, EJ, singlet=True, atmlst=None, verbose=logger.INFO):
    if singlet is None:
        singlet = True
    mol = td_nac.mol
    mf = td_nac.base._scf
    mf_grad = mf.nuc_grad_method()
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    orbv = mo_coeff[:, nocc:]
    orbo = mo_coeff[:, :nocc]

    xI, yI = x_yI
    xJ, yJ = x_yJ
    xI = cp.asarray(xI).reshape(nocc, nvir).T
    if not isinstance(yI, np.ndarray):
        yI = xI * 0.0
    yI = cp.asarray(yI).reshape(nocc, nvir).T
    xJ = cp.asarray(xJ).reshape(nocc, nvir).T
    if not isinstance(yI, np.ndarray):
        yJ = xJ * 0.0
    yJ = cp.asarray(yJ).reshape(nocc, nvir).T
    LI = xI-yI

    vresp = mf.gen_response(singlet=None, hermi=1)

    def fvind(x):
        dm = reduce(cp.dot, (orbv, x.reshape(nvir, nocc) * 2, orbo.T))
        v1ao = vresp(dm + dm.T)
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()

    z1 = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        LI*2.0,
        max_cycle=td_nac.cphf_max_cycle,
        tol=td_nac.cphf_conv_tol)[0]
    z1 = z1.reshape(nvir, nocc)
    z1ao = reduce(cp.dot, (orbv, z1, orbo.T))
    z1aoS = (z1ao + z1ao.T)*0.5
    GZS = vresp(z1aoS)
    GZS_mo = reduce(cp.dot, (mo_coeff.T, GZS, mo_coeff))
    W = cp.zeros((nmo, nmo))
    W[:nocc, :nocc] = GZS_mo[:nocc, :nocc]
    zeta0 = mo_energy[nocc:, cp.newaxis]
    zeta0 = z1 * zeta0
    W[:nocc, nocc:] = GZS_mo[:nocc, nocc:] + 0.5*xI.T + 0.5*zeta0.T
    zeta1 = mo_energy[cp.newaxis, :nocc]
    zeta1 = z1 * zeta1
    W[nocc:, :nocc] = 0.5*yI + 0.5*zeta1
    W = reduce(cp.dot, (mo_coeff, W , mo_coeff.T)) * 2.0

    mf_grad = mf.nuc_grad_method()
    s1 = mf_grad.get_ovlp(mol)

    dmz1doo = z1ao
    oo0 = reduce(cp.dot, (orbo, orbo.T))

    if atmlst is None:
        atmlst = range(mol.natm)

    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_td = contract("xij,ij->xi", h1, (dmz1doo + dmz1doo.T) * 0.5)
    ds = contract("xij,ij->xi", s1, (W + W.T) * 0.5)

    dh1e_td = int3c2e.get_dh1e(mol, (dmz1doo + dmz1doo.T) * 0.5)  # 1/r like terms
    if mol.has_ecp():
        dh1e_td += rhf_grad.get_dh1e_ecp(mol, (dmz1doo + dmz1doo.T) * 0.5)  # 1/r like terms
    extra_force = cp.zeros((len(atmlst), 3))

    dvhf_all = 0
    dvhf = td_nac.get_veff(mol, (dmz1doo + dmz1doo.T) * 0.5 + oo0 * 2) 
    for k, ia in enumerate(atmlst):
        extra_force[k] += mf_grad.extra_force(ia, locals())
    dvhf_all += dvhf
    dvhf = td_nac.get_veff(mol, (dmz1doo + dmz1doo.T) * 0.5)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= mf_grad.extra_force(ia, locals())
    dvhf_all -= dvhf
    dvhf = td_nac.get_veff(mol, oo0 * 2)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= mf_grad.extra_force(ia, locals())
    dvhf_all -= dvhf

    delec = 2.0 * (dh_td - ds)
    aoslices = mol.aoslice_by_atom()
    delec = cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])
    de = 2.0 * dvhf_all + dh1e_td + delec + extra_force

    offsetdic = mol.offset_nr_by_atom()
    ds1 = s1.transpose(0,2,1)
    xIao = reduce(cp.dot, (orbo, xI.T, orbv.T)) * 2
    yIao = reduce(cp.dot, (orbv, yI, orbo.T)) * 2
    for ia in range(mol.natm):
        shl0, shl1, p0, p1 = offsetdic[ia]
        de[ia] += cp.einsum('xij,ij->x', ds1[:, :, p0:p1], xIao[:, p0:p1])
        de[ia] += cp.einsum('xij,ij->x', ds1[:, :, p0:p1], yIao[:, p0:p1])

    return -de.get() # derivetive couplings


class NAC(rhf_grad.GradientsBase):

    cphf_max_cycle = getattr(__config__, "grad_tdrhf_Gradients_cphf_max_cycle", 20)
    cphf_conv_tol = getattr(__config__, "grad_tdrhf_Gradients_cphf_conv_tol", 1e-8)

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    _keys = {
        "cphf_max_cycle",
        "cphf_conv_tol",
        "mol",
        "base",
        "chkfile",
        "state",
        "atmlst",
        "de",
        "numerical",
    }

    def __init__(self, td):
        super().__init__(td)
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.base = td
        self.chkfile = td.chkfile
        self.state = (0, 1)  # of which the gradients to be computed.
        self.atmlst = None
        self.de = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("\n")
        log.info(
            "******** LR %s gradients for %s ********",
            self.base.__class__,
            self.base._scf.__class__,
        )
        log.info("cphf_conv_tol = %g", self.cphf_conv_tol)
        log.info("cphf_max_cycle = %d", self.cphf_max_cycle)
        log.info("chkfile = %s", self.chkfile)
        log.info(f"State ID = {self.state}")
        log.info("\n")
        return self

    @lib.with_doc(get_nacv.__doc__)
    def get_nacv(self, x_yI, x_yJ, EI, EJ, singlet, atmlst=None, verbose=logger.INFO):
        return get_nacv(self, x_yI, x_yJ, EI, EJ, singlet, atmlst, verbose)

    def kernel(self, xy_I=None, xy_J=None, E_I=None, E_J=None, state=None, singlet=None, atmlst=None):
        """
        Args:
            state : int
                Excited state ID.  state = 1 means the first excited state.
        """
        if xy_I is None or xy_J is None:
            if state is None:
                state = self.state
            else:
                self.state = state

            if state[1] == 0:
                x_yJ = self.base.xy[state[1]]
                E_J = self.base.e[state[1]]
            else:
                x_yJ = self.base.xy[state[1] - 1]
                E_J = self.base.e[state[1] - 1]
            x_yI = self.base.xy[state[0] - 1]
            E_I = self.base.e[state[0] - 1]

        if singlet is None:
            singlet = self.base.singlet
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        self.de = self.get_nacv(x_yI, x_yJ, E_I, E_J, singlet, atmlst, verbose=self.verbose)
        self._finalize()
        return self.de
    
    def get_veff(self, mol=None, dm=None, j_factor=1.0, k_factor=1.0, omega=0.0, hermi=0, verbose=None):
        """
        Computes the first-order derivatives of the energy contributions from
        Veff per atom.

        NOTE: This function is incompatible to the one implemented in PySCF CPU version.
        In the CPU version, get_veff returns the first order derivatives of Veff matrix.
        """
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.base.make_rdm1()
        if omega == 0.0:
            vhfopt = self.base._scf._opt_gpu.get(None, None)
            return rhf_grad._jk_energy_per_atom(mol, dm, vhfopt, j_factor=j_factor, k_factor=k_factor, verbose=verbose)
        else:
            vhfopt = self.base._scf._opt_gpu.get(omega, None)
            with mol.with_range_coulomb(omega):
                return rhf_grad._jk_energy_per_atom(
                    mol, dm, vhfopt, j_factor=j_factor, k_factor=k_factor, verbose=verbose)


    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(
                self,
                "--------- %s nonadiabatic coupling vector for state %d and %d----------",
                self.base.__class__.__name__,
                self.state[0],
                self.state[1],
            )
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, "----------------------------------------------")

    def solvent_response(self, dm):
        return 0.0

    as_scanner = NotImplemented

    to_gpu = lib.to_gpu


Grad = NAC

tdscf.rhf.TDA.NAC = tdscf.rhf.TDHF.NAC = lib.class_as_method(NAC)
