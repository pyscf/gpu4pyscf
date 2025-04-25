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


def get_jk(mol, dm):
    '''J = ((-nabla i) j| kl) D_lk
    K = ((-nabla i) j| kl) D_jk
    '''
    if not isinstance(dm, np.ndarray): dm = dm.get()
    # vhfopt = _VHFOpt(mol, 'int2e_ip1').build()
    intor = mol._add_suffix('int2e_ip1')
    vj, vk = _vhf.direct_mapdm(intor,  # (nabla i,j|k,l)
                               's2kl', # ip1_sph has k>=l,
                               ('lk->s1ij', 'jk->s1il'),
                               dm, 3, # xyz, 3 components
                               mol._atm, mol._bas, mol._env)
    return -vj, -vk


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
    if yI == 0:
        yI = xI * 0.0
    yI = cp.asarray(yI).reshape(nocc, nvir).T
    xJ = cp.asarray(xJ).reshape(nocc, nvir).T
    if yJ == 0:
        yJ = xJ * 0.0
    yJ = cp.asarray(yJ).reshape(nocc, nvir).T
    RxI = reduce(cp.dot, (orbv, xI, orbo.T)) * 2.0
    RyI = reduce(cp.dot, (orbv, yI, orbo.T)) * 2.0
    RxJ = reduce(cp.dot, (orbv, xJ, orbo.T)) * 2.0
    RyJ = reduce(cp.dot, (orbv, yJ, orbo.T)) * 2.0
    DIJ_mo = 2*cp.dot(xI,xJ.T)
    DIJ_mo+= 2*cp.dot(yJ,yI.T)
    DIJ = reduce(cp.dot, (orbv, DIJ_mo, orbv.T))
    DIJ_mo = 2*cp.dot(xJ.T,xI)
    DIJ_mo+= 2*cp.dot(yI.T,yJ)
    DIJ -= reduce(cp.dot, (orbo, DIJ_mo, orbo.T))
    
    vjxI1, vkxI1 = mf.get_jk(mol, RxI, hermi=0)    #nu sigma in article
    vjxI2, vkxI2 = mf.get_jk(mol, RxI.T, hermi=0)
    vjyI1, vkyI1 = mf.get_jk(mol, RyI, hermi=0)
    vjyI2, vkyI2 = mf.get_jk(mol, RyI.T, hermi=0)
    vjxJ1, vkxJ1 = mf.get_jk(mol, RxJ, hermi=0)
    vjxJ2, vkxJ2 = mf.get_jk(mol, RxJ.T, hermi=0)
    vjyJ1, vkyJ1 = mf.get_jk(mol, RyJ, hermi=0)
    vjyJ2, vkyJ2 = mf.get_jk(mol, RyJ.T, hermi=0)
    vjDIJ, vkDIJ = mf.get_jk(mol, DIJ+DIJ.T, hermi=0)
    if not isinstance(vjxI1, cp.ndarray):
        vjxI1 = cp.asarray(vjxI1)
        vkxI1 = cp.asarray(vkxI1)
        vjxI2 = cp.asarray(vjxI2)
        vkxI2 = cp.asarray(vkxI2)
        vjyI1 = cp.asarray(vjyI1)
        vkyI1 = cp.asarray(vkyI1)
        vjyI2 = cp.asarray(vjyI2)
        vkyI2 = cp.asarray(vkyI2)
        vjxJ1 = cp.asarray(vjxJ1)
        vkxJ1 = cp.asarray(vkxJ1)
        vjxJ2 = cp.asarray(vjxJ2)
        vkxJ2 = cp.asarray(vkxJ2)
        vjyJ1 = cp.asarray(vjyJ1)
        vkyJ1 = cp.asarray(vkyJ1)
        vjyJ2 = cp.asarray(vjyJ2)
        vkyJ2 = cp.asarray(vkyJ2)
        vjDIJ = cp.asarray(vjDIJ)
        vkDIJ = cp.asarray(vkDIJ)
    veffxI1 = 2 * vjxI1 - vkxI1
    veffxI2 = 2 * vjxI2 - vkxI2
    veffyI1 = 2 * vjyI1 - vkyI1
    veffyI2 = 2 * vjyI2 - vkyI2
    veffxJ1 = 2 * vjxJ1 - vkxJ1
    veffxJ2 = 2 * vjxJ2 - vkxJ2
    veffyJ1 = 2 * vjyJ1 - vkyJ1
    veffyJ2 = 2 * vjyJ2 - vkyJ2
    veffDIJ = 2 * vjDIJ - vkDIJ
    # first term of L
    wvo = reduce(cp.dot, (orbv.T, veffxI1, orbv, xJ)) * 2.0
    wvo+= reduce(cp.dot, (orbv.T, veffyI1, orbv, yJ)) * 2.0
    wvo+= reduce(cp.dot, (orbv.T, veffxJ1, orbv, xI)) * 2.0
    wvo+= reduce(cp.dot, (orbv.T, veffyJ1, orbv, yI)) * 2.0
    wvo+= reduce(cp.dot, (orbv.T, veffxI2, orbv, yJ)) * 2.0
    wvo+= reduce(cp.dot, (orbv.T, veffyI2, orbv, xJ)) * 2.0
    wvo+= reduce(cp.dot, (orbv.T, veffxJ2, orbv, yI)) * 2.0
    wvo+= reduce(cp.dot, (orbv.T, veffyJ2, orbv, xI)) * 2.0
    # second term of L
    wvo-= reduce(cp.dot, (xJ, orbo.T, veffxI1, orbo)) * 2.0
    wvo-= reduce(cp.dot, (yJ, orbo.T, veffyI1, orbo)) * 2.0
    wvo-= reduce(cp.dot, (xI, orbo.T, veffxJ1, orbo)) * 2.0
    wvo-= reduce(cp.dot, (yI, orbo.T, veffyJ1, orbo)) * 2.0
    wvo-= reduce(cp.dot, (yJ, orbo.T, veffxI2, orbo)) * 2.0
    wvo-= reduce(cp.dot, (xJ, orbo.T, veffyI2, orbo)) * 2.0
    wvo-= reduce(cp.dot, (yI, orbo.T, veffxJ2, orbo)) * 2.0
    wvo-= reduce(cp.dot, (xI, orbo.T, veffyJ2, orbo)) * 2.0
    # forth term of L
    wvo+= reduce(cp.dot, (orbv.T, veffDIJ, orbo))
    # wvo *= 2.0

    vresp = td_nac.base._scf.gen_response(singlet=None, hermi=0)
    def fvind(x):
        dm = reduce(cp.dot, (orbv, x.reshape(nvir, nocc) * 2, orbo.T))
        v1ao = vresp(dm)
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()
    z1 = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        wvo,
        max_cycle=td_nac.cphf_max_cycle,
        tol=td_nac.cphf_conv_tol)[0]
    z1 = z1.reshape(nvir, nocc)
    z1ao = 2*reduce(cp.dot, (orbv, z1, orbo.T))

    DIJtilde = DIJ - z1ao - z1ao.T # .T is included in z1ao
    P = mf.make_rdm1()
    Ptilde = P + cp.dot(orbv, orbv.T)*2.0
    fmat = mf.get_fock()
    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))

    de1 = contract("xij,ij->xi", h1, (DIJtilde+DIJtilde.T)*0.5)
    de1_r = int3c2e.get_dh1e(mol, (DIJtilde+DIJtilde.T)*0.5)  # 1/r like terms
    if mol.has_ecp():
        de1_r  = rhf_grad.get_dh1e_ecp(mol, (DIJtilde+DIJtilde.T)*0.5)  # 1/r like terms

    Ds = reduce(cp.dot, (Ptilde, fmat, DIJtilde+DIJtilde.T))
    des1 = contract("xij,ij->xi", s1, Ds)*0.5

    Ds2 = reduce(cp.dot, (Ptilde, veffxJ2, RxI.T))
    Ds2+= reduce(cp.dot, (Ptilde, veffyJ2, RyI.T))
    Ds2+= reduce(cp.dot, (Ptilde, veffxI2, RxJ.T))
    Ds2+= reduce(cp.dot, (Ptilde, veffyI2, RyJ.T))
    Ds2+= reduce(cp.dot, (Ptilde, veffyJ1, RxI.T))
    Ds2+= reduce(cp.dot, (Ptilde, veffxJ1, RyI.T))
    Ds2+= reduce(cp.dot, (Ptilde, veffyI1, RxJ.T))
    Ds2+= reduce(cp.dot, (Ptilde, veffxI1, RyJ.T))
    des2 = contract("xij,ij->xi", s1, Ds2)*0.5

    Ds3 = reduce(cp.dot, (Ptilde, veffxJ1, RxI))
    Ds3+= reduce(cp.dot, (Ptilde, veffyJ1, RyI))
    Ds3+= reduce(cp.dot, (Ptilde, veffxI1, RxJ))
    Ds3+= reduce(cp.dot, (Ptilde, veffyI1, RyJ))
    Ds3+= reduce(cp.dot, (Ptilde, veffyJ2, RxI))
    Ds3+= reduce(cp.dot, (Ptilde, veffxJ2, RyI))
    Ds3+= reduce(cp.dot, (Ptilde, veffyI2, RxJ))
    Ds3+= reduce(cp.dot, (Ptilde, veffxI2, RyJ))
    des3 = contract("xij,ij->xi", s1, Ds3)*0.5
    
    Ds5 = reduce(cp.dot, (Ptilde, veffDIJ, P.T))
    des5 = contract("xij,ij->xi", s1, Ds5)*0.5
    
    delec = de1 - des1 - des2 - des3 + des5
    aoslices = mol.aoslice_by_atom()
    delec = cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])
    de = (delec + de1_r)/(EJ - EI)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    s1int = mol.intor('int1e_ipovlp', comp=3)
    s1int = cp.asarray(s1int)
    xx = 2 * cp.dot(xI,xJ.T)
    yy = 2 * cp.dot(yI,yJ.T)
    DS6 = reduce(cp.dot, (orbv, xx-yy, orbv.T))
    xx = 2 * cp.dot(xI.T,xJ)
    yy = 2 * cp.dot(yI.T,yJ)
    DS6+= reduce(cp.dot, (orbo, xx-yy, orbo.T))

    eri1 = mol.intor('int2e_ip1', aosym='s1', comp=3)
    eri1 = cp.asarray(eri1)
    eri1 = eri1.reshape(3,nao,nao,nao,nao)
    
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        SA = s1int.copy()
        SA[:, :p0] = 0
        SA[:, p1:] = 0
        SA -= 0.5*s1
        de[k] -= cp.einsum("xij,ji->x", SA[:, p0:p1], DS6[:, p0:p1])

        eri1a = eri1.copy()
        eri1a[:,:p0] = 0
        eri1a[:,p1:] = 0
        eri1a = eri1a + eri1a.transpose(0,2,1,3,4)
        eri1a = eri1a + eri1a.transpose(0,3,4,1,2)
        veff1 = eri1a * 2 - eri1a.transpose(0,1,4,3,2)
        de_vjk1 = cp.einsum('xuvls,ul,sv->x', veff1, RxI, RxJ)
        de_vjk1+= cp.einsum('xuvls,ul,sv->x', veff1, RyI, RyJ)
        de_vjk1+= cp.einsum('xuvls,ul,vs->x', veff1, RxI, RyJ)
        de_vjk1+= cp.einsum('xuvls,ul,vs->x', veff1, RyI, RxJ)
        de_vjk1+= cp.einsum('xuvls,ul,sv->x', veff1, DIJtilde, P)
        de[k] += de_vjk1/(EJ - EI)

    return de


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
    }

    def __init__(self, td):
        super().__init__(td)
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.base = td
        self.chkfile = td.chkfile
        self.state = (1, 2)  # of which the gradients to be computed.
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

            if state == 0:
                raise ValueError("State ID cannot be 0.")

            x_yI = self.base.xy[state[0] - 1]
            x_yJ = self.base.xy[state[1] - 1]
            E_I = self.base.e[state[0] - 1]
            E_J = self.base.e[state[1] - 1]

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

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(
                self,
                "--------- %s gradients for state %d and %d----------",
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
