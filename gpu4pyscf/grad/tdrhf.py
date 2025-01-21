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


from functools import reduce
import cupy as cp
import numpy as np
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e     
from gpu4pyscf.lib.cupy_helper import  contract
from gpu4pyscf.scf import cphf
from gpu4pyscf import lib as lib_gpu
from pyscf import __config__
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

def grad_elec(td_grad, x_y, singlet=True, atmlst=None,
              max_memory=2000, verbose=logger.INFO):
    '''
    Electronic part of TDA, TDHF nuclear gradients

    Args:
        td_grad : grad.tdrhf.Gradients or grad.tdrks.Gradients object.

        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ>0).sum())
    nvir = nmo - nocc
    x, y = x_y
    x = cp.asarray(x)
    y = cp.asarray(y)
    xpy = (x+y).reshape(nocc,nvir).T
    xmy = (x-y).reshape(nocc,nvir).T
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]
    dvv = cp.einsum('ai,bi->ab', xpy, xpy) + cp.einsum('ai,bi->ab', xmy, xmy) # 2 T_{ab}
    doo =-cp.einsum('ai,aj->ij', xpy, xpy) - cp.einsum('ai,aj->ij', xmy, xmy) # 2 T_{ij}
    dmxpy = reduce(cp.dot, (orbv, xpy, orbo.T)) # (X+Y) in ao basis
    dmxmy = reduce(cp.dot, (orbv, xmy, orbo.T)) # (X-Y) in ao basis
    dmzoo = reduce(cp.dot, (orbo, doo, orbo.T)) # T_{ij}*2 in ao basis
    dmzoo+= reduce(cp.dot, (orbv, dvv, orbv.T)) # T_{ij}*2 + T_{ab}*2 in ao basis

    vj, vk = mf.get_jk(mol, (dmzoo, dmxpy+dmxpy.T, dmxmy-dmxmy.T), hermi=0)
    if not isinstance(vj, cp.ndarray): vj = cp.asarray(vj)
    if not isinstance(vk, cp.ndarray): vk = cp.asarray(vk)
    veff0doo = vj[0] * 2 - vk[0] # 2 for alpha and beta
    wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
    if singlet:
        veff = vj[1] * 2 - vk[1]
    else:
        veff = -vk[1]
    veff0mop = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= cp.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy) * 2 # 2 for dm + dm.T
    wvo += cp.einsum('ac,ai->ci', veff0mop[nocc:,nocc:], xpy) * 2 
    veff = -vk[2]
    veff0mom = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= cp.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy) * 2
    wvo += cp.einsum('ac,ai->ci', veff0mom[nocc:,nocc:], xmy) * 2 

    # set singlet=None, generate function for CPHF type response kernel
    vresp = mf.gen_response(singlet=None, hermi=1)
    def fvind(x):  # For singlet, closed shell ground state
        dm = reduce(cp.dot, (orbv, x.reshape(nvir,nocc)*2, orbo.T)) # 2 for double occupancy
        v1ao = vresp(dm+dm.T) # for the upused 2
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()
    z1 = cphf.solve(fvind, mo_energy, mo_occ, wvo,
                    max_cycle=td_grad.cphf_max_cycle,
                    tol=td_grad.cphf_conv_tol)[0]
    z1 = z1.reshape(nvir,nocc)
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    z1ao = reduce(cp.dot, (orbv, z1, orbo.T))
    veff = vresp(z1ao+z1ao.T)

    im0 = cp.zeros((nmo,nmo))
    # in the following, all should be doubled, due to double occupancy
    # and 0.5 for i<=j and a<= b
    # but this is reduced.
    im0[:nocc,:nocc] = reduce(cp.dot, (orbo.T, veff0doo+veff, orbo)) # H_{ij}^+[T] + H_{ij}^+[Z] # 
    im0[:nocc,:nocc]+= cp.einsum('ak,ai->ki', veff0mop[nocc:,:nocc], xpy) # H_{ij}^+[T] + H_{ij}^+[Z] + sum_{a} (X+Y)_{aj}H_{ai}^+[(X+Y)]
    im0[:nocc,:nocc]+= cp.einsum('ak,ai->ki', veff0mom[nocc:,:nocc], xmy) # H_{ij}^+[T] + H_{ij}^+[Z] + sum_{a} (X+Y)_{aj}H_{ai}^+[(X+Y)] + sum_{a} (X-Y)_{aj}H_{ai}^-[(X-Y)]
    im0[nocc:,nocc:] = cp.einsum('ci,ai->ac', veff0mop[nocc:,:nocc], xpy) #  sum_{i} (X+Y)_{ci}H_{ai}^+[(X+Y)]
    im0[nocc:,nocc:]+= cp.einsum('ci,ai->ac', veff0mom[nocc:,:nocc], xmy) #  sum_{i} (X+Y)_{ci}H_{ai}^+[(X+Y)] + sum_{i} (X-Y)_{cj}H_{ai}^-[(X-Y)]
    im0[nocc:,:nocc] = cp.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy)*2 #  sum_{i} (X+Y)_{ki}H_{ai}^+[(X+Y)] * 2
    im0[nocc:,:nocc]+= cp.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy)*2 #  sum_{i} (X+Y)_{ki}H_{ai}^+[(X+Y)] + sum_{i} (X-Y)_{ki}H_{ai}^-[(X-Y)] * 2

    zeta = lib_gpu.cupy_helper.direct_sum('i+j->ij', mo_energy, mo_energy) * .5 
    zeta[nocc:,:nocc] = mo_energy[:nocc] 
    zeta[:nocc,nocc:] = mo_energy[nocc:] 
    dm1 = cp.zeros((nmo,nmo))
    dm1[:nocc,:nocc] = doo 
    dm1[nocc:,nocc:] = dvv 
    dm1[nocc:,:nocc] = z1
    dm1[:nocc,:nocc] += cp.eye(nocc)*2 # for ground state
    im0 = reduce(cp.dot, (mo_coeff, im0+zeta*dm1, mo_coeff.T)) 

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td_grad.base._scf.nuc_grad_method()
    s1 = mf_grad.get_ovlp(mol)

    dmz1doo = z1ao + dmzoo # P
    oo0 = reduce(cp.dot, (orbo, orbo.T)) #D

    if atmlst is None:
        atmlst = range(mol.natm)
    # offsetdic = mol.offset_nr_by_atom()
    de = cp.zeros((len(atmlst),3))
    h1 = cp.asarray(mf_grad.get_hcore(mol)) # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_ground = contract('xij,ij->xi', h1, oo0*2)
    dh_td = contract('xij,ij->xi', h1, (dmz1doo+dmz1doo.T)*0.5)
    ds = contract('xij,ij->xi', s1, (im0+im0.T)*0.5)

    dh1e_ground = int3c2e.get_dh1e(mol, oo0*2) # 1/r like terms
    if mol.has_ecp():
        dh1e_ground += rhf_grad.get_dh1e_ecp(mol, oo0*2) # 1/r like terms
    dh1e_td = int3c2e.get_dh1e(mol, (dmz1doo+dmz1doo.T)*0.5) # 1/r like terms
    if mol.has_ecp():
        dh1e_td += rhf_grad.get_dh1e_ecp(mol, (dmz1doo+dmz1doo.T)*0.5) # 1/r like terms

    dvhf_DD_DP = mf_grad.get_veff(mol, (dmz1doo+dmz1doo.T)*0.5 + oo0*2)
    dvhf_DD_DP -= mf_grad.get_veff(mol, (dmz1doo+dmz1doo.T)*0.5)
    dvhf_xpy = mf_grad.get_veff(mol, dmxpy+dmxpy.T)*2
    dvhf_xmy = mf_grad.get_veff(mol, dmxmy-dmxmy.T)*2
    vj, vk = get_jk(mol, (dmxmy-dmxmy.T)) #D, P, (X+Y), (X-Y)
    if not isinstance(vj, cp.ndarray): vj = cp.asarray(vj)
    if not isinstance(vk, cp.ndarray): vk = cp.asarray(vk)
    vj = vj.reshape(3,nao,nao)
    vk = vk.reshape(3,nao,nao)
    vhf1 = -vk
    if singlet:
        vhf1 += vj * 2
    else:
        vhf1 += vj*2
    extra_force = cp.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        extra_force[k] += mf_grad.extra_force(ia, locals())
    
    delec = 2.0*(dh_ground + dh_td - ds)
    aoslices = mol.aoslice_by_atom()
    delec= cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:,2:]])
    de = 2.0 * (dvhf_DD_DP + dvhf_xpy) + dh1e_ground + dh1e_td + delec + extra_force
    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        de[k] += cp.einsum('xij,ij->x', vhf1[:,p0:p1], dmxmy[p0:p1,:]) * 2
        de[k] -= cp.einsum('xji,ij->x', vhf1[:,p0:p1], dmxmy[:,p0:p1]) * 2

    log.timer('TDHF nuclear gradients', *time0)
    return de.get()


# def as_scanner(td_grad, state=1):
#     '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

#     The returned solver is a function. This function requires one argument
#     "mol" as input and returns energy and first order nuclear derivatives.

#     The solver will automatically use the results of last calculation as the
#     initial guess of the new calculation.  All parameters assigned in the
#     nuc-grad object and SCF object (DIIS, conv_tol, max_memory etc) are
#     automatically applied in the solver.

#     Note scanner has side effects.  It may change many underlying objects
#     (_scf, with_df, with_x2c, ...) during calculation.

#     Examples::

#     >>> from pyscf import gto, scf, tdscf, grad
#     >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
#     >>> td_grad_scanner = scf.RHF(mol).apply(tdscf.TDA).nuc_grad_method().as_scanner()
#     >>> e_tot, grad = td_grad_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
#     >>> e_tot, grad = td_grad_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
#     '''
#     from pyscf import gto
#     if isinstance(td_grad, lib.GradScanner):
#         return td_grad

#     if state == 0:
#         return td_grad.base._scf.nuc_grad_method().as_scanner()

#     logger.info(td_grad, 'Create scanner for %s', td_grad.__class__)
#     name = td_grad.__class__.__name__ + TDSCF_GradScanner.__name_mixin__
#     return lib.set_class(TDSCF_GradScanner(td_grad, state),
#                          (TDSCF_GradScanner, td_grad.__class__), name)

# class TDSCF_GradScanner(lib.GradScanner):
#     _keys = {'e_tot'}

#     def __init__(self, g, state):
#         lib.GradScanner.__init__(self, g)
#         if state is not None:
#             self.state = state

#     def __call__(self, mol_or_geom, state=None, **kwargs):
#         if isinstance(mol_or_geom, gto.MoleBase):
#             assert mol_or_geom.__class__ == gto.Mole
#             mol = mol_or_geom
#         else:
#             mol = self.mol.set_geom_(mol_or_geom, inplace=False)
#         self.reset(mol)

#         if state is None:
#             state = self.state
#         else:
#             self.state = state

#         td_scanner = self.base
#         td_scanner(mol)
# # TODO: Check root flip.  Maybe avoid the initial guess in TDHF otherwise
# # large error may be found in the excited states amplitudes
#         de = self.kernel(state=state, **kwargs)
#         e_tot = self.e_tot[state-1]
#         return e_tot, de

#     @property
#     def converged(self):
#         td_scanner = self.base
#         return all((td_scanner._scf.converged,
#                     td_scanner.converged[self.state]))


class Gradients(rhf_grad.GradientsBase):

    cphf_max_cycle = getattr(__config__, 'grad_tdrhf_Gradients_cphf_max_cycle', 20)
    cphf_conv_tol = getattr(__config__, 'grad_tdrhf_Gradients_cphf_conv_tol', 1e-8)

    _keys = {
        'cphf_max_cycle', 'cphf_conv_tol',
        'mol', 'base', 'chkfile', 'state', 'atmlst', 'de',
    }

    def __init__(self, td):
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.base = td
        self.chkfile = td.chkfile
        self.max_memory = td.max_memory
        self.state = 1  # of which the gradients to be computed.
        self.atmlst = None
        self.de = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** LR %s gradients for %s ********',
                 self.base.__class__, self.base._scf.__class__)
        log.info('cphf_conv_tol = %g', self.cphf_conv_tol)
        log.info('cphf_max_cycle = %d', self.cphf_max_cycle)
        log.info('chkfile = %s', self.chkfile)
        log.info('State ID = %d', self.state)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        log.info('\n')
        return self

    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet, atmlst=None):
        return grad_elec(self, xy, singlet, atmlst, self.max_memory, self.verbose)

    def kernel(self, xy=None, state=None, singlet=None, atmlst=None):
        '''
        Args:
            state : int
                Excited state ID.  state = 1 means the first excited state.
        '''
        if xy is None:
            if state is None:
                state = self.state
            else:
                self.state = state

            if state == 0:
                logger.warn(self, 'state=0 found in the input. '
                            'Gradients of ground state is computed.')
                return self.base._scf.nuc_grad_method().kernel(atmlst=atmlst)

            xy = self.base.xy[state-1]

        if singlet is None: singlet = self.base.singlet
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(xy, singlet, atmlst)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de

    # Calling the underlying SCF nuclear gradients because it may be modified
    # by external modules (e.g. QM/MM, solvent)
    def grad_nuc(self, mol=None, atmlst=None):
        mf_grad = self.base._scf.nuc_grad_method()
        return mf_grad.grad_nuc(mol, atmlst)

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------- %s gradients for state %d ----------',
                        self.base.__class__.__name__, self.state)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    # as_scanner = as_scanner

    to_gpu = lib.to_gpu

Grad = Gradients

from pyscf import tdscf
tdscf.rhf.TDA.Gradients = tdscf.rhf.TDHF.Gradients = lib.class_as_method(Gradients)