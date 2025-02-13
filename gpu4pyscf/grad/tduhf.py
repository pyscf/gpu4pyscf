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
from pyscf import lib
from pyscf.lib import logger
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.df import int3c2e     
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import ucphf
from pyscf import __config__
from gpu4pyscf.lib import utils


def grad_elec(td_grad, x_y, atmlst=None, max_memory=2000, verbose=logger.INFO):
    '''
    Electronic part of TDA, TDHF nuclear gradients

    Args:
        td_grad : grad.tduhf.Gradients or grad.tduks.Gradients object.

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
    occidxa = cp.where(mo_occ[0]>0)[0]
    occidxb = cp.where(mo_occ[1]>0)[0]
    viridxa = cp.where(mo_occ[0]==0)[0]
    viridxb = cp.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]
    nao = mo_coeff[0].shape[0]
    nmoa = nocca + nvira
    nmob = noccb + nvirb

    (xa, xb), (ya, yb) = x_y
    xa = cp.asarray(xa)
    xb = cp.asarray(xb)
    ya = cp.asarray(ya)
    yb = cp.asarray(yb)
    xpya = (xa+ya).reshape(nocca,nvira).T
    xpyb = (xb+yb).reshape(noccb,nvirb).T
    xmya = (xa-ya).reshape(nocca,nvira).T
    xmyb = (xb-yb).reshape(noccb,nvirb).T

    dvva = cp.einsum('ai,bi->ab', xpya, xpya) + cp.einsum('ai,bi->ab', xmya, xmya)
    dvvb = cp.einsum('ai,bi->ab', xpyb, xpyb) + cp.einsum('ai,bi->ab', xmyb, xmyb)
    dooa =-cp.einsum('ai,aj->ij', xpya, xpya) - cp.einsum('ai,aj->ij', xmya, xmya)
    doob =-cp.einsum('ai,aj->ij', xpyb, xpyb) - cp.einsum('ai,aj->ij', xmyb, xmyb)
    dmxpya = reduce(cp.dot, (orbva, xpya, orboa.T))
    dmxpyb = reduce(cp.dot, (orbvb, xpyb, orbob.T))
    dmxmya = reduce(cp.dot, (orbva, xmya, orboa.T))
    dmxmyb = reduce(cp.dot, (orbvb, xmyb, orbob.T))
    dmzooa = reduce(cp.dot, (orboa, dooa, orboa.T))
    dmzoob = reduce(cp.dot, (orbob, doob, orbob.T))
    dmzooa+= reduce(cp.dot, (orbva, dvva, orbva.T))
    dmzoob+= reduce(cp.dot, (orbvb, dvvb, orbvb.T))

    vj, vk = mf.get_jk(mol, (dmzooa, dmxpya+dmxpya.T, dmxmya-dmxmya.T,
                             dmzoob, dmxpyb+dmxpyb.T, dmxmyb-dmxmyb.T), hermi=0)
    if not isinstance(vj, cp.ndarray): vj = cp.asarray(vj)
    if not isinstance(vk, cp.ndarray): vk = cp.asarray(vk)
    vj = vj.reshape(2,3,nao,nao)
    vk = vk.reshape(2,3,nao,nao)
    veff0doo = vj[0,0]+vj[1,0] - vk[:,0]
    wvoa = reduce(cp.dot, (orbva.T, veff0doo[0], orboa)) * 2
    wvob = reduce(cp.dot, (orbvb.T, veff0doo[1], orbob)) * 2
    veff = vj[0,1]+vj[1,1] - vk[:,1]
    veff0mopa = reduce(cp.dot, (mo_coeff[0].T, veff[0], mo_coeff[0]))
    veff0mopb = reduce(cp.dot, (mo_coeff[1].T, veff[1], mo_coeff[1]))
    wvoa -= cp.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], xpya) * 2
    wvob -= cp.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], xpyb) * 2
    wvoa += cp.einsum('ac,ai->ci', veff0mopa[nocca:,nocca:], xpya) * 2
    wvob += cp.einsum('ac,ai->ci', veff0mopb[noccb:,noccb:], xpyb) * 2
    veff = -vk[:,2]
    veff0moma = reduce(cp.dot, (mo_coeff[0].T, veff[0], mo_coeff[0]))
    veff0momb = reduce(cp.dot, (mo_coeff[1].T, veff[1], mo_coeff[1]))
    wvoa -= cp.einsum('ki,ai->ak', veff0moma[:nocca,:nocca], xmya) * 2
    wvob -= cp.einsum('ki,ai->ak', veff0momb[:noccb,:noccb], xmyb) * 2
    wvoa += cp.einsum('ac,ai->ci', veff0moma[nocca:,nocca:], xmya) * 2
    wvob += cp.einsum('ac,ai->ci', veff0momb[noccb:,noccb:], xmyb) * 2

    vresp = mf.gen_response(hermi=1)
    def fvind(x):
        dm1 = cp.empty((2,nao,nao))
        xa = x[0,:nvira*nocca].reshape(nvira,nocca)
        xb = x[0,nvira*nocca:].reshape(nvirb,noccb)
        dma = reduce(cp.dot, (orbva, xa, orboa.T))
        dmb = reduce(cp.dot, (orbvb, xb, orbob.T))
        dm1[0] = dma + dma.T
        dm1[1] = dmb + dmb.T
        v1 = vresp(dm1)
        v1a = reduce(cp.dot, (orbva.T, v1[0], orboa))
        v1b = reduce(cp.dot, (orbvb.T, v1[1], orbob))
        return cp.hstack((v1a.ravel(), v1b.ravel()))
    z1a, z1b = ucphf.solve(fvind, mo_energy, mo_occ, (wvoa,wvob),
                           max_cycle=td_grad.cphf_max_cycle,
                           tol=td_grad.cphf_conv_tol)[0]

    z1ao = cp.empty((2,nao,nao))
    z1ao[0] = reduce(cp.dot, (orbva, z1a, orboa.T))
    z1ao[1] = reduce(cp.dot, (orbvb, z1b, orbob.T))
    veff = vresp((z1ao+z1ao.transpose(0,2,1)) * .5)

    im0a = cp.zeros((nmoa,nmoa))
    im0b = cp.zeros((nmob,nmob))
    im0a[:nocca,:nocca] = reduce(cp.dot, (orboa.T, veff0doo[0]+veff[0], orboa)) * .5
    im0b[:noccb,:noccb] = reduce(cp.dot, (orbob.T, veff0doo[1]+veff[1], orbob)) * .5
    im0a[:nocca,:nocca]+= cp.einsum('ak,ai->ki', veff0mopa[nocca:,:nocca], xpya) * .5
    im0b[:noccb,:noccb]+= cp.einsum('ak,ai->ki', veff0mopb[noccb:,:noccb], xpyb) * .5
    im0a[:nocca,:nocca]+= cp.einsum('ak,ai->ki', veff0moma[nocca:,:nocca], xmya) * .5
    im0b[:noccb,:noccb]+= cp.einsum('ak,ai->ki', veff0momb[noccb:,:noccb], xmyb) * .5
    im0a[nocca:,nocca:] = cp.einsum('ci,ai->ac', veff0mopa[nocca:,:nocca], xpya) * .5
    im0b[noccb:,noccb:] = cp.einsum('ci,ai->ac', veff0mopb[noccb:,:noccb], xpyb) * .5
    im0a[nocca:,nocca:]+= cp.einsum('ci,ai->ac', veff0moma[nocca:,:nocca], xmya) * .5
    im0b[noccb:,noccb:]+= cp.einsum('ci,ai->ac', veff0momb[noccb:,:noccb], xmyb) * .5
    im0a[nocca:,:nocca] = cp.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], xpya)
    im0b[noccb:,:noccb] = cp.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], xpyb)
    im0a[nocca:,:nocca]+= cp.einsum('ki,ai->ak', veff0moma[:nocca,:nocca], xmya)
    im0b[noccb:,:noccb]+= cp.einsum('ki,ai->ak', veff0momb[:noccb,:noccb], xmyb)

    zeta_a = (mo_energy[0][:,None] + mo_energy[0]) * .5
    zeta_b = (mo_energy[1][:,None] + mo_energy[1]) * .5
    zeta_a[nocca:,:nocca] = mo_energy[0][:nocca]
    zeta_b[noccb:,:noccb] = mo_energy[1][:noccb]
    zeta_a[:nocca,nocca:] = mo_energy[0][nocca:]
    zeta_b[:noccb,noccb:] = mo_energy[1][noccb:]
    dm1a = cp.zeros((nmoa,nmoa))
    dm1b = cp.zeros((nmob,nmob))
    dm1a[:nocca,:nocca] = dooa * .5
    dm1b[:noccb,:noccb] = doob * .5
    dm1a[nocca:,nocca:] = dvva * .5
    dm1b[noccb:,noccb:] = dvvb * .5
    dm1a[nocca:,:nocca] = z1a * .5
    dm1b[noccb:,:noccb] = z1b * .5
    dm1a[:nocca,:nocca] += cp.eye(nocca) # for ground state
    dm1b[:noccb,:noccb] += cp.eye(noccb)
    im0a = reduce(cp.dot, (mo_coeff[0], im0a+zeta_a*dm1a, mo_coeff[0].T))
    im0b = reduce(cp.dot, (mo_coeff[1], im0b+zeta_b*dm1b, mo_coeff[1].T))
    im0 = im0a + im0b

    dmz1dooa = z1ao[0] + dmzooa
    dmz1doob = z1ao[1] + dmzoob
    oo0a = reduce(cp.dot, (orboa, orboa.T))
    oo0b = reduce(cp.dot, (orbob, orbob.T))

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td_grad.base._scf.nuc_grad_method()
    h1 = cp.asarray(mf_grad.get_hcore(mol)) # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_ground = contract('xij,ij->xi', h1, oo0a + oo0b)
    dh_td = contract('xij,ij->xi', h1, (dmz1dooa + dmz1doob) * .25 
                                        + (dmz1dooa + dmz1doob).T * .25)
    ds = contract('xij,ij->xi', s1, (im0+im0.T)*0.5)

    dh1e_ground = int3c2e.get_dh1e(mol, oo0a + oo0b) # 1/r like terms
    if mol.has_ecp():
        dh1e_ground += rhf_grad.get_dh1e_ecp(mol, oo0a + oo0b) # 1/r like terms
    dh1e_td = int3c2e.get_dh1e(mol, (dmz1dooa + dmz1doob) * .25
                               + (dmz1dooa + dmz1doob).T * .25) # 1/r like terms
    if mol.has_ecp():
        dh1e_td += rhf_grad.get_dh1e_ecp(mol, (dmz1dooa + dmz1doob) * .25 
                                                + (dmz1dooa + dmz1doob).T * .25) # 1/r like terms
    vhfopt = mf._opt_gpu.get(None, None)
    dvhf_DD_DP = rhf_grad._jk_energy_per_atom(mol, ((dmz1dooa+dmz1dooa.T)*0.25 + oo0a,
                                        (dmz1doob+dmz1doob.T)*0.25 + oo0b), vhfopt)
    dvhf_DD_DP -= rhf_grad._jk_energy_per_atom(mol, ((dmz1dooa+dmz1dooa.T)*0.25,
                                         (dmz1doob+dmz1doob.T)*0.25), vhfopt)
    dvhf_xpy = rhf_grad._jk_energy_per_atom(mol, ((dmxpya+dmxpya.T)*0.5,
                                      (dmxpyb+dmxpyb.T)*0.5), vhfopt)*2
    dvhf_xmy = rhf_grad._jk_energy_per_atom(mol, ((dmxmya-dmxmya.T)*0.5,
                                      (dmxmyb-dmxmyb.T)*0.5), vhfopt, j_factor=0.0)*2
    
    if atmlst is None:
        atmlst = range(mol.natm)
    extra_force = cp.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        extra_force[k] += mf_grad.extra_force(ia, locals())
    
    delec = 2.0*(dh_ground + dh_td - ds)
    aoslices = mol.aoslice_by_atom()
    delec = cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:,2:]])
    de = 2.0 * (dvhf_DD_DP + dvhf_xpy + dvhf_xmy) + dh1e_ground + dh1e_td + delec + extra_force
    
    log.timer('TDUHF nuclear gradients', *time0)
    return de.get()


class Gradients(tdrhf_grad.Gradients):

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet=None, atmlst=None):
        return grad_elec(self, xy, atmlst, self.max_memory, self.verbose)

Grad = Gradients

from gpu4pyscf import tdscf
tdscf.uhf.TDA.Gradients = tdscf.uhf.TDHF.Gradients = lib.class_as_method(Gradients)