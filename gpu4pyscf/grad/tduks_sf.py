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
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ucphf
from pyscf.dft import numint
from pyscf.dft import numint2c
from pyscf.grad import rks as rks_grad
from pyscf.grad import tdrhf as tdrhf_grad
from pyscf.tdscf._uhf_resp_sf import cache_xc_kernel_sf

def grad_elec(td_grad, x_y, atmlst=None, max_memory=2000, verbose=logger.INFO):
    ''' Spin flip TDDFT gradient in UKS framework. Note: This function supports
    both TDA or TDDFT results.

    Parameters
    ----------
    Args:
        td_grad : sftda.TDA_SF object.

    Returns:
        The gradient of excited states: Ei^{\\xi} = E0^{\\xi} + wi^{\\xi}
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    mf = td_grad.base._scf

    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidxa = np.where(mo_occ[0]>0)[0]
    occidxb = np.where(mo_occ[1]>0)[0]
    viridxa = np.where(mo_occ[0]==0)[0]
    viridxb = np.where(mo_occ[1]==0)[0]
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

    if td_grad.base.extype==0 or 1:
        # x_ab, a means vira, b means occb
        (x_ab, x_ba), (y_ab, y_ba) = x_y
        xpy_ab = (x_ab + y_ab).T
        xpy_ba = (x_ba + y_ba).T
        xmy_ab = (x_ab - y_ab).T
        xmy_ba = (x_ba - y_ba).T

        dvv_a = np.einsum('ai,bi->ab', xpy_ab, xpy_ab) + np.einsum('ai,bi->ab', xmy_ab, xmy_ab) # T^{ab \alpha \beta}*2
        dvv_b = np.einsum('ai,bi->ab', xpy_ba, xpy_ba) + np.einsum('ai,bi->ab', xmy_ba, xmy_ba) # T^{ab \beta \alpha}*2
        doo_b =-np.einsum('ai,aj->ij', xpy_ab, xpy_ab) - np.einsum('ai,aj->ij', xmy_ab, xmy_ab) # T^{ij \alpha \beta}*2
        doo_a =-np.einsum('ai,aj->ij', xpy_ba, xpy_ba) - np.einsum('ai,aj->ij', xmy_ba, xmy_ba) # T^{ij \beta \alpha}*2

        dmxpy_ab = reduce(np.dot, (orbva, xpy_ab, orbob.T)) # ua ai iv -> uv -> (X+Y)_{uv \alpha \beta}
        dmxpy_ba = reduce(np.dot, (orbvb, xpy_ba, orboa.T)) # ua ai iv -> uv -> (X+Y)_{uv \beta \alpha}
        dmxmy_ab = reduce(np.dot, (orbva, xmy_ab, orbob.T)) # ua ai iv -> uv -> (X-Y)_{uv \alpha \beta}
        dmxmy_ba = reduce(np.dot, (orbvb, xmy_ba, orboa.T)) # ua ai iv -> uv -> (X-Y)_{uv \beta \alpha}

        dmzoo_a = reduce(np.dot, (orboa, doo_a, orboa.T)) # \sum_{\sigma ab} 2*Tab \sigma C_{au} C_{bu}
        dmzoo_b = reduce(np.dot, (orbob, doo_b, orbob.T)) # \sum_{\sigma ab} 2*Tij \sigma C_{iu} C_{iu}
        dmzoo_a+= reduce(np.dot, (orbva, dvv_a, orbva.T))
        dmzoo_b+= reduce(np.dot, (orbvb, dvv_b, orbvb.T))

        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        # # used by mcfun.
        # rho0, vxc, fxc = ni.cache_xc_kernel(mf.mol, mf.grids, mf.xc,
        #                                 mo_coeff, mo_occ, spin=1)

        f1vo, f1oo, vxc1, k1ao = \
                _contract_xc_kernel(td_grad, mf.xc, ((dmxpy_ab,dmxpy_ba),(dmxmy_ab,dmxmy_ba)),
                                    (dmzoo_a,dmzoo_b), True, True, max_memory)
        k1ao_xpy, k1ao_xmy = k1ao

        # f1vo, (2,2,4,nao,nao), (X+Y) and (X-Y) with fxc_sf
        # f1oo, (2,4,nao,nao), 2T with fxc_sc
        # vxc1, ao with v1^{\sigma}
        # k1ao_xpy，(2,2,4,nao,nao), (X+Y)(X+Y) and (X-Y)(X-Y) with gxc

        if abs(hyb) > 1e-10:
            dm = (dmzoo_a, dmxpy_ba+dmxpy_ab.T, dmxmy_ba-dmxmy_ab.T,
                  dmzoo_b, dmxpy_ab+dmxpy_ba.T, dmxmy_ab-dmxmy_ba.T)
            vj, vk = mf.get_jk(mol, dm, hermi=0)
            vk *= hyb
            if abs(omega) > 1e-10:
                vk += mf.get_k(mol, dm, hermi=0, omega=omega) * (alpha-hyb)
            vj = vj.reshape(2,3,nao,nao)
            vk = vk.reshape(2,3,nao,nao)

            veff0doo = vj[0,0]+vj[1,0] - vk[:,0]+ f1oo[:,0]
            veff0doo[0] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] + k1ao_xpy[1,0,0] + k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] + k1ao_xmy[1,0,0] + k1ao_xmy[1,1,0])
            veff0doo[1] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] - k1ao_xpy[1,0,0] - k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] - k1ao_xmy[1,0,0] - k1ao_xmy[1,1,0])

            wvoa = reduce(np.dot, (orbva.T, veff0doo[0], orboa)) *2
            wvob = reduce(np.dot, (orbvb.T, veff0doo[1], orbob)) *2

            veff = - vk[:,1] + f1vo[0,:,0]
            veff0mop_ba = reduce(np.dot, (mo_coeff[1].T, veff[0], mo_coeff[0]))
            veff0mop_ab = reduce(np.dot, (mo_coeff[0].T, veff[1], mo_coeff[1]))

            wvoa += np.einsum('ca,ci->ai', veff0mop_ba[noccb:,nocca:], xpy_ba) *2
            wvob += np.einsum('ca,ci->ai', veff0mop_ab[nocca:,noccb:], xpy_ab) *2

            wvoa -= np.einsum('il,al->ai', veff0mop_ab[:nocca,:noccb], xpy_ab) *2
            wvob -= np.einsum('il,al->ai', veff0mop_ba[:noccb,:nocca], xpy_ba) *2

            veff = -vk[:,2] + f1vo[1,:,0]
            veff0mom_ba = reduce(np.dot, (mo_coeff[1].T, veff[0], mo_coeff[0]))
            veff0mom_ab = reduce(np.dot, (mo_coeff[0].T, veff[1], mo_coeff[1]))

            wvoa += np.einsum('ca,ci->ai', veff0mom_ba[noccb:,nocca:], xmy_ba) *2
            wvob += np.einsum('ca,ci->ai', veff0mom_ab[nocca:,noccb:], xmy_ab) *2

            wvoa -= np.einsum('il,al->ai', veff0mom_ab[:nocca,:noccb], xmy_ab) *2
            wvob -= np.einsum('il,al->ai', veff0mom_ba[:noccb,:nocca], xmy_ba) *2

        else:
            dm = (dmzoo_a, dmxpy_ba+dmxpy_ab.T, dmxmy_ba-dmxmy_ab.T,
                  dmzoo_b, dmxpy_ab+dmxpy_ba.T, dmxmy_ab-dmxmy_ba.T)
            vj = mf.get_j(mol, dm, hermi=0).reshape(2,3,nao,nao)

            veff0doo = vj[0,0]+vj[1,0] + f1oo[:,0]
            veff0doo[0] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] + k1ao_xpy[1,0,0] + k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] + k1ao_xmy[1,0,0] + k1ao_xmy[1,1,0])
            veff0doo[1] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] - k1ao_xpy[1,0,0] - k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] - k1ao_xmy[1,0,0] - k1ao_xmy[1,1,0])

            wvoa = reduce(np.dot, (orbva.T, veff0doo[0], orboa)) *2
            wvob = reduce(np.dot, (orbvb.T, veff0doo[1], orbob)) *2

            veff = f1vo[0,:,0]
            veff0mop_ba = reduce(np.dot, (mo_coeff[1].T, veff[0], mo_coeff[0]))
            veff0mop_ab = reduce(np.dot, (mo_coeff[0].T, veff[1], mo_coeff[1]))

            wvoa += np.einsum('ca,ci->ai', veff0mop_ba[noccb:,nocca:], xpy_ba) *2
            wvob += np.einsum('ca,ci->ai', veff0mop_ab[nocca:,noccb:], xpy_ab) *2

            wvoa -= np.einsum('il,al->ai', veff0mop_ab[:nocca,:noccb], xpy_ab) *2
            wvob -= np.einsum('il,al->ai', veff0mop_ba[:noccb,:nocca], xpy_ba) *2

            veff = f1vo[1,:,0]
            veff0mom_ba = reduce(np.dot, (mo_coeff[1].T, veff[0], mo_coeff[0]))
            veff0mom_ab = reduce(np.dot, (mo_coeff[0].T, veff[1], mo_coeff[1]))

            wvoa += np.einsum('ca,ci->ai', veff0mom_ba[noccb:,nocca:], xmy_ba) *2
            wvob += np.einsum('ca,ci->ai', veff0mom_ab[nocca:,noccb:], xmy_ab) *2

            wvoa -= np.einsum('il,al->ai', veff0mom_ab[:nocca,:noccb], xmy_ab) *2
            wvob -= np.einsum('il,al->ai', veff0mom_ba[:noccb,:nocca], xmy_ba) *2

    vresp = mf.gen_response(hermi=1)

    def fvind(x):
        dm1 = np.empty((2,nao,nao))
        x_a = x[0,:nvira*nocca].reshape(nvira,nocca)
        x_b = x[0,nvira*nocca:].reshape(nvirb,noccb)
        dm_a = reduce(np.dot, (orbva, x_a, orboa.T))
        dm_b = reduce(np.dot, (orbvb, x_b, orbob.T))
        dm1[0] = (dm_a + dm_a.T).real
        dm1[1] = (dm_b + dm_b.T).real

        v1 = vresp(dm1)
        v1a = reduce(np.dot, (orbva.T, v1[0], orboa))
        v1b = reduce(np.dot, (orbvb.T, v1[1], orbob))
        return np.hstack((v1a.ravel(), v1b.ravel()))

    z1a, z1b = ucphf.solve(fvind, mo_energy, mo_occ, (wvoa,wvob),
                           max_cycle=td_grad.cphf_max_cycle,
                           tol=td_grad.cphf_conv_tol)[0]

    time1 = log.timer('Z-vector using UCPHF solver', *time0)

    z1ao = np.zeros((2,nao,nao))
    z1ao[0] += reduce(np.dot, (orbva, z1a, orboa.T))
    z1ao[1] += reduce(np.dot, (orbvb, z1b, orbob.T))

    veff = vresp((z1ao+z1ao.transpose(0,2,1))*0.5)

    im0a = np.zeros((nmoa,nmoa))
    im0b = np.zeros((nmob,nmob))

    im0a[:nocca,:nocca] = reduce(np.dot, (orboa.T, veff0doo[0]+veff[0], orboa)) *.5
    im0b[:noccb,:noccb] = reduce(np.dot, (orbob.T, veff0doo[1]+veff[1], orbob)) *.5
    im0a[:nocca,:nocca] += np.einsum('aj,ai->ij', veff0mop_ba[noccb:,:nocca], xpy_ba) *0.5
    im0b[:noccb,:noccb] += np.einsum('aj,ai->ij', veff0mop_ab[nocca:,:noccb], xpy_ab) *0.5
    im0a[:nocca,:nocca] += np.einsum('aj,ai->ij', veff0mom_ba[noccb:,:nocca], xmy_ba) *0.5
    im0b[:noccb,:noccb] += np.einsum('aj,ai->ij', veff0mom_ab[nocca:,:noccb], xmy_ab) *0.5

    im0a[nocca:,nocca:]  = np.einsum('bi,ai->ab', veff0mop_ab[nocca:,:noccb], xpy_ab) *0.5
    im0b[noccb:,noccb:]  = np.einsum('bi,ai->ab', veff0mop_ba[noccb:,:nocca], xpy_ba) *0.5
    im0a[nocca:,nocca:] += np.einsum('bi,ai->ab', veff0mom_ab[nocca:,:noccb], xmy_ab) *0.5
    im0b[noccb:,noccb:] += np.einsum('bi,ai->ab', veff0mom_ba[noccb:,:nocca], xmy_ba) *0.5

    im0a[nocca:,:nocca]  = np.einsum('il,al->ai', veff0mop_ab[:nocca,:noccb], xpy_ab)
    im0b[noccb:,:noccb]  = np.einsum('il,al->ai', veff0mop_ba[:noccb,:nocca], xpy_ba)
    im0a[nocca:,:nocca] += np.einsum('il,al->ai', veff0mom_ab[:nocca,:noccb], xmy_ab)
    im0b[noccb:,:noccb] += np.einsum('il,al->ai', veff0mom_ba[:noccb,:nocca], xmy_ba)

    zeta_a = (mo_energy[0][:,None] + mo_energy[0]) * .5
    zeta_b = (mo_energy[1][:,None] + mo_energy[1]) * .5
    zeta_a[nocca:,:nocca] = mo_energy[0][:nocca]
    zeta_b[noccb:,:noccb] = mo_energy[1][:noccb]
    zeta_a[:nocca,nocca:] = mo_energy[0][nocca:]
    zeta_b[:noccb,noccb:] = mo_energy[1][noccb:]

    dm1a = np.zeros((nmoa,nmoa))
    dm1b = np.zeros((nmob,nmob))
    dm1a[:nocca,:nocca] = doo_a * .5
    dm1b[:noccb,:noccb] = doo_b * .5
    dm1a[nocca:,nocca:] = dvv_a * .5
    dm1b[noccb:,noccb:] = dvv_b * .5

    dm1a[nocca:,:nocca] = z1a *.5
    dm1b[noccb:,:noccb] = z1b *.5

    dm1a[:nocca,:nocca] += np.eye(nocca) # for ground state
    dm1b[:noccb,:noccb] += np.eye(noccb)

    im0a = reduce(np.dot, (mo_coeff[0], im0a+zeta_a*dm1a, mo_coeff[0].T))
    im0b = reduce(np.dot, (mo_coeff[1], im0b+zeta_b*dm1b, mo_coeff[1].T))
    im0 = im0a + im0b

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = mf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)

    # -mol.intor('int1e_ipovlp', comp=3)
    s1 = mf_grad.get_ovlp(mol)

    dmz1doo_a = z1ao[0] + dmzoo_a
    dmz1doo_b = z1ao[1] + dmzoo_b
    oo0a = reduce(np.dot, (orboa, orboa.T))
    oo0b = reduce(np.dot, (orbob, orbob.T))

    as_dm1 = oo0a + oo0b + (dmz1doo_a + dmz1doo_b) * .5

    if abs(hyb) > 1e-10:
        dm = (oo0a, dmz1doo_a+dmz1doo_a.T, dmxpy_ba+dmxpy_ab.T, dmxmy_ba-dmxmy_ab.T,
              oo0b, dmz1doo_b+dmz1doo_b.T, dmxpy_ab+dmxpy_ba.T, dmxmy_ab-dmxmy_ba.T)
        vj, vk = td_grad.get_jk(mol, dm)
        vj = vj.reshape(2,4,3,nao,nao)
        vk = vk.reshape(2,4,3,nao,nao) * hyb
        vj[:,2:4] *= 0.0
        if abs(omega) > 1e-10:
            with mol.with_range_coulomb(omega):
                vk += td_grad.get_k(mol, dm).reshape(2,4,3,nao,nao) * (alpha-hyb)

        veff1 = np.zeros((2,4,3,nao,nao))
        veff1[:,:2] = vj[0,:2] + vj[1,:2] - vk[:,:2]
    else:
        dm = (oo0a, dmz1doo_a+dmz1doo_a.T, dmxpy_ba+dmxpy_ab.T,
              oo0b, dmz1doo_b+dmz1doo_b.T, dmxpy_ab+dmxpy_ba.T)
        vj = td_grad.get_j(mol, dm).reshape(2,3,3,nao,nao)
        vj[:,2] *= 0.0
        veff1 = np.zeros((2,4,3,nao,nao))
        veff1[:,:3] = vj[0] + vj[1]

    fxcz1 = _contract_xc_kernel_z(td_grad, mf.xc, z1ao, max_memory)

    veff1[:,0] += vxc1[:,1:]
    veff1[:,1] += (f1oo[:,1:] + fxcz1[:,1:])*2
    veff1[0,1] += (k1ao_xpy[0,0,1:] + k1ao_xpy[0,1,1:] + k1ao_xpy[1,0,1:] + k1ao_xpy[1,1,1:]
                  +k1ao_xmy[0,0,1:] + k1ao_xmy[0,1,1:] + k1ao_xmy[1,0,1:] + k1ao_xmy[1,1,1:])*2
    veff1[1,1] += (k1ao_xpy[0,0,1:] + k1ao_xpy[0,1,1:] - k1ao_xpy[1,0,1:] - k1ao_xpy[1,1,1:]
                  +k1ao_xmy[0,0,1:] + k1ao_xmy[0,1,1:] - k1ao_xmy[1,0,1:] - k1ao_xmy[1,1,1:])*2

    veff1[:,2] += f1vo[0,:,1:]
    veff1[:,3] += f1vo[1,:,1:]
    veff1a, veff1b = veff1
    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = np.zeros((len(atmlst),3))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        # Ground state gradients
        h1ao = hcore_deriv(ia)
        de[k] = np.einsum('xpq,pq->x', h1ao, as_dm1)
        de[k] += np.einsum('xpq,pq->x', veff1a[0,:,p0:p1], oo0a[p0:p1])
        de[k] += np.einsum('xpq,pq->x', veff1b[0,:,p0:p1], oo0b[p0:p1])
        de[k] += np.einsum('xpq,qp->x', veff1a[0,:,p0:p1], oo0a[:,p0:p1])
        de[k] += np.einsum('xpq,qp->x', veff1b[0,:,p0:p1], oo0b[:,p0:p1])

        de[k] += np.einsum('xpq,pq->x', veff1a[0,:,p0:p1], dmz1doo_a[p0:p1]) *.5
        de[k] += np.einsum('xpq,pq->x', veff1b[0,:,p0:p1], dmz1doo_b[p0:p1]) *.5
        de[k] += np.einsum('xpq,qp->x', veff1a[0,:,p0:p1], dmz1doo_a[:,p0:p1]) *.5
        de[k] += np.einsum('xpq,qp->x', veff1b[0,:,p0:p1], dmz1doo_b[:,p0:p1]) *.5

        de[k] -= np.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        de[k] -= np.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])

        de[k] += np.einsum('xij,ij->x', veff1a[1,:,p0:p1], oo0a[p0:p1]) *0.5
        de[k] += np.einsum('xij,ij->x', veff1b[1,:,p0:p1], oo0b[p0:p1]) *0.5

        de[k] += np.einsum('xij,ij->x', veff1b[2,:,p0:p1], dmxpy_ab[p0:p1,:])
        de[k] += np.einsum('xij,ij->x', veff1a[2,:,p0:p1], dmxpy_ba[p0:p1,:])
        de[k] += np.einsum('xji,ij->x', veff1b[2,:,p0:p1], dmxpy_ab[:,p0:p1])
        de[k] += np.einsum('xji,ij->x', veff1a[2,:,p0:p1], dmxpy_ba[:,p0:p1])

        de[k] += np.einsum('xij,ij->x', veff1b[3,:,p0:p1], dmxmy_ab[p0:p1,:])
        de[k] += np.einsum('xij,ij->x', veff1a[3,:,p0:p1], dmxmy_ba[p0:p1,:])
        de[k] += np.einsum('xji,ij->x', veff1b[3,:,p0:p1], dmxmy_ab[:,p0:p1])
        de[k] += np.einsum('xji,ij->x', veff1a[3,:,p0:p1], dmxmy_ba[:,p0:p1])

        if abs(hyb) > 1e-10:
            de[k] -= np.einsum('xij,ij->x', vk[1,2,:,p0:p1], dmxpy_ab[p0:p1,:])
            de[k] -= np.einsum('xij,ij->x', vk[0,2,:,p0:p1], dmxpy_ba[p0:p1,:])
            de[k] -= np.einsum('xji,ij->x', vk[0,2,:,p0:p1], dmxpy_ab[:,p0:p1])
            de[k] -= np.einsum('xji,ij->x', vk[1,2,:,p0:p1], dmxpy_ba[:,p0:p1])

            de[k] -= np.einsum('xij,ij->x', vk[1,3,:,p0:p1], dmxmy_ab[p0:p1,:])
            de[k] -= np.einsum('xij,ij->x', vk[0,3,:,p0:p1], dmxmy_ba[p0:p1,:])
            de[k] += np.einsum('xji,ij->x', vk[0,3,:,p0:p1], dmxmy_ab[:,p0:p1])
            de[k] += np.einsum('xji,ij->x', vk[1,3,:,p0:p1], dmxmy_ba[:,p0:p1])

        # de[k] += td_grad.extra_force(ia, locals())
    log.timer('TDUKS nuclear gradients', *time0)
    return de

def _contract_xc_kernel(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True,
                        with_kxc=True, max_memory=2000):
    mol = td_grad.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao = mo_coeff[0].shape[0]

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    f1vo = np.zeros((2,2,4,nao,nao))
    deriv = 2

    if dmoo is not None:
        f1oo = np.zeros((2,4,nao,nao))
    else:
        f1oo = None
    if with_vxc:
        v1ao = np.zeros((2,4,nao,nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao_xpy = np.zeros((2,2,4,nao,nao))
        k1ao_xmy = np.zeros((2,2,4,nao,nao))
        deriv = 3
    else:
        k1ao_xpy = k1ao_xmy = None

    # create a mc object to use mcfun.
    # nimc = numint2c.NumInt2C()
    # nimc.collinear = 'mcol'
    # nimc.collinear_samples=td_grad.base.collinear_samples
    collinear_samples=td_grad.base.collinear_samples
    ni = mf._numint

    # calculate the derivatives.
    fxc_sf,kxc_sf = cache_xc_kernel_sf(ni,mol,mf.grids,mf.xc,mo_coeff,mo_occ,collinear_samples,deriv=3)[2:]
    p0,p1=0,0 # the two parameters are used for counts the batch of grids.

    if xctype == 'LDA':
        def lda_sum_(vmat, ao, wv, mask):
            aow = numint._scale_ao(ao[0], wv)
            for k in range(4):
                vmat[k] += numint._dot_ao_ao(mol, ao[k], aow, mask, shls_slice, ao_loc)

        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            p0 = p1
            p1+= weight.shape[0]
            s_s = fxc_sf[...,p0:p1] * weight

            rho1_ab = ni.eval_rho(mol, ao[0], dmvo[0][0], mask, xctype)
            rho1_ba = ni.eval_rho(mol, ao[0], dmvo[0][1], mask, xctype)
            # s_s*2 because of \sigma_x \sigma_x + \sigma_y \sigma_y
            lda_sum_(f1vo[0][1], ao, (rho1_ab+rho1_ba)*s_s*2, mask)
            lda_sum_(f1vo[0][0], ao, (rho1_ba+rho1_ab)*s_s*2, mask)

            if with_kxc:
                s_s_n = kxc_sf[:,:,0][...,p0:p1] * weight
                s_s_s = kxc_sf[:,:,1][...,p0:p1] * weight
                lda_sum_(k1ao_xpy[0][0], ao, s_s_n*2*rho1_ab*(rho1_ab+rho1_ba), mask)
                lda_sum_(k1ao_xpy[0][1], ao, s_s_n*2*rho1_ba*(rho1_ba+rho1_ab), mask)
                lda_sum_(k1ao_xpy[1][0], ao, s_s_s*2*rho1_ab*(rho1_ab+rho1_ba), mask)
                lda_sum_(k1ao_xpy[1][1], ao, s_s_s*2*rho1_ba*(rho1_ba+rho1_ab), mask)

            rho1_ab = ni.eval_rho(mol, ao[0], dmvo[1][0], mask, xctype)
            rho1_ba = ni.eval_rho(mol, ao[0], dmvo[1][1], mask, xctype)

            # py attention to the order of f1vo[1][1] and f1vo[1][0]
            lda_sum_(f1vo[1][1], ao, (rho1_ab-rho1_ba)*s_s*2, mask)
            lda_sum_(f1vo[1][0], ao, (rho1_ba-rho1_ab)*s_s*2, mask)

            if with_kxc:
                # Note the "-"
                lda_sum_(k1ao_xmy[0][0], ao, s_s_n*2*rho1_ab*(rho1_ab-rho1_ba), mask)
                lda_sum_(k1ao_xmy[0][1], ao, s_s_n*2*rho1_ba*(rho1_ba-rho1_ab), mask)
                lda_sum_(k1ao_xmy[1][0], ao, s_s_s*2*rho1_ab*(rho1_ab-rho1_ba), mask)
                lda_sum_(k1ao_xmy[1][1], ao, s_s_s*2*rho1_ba*(rho1_ba-rho1_ab), mask)

            rho = (ni.eval_rho2(mol, ao[0], mo_coeff[0], mo_occ[0], mask, xctype),
                   ni.eval_rho2(mol, ao[0], mo_coeff[1], mo_occ[1], mask, xctype))
            vxc, fxc, kxc = ni.eval_xc(xc_code, rho, 1, deriv=deriv)[1:]
            u_u, u_d, d_d = fxc[0].T * weight
            if dmoo is not None:
                rho2a = ni.eval_rho(mol, ao[0], dmoo[0], mask, xctype, hermi=1)
                rho2b = ni.eval_rho(mol, ao[0], dmoo[1], mask, xctype, hermi=1)
                lda_sum_(f1oo[0], ao, u_u*rho2a+u_d*rho2b, mask)
                lda_sum_(f1oo[1], ao, u_d*rho2a+d_d*rho2b, mask)
            if with_vxc:
                vrho = vxc[0].T * weight
                lda_sum_(v1ao[0], ao, vrho[0], mask)
                lda_sum_(v1ao[1], ao, vrho[1], mask)

    elif xctype == 'GGA':
        def gga_sum_(vmat, ao, wv, mask):
            aow = numint._scale_ao(ao[:4], wv[:4])
            tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
            vmat[0] += tmp + tmp.T
            rks_grad._gga_grad_sum_(vmat[1:], mol, ao, wv, mask, ao_loc)

        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            p0 = p1
            p1+= weight.shape[0]

            rho1_ab = ni.eval_rho(mol, ao, dmvo[0][0], mask, xctype)
            rho1_ba = ni.eval_rho(mol, ao, dmvo[0][1], mask, xctype)

            wv_sf = uks_sf_gga_wv1((rho1_ab,rho1_ba),fxc_sf[...,p0:p1],weight)
            gga_sum_(f1vo[0][1], ao, wv_sf[0]+wv_sf[1], mask)
            gga_sum_(f1vo[0][0], ao, wv_sf[1]+wv_sf[0], mask)

            if with_kxc:
                gv_sf = uks_sf_gga_wv2_p((rho1_ab,rho1_ba),kxc_sf[...,p0:p1],weight)
                gga_sum_(k1ao_xpy[0][0], ao, gv_sf[0][0], mask)
                gga_sum_(k1ao_xpy[0][1], ao, gv_sf[1][0], mask)
                gga_sum_(k1ao_xpy[1][0], ao, gv_sf[0][1], mask)
                gga_sum_(k1ao_xpy[1][1], ao, gv_sf[1][1], mask)

            rho1_ab = ni.eval_rho(mol, ao, dmvo[1][0], mask, xctype)
            rho1_ba = ni.eval_rho(mol, ao, dmvo[1][1], mask, xctype)

            wv_sf = uks_sf_gga_wv1((rho1_ab,rho1_ba),fxc_sf[...,p0:p1],weight)
            gga_sum_(f1vo[1][1], ao, wv_sf[0]-wv_sf[1], mask)
            gga_sum_(f1vo[1][0], ao, wv_sf[1]-wv_sf[0], mask)

            if with_kxc:
                gv_sf = uks_sf_gga_wv2_m((rho1_ab,rho1_ba),kxc_sf[...,p0:p1],weight)
                gga_sum_(k1ao_xmy[0][0], ao, gv_sf[0][0], mask)
                gga_sum_(k1ao_xmy[0][1], ao, gv_sf[1][0], mask)
                gga_sum_(k1ao_xmy[1][0], ao, gv_sf[0][1], mask)
                gga_sum_(k1ao_xmy[1][1], ao, gv_sf[1][1], mask)

            rho = (ni.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype),
                   ni.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype))
            vxc, fxc, kxc = ni.eval_xc(xc_code, rho, 1, deriv=deriv)[1:]

            if dmoo is not None:
                rho2 = (ni.eval_rho(mol, ao, dmoo[0], mask, xctype, hermi=1),
                        ni.eval_rho(mol, ao, dmoo[1], mask, xctype, hermi=1))
                wv = numint._uks_gga_wv1(rho, rho2, vxc, fxc, weight)
                gga_sum_(f1oo[0], ao, wv[0], mask)
                gga_sum_(f1oo[1], ao, wv[1], mask)
            if with_vxc:
                wv = numint._uks_gga_wv0(rho, vxc, weight)
                gga_sum_(v1ao[0], ao, wv[0], mask)
                gga_sum_(v1ao[1], ao, wv[1], mask)

    elif xctype == 'MGGA':
        def mgga_sum_(vmat, ao, wv, mask):
            aow = numint._scale_ao(ao[:4], wv[:4])
            tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

            aow = numint._scale_ao(ao[1], wv[4], aow)
            tmp += numint._dot_ao_ao(mol, ao[1], aow, mask, shls_slice, ao_loc)
            aow = numint._scale_ao(ao[2], wv[4], aow)
            tmp += numint._dot_ao_ao(mol, ao[2], aow, mask, shls_slice, ao_loc)
            aow = numint._scale_ao(ao[3], wv[4], aow)
            tmp += numint._dot_ao_ao(mol, ao[3], aow, mask, shls_slice, ao_loc)
            vmat[0] += tmp + tmp.T

            rks_grad._gga_grad_sum_(vmat[1:], mol, ao, wv[:4], mask, ao_loc)
            rks_grad._tau_grad_dot_(vmat[1:], mol, ao, wv[4]*2, mask, ao_loc, True)

        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            p0 = p1
            p1+= weight.shape[0]
            ngrid=weight.shape[-1]

            rho1_ab_tmp = ni.eval_rho(mol, ao, dmvo[0][0], mask, xctype)
            rho1_ba_tmp = ni.eval_rho(mol, ao, dmvo[0][1], mask, xctype)
            # Padding for laplacian
            rho1_ab = np.empty((5, ngrid))
            rho1_ba = np.empty((5, ngrid))
            rho1_ab[:4] = rho1_ab_tmp[:4]
            rho1_ba[:4] = rho1_ba_tmp[:4]
            rho1_ab[4] = rho1_ab_tmp[5]
            rho1_ba[4] = rho1_ba_tmp[5]

            wv_sf = uks_sf_mgga_wv1((rho1_ab,rho1_ba), fxc_sf[...,p0:p1],weight)
            mgga_sum_(f1vo[0][1], ao, wv_sf[0]+wv_sf[1], mask)
            mgga_sum_(f1vo[0][0], ao, wv_sf[1]+wv_sf[0], mask)

            if with_kxc:
                gv_sf = uks_sf_mgga_wv2_p((rho1_ab,rho1_ba), kxc_sf[...,p0:p1], weight)
                mgga_sum_(k1ao_xpy[0][0], ao, gv_sf[0][0], mask)
                mgga_sum_(k1ao_xpy[0][1], ao, gv_sf[1][0], mask)
                mgga_sum_(k1ao_xpy[1][0], ao, gv_sf[0][1], mask)
                mgga_sum_(k1ao_xpy[1][1], ao, gv_sf[1][1], mask)

            rho1_ab_tmp = ni.eval_rho(mol, ao, dmvo[1][0], mask, xctype)
            rho1_ba_tmp = ni.eval_rho(mol, ao, dmvo[1][1], mask, xctype)
            # Padding for laplacian
            rho1_ab = np.empty((5, ngrid))
            rho1_ba = np.empty((5, ngrid))
            rho1_ab[:4] = rho1_ab_tmp[:4]
            rho1_ba[:4] = rho1_ba_tmp[:4]
            rho1_ab[4] = rho1_ab_tmp[5]
            rho1_ba[4] = rho1_ba_tmp[5]

            wv_sf = uks_sf_mgga_wv1((rho1_ab,rho1_ba), fxc_sf[...,p0:p1],weight)
            mgga_sum_(f1vo[1][1], ao, wv_sf[0]-wv_sf[1], mask)
            mgga_sum_(f1vo[1][0], ao, wv_sf[1]-wv_sf[0], mask)

            if with_kxc:
                gv_sf = uks_sf_mgga_wv2_m((rho1_ab,rho1_ba), kxc_sf[...,p0:p1], weight)
                mgga_sum_(k1ao_xmy[0][0], ao, gv_sf[0][0], mask)
                mgga_sum_(k1ao_xmy[0][1], ao, gv_sf[1][0], mask)
                mgga_sum_(k1ao_xmy[1][0], ao, gv_sf[0][1], mask)
                mgga_sum_(k1ao_xmy[1][1], ao, gv_sf[1][1], mask)

            rho = (ni.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype),
                   ni.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype))
            vxc, fxc, kxc = ni.eval_xc(xc_code, rho, 1, deriv=deriv)[1:]

            if dmoo is not None:
                rho2 = (ni.eval_rho(mol, ao, dmoo[0], mask, xctype, hermi=1),
                        ni.eval_rho(mol, ao, dmoo[1], mask, xctype, hermi=1))
                wv_tmp = numint._uks_mgga_wv1(rho, rho2, vxc, fxc, weight)
                # # Padding for laplacian
                wv = np.empty((2,5,ngrid))
                wv[0][:4] = wv_tmp[0][:4]
                wv[0][4]  = wv_tmp[0][5]
                wv[1][:4] = wv_tmp[1][:4]
                wv[1][4]  = wv_tmp[1][5]

                mgga_sum_(f1oo[0], ao, wv[0], mask)
                mgga_sum_(f1oo[1], ao, wv[1], mask)

            if with_vxc:
                wv_tmp = numint._uks_mgga_wv0(rho, vxc, weight)
                # # Padding for laplacian
                wv = np.empty((2,5,ngrid))
                wv[0][:4] = wv_tmp[0][:4]
                wv[0][4]  = wv_tmp[0][5]
                wv[1][:4] = wv_tmp[1][:4]
                wv[1][4]  = wv_tmp[1][5]

                mgga_sum_(v1ao[0], ao, wv[0], mask)
                mgga_sum_(v1ao[1], ao, wv[1], mask)

    else:
        raise NotImplementedError(f'td-uks for functional {xc_code}')

    f1vo[:,:,1:] *= -1
    if f1oo is not None: f1oo[:,1:] *= -1
    if v1ao is not None: v1ao[:,1:] *= -1
    if with_kxc:
        k1ao_xpy[:,:,1:] *= -1
        k1ao_xmy[:,:,1:] *= -1
    return f1vo, f1oo, v1ao, (k1ao_xpy,k1ao_xmy)

def uks_sf_gga_wv1(rho1, fxc_sf,weight):
    # fxc_sf with a shape (4,4,ngrid), 4 means I, \nabla_x,y,z.
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    wv_ab, wv_ba = np.empty((2,4,ngrid))
    wv_ab = np.einsum('yp,xyp->xp',  rho1_ab,fxc_sf)
    wv_ba = np.einsum('yp,xyp->xp',  rho1_ba,fxc_sf)
    # wv_ab[0] = wv_ab[0] *2 *.5 # *2 bacause of kernel, *0.5 for the (x + x.T)*0.5
    # wv_ba[0] = wv_ba[0] *2 *.5

    # Don't forget (sigma_x sigma_x + sigma_y sigma_y) needs *2 for kernel term.
    wv_ab[1:] *=2.0
    wv_ba[1:] *=2.0
    return wv_ab*weight, wv_ba*weight

def uks_sf_gga_wv2_p(rho1, kxc_sf,weight):
    # kxc_sf with a shape (4,4,2,4,ngrid), 4 means I,\nabla_x,y,z,
    # 0: n, \nabla_x,y,z n;  1: s, \nabla_x,y,z s.
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    gv_ab, gv_ba = np.empty((2,2,4,ngrid))
    # Note *2 and *0.5 like in function uks_sf_gga_wv1
    gv_ab = np.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab+rho1_ba, kxc_sf, optimize=True)
    gv_ba = np.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba+rho1_ab, kxc_sf, optimize=True)

    gv_ab[0,1:] *=2.0
    gv_ab[1,1:] *=2.0
    gv_ba[0,1:] *=2.0
    gv_ba[1,1:] *=2.0
    return gv_ab*weight, gv_ba*weight

def uks_sf_gga_wv2_m(rho1, kxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    gv_ab, gv_ba = np.empty((2,2,5,ngrid))
    # Note *2 and *0.5 like in function uks_sf_mgga_wv1
    gv_ab = np.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab-rho1_ba, kxc_sf , optimize=True)
    gv_ba = np.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba-rho1_ab, kxc_sf , optimize=True)

    gv_ab[:,1:] *=2.0
    gv_ba[:,1:] *=2.0
    return gv_ab*weight, gv_ba*weight

def uks_sf_mgga_wv1(rho1, fxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    # fxc_sf with a shape (5,5,ngrid), 5 means I, \nabla_x,y,z s, u
    # s_s, s_Ns, Ns_s, Ns_Ns, s_u, u_s, u_Ns, Ns_u, u_u
    ngrid = weight.shape[-1]
    wv_ab, wv_ba = np.empty((2,5,ngrid))
    wv_ab = np.einsum('yp,xyp->xp',  rho1_ab,fxc_sf)
    wv_ba = np.einsum('yp,xyp->xp',  rho1_ba,fxc_sf)
    # wv_ab[0] = wv_ab[0] *2 *.5 # *2 bacause of kernel, *0.5 for the (x + x.T)*0.5
    # wv_ba[0] = wv_ba[0] *2 *.5

    # Don't forget (sigma_x sigma_x + sigma_y sigma_y) needs *2 for kernel term.
    wv_ab[1:4] *=2.0
    wv_ba[1:4] *=2.0
    # *0.5 below is for tau->ao
    wv_ab[4] *= 0.5
    wv_ba[4] *= 0.5
    return wv_ab*weight, wv_ba*weight

def uks_sf_mgga_wv2_p(rho1, kxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    # kxc_sf with a shape (5,5,2,5,ngrid), 5 means s \nabla_x,y,z s, u
    # s_s    ->  0: n, \nabla_x,y,z n, tau ;  1: s, \nabla_x,y,z s, u
    # s_Ns   ->
    # Ns_s   ->
    # Ns_Ns  ->
    # s_u    ->
    # u_s    ->
    # u_Ns   ->
    # Ns_u   ->
    # u_u    ->
    ngrid = weight.shape[-1]
    gv_ab, gv_ba = np.empty((2,2,5,ngrid))
    # Note *2 and *0.5 like in function uks_sf_mgga_wv1
    gv_ab = np.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab+rho1_ba, kxc_sf, optimize=True)
    gv_ba = np.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba+rho1_ab, kxc_sf, optimize=True)

    gv_ab[:,1:4] *=2.0
    gv_ba[:,1:4] *=2.0
    gv_ab[:,4] *= 0.5
    gv_ba[:,4] *= 0.5
    return gv_ab*weight, gv_ba*weight

def uks_sf_mgga_wv2_m(rho1, kxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    gv_ab, gv_ba = np.empty((2,2,5,ngrid))
    # Note *2 and *0.5 like in function uks_sf_mgga_wv1
    gv_ab = np.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab-rho1_ba, kxc_sf , optimize=True)
    gv_ba = np.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba-rho1_ab, kxc_sf , optimize=True)

    gv_ab[:,1:4] *=2.0
    gv_ba[:,1:4] *=2.0
    gv_ab[:,4] *= 0.5
    gv_ba[:,4] *= 0.5
    return gv_ab*weight, gv_ba*weight

def _contract_xc_kernel_z(td_grad, xc_code, dmvo, max_memory=2000):
    mol = td_grad.base._scf.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao = mo_coeff[0].shape[0]

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dmvo = [(dmvo[0]+dmvo[0].T)*.5,
            (dmvo[1]+dmvo[1].T)*.5]

    f1vo = np.zeros((2,4,nao,nao))
    deriv = 2

    if xctype == 'LDA':
        def lda_sum_(vmat, ao, wv, mask):
            aow = numint._scale_ao(ao[0], wv)
            for k in range(4):
                vmat[k] += numint._dot_ao_ao(mol, ao[k], aow, mask, shls_slice, ao_loc)

        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = (ni.eval_rho2(mol, ao[0], mo_coeff[0], mo_occ[0], mask, xctype),
                   ni.eval_rho2(mol, ao[0], mo_coeff[1], mo_occ[1], mask, xctype))
            vxc, fxc = ni.eval_xc(xc_code, rho, 1, deriv=deriv)[1:3]
            u_u, u_d, d_d = fxc[0].T * weight
            rho1a = ni.eval_rho(mol, ao[0], dmvo[0], mask, xctype, hermi=1)
            rho1b = ni.eval_rho(mol, ao[0], dmvo[1], mask, xctype, hermi=1)

            lda_sum_(f1vo[0], ao, u_u*rho1a+u_d*rho1b, mask)
            lda_sum_(f1vo[1], ao, u_d*rho1a+d_d*rho1b, mask)

    elif xctype == 'GGA':
        def gga_sum_(vmat, ao, wv, mask):
            aow = numint._scale_ao(ao[:4], wv[:4])
            tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
            vmat[0] += tmp + tmp.T
            rks_grad._gga_grad_sum_(vmat[1:], mol, ao, wv, mask, ao_loc)
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = (ni.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype),
                   ni.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype))
            vxc, fxc = ni.eval_xc(xc_code, rho, 1, deriv=deriv)[1:3]

            rho1 = (ni.eval_rho(mol, ao, dmvo[0], mask, xctype, hermi=1),
                    ni.eval_rho(mol, ao, dmvo[1], mask, xctype, hermi=1))
            wv = numint._uks_gga_wv1(rho, rho1, vxc, fxc, weight)
            gga_sum_(f1vo[0], ao, wv[0], mask)
            gga_sum_(f1vo[1], ao, wv[1], mask)

    elif xctype == 'MGGA':
        def mgga_sum_(vmat, ao, wv, mask):
            aow = numint._scale_ao(ao[:4], wv[:4])
            tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

            aow = numint._scale_ao(ao[1], wv[5], aow)
            tmp += numint._dot_ao_ao(mol, ao[1], aow, mask, shls_slice, ao_loc)
            aow = numint._scale_ao(ao[2], wv[5], aow)
            tmp += numint._dot_ao_ao(mol, ao[2], aow, mask, shls_slice, ao_loc)
            aow = numint._scale_ao(ao[3], wv[5], aow)
            tmp += numint._dot_ao_ao(mol, ao[3], aow, mask, shls_slice, ao_loc)
            vmat[0] += tmp + tmp.T

            rks_grad._gga_grad_sum_(vmat[1:], mol, ao, wv[:4], mask, ao_loc)
            rks_grad._tau_grad_dot_(vmat[1:], mol, ao, wv[5]*2, mask, ao_loc, True)

        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = (ni.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype),
                   ni.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype))
            vxc, fxc, kxc = ni.eval_xc(xc_code, rho, 1, deriv=deriv)[1:]

            rho1 = (ni.eval_rho(mol, ao, dmvo[0], mask, xctype, hermi=1),
                    ni.eval_rho(mol, ao, dmvo[1], mask, xctype, hermi=1))
            wv = numint._uks_mgga_wv1(rho, rho1, vxc, fxc, weight)
            mgga_sum_(f1vo[0], ao, wv[0], mask)
            mgga_sum_(f1vo[1], ao, wv[1], mask)

            vxc = fxc = rho = rho1 = None

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'td-uks for functional {xc_code}')

    f1vo[:,1:] *= -1
    return f1vo

class Gradients(tdrhf_grad.Gradients):
    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet=None, atmlst=None):
        return grad_elec(self, xy, atmlst, self.max_memory, self.verbose)

Grad = Gradients

from pyscf import sftda
sftda.uks_sf.TDA_SF.Gradients = sftda.uks_sf.TDDFT_SF.Gradients = lib.class_as_method(Gradients)