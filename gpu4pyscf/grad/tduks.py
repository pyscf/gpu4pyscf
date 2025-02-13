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
from gpu4pyscf import lib as lib_gpu
from gpu4pyscf.lib.cupy_helper import  contract
from gpu4pyscf.lib import logger
from gpu4pyscf.df import int3c2e
from gpu4pyscf.dft import numint
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.grad import tdrks as tdrks_grad
from gpu4pyscf.scf import ucphf


#
# Given Y = 0, TDHF gradients (XAX+XBY+YBX+YAY)^1 turn to TDA gradients (XAX)^1
#
def grad_elec(td_grad, x_y, atmlst=None, verbose=logger.INFO):
    '''
    Electronic part of TDA, TDDFT nuclear gradients

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

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    # dm0 = mf.make_rdm1(mo_coeff, mo_occ), but it is not used when computing
    # fxc since rho0 is passed to fxc function.
    f1vo, f1oo, vxc1, k1ao = \
            _contract_xc_kernel(td_grad, mf.xc, cp.stack((dmxpya,dmxpyb)),
                                cp.stack((dmzooa,dmzoob)), True, True)

    if ni.libxc.is_hybrid_xc(mf.xc):
        dm = (dmzooa, dmxpya+dmxpya.T, dmxmya-dmxmya.T,
              dmzoob, dmxpyb+dmxpyb.T, dmxmyb-dmxmyb.T)
        vj, vk = mf.get_jk(mol, dm, hermi=0)
        if not isinstance(vj, cp.ndarray): vj = cp.asarray(vj)
        if not isinstance(vk, cp.ndarray): vk = cp.asarray(vk)
        vk *= hyb
        if omega != 0:
            vk += mf.get_k(mol, dm, hermi=0, omega=omega) * (alpha-hyb)
        vj = vj.reshape(2,3,nao,nao)
        vk = vk.reshape(2,3,nao,nao)

        veff0doo = vj[0,0]+vj[1,0] - vk[:,0] + f1oo[:,0] + k1ao[:,0] * 2
        wvoa = reduce(cp.dot, (orbva.T, veff0doo[0], orboa)) * 2
        wvob = reduce(cp.dot, (orbvb.T, veff0doo[1], orbob)) * 2
        veff = vj[0,1]+vj[1,1] - vk[:,1] + f1vo[:,0] * 2
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
    else:
        dm = (dmzooa, dmxpya+dmxpya.T,
              dmzoob, dmxpyb+dmxpyb.T)
        vj = mf.get_j(mol, dm, hermi=1).reshape(2,2,nao,nao)
        if not isinstance(vj, cp.ndarray): vj = cp.asarray(vj)

        veff0doo = vj[0,0]+vj[1,0] + f1oo[:,0] + k1ao[:,0] * 2
        wvoa = reduce(cp.dot, (orbva.T, veff0doo[0], orboa)) * 2
        wvob = reduce(cp.dot, (orbvb.T, veff0doo[1], orbob)) * 2
        veff = vj[0,1]+vj[1,1] + f1vo[:,0] * 2
        veff0mopa = reduce(cp.dot, (mo_coeff[0].T, veff[0], mo_coeff[0]))
        veff0mopb = reduce(cp.dot, (mo_coeff[1].T, veff[1], mo_coeff[1]))
        wvoa -= cp.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], xpya) * 2
        wvob -= cp.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], xpyb) * 2
        wvoa += cp.einsum('ac,ai->ci', veff0mopa[nocca:,nocca:], xpya) * 2
        wvob += cp.einsum('ac,ai->ci', veff0mopb[noccb:,noccb:], xpyb) * 2
        veff0moma = cp.zeros((nmoa,nmoa))
        veff0momb = cp.zeros((nmob,nmob))

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
    time1 = log.timer('Z-vector using UCPHF solver', *time0)

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

    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    vhfopt = mf._opt_gpu.get(None, None)
    j_factor = 1.
    k_factor = 0.

    if with_k:
        if omega == 0:
            k_factor = hyb
        elif alpha == 0: # LR=0, only SR exchange
            pass
        elif hyb == 0: # SR=0, only LR exchange
            k_factor = alpha
        else: # SR and LR exchange with different ratios
            k_factor = alpha

    dvhf_DD_DP = rhf_grad._jk_energy_per_atom(mol, ((dmz1dooa+dmz1dooa.T)*0.25 + oo0a,
                                        (dmz1doob+dmz1doob.T)*0.25 + oo0b), vhfopt, j_factor, k_factor, verbose=verbose)
    dvhf_DD_DP-= rhf_grad._jk_energy_per_atom(mol, ((dmz1dooa+dmz1dooa.T)*0.25,
                                         (dmz1doob+dmz1doob.T)*0.25), vhfopt, j_factor, k_factor, verbose=verbose)
    dvhf_xpy = rhf_grad._jk_energy_per_atom(mol, ((dmxpya+dmxpya.T)*0.5,
                                      (dmxpyb+dmxpyb.T)*0.5), vhfopt, j_factor, k_factor, verbose=verbose)*2
    dvhf_xmy = rhf_grad._jk_energy_per_atom(mol, ((dmxmya-dmxmya.T)*0.5,
                                      (dmxmyb-dmxmyb.T)*0.5), vhfopt, j_factor=0.0, k_factor=k_factor)*2
        
    if with_k and omega != 0:
        j_factor = 0.
        omega = -omega # Prefer computing the SR part
        if alpha == 0: # LR=0, only SR exchange
            k_factor = hyb
        elif hyb == 0: # SR=0, only LR exchange
            # full range exchange was computed in the previous step
            k_factor = -alpha
        else: # SR and LR exchange with different ratios
            k_factor = hyb - alpha # =beta
        vhfopt = mf._opt_gpu.get(omega, None)
        with mol.with_range_coulomb(omega):
            dvhf_DD_DP += rhf_grad._jk_energy_per_atom(mol, ((dmz1dooa+dmz1dooa.T)*0.25 + oo0a,
                                        (dmz1doob+dmz1doob.T)*0.25 + oo0b), vhfopt, j_factor, k_factor, verbose=verbose)
            dvhf_DD_DP -= rhf_grad._jk_energy_per_atom(mol, ((dmz1dooa+dmz1dooa.T)*0.25,
                                         (dmz1doob+dmz1doob.T)*0.25), vhfopt, j_factor, k_factor, verbose=verbose)
            dvhf_xpy += rhf_grad._jk_energy_per_atom(mol, ((dmxpya+dmxpya.T)*0.5,
                                      (dmxpyb+dmxpyb.T)*0.5), vhfopt, j_factor, k_factor, verbose=verbose)*2
            dvhf_xmy += rhf_grad._jk_energy_per_atom(mol, ((dmxmya-dmxmya.T)*0.5,
                                      (dmxmyb-dmxmyb.T)*0.5), vhfopt, j_factor=0.0, k_factor=k_factor)*2

    fxcz1 = _contract_xc_kernel(td_grad, mf.xc, z1ao, None,
                                False, False)[0]
    
    veff1_0 = vxc1[:, 1:]
    veff1_1 =(f1oo[:, 1:] + fxcz1[:, 1:] + k1ao[:, 1:]*2)*2 # *2 for dmz1doo+dmz1oo.T
    veff1_2 = f1vo[:, 1:] * 2
    veff1_0_a, veff1_0_b = veff1_0
    veff1_1_a, veff1_1_b = veff1_1
    veff1_2_a, veff1_2_b = veff1_2

    time1 = log.timer('2e AO integral derivatives', *time1)
    if atmlst is None:
        atmlst = range(mol.natm)
    extra_force = cp.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        extra_force[k] += mf_grad.extra_force(ia, locals())

    delec = 2.0*(dh_ground + dh_td - ds)
    aoslices = mol.aoslice_by_atom()
    delec= cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:,2:]])
    de = 2.0 * (dvhf_DD_DP + dvhf_xpy + dvhf_xmy) + dh1e_ground + dh1e_td + delec + extra_force
    
    offsetdic = mol.offset_nr_by_atom()

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        
        de[k] += cp.einsum('xpq,pq->x', veff1_0_a[:,p0:p1], oo0a[p0:p1])
        de[k] += cp.einsum('xpq,pq->x', veff1_0_b[:,p0:p1], oo0b[p0:p1])
        de[k] += cp.einsum('xpq,qp->x', veff1_0_a[:,p0:p1], oo0a[:,p0:p1])
        de[k] += cp.einsum('xpq,qp->x', veff1_0_b[:,p0:p1], oo0b[:,p0:p1])

        de[k] += cp.einsum('xpq,pq->x', veff1_0_a[:,p0:p1], dmz1dooa[p0:p1]) * .5
        de[k] += cp.einsum('xpq,pq->x', veff1_0_b[:,p0:p1], dmz1doob[p0:p1]) * .5
        de[k] += cp.einsum('xpq,qp->x', veff1_0_a[:,p0:p1], dmz1dooa[:,p0:p1]) * .5
        de[k] += cp.einsum('xpq,qp->x', veff1_0_b[:,p0:p1], dmz1doob[:,p0:p1]) * .5


        de[k] += cp.einsum('xij,ij->x', veff1_1_a[:,p0:p1], oo0a[p0:p1]) * .5
        de[k] += cp.einsum('xij,ij->x', veff1_1_b[:,p0:p1], oo0b[p0:p1]) * .5
        de[k] += cp.einsum('xij,ij->x', veff1_2_a[:,p0:p1], dmxpya[p0:p1,:])
        de[k] += cp.einsum('xij,ij->x', veff1_2_b[:,p0:p1], dmxpyb[p0:p1,:])
        de[k] += cp.einsum('xji,ij->x', veff1_2_a[:,p0:p1], dmxpya[:,p0:p1])
        de[k] += cp.einsum('xji,ij->x', veff1_2_b[:,p0:p1], dmxpyb[:,p0:p1])

    log.timer('TDRKS nuclear gradients', *time0)
    return de.get()


# dmov, dmoo in AO-representation
# Note spin-trace is applied for fxc, kxc
#TODO: to include the response of grids
def _contract_xc_kernel(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True,
                        with_kxc=True):
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

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])

    # dmvo ~ reduce(cp.dot, (orbv, Xai, orbo.T))
    dmvo = cp.array([(dmvo[0] + dmvo[0].T) * .5, # because K_{ia,jb} == K_{ia,jb}
                     (dmvo[1] + dmvo[1].T) * .5])
    dmvo = opt.sort_orbitals(dmvo, axis=[1,2])

    f1vo = cp.zeros((2,4,nao,nao))
    deriv = 2
    if dmoo is not None:
        f1oo = cp.zeros((2,4,nao,nao))
        dmoo = opt.sort_orbitals(dmoo, axis=[1,2])
    else:
        f1oo = None
    if with_vxc:
        v1ao = cp.zeros((2,4,nao,nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = cp.zeros((2,4,nao,nao))
        deriv = 3
    else:
        k1ao = None

    if xctype == 'HF':
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == 'LDA':
        fmat_, ao_deriv = tdrks_grad._lda_eval_mat_, 1
    elif xctype == 'GGA':
        fmat_, ao_deriv = tdrks_grad._gga_eval_mat_, 2
    elif xctype == 'MGGA':
        fmat_, ao_deriv = tdrks_grad._mgga_eval_mat_, 2
        logger.warn(td_grad, 'TDUKS-MGGA Gradients may be inaccurate due to grids response')
    else:
        raise NotImplementedError(f'td-uks for functional {xc_code}')

    for ao, mask, weight, coords \
            in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
        if xctype == 'LDA':
            ao0 = ao[0]
        else:
            ao0 = ao
        rho = cp.asarray((ni.eval_rho2(_sorted_mol, ao0, mo_coeff[0], mo_occ[0], mask, xctype, with_lapl=False),
               ni.eval_rho2(_sorted_mol, ao0, mo_coeff[1], mo_occ[1], mask, xctype, with_lapl=False)))
        vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[1:]

        rho1 = cp.asarray((
            ni.eval_rho(_sorted_mol, ao0, dmvo[0], mask, xctype, hermi=1, with_lapl=False),
            ni.eval_rho(_sorted_mol, ao0, dmvo[1], mask, xctype, hermi=1, with_lapl=False)))
        if xctype == 'LDA':
            rho1 = rho1[:,cp.newaxis]
        wv = cp.einsum('axg,axbyg,g->byg', rho1, fxc, weight)
        fmat_(_sorted_mol, f1vo[0], ao, wv[0], mask, shls_slice, ao_loc)
        fmat_(_sorted_mol, f1vo[1], ao, wv[1], mask, shls_slice, ao_loc)

        if dmoo is not None:
            rho2 = cp.asarray((
                ni.eval_rho(_sorted_mol, ao0, dmoo[0], mask, xctype, hermi=1, with_lapl=False),
                ni.eval_rho(_sorted_mol, ao0, dmoo[1], mask, xctype, hermi=1, with_lapl=False)))
            if xctype == 'LDA':
                rho2 = rho2[:,cp.newaxis]
            wv = cp.einsum('axg,axbyg,g->byg', rho2, fxc, weight)
            fmat_(_sorted_mol, f1oo[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(_sorted_mol, f1oo[1], ao, wv[1], mask, shls_slice, ao_loc)
        if with_vxc:
            wv = vxc * weight
            fmat_(_sorted_mol, v1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(_sorted_mol, v1ao[1], ao, wv[1], mask, shls_slice, ao_loc)
        if with_kxc:
            wv = cp.einsum('axg,byg,axbyczg,g->czg', rho1, rho1, kxc, weight)
            fmat_(_sorted_mol, k1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(_sorted_mol, k1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

    f1vo[:,1:] *= -1
    f1vo = opt.unsort_orbitals(f1vo, axis=[2,3])
    if f1oo is not None: 
        f1oo[:,1:] *= -1
        f1oo = opt.unsort_orbitals(f1oo, axis=[2,3])
    if v1ao is not None: 
        v1ao[:,1:] *= -1
        v1ao = opt.unsort_orbitals(v1ao, axis=[2,3])
    if k1ao is not None: 
        k1ao[:,1:] *= -1
        k1ao = opt.unsort_orbitals(k1ao, axis=[2,3])

    return f1vo, f1oo, v1ao, k1ao


class Gradients(tdrhf_grad.Gradients):
    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet=None, atmlst=None):
        return grad_elec(self, xy, atmlst, self.verbose)

Grad = Gradients

from gpu4pyscf import tdscf
tdscf.uks.TDA.Gradients = tdscf.uks.TDDFT.Gradients = lib.class_as_method(Gradients)
