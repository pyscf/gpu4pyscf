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

import copy
import numpy
from cupyx.scipy.linalg import solve_triangular
from pyscf import scf, gto
from gpu4pyscf.df import int3c2e, df
from gpu4pyscf.lib.cupy_helper import tag_array, contract, cholesky
from gpu4pyscf.df.grad import rhf as rhf_grad_df
from gpu4pyscf.tdscf import rhf as tdrhf
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf import __config__
from gpu4pyscf.lib import logger
from gpu4pyscf.df.grad.jk import get_rhojk_mo, get_grad_vjk_mo, get_rhojk, get_grad_vjk
from functools import reduce
import cupy as cp
from pyscf import lib
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import cphf
from gpu4pyscf import lib as lib_gpu
from pyscf import __config__
from gpu4pyscf.lib import utils
from gpu4pyscf import tdscf


def grad_elec(td_grad, x_y, singlet=True, atmlst=None, verbose=logger.INFO):
    """
    Electronic part of TDA, TDHF nuclear gradients

    Args:
        td_grad : grad.tdrhf.Gradients or grad.tdrks.Gradients object.

        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    """
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()
    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nmo = mo_coeff.shape[1]
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    x, y = x_y
    x = cp.asarray(x)
    y = cp.asarray(y)
    xpy = (x + y).reshape(nocc, nvir).T
    xmy = (x - y).reshape(nocc, nvir).T
    orbv = mo_coeff[:, nocc:]
    orbo = mo_coeff[:, :nocc]
    dvv = cp.einsum("ai,bi->ab", xpy, xpy) + cp.einsum("ai,bi->ab", xmy, xmy)  # 2 T_{ab}
    doo = -cp.einsum("ai,aj->ij", xpy, xpy) - cp.einsum("ai,aj->ij", xmy, xmy)  # 2 T_{ij}
    dmxpy = reduce(cp.dot, (orbv, xpy, orbo.T))  # (X+Y) in ao basis
    dmxmy = reduce(cp.dot, (orbv, xmy, orbo.T))  # (X-Y) in ao basis
    dmzoo = reduce(cp.dot, (orbo, doo, orbo.T))  # T_{ij}*2 in ao basis
    dmzoo += reduce(cp.dot, (orbv, dvv, orbv.T))  # T_{ij}*2 + T_{ab}*2 in ao basis

    vj0, vk0 = mf.get_jk(mol, dmzoo, hermi=0)
    vj1, vk1 = mf.get_jk(mol, dmxpy + dmxpy.T, hermi=0)
    vj2, vk2 = mf.get_jk(mol, dmxmy - dmxmy.T, hermi=0)
    if not isinstance(vj0, cp.ndarray):
        vj0 = cp.asarray(vj0)
    if not isinstance(vk0, cp.ndarray):
        vk0 = cp.asarray(vk0)
    if not isinstance(vj1, cp.ndarray):
        vj1 = cp.asarray(vj1)
    if not isinstance(vk1, cp.ndarray):
        vk1 = cp.asarray(vk1)
    if not isinstance(vj2, cp.ndarray):
        vj2 = cp.asarray(vj2)
    if not isinstance(vk2, cp.ndarray):
        vk2 = cp.asarray(vk2)
    vj = cp.stack((vj0, vj1, vj2))
    vk = cp.stack((vk0, vk1, vk2))
    veff0doo = vj[0] * 2 - vk[0]  # 2 for alpha and beta
    wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
    if singlet:
        veff = vj[1] * 2 - vk[1]
    else:
        veff = -vk[1]
    veff0mop = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= cp.einsum("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2  # 2 for dm + dm.T
    wvo += cp.einsum("ac,ai->ci", veff0mop[nocc:, nocc:], xpy) * 2
    veff = -vk[2]
    veff0mom = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= cp.einsum("ki,ai->ak", veff0mom[:nocc, :nocc], xmy) * 2
    wvo += cp.einsum("ac,ai->ci", veff0mom[nocc:, nocc:], xmy) * 2

    # set singlet=None, generate function for CPHF type response kernel
    vresp = mf.gen_response(singlet=None, hermi=1)

    def fvind(x):  # For singlet, closed shell ground state
        dm = reduce(cp.dot, (orbv, x.reshape(nvir, nocc) * 2, orbo.T))  # 2 for double occupancy
        v1ao = vresp(dm + dm.T)  # for the upused 2
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()

    z1 = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        wvo,
        max_cycle=td_grad.cphf_max_cycle,
        tol=td_grad.cphf_conv_tol)[0]
    z1 = z1.reshape(nvir, nocc)
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    z1ao = reduce(cp.dot, (orbv, z1, orbo.T))
    veff = vresp(z1ao + z1ao.T)

    im0 = cp.zeros((nmo, nmo))
    # in the following, all should be doubled, due to double occupancy
    # and 0.5 for i<=j and a<= b
    # but this is reduced.
    # H_{ij}^+[T] + H_{ij}^+[Z] #
    im0[:nocc, :nocc] = reduce(cp.dot, (orbo.T, veff0doo + veff, orbo))
    # H_{ij}^+[T] + H_{ij}^+[Z] + sum_{a} (X+Y)_{aj}H_{ai}^+[(X+Y)]
    im0[:nocc, :nocc] += cp.einsum("ak,ai->ki", veff0mop[nocc:, :nocc], xpy)
    # H_{ij}^+[T] + H_{ij}^+[Z] + sum_{a} (X+Y)_{aj}H_{ai}^+[(X+Y)]
    #  + sum_{a} (X-Y)_{aj}H_{ai}^-[(X-Y)]
    im0[:nocc, :nocc] += cp.einsum("ak,ai->ki", veff0mom[nocc:, :nocc], xmy)
    #  sum_{i} (X+Y)_{ci}H_{ai}^+[(X+Y)]
    im0[nocc:, nocc:] = cp.einsum("ci,ai->ac", veff0mop[nocc:, :nocc], xpy)
    #  sum_{i} (X+Y)_{ci}H_{ai}^+[(X+Y)] + sum_{i} (X-Y)_{cj}H_{ai}^-[(X-Y)]
    im0[nocc:, nocc:] += cp.einsum("ci,ai->ac", veff0mom[nocc:, :nocc], xmy)
    #  sum_{i} (X+Y)_{ki}H_{ai}^+[(X+Y)] * 2
    im0[nocc:, :nocc] = cp.einsum("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2
    #  sum_{i} (X+Y)_{ki}H_{ai}^+[(X+Y)] + sum_{i} (X-Y)_{ki}H_{ai}^-[(X-Y)] * 2
    im0[nocc:, :nocc] += cp.einsum("ki,ai->ak", veff0mom[:nocc, :nocc], xmy) * 2

    zeta = lib_gpu.cupy_helper.direct_sum("i+j->ij", mo_energy, mo_energy) * 0.5
    zeta[nocc:, :nocc] = mo_energy[:nocc]
    zeta[:nocc, nocc:] = mo_energy[nocc:]
    dm1 = cp.zeros((nmo, nmo))
    dm1[:nocc, :nocc] = doo
    dm1[nocc:, nocc:] = dvv
    dm1[nocc:, :nocc] = z1
    dm1[:nocc, :nocc] += cp.eye(nocc) * 2  # for ground state
    im0 = reduce(cp.dot, (mo_coeff, im0 + zeta * dm1, mo_coeff.T))

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td_grad.base._scf.nuc_grad_method()
    s1 = mf_grad.get_ovlp(mol)

    dmz1doo = z1ao + dmzoo  # P
    oo0 = reduce(cp.dot, (orbo, orbo.T))  # D

    if atmlst is None:
        atmlst = range(mol.natm)

    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_ground = contract("xij,ij->xi", h1, oo0 * 2)
    dh_td = contract("xij,ij->xi", h1, (dmz1doo + dmz1doo.T) * 0.5)
    ds = contract("xij,ij->xi", s1, (im0 + im0.T) * 0.5)

    dh1e_ground = int3c2e.get_dh1e(mol, oo0 * 2)  # 1/r like terms
    if mol.has_ecp():
        dh1e_ground += rhf_grad.get_dh1e_ecp(mol, oo0 * 2)  # 1/r like terms
    dh1e_td = int3c2e.get_dh1e(mol, (dmz1doo + dmz1doo.T) * 0.5)  # 1/r like terms
    if mol.has_ecp():
        dh1e_td += rhf_grad.get_dh1e_ecp(mol, (dmz1doo + dmz1doo.T) * 0.5)  # 1/r like terms
    extra_force = cp.zeros((len(atmlst), 3))

    dvhf_all = 0
    dvhf = td_grad.get_veff(mol, (dmz1doo + dmz1doo.T) * 0.5 + oo0 * 2)
    for k, ia in enumerate(atmlst):
        extra_force[k] += mf_grad.extra_force(ia, locals())
    dvhf_all += dvhf
    dvhf = td_grad.get_veff(mol, (dmz1doo + dmz1doo.T) * 0.5)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= mf_grad.extra_force(ia, locals())
    dvhf_all -= dvhf

    if singlet:
        j_factor=1.0
        k_factor=1.0
    else:
        j_factor=0.0
        k_factor=1.0
    dvhf = td_grad.get_veff(mol, (dmxpy + dmxpy.T), j_factor * 2, k_factor * 2, dm_mo=xpy)
    for k, ia in enumerate(atmlst):
        extra_force[k] += mf_grad.extra_force(ia, locals())
    dvhf_all += dvhf
    dvhf = td_grad.get_veff(mol, (dmxmy - dmxmy.T), 0.0, k_factor * 2, dm_mo=xmy)
    for k, ia in enumerate(atmlst):
        extra_force[k] += mf_grad.extra_force(ia, locals())
    dvhf_all += dvhf
    time1 = log.timer('2e AO integral derivatives', *time1)

    delec = 2.0 * (dh_ground + dh_td - ds)
    aoslices = mol.aoslice_by_atom()
    delec = cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])
    de = 2.0 * dvhf_all + dh1e_ground + dh1e_td + delec + extra_force

    log.timer('TDHF nuclear gradients', *time0)
    return de.get()


LINEAR_DEP_THRESHOLD = df.LINEAR_DEP_THR
MIN_BLK_SIZE = getattr(__config__, 'min_ao_blksize', 128)
ALIGNED = getattr(__config__, 'ao_aligned', 64)


def get_jk(tdgrad, mol=None, dm0=None, hermi=0, with_j=True, with_k=True, omega=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.

    NOTE: This function is incompatible to the one implemented in PySCF CPU version.
    In the CPU version, get_jk returns the first order derivatives of J/K matrices.
    '''
    if mol is None: mol = tdgrad.mol
    #TODO: dm has to be the SCF density matrix in this version.  dm should be
    # extended to any 1-particle density matrix
    mf = tdgrad.base._scf
    assert dm0 is not None
    if omega is None:
        with_df = mf.with_df
    else:
        key = '%.6f' % omega
        if key in mf.with_df._rsh_df:
            with_df = mf.with_df._rsh_df[key]
        else:
            dfobj = mf.with_df
            with_df = dfobj._rsh_df[key] = dfobj.copy().reset()

    auxmol = with_df.auxmol
    if not hasattr(with_df, 'intopt') or with_df._cderi is None:
        with_df.build(omega=omega)
    intopt = with_df.intopt
    naux = with_df.naux

    log = logger.new_logger(mol, mol.verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    if isinstance(mf, scf.rohf.ROHF):
        raise NotImplementedError()
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)

    dm = intopt.sort_orbitals(dm0, axis=[0,1])
    orbo = mo_coeff[:,mo_occ>0] * mo_occ[mo_occ>0] ** 0.5
    mo_coeff = None
    orbo = intopt.sort_orbitals(orbo, axis=[0])

    rhoj, rhok = get_rhojk(with_df, dm, orbo, with_j=with_j, with_k=with_k)
    
    # (d/dX P|Q) contributions
    if omega and omega > 1e-10:
        with auxmol.with_range_coulomb(omega):
            int2c_e1 = auxmol.intor('int2c2e_ip1')
    else:
        int2c_e1 = auxmol.intor('int2c2e_ip1')
    int2c_e1 = cp.asarray(int2c_e1)

    rhoj_cart = rhok_cart = None
    auxslices = auxmol.aoslice_by_atom()
    aux_cart2sph = intopt.aux_cart2sph
    low = with_df.cd_low
    low_t = low.T.copy()
    if with_j:
        if low.tag == 'eig':
            rhoj = cp.dot(low_t.T, rhoj)
        elif low.tag == 'cd':
            #rhoj = solve_triangular(low_t, rhoj, lower=False)
            rhoj = solve_triangular(low_t, rhoj, lower=False, overwrite_b=True)
        if not auxmol.cart:
            rhoj_cart = contract('pq,q->p', aux_cart2sph, rhoj)
        else:
            rhoj_cart = rhoj

        rhoj = intopt.unsort_orbitals(rhoj, aux_axis=[0])
        tmp = contract('xpq,q->xp', int2c_e1, rhoj)
        vjaux = -contract('xp,p->xp', tmp, rhoj)
        vjaux_2c = cp.array([-vjaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]])
        rhoj = vjaux = tmp = None
    if with_k:
        nocc = orbo.shape[-1]
        if low.tag == 'eig':
            rhok = contract('pq,qij->pij', low_t.T, rhok)
        elif low.tag == 'cd':
            #rhok = solve_triangular(low_t, rhok, lower=False)
            rhok = solve_triangular(low_t, rhok.reshape(naux, -1), lower=False, overwrite_b=True).reshape(naux, nocc, nocc)
            rhok = rhok.copy(order='C')
        tmp = contract('pij,qij->pq', rhok, rhok)
        tmp = intopt.unsort_orbitals(tmp, aux_axis=[0,1])
        vkaux = -contract('xpq,pq->xp', int2c_e1, tmp)
        vkaux_2c = cp.array([-vkaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]])
        vkaux = tmp = None
        if not auxmol.cart:
            rhok_cart = contract('pq,qkl->pkl', aux_cart2sph, rhok)
        else:
            rhok_cart = rhok
        rhok = None
    low_t = None
    t0 = log.timer_debug1('rhoj and rhok', *t0)
    int2c_e1 = None

    dm_cart = dm
    orbo_cart = orbo
    if not mol.cart:
        # sph2cart for ao
        cart2sph = intopt.cart2sph
        orbo_cart = cart2sph @ orbo
        dm_cart = cart2sph @ dm @ cart2sph.T
        
    with_df._cderi = None # release GPU memory
    vj, vk, vjaux, vkaux = get_grad_vjk(with_df, mol, auxmol, rhoj_cart, dm_cart, rhok_cart, orbo_cart,
                                        with_j=with_j, with_k=with_k, omega=omega)
    # NOTE: vj and vk are still in cartesian
    _sorted_mol = intopt._sorted_mol
    natm = _sorted_mol.natm
    nao_cart = _sorted_mol.nao
    ao2atom = numpy.zeros([nao_cart, natm])
    ao_loc = _sorted_mol.ao_loc
    for ibas, iatm in enumerate(_sorted_mol._bas[:,gto.ATOM_OF]):
        ao2atom[ao_loc[ibas]:ao_loc[ibas+1],iatm] = 1
    ao2atom = cp.asarray(ao2atom)
    if with_j:
        vj = -ao2atom.T @ vj.T
    if with_k:
        vk = -ao2atom.T @ vk.T
    t0 = log.timer_debug1('(di,j|P) and (i,j|dP)', *t0)

    _sorted_auxmol = intopt._sorted_auxmol
    natm = _sorted_auxmol.natm
    naux_cart = _sorted_auxmol.nao
    aux2atom = numpy.zeros([naux_cart, natm])
    ao_loc = _sorted_auxmol.ao_loc
    for ibas, iatm in enumerate(_sorted_auxmol._bas[:,gto.ATOM_OF]):
        aux2atom[ao_loc[ibas]:ao_loc[ibas+1],iatm] = 1
    aux2atom = cp.asarray(aux2atom)
    if with_j:
        vjaux_3c = aux2atom.T @ vjaux.T
        vjaux = vjaux_2c - vjaux_3c

    if with_k:
        vkaux_3c = aux2atom.T @ vkaux.T
        vkaux = vkaux_2c - vkaux_3c
    return vj, vk, vjaux, vkaux


def get_jk_mo(tdgrad, mol=None, dm0=None, dm_mo=None, hermi=0, with_j=True, with_k=True, omega=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.

    NOTE: This function is incompatible to the one implemented in PySCF CPU version.
    In the CPU version, get_jk returns the first order derivatives of J/K matrices.
    '''
    if mol is None: mol = tdgrad.mol
    #TODO: dm has to be the SCF density matrix in this version.  dm should be
    # extended to any 1-particle density matrix
    mf = tdgrad.base._scf
    assert dm0 is not None
    if omega is None:
        with_df = mf.with_df
    else:
        key = '%.6f' % omega
        if key in mf.with_df._rsh_df:
            with_df = mf.with_df._rsh_df[key]
        else:
            dfobj = mf.with_df
            with_df = dfobj._rsh_df[key] = dfobj.copy().reset()

    auxmol = with_df.auxmol
    if not hasattr(with_df, 'intopt') or with_df._cderi is None:
        with_df.build(omega=omega)
    intopt = with_df.intopt
    naux = with_df.naux

    log = logger.new_logger(mol, mol.verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    if isinstance(mf, scf.rohf.ROHF):
        raise NotImplementedError()
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)

    dm = intopt.sort_orbitals(dm0, axis=[0,1])
    orbo = mo_coeff[:,mo_occ>0] * mo_occ[mo_occ>0] ** 0.5
    orbv = mo_coeff[:,mo_occ==0]
    mo_coeff = None
    orbo = intopt.sort_orbitals(orbo, axis=[0])
    orbv = intopt.sort_orbitals(orbv, axis=[0])

    rhoj, rhok = get_rhojk_mo(with_df, dm, orbo, orbv, with_j=with_j, with_k=with_k)
    
    # (d/dX P|Q) contributions
    if omega and omega > 1e-10:
        with auxmol.with_range_coulomb(omega):
            int2c_e1 = auxmol.intor('int2c2e_ip1')
    else:
        int2c_e1 = auxmol.intor('int2c2e_ip1')
    int2c_e1 = cp.asarray(int2c_e1)

    rhoj_cart = rhok_cart = None
    auxslices = auxmol.aoslice_by_atom()
    aux_cart2sph = intopt.aux_cart2sph
    low = with_df.cd_low
    low_t = low.T.copy()
    if with_j:
        if low.tag == 'eig':
            rhoj = cp.dot(low_t.T, rhoj)
        elif low.tag == 'cd':
            #rhoj = solve_triangular(low_t, rhoj, lower=False)
            rhoj = solve_triangular(low_t, rhoj, lower=False, overwrite_b=True)
        if not auxmol.cart:
            rhoj_cart = contract('pq,q->p', aux_cart2sph, rhoj)
        else:
            rhoj_cart = rhoj

        rhoj = intopt.unsort_orbitals(rhoj, aux_axis=[0])
        tmp = contract('xpq,q->xp', int2c_e1, rhoj)
        vjaux = -contract('xp,p->xp', tmp, rhoj)
        vjaux_2c = cp.array([-vjaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]])
        rhoj = vjaux = tmp = None
    if with_k:
        nocc = orbo.shape[-1]
        nvir = orbv.shape[-1]
        if low.tag == 'eig':
            rhok = contract('pq,qij->pij', low_t.T, rhok)
        elif low.tag == 'cd':
            #rhok = solve_triangular(low_t, rhok, lower=False)
            rhok = solve_triangular(low_t, rhok.reshape(naux, -1), lower=False, overwrite_b=True).reshape(naux, nocc, nvir)
            rhok = rhok.copy(order='C')
        tmp = cp.einsum('pja,qib,ai,bj->pq', rhok, rhok, dm_mo, dm_mo)
        tmp = intopt.unsort_orbitals(tmp, aux_axis=[0,1])
        vkaux = -contract('xpq,pq->xp', int2c_e1, tmp)
        vkaux_2c = cp.array([-vkaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]])
        vkaux = tmp = None
        if not auxmol.cart:
            rhok_cart = contract('pq,qkl->pkl', aux_cart2sph, rhok)
        else:
            rhok_cart = rhok
        rhok = None
    low_t = None
    t0 = log.timer_debug1('rhoj and rhok', *t0)
    int2c_e1 = None

    dm_cart = dm
    orbo_cart = orbo
    if not mol.cart:
        # sph2cart for ao
        cart2sph = intopt.cart2sph
        orbo_cart = cart2sph @ orbo
        orbv_cart = cart2sph @ orbv
        dm_cart = cart2sph @ dm @ cart2sph.T
        
    with_df._cderi = None # release GPU memory
    vj, vk, vjaux, vkaux = get_grad_vjk_mo(with_df, mol, auxmol, rhoj_cart, dm_cart, dm_mo, rhok_cart, orbo_cart, orbv_cart,
                                            with_j=with_j, with_k=with_k, omega=omega)
    # NOTE: vj and vk are still in cartesian
    _sorted_mol = intopt._sorted_mol
    natm = _sorted_mol.natm
    nao_cart = _sorted_mol.nao
    ao2atom = numpy.zeros([nao_cart, natm])
    ao_loc = _sorted_mol.ao_loc
    for ibas, iatm in enumerate(_sorted_mol._bas[:,gto.ATOM_OF]):
        ao2atom[ao_loc[ibas]:ao_loc[ibas+1],iatm] = 1
    ao2atom = cp.asarray(ao2atom)
    if with_j:
        vj = -ao2atom.T @ vj.T
    if with_k:
        vk = -ao2atom.T @ vk.T
    t0 = log.timer_debug1('(di,j|P) and (i,j|dP)', *t0)

    _sorted_auxmol = intopt._sorted_auxmol
    natm = _sorted_auxmol.natm
    naux_cart = _sorted_auxmol.nao
    aux2atom = numpy.zeros([naux_cart, natm])
    ao_loc = _sorted_auxmol.ao_loc
    for ibas, iatm in enumerate(_sorted_auxmol._bas[:,gto.ATOM_OF]):
        aux2atom[ao_loc[ibas]:ao_loc[ibas+1],iatm] = 1
    aux2atom = cp.asarray(aux2atom)
    if with_j:
        vjaux_3c = aux2atom.T @ vjaux.T
        vjaux = vjaux_2c - vjaux_3c

    if with_k:
        vkaux_3c = aux2atom.T @ vkaux.T
        vkaux = vkaux_2c - vkaux_3c
    return vj, vk, vjaux, vkaux


class Gradients(tdrhf_grad.Gradients):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'with_df', 'auxbasis_response'}
    def __init__(self, td):
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        tdrhf_grad.Gradients.__init__(self, td)

    auxbasis_response = True
    get_jk = get_jk
    get_jk_mo = get_jk_mo

    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet, atmlst=None, verbose=logger.INFO):
        return grad_elec(self, xy, singlet, atmlst, verbose)

    def check_sanity(self):
        assert isinstance(self.base._scf, df.df_jk._DFHF)
        assert isinstance(self.base, tdrhf.TDHF) or isinstance(self.base, tdrhf.TDA)

    def get_j(self, mol=None, dm=None, hermi=0):
        vj, _, vjaux, _ = self.get_jk(mol, dm, with_k=False)
        return vj, vjaux

    def get_k(self, mol=None, dm=None, hermi=0):
        _, vk, _, vkaux = self.get_jk(mol, dm, with_j=False)
        return vk, vkaux

    def get_veff(self, mol=None, dm=None, j_factor=1.0, k_factor=1.0, omega=0.0, dm_mo = None, verbose=None):
        if dm_mo is None:
            if omega != 0.0:
                vj, vk, vjaux, vkaux = self.get_jk(mol, dm, omega=omega)
            else:
                vj, vk, vjaux, vkaux = self.get_jk(mol, dm)
            vhf = vj * j_factor - vk * .5 * k_factor
            if self.auxbasis_response:
                e1_aux = vjaux * j_factor - vkaux * .5 * k_factor
                logger.debug1(self, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
            else:
                e1_aux = None
        else:
            if omega != 0.0:
                vj, vk, vjaux, vkaux = self.get_jk_mo(mol, dm, omega=omega, dm_mo = dm_mo)
            else:
                vj, vk, vjaux, vkaux = self.get_jk_mo(mol, dm, dm_mo = dm_mo)
            vhf = vj * j_factor - vk * .5 * k_factor
            if self.auxbasis_response:
                e1_aux = vjaux * j_factor - vkaux * .5 * k_factor
                logger.debug1(self, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
            else:
                e1_aux = None

        vhf = tag_array(vhf, aux=e1_aux)
        
        return vhf

Grad = Gradients
