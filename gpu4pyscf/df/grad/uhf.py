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
import cupy
import copy
from cupyx.scipy.linalg import solve_triangular
from pyscf import scf
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import tag_array, contract, load_library, take_last2d
from gpu4pyscf.grad import uhf as uhf_grad
from gpu4pyscf import __config__
from gpu4pyscf.lib import logger

libgvhf = load_library('libgvhf')
FREE_CUPY_CACHE = True
BINSIZE = 128

def get_jk(mf_grad, mol=None, dm0=None, hermi=0, with_j=True, with_k=True, omega=None, mo_coeff=None, mo_occ=None, dm2 = None):
    if mol is None: mol = mf_grad.mol
    #TODO: dm has to be the SCF density matrix in this version.  dm should be
    # extended to any 1-particle density matrix

    if(dm0 is None): dm0 = mf_grad.base.make_rdm1()
    mf = mf_grad.base
    if omega is None:
        with_df = mf_grad.base.with_df
    else:
        key = '%.6f' % omega
        if key in mf_grad.base.with_df._rsh_df:
            with_df = mf_grad.base.with_df._rsh_df[key]
        else:
            dfobj = mf_grad.base.with_df
            with_df = dfobj._rsh_df[key] = copy.copy(dfobj).reset()

    auxmol = with_df.auxmol
    if not hasattr(with_df, 'intopt') or with_df._cderi is None:
        with_df.build(omega=omega)
    intopt = with_df.intopt

    naux = with_df.naux

    log = logger.new_logger(mol, mol.verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    if isinstance(mf_grad.base, scf.rohf.ROHF):
        raise NotImplementedError()
    if mo_coeff is None:
        mo_coeff = cupy.asarray(mf_grad.base.mo_coeff)
    if mo_occ is None:
        mo_occ = cupy.asarray(mf_grad.base.mo_occ)
    ao_idx = intopt.ao_idx
    dm = take_last2d(dm0, ao_idx)
    if dm2 is not None:
        dm2_tmp = take_last2d(dm2, ao_idx)
    # (L|ij) -> rhoj: (L), rhok: (L|oo)
    orbo = mo_coeff[:,mo_occ>0] * mo_occ[mo_occ>0] ** 0.5
    orbo = orbo[ao_idx, :]
    nocc = orbo.shape[-1]

    # (L|ij) -> rhoj: (L), rhok: (L|oo)
    low = with_df.cd_low
    rows = with_df.intopt.cderi_row
    cols = with_df.intopt.cderi_col
    dm_sparse = dm[rows, cols]
    dm_sparse[with_df.intopt.cderi_diag] *= .5
    if dm2 is not None:
        dm2_sparse = dm2_tmp[rows, cols]
        dm2_sparse[with_df.intopt.cderi_diag] *= .5

    blksize = with_df.get_blksize()
    if with_j:
        rhoj = cupy.empty([naux])
        if dm2 is not None:
            rhoj2 = cupy.empty([naux])
    if with_k:
        rhok = cupy.empty([naux, nocc, nocc], order='C')
    p0 = p1 = 0

    for cderi, cderi_sparse in with_df.loop(blksize=blksize):
        p1 = p0 + cderi.shape[0]
        if with_j:
            rhoj[p0:p1] = 2.0*dm_sparse.dot(cderi_sparse)
            if dm2 is not None:
                rhoj2[p0:p1] = 2.0*dm2_sparse.dot(cderi_sparse)
        if with_k:
            tmp = contract('Lij,jk->Lki', cderi, orbo)
            contract('Lki,il->Lkl', tmp, orbo, out=rhok[p0:p1])
        p0 = p1
    tmp = dm_sparse = cderi_sparse = cderi = None

    # (d/dX P|Q) contributions
    if omega and omega > 1e-10:
        with auxmol.with_range_coulomb(omega):
            int2c_e1 = auxmol.intor('int2c2e_ip1')
    else:
        int2c_e1 = auxmol.intor('int2c2e_ip1')
    int2c_e1 = cupy.asarray(int2c_e1)
    aux_ao_idx = intopt.aux_ao_idx
    rev_aux_idx = np.argsort(aux_ao_idx)
    auxslices = auxmol.aoslice_by_atom()
    aux_cart2sph = intopt.aux_cart2sph
    low_t = low.T.copy()
    if with_j:
        if low.tag == 'eig':
            rhoj = cupy.dot(low_t.T, rhoj)
            if dm2 is not None:
                rhoj2 = cupy.dot(low_t.T, rhoj2)
        elif low.tag == 'cd':
            rhoj = solve_triangular(low_t, rhoj, lower=False, overwrite_b=True)
            if dm2 is not None:
                rhoj2 = solve_triangular(low_t, rhoj2, lower=False, overwrite_b=True)
        if not auxmol.cart:
            rhoj_cart = contract('pq,q->p', aux_cart2sph, rhoj)
        else:
            rhoj_cart = rhoj

        rhoj = rhoj[rev_aux_idx]

        if dm2 is not None:
            rhoj2 = rhoj2[rev_aux_idx]
        tmp = contract('xpq,q->xp', int2c_e1, rhoj)
        if dm2 is not None:
            vjaux = -contract('xp,p->xp', tmp, rhoj2)
        else:
            vjaux = -contract('xp,p->xp', tmp, rhoj)
        vjaux_2c = cupy.array([-vjaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]])
        rhoj = vjaux = tmp = None
    if with_k:
        if low.tag == 'eig':
            rhok = contract('pq,qij->pij', low_t.T, rhok)
        elif low.tag == 'cd':
            rhok = solve_triangular(low_t, rhok.reshape(naux, -1), lower=False, overwrite_b=True).reshape(naux, nocc, nocc)
        tmp = contract('pij,qij->pq', rhok, rhok)
        tmp = take_last2d(tmp, rev_aux_idx)
        vkaux = -contract('xpq,pq->xp', int2c_e1, tmp)
        vkaux_2c = cupy.array([-vkaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]])
        vkaux = tmp = None
        if not auxmol.cart:
            rhok_cart = contract('pq,qkl->pkl', aux_cart2sph, rhok)
        else:
            rhok_cart = rhok
        rhok = None
    low_t = None
    t0 = log.timer_debug1('rhoj and rhok', *t0)
    int2c_e1 = None

    nao_cart = intopt.mol.nao
    block_size = with_df.get_blksize(nao=nao_cart)
    intopt.clear()
    # rebuild with aosym
    intopt.build(mf.direct_scf_tol, diag_block_with_triu=True, aosym=False,
                 group_size_aux=block_size)#, group_size=block_size)
    if not intopt._mol.cart:
        # sph2cart for ao
        cart2sph = intopt.cart2sph
        orbo_cart = cart2sph @ orbo
        if dm2 is None:
            dm_cart = cart2sph @ dm @ cart2sph.T
        else:
            dm2_tmp = take_last2d(dm2, ao_idx)
            dm_cart = cart2sph @ dm2_tmp @ cart2sph.T
    else:
        if dm2 is None:
            dm_cart = dm
        else:
            dm_cart = take_last2d(dm2, ao_idx)
        orbo_cart = orbo
    dm = orbo = None

    vj = vk = rhoj_tmp = rhok_tmp = None
    vjaux = vkaux = None

    naux_cart = intopt.auxmol.nao
    if with_j:
        vj = cupy.zeros((3,nao_cart), order='C')
        vjaux = cupy.zeros((3,naux_cart))
    if with_k:
        vk = cupy.zeros((3,nao_cart), order='C')
        vkaux = cupy.zeros((3,naux_cart))
    cupy.get_default_memory_pool().free_all_blocks()
    for cp_kl_id in range(len(intopt.aux_log_qs)):
        t1 = log.init_timer()
        k0, k1 = intopt.cart_aux_loc[cp_kl_id], intopt.cart_aux_loc[cp_kl_id+1]
        assert k1-k0 <= block_size
        if with_j:
            rhoj_tmp = rhoj_cart[k0:k1]
        if with_k:
            rhok_tmp = contract('por,ir->pio', rhok_cart[k0:k1], orbo_cart)
            rhok_tmp = contract('pio,jo->pji', rhok_tmp, orbo_cart)
        '''
        if(rhoj_tmp.flags['C_CONTIGUOUS'] == False):
            rhoj_tmp = rhoj_tmp.astype(cupy.float64, order='C')

        if(rhok_tmp.flags['C_CONTIGUOUS'] == False):
            rhok_tmp = rhok_tmp.astype(cupy.float64, order='C')
        '''
        '''
        # outcore implementation
        int3c2e.get_int3c2e_ip_slice(intopt, cp_kl_id, 1, out=buf)
        size = 3*(k1-k0)*nao_cart*nao_cart
        int3c_ip = buf[:size].reshape([3,k1-k0,nao_cart,nao_cart], order='C')
        rhoj_tmp = contract('xpji,ij->xip', int3c_ip, dm_cart)
        vj += contract('xip,p->xi', rhoj_tmp, rhoj_cart[k0:k1])
        vk += contract('pji,xpji->xi', rhok_tmp, int3c_ip)

        int3c2e.get_int3c2e_ip_slice(intopt, cp_kl_id, 2, out=buf)
        rhoj_tmp = contract('xpji,ji->xp', int3c_ip, dm_cart)
        vjaux[:, k0:k1] = contract('xp,p->xp', rhoj_tmp, rhoj_cart[k0:k1])
        vkaux[:, k0:k1] = contract('xpji,pji->xp', int3c_ip, rhok_tmp)
        '''
        vj_tmp, vk_tmp = int3c2e.get_int3c2e_ip_jk(intopt, cp_kl_id, 'ip1', rhoj_tmp, rhok_tmp, dm_cart, omega=omega)
        if with_j: vj += vj_tmp
        if with_k: vk += vk_tmp

        vj_tmp, vk_tmp = int3c2e.get_int3c2e_ip_jk(intopt, cp_kl_id, 'ip2', rhoj_tmp, rhok_tmp, dm_cart, omega=omega)
        if with_j: vjaux[:, k0:k1] = vj_tmp
        if with_k: vkaux[:, k0:k1] = vk_tmp

        rhoj_tmp = rhok_tmp = vj_tmp = vk_tmp = None
        t1 = log.timer_debug1(f'calculate {cp_kl_id:3d} / {len(intopt.aux_log_qs):3d}, {k1-k0:3d} slices', *t1)

    cart_ao_idx = intopt.cart_ao_idx
    rev_cart_ao_idx = np.argsort(cart_ao_idx)
    aoslices = intopt.mol.aoslice_by_atom()
    if with_j:
        vj = vj[:, rev_cart_ao_idx]
        vj = [-vj[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]]
        vj = cupy.asarray(vj)
    if with_k:
        vk = vk[:, rev_cart_ao_idx]
        vk = [-vk[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]]
        vk = cupy.asarray(vk)
    t0 = log.timer_debug1('(di,j|P) and (i,j|dP)', *t0)

    cart_aux_idx = intopt.cart_aux_idx
    rev_cart_aux_idx = np.argsort(cart_aux_idx)
    auxslices = intopt.auxmol.aoslice_by_atom()

    if with_j:
        vjaux = vjaux[:, rev_cart_aux_idx]
        vjaux_3c = cupy.asarray([-vjaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]])
        vjaux = vjaux_2c + vjaux_3c

    if with_k:
        vkaux = vkaux[:, rev_cart_aux_idx]
        vkaux_3c = cupy.asarray([-vkaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]])
        vkaux = vkaux_2c + vkaux_3c
    return vj, vk, vjaux, vkaux

class Gradients(uhf_grad.Gradients):
    '''Unrestricted density-fitting Hartree-Fock gradients'''

    _keys = {'with_df', 'auxbasis_response'}

    def __init__(self, mf):
        self._keys = self._keys.union(['auxbasis_response'])
        uhf_grad.Gradients.__init__(self, mf)

    # Whether to include the response of DF auxiliary basis when computing
    # nuclear gradients of J/K matrices
    auxbasis_response = True
    get_jk = get_jk

    # TODO: finish these two functions
    def get_j(self, mol=None, dm=None, hermi=0, mo_coeff=None, mo_occ=None, dm2 = None):
        vj, _, vjaux, _ =  self.get_jk(mol, dm, with_k=False, mo_coeff=mo_coeff, mo_occ=mo_occ, dm2=dm2)
        return vj, vjaux

    def get_k(self, mol=None, dm=None, hermi=0, mo_coeff=None, mo_occ=None, dm2 = None):
        _, vk, _, vkaux = self.get_jk(mol, dm, with_j=False, mo_coeff=mo_coeff, mo_occ=mo_occ, dm2=dm2)
        return vk, vkaux


    def get_veff(self, mol=None, dm=None):
        mo_a, mo_b = self.base.mo_coeff
        mo_occa, mo_occb = self.base.mo_occ
        vj0, vk0, vjaux0, vkaux0 = self.get_jk(mol, dm[0], mo_coeff=mo_a, mo_occ=mo_occa)
        vj1, vk1, vjaux1, vkaux1 = self.get_jk(mol, dm[1], mo_coeff=mo_b, mo_occ=mo_occb)
        vj0_m1, vjaux0_m1 = self.get_j(mol, dm[0], mo_coeff=mo_a, mo_occ=mo_occa, dm2=dm[1])
        vj1_m0, vjaux1_m0 = self.get_j(mol, dm[1], mo_coeff=mo_b, mo_occ=mo_occb, dm2=dm[0])
        vhf = vj0 + vj1 + vj0_m1 + vj1_m0 - vk0 - vk1
        if self.auxbasis_response:
            e1_aux = vjaux0 + vjaux1 + vjaux0_m1 + vjaux1_m0 - vkaux0 - vkaux1
            logger.debug1(self, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
        else:
            e1_aux = None
        vhf = tag_array(vhf, aux=e1_aux)
        return vhf


    def extra_force(self, atom_id, envs):
        if self.auxbasis_response:
            return envs['dvhf'].aux[atom_id]
        else:
            return 0

Grad = Gradients