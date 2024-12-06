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

import copy
import numpy
import cupy
from cupyx.scipy.linalg import solve_triangular
from pyscf import scf, gto
from gpu4pyscf.df import int3c2e, df
from gpu4pyscf.lib.cupy_helper import tag_array, contract, cholesky
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf import __config__
from gpu4pyscf.lib import logger
from gpu4pyscf.df.grad.jk import get_rhoj_rhok

LINEAR_DEP_THRESHOLD = df.LINEAR_DEP_THR
MIN_BLK_SIZE = getattr(__config__, 'min_ao_blksize', 128)
ALIGNED = getattr(__config__, 'ao_aligned', 64)

def _gen_metric_solver(int2c, decompose_j2c='CD', lindep=LINEAR_DEP_THRESHOLD):
    ''' generate a solver to solve Ax = b, RHS must be in (n,....) '''
    if decompose_j2c.upper() == 'CD':
        try:
            j2c = cholesky(int2c, lower=True)
            def j2c_solver(v):
                return solve_triangular(j2c, v, overwrite_b=False)
            return j2c_solver

        except Exception:
            pass

    w, v = cupy.linalg.eigh(int2c)
    mask = w > lindep
    v1 = v[:,mask]
    j2c = cupy.dot(v1/w[mask], v1.conj().T)
    def j2c_solver(b): # noqa: F811
        return j2c.dot(b.reshape(j2c.shape[0],-1)).reshape(b.shape)
    return j2c_solver

def get_jk(mf_grad, mol=None, dm0=None, hermi=0, with_j=True, with_k=True, omega=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.

    NOTE: This function is incompatible to the one implemented in PySCF CPU version.
    In the CPU version, get_jk returns the first order derivatives of J/K matrices.
    '''
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
            with_df = dfobj._rsh_df[key] = dfobj.copy().reset()

    auxmol = with_df.auxmol
    if not hasattr(with_df, 'intopt') or with_df._cderi is None:
        with_df.build(omega=omega)
    intopt = with_df.intopt
    naux = with_df.naux

    log = logger.new_logger(mol, mol.verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    if isinstance(mf_grad.base, scf.rohf.ROHF):
        raise NotImplementedError()
    mo_coeff = cupy.asarray(mf_grad.base.mo_coeff)
    mo_occ = cupy.asarray(mf_grad.base.mo_occ)

    dm = intopt.sort_orbitals(dm0, axis=[0,1])
    orbo = mo_coeff[:,mo_occ>0] * mo_occ[mo_occ>0] ** 0.5
    mo_coeff = None
    orbo = intopt.sort_orbitals(orbo, axis=[0])

    rhoj, rhok = get_rhoj_rhok(with_df, dm, orbo, with_j=with_j, with_k=with_k)
    
    # (d/dX P|Q) contributions
    if omega and omega > 1e-10:
        with auxmol.with_range_coulomb(omega):
            int2c_e1 = auxmol.intor('int2c2e_ip1')
    else:
        int2c_e1 = auxmol.intor('int2c2e_ip1')
    int2c_e1 = cupy.asarray(int2c_e1)

    auxslices = auxmol.aoslice_by_atom()
    aux_cart2sph = intopt.aux_cart2sph
    low = with_df.cd_low
    low_t = low.T.copy()
    if with_j:
        if low.tag == 'eig':
            rhoj = cupy.dot(low_t.T, rhoj)
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
        vjaux_2c = cupy.array([-vjaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]])
        rhoj = vjaux = tmp = None
    if with_k:
        nocc = orbo.shape[-1]
        if low.tag == 'eig':
            rhok = contract('pq,qij->pij', low_t.T, rhok)
        elif low.tag == 'cd':
            #rhok = solve_triangular(low_t, rhok, lower=False)
            rhok = solve_triangular(low_t, rhok.reshape(naux, -1), lower=False, overwrite_b=True).reshape(naux, nocc, nocc)
        tmp = contract('pij,qij->pq', rhok, rhok)
        tmp = intopt.unsort_orbitals(tmp, aux_axis=[0,1])
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

    nao_cart = intopt._sorted_mol.nao
    block_size = with_df.get_blksize(nao=nao_cart)

    intopt = int3c2e.VHFOpt(mol, auxmol, 'int2e')
    intopt.build(mf.direct_scf_tol, diag_block_with_triu=True, aosym=False,
                 group_size_aux=block_size)#, group_size=block_size)
    dm_cart = dm
    orbo_cart = orbo
    if not mol.cart:
        # sph2cart for ao
        cart2sph = intopt.cart2sph
        orbo_cart = cart2sph @ orbo
        dm_cart = cart2sph @ dm @ cart2sph.T

    dm = orbo = None
    vj = vk = rhoj_tmp = rhok_tmp = None
    vjaux = vkaux = None

    naux_cart = intopt._sorted_auxmol.nao
    if with_j:
        vj = cupy.zeros((3,nao_cart), order='C')
        vjaux = cupy.zeros((3,naux_cart))
    if with_k:
        vk = cupy.zeros((3,nao_cart), order='C')
        vkaux = cupy.zeros((3,naux_cart))
    cupy.get_default_memory_pool().free_all_blocks()
    t1 = log.init_timer()
    for cp_kl_id in range(len(intopt.aux_log_qs)):
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
        buf = int3c2e.get_int3c2e_ip_slice(intopt, cp_kl_id, 1)
        size = 3*(k1-k0)*nao_cart*nao_cart
        int3c_ip = buf[:size].reshape([3,k1-k0,nao_cart,nao_cart], order='C')
        rhoj_tmp0 = contract('xpji,ij->xip', int3c_ip, dm_cart)
        vj_outcore = contract('xip,p->xi', rhoj_tmp0, rhoj_cart[k0:k1])
        vk_outcore = contract('pji,xpji->xi', rhok_tmp, int3c_ip)

        buf = int3c2e.get_int3c2e_ip_slice(intopt, cp_kl_id, 2)
        int3c_ip = buf[:size].reshape([3,k1-k0,nao_cart,nao_cart], order='C')
        rhoj_tmp0 = contract('xpji,ji->xp', int3c_ip, dm_cart)
        vjaux_outcore = contract('xp,p->xp', rhoj_tmp0, rhoj_cart[k0:k1])
        vkaux_outcore = contract('xpji,pji->xp', int3c_ip, rhok_tmp)
        '''
        vj_tmp, vk_tmp = int3c2e.get_int3c2e_ip_jk(intopt, cp_kl_id, 'ip1', rhoj_tmp, rhok_tmp, dm_cart, omega=omega)
        if with_j: vj += vj_tmp
        if with_k: vk += vk_tmp
        vj_tmp, vk_tmp = int3c2e.get_int3c2e_ip_jk(intopt, cp_kl_id, 'ip2', rhoj_tmp, rhok_tmp, dm_cart, omega=omega)
        if with_j: vjaux[:, k0:k1] = vj_tmp
        if with_k: vkaux[:, k0:k1] = vk_tmp

        rhoj_tmp = rhok_tmp = vj_tmp = vk_tmp = None
        t1 = log.timer_debug1(f'calculate {cp_kl_id:3d} / {len(intopt.aux_log_qs):3d}, {k1-k0:3d} slices', *t1)
    
    # NOTE: vj and vk are still in cartesian
    _sorted_mol = intopt._sorted_mol
    natm = _sorted_mol.natm
    ao2atom = numpy.zeros([nao_cart, natm])
    ao_loc = _sorted_mol.ao_loc
    for ibas, iatm in enumerate(_sorted_mol._bas[:,gto.ATOM_OF]):
        ao2atom[ao_loc[ibas]:ao_loc[ibas+1],iatm] = 1
    ao2atom = cupy.asarray(ao2atom)
    if with_j:
        vj = -ao2atom.T @ vj.T
    if with_k:
        vk = -ao2atom.T @ vk.T
    t0 = log.timer_debug1('(di,j|P) and (i,j|dP)', *t0)

    _sorted_auxmol = intopt._sorted_auxmol
    natm = _sorted_auxmol.natm
    aux2atom = numpy.zeros([naux_cart, natm])
    ao_loc = _sorted_auxmol.ao_loc
    for ibas, iatm in enumerate(_sorted_auxmol._bas[:,gto.ATOM_OF]):
        aux2atom[ao_loc[ibas]:ao_loc[ibas+1],iatm] = 1
    aux2atom = cupy.asarray(aux2atom)
    if with_j:
        vjaux_3c = aux2atom.T @ vjaux.T
        vjaux = vjaux_2c - vjaux_3c

    if with_k:
        vkaux_3c = aux2atom.T @ vkaux.T
        vkaux = vkaux_2c - vkaux_3c
    
    return vj, vk, vjaux, vkaux


class Gradients(rhf_grad.Gradients):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'with_df', 'auxbasis_response'}
    def __init__(self, mf):
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        rhf_grad.Gradients.__init__(self, mf)

    auxbasis_response = True
    get_jk = get_jk

    def check_sanity(self):
        assert isinstance(self.base, df.df_jk._DFHF)

    def get_j(self, mol=None, dm=None, hermi=0):
        vj, _, vjaux, _ = self.get_jk(mol, dm, with_k=False)
        return vj, vjaux

    def get_k(self, mol=None, dm=None, hermi=0):
        _, vk, _, vkaux = self.get_jk(mol, dm, with_j=False)
        return vk, vkaux

    def get_veff(self, mol=None, dm=None, verbose=None):
        vj, vk, vjaux, vkaux = self.get_jk(mol, dm)
        vhf = vj - vk*.5
        if self.auxbasis_response:
            e1_aux = vjaux - vkaux*.5
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
