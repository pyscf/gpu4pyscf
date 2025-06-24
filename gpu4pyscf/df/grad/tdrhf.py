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

import cupy
import numpy
from pyscf import gto
from cupyx.scipy.linalg import solve_triangular
from gpu4pyscf import scf as scf_gpu
from gpu4pyscf.df import int3c2e, df
from gpu4pyscf.lib.cupy_helper import tag_array, contract
from gpu4pyscf.tdscf import rhf as tdrhf
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf import __config__
from gpu4pyscf.lib import logger
from gpu4pyscf.df.grad.jk import get_rhojk_td, get_grad_vjk_td


def _decompose_rdm1_svd(dm):
    '''Decompose density matrix as U.Vh using SVD

    Args:
        dm : ndarray or sequence of ndarrays of shape (nao,nao)
            Density matrices

    Returns:
        orbol : list of ndarrays of shape (nao,*)
            Contains non-null eigenvectors of density matrix
        orbor : list of ndarrays of shape (nao,*)
            Contains orbol * eigenvalues (occupancies)
    '''
    u, s, vh = cupy.linalg.svd(dm)
    idx = cupy.abs(s)>1e-8
    return cupy.asfortranarray(u[:,idx]), cupy.asfortranarray(contract('i,ip->pi', s[idx], vh[idx]))


def get_jk(mf_grad, mol=None, dm0=None, hermi=0, with_j=True, with_k=True, omega=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.

    NOTE: This function is incompatible to the one implemented in PySCF CPU version.
    In the CPU version, get_jk returns the first order derivatives of J/K matrices.
    '''
    if mol is None: mol = mf_grad.mol
    if getattr(mf_grad, "base", None) is not None:
        if isinstance(mf_grad.base, scf_gpu.rohf.ROHF):
            raise NotImplementedError()
        elif isinstance(mf_grad.base, scf_gpu.hf.SCF):
            mf = mf_grad.base
        else:
            mf = mf_grad.base._scf 
    else:
        if isinstance(mf_grad, scf_gpu.hf.SCF):
            mf = mf_grad
        else:
            raise RuntimeError("Should not reach here.")
    if(dm0 is None): dm0 = mf.make_rdm1()
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
    
    dm = intopt.sort_orbitals(dm0, axis=[0,1])

    orbol, orbor = _decompose_rdm1_svd(dm)
    nl = orbol.shape[-1]
    nr = orbor.shape[-1]
    rhoj, rhok = get_rhojk_td(with_df, dm, orbol, orbor, with_j=with_j, with_k=with_k)
    
    # (d/dX P|Q) contributions
    if omega and omega > 1e-10:
        with auxmol.with_range_coulomb(omega):
            int2c_e1 = auxmol.intor('int2c2e_ip1')
    else:
        int2c_e1 = auxmol.intor('int2c2e_ip1')
    int2c_e1 = cupy.asarray(int2c_e1)

    rhoj_cart = rhok_cart = None
    auxslices = auxmol.aoslice_by_atom()
    aux_cart2sph = intopt.aux_cart2sph
    low = with_df.cd_low
    low_t = low.T.copy()
    ejaux = ekaux = None
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
        ejaux = cupy.array([-vjaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]])
        rhoj = vjaux = tmp = None
    if with_k:
        if low.tag == 'eig':
            rhok = contract('pq,qij->pij', low_t.T, rhok)
        elif low.tag == 'cd':
            #rhok = solve_triangular(low_t, rhok, lower=False)
            rhok = solve_triangular(low_t, rhok.reshape(naux, -1), lower=False, overwrite_b=True).reshape(naux, nl, nr)
            rhok = rhok.copy(order='C')
        tmp = contract('pij,qji->pq', rhok, rhok)
        tmp = intopt.unsort_orbitals(tmp, aux_axis=[0,1])
        vkaux = -contract('xpq,pq->xp', int2c_e1, tmp)
        ekaux = cupy.array([-vkaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]])
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
    orbol_cart = orbol
    orbor_cart = orbor
    if not mol.cart:
        # sph2cart for ao
        cart2sph = intopt.cart2sph
        orbol_cart = cart2sph @ orbol
        orbor_cart = cart2sph @ orbor
        dm_cart = cart2sph @ dm @ cart2sph.T
        
    with_df._cderi = None # release GPU memory
    ej, ek, ejaux_3c, ekaux_3c = get_grad_vjk_td(with_df, mol, auxmol, rhoj_cart, dm_cart, rhok_cart, orbol_cart, orbor_cart,
                                        with_j=with_j, with_k=with_k, omega=omega)
    if with_j:
        ej = -ej
        ejaux -= ejaux_3c
    if with_k:
        ek = -ek
        ekaux -= ekaux_3c
        if hermi == 2:
            if ekaux is not None:
                ekaux *= -1
            ek *= -1
    t0 = log.timer_debug1('(di,j|P) and (i,j|dP)', *t0)
    return ej, ek, ejaux, ekaux


def get_veff(td_grad, mol=None, dm=None, j_factor=1.0, k_factor=1.0, omega=0.0, hermi=0, verbose=None):
    if omega != 0.0:
        vj, vk, vjaux, vkaux = td_grad.get_jk(mol, dm, omega=omega, hermi=hermi)
    else:
        vj, vk, vjaux, vkaux = td_grad.get_jk(mol, dm, hermi=hermi)
    vhf = vj * j_factor - vk * .5 * k_factor
    if td_grad.auxbasis_response:
        e1_aux = vjaux * j_factor - vkaux * .5 * k_factor
        logger.debug1(td_grad, 'sum(auxbasis response) %s', e1_aux.sum(axis=0))
    else:
        e1_aux = None
    vhf = tag_array(vhf, aux=e1_aux)
    
    return vhf


class Gradients(tdrhf_grad.Gradients):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'with_df', 'auxbasis_response'}
    def __init__(self, td):
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        tdrhf_grad.Gradients.__init__(self, td)

    auxbasis_response = True
    get_jk = get_jk

    def check_sanity(self):
        assert isinstance(self.base._scf, df.df_jk._DFHF)
        assert isinstance(self.base, tdrhf.TDHF) or isinstance(self.base, tdrhf.TDA)

    get_veff = get_veff

Grad = Gradients
