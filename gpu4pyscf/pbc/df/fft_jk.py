#!/usr/bin/env python
#
# Copyright 2024 The GPU4PySCF Developers. All Rights Reserved.
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

'''
JK with GPW
'''

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero, member
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc import tools

__all__ = [
    'get_j_kpts', 'get_k_kpts', 'get_jk', 'get_j', 'get_k',
    'get_j_e1_kpts', 'get_k_e1_kpts'
]

def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    cell = mydf.cell
    mesh = mydf.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1

    ni = mydf._numint
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    coulG = tools.get_coulG(cell, mesh=mesh)
    ngrids = len(coulG)

    if hermi == 1 or is_zero(kpts):
        vR = cp.zeros((nset,ngrids))
        ao_ks = ni.eval_ao(cell, mydf.grids.coords, kpts)
        for i in range(nset):
            rhoR = ni.eval_rho(cell, ao_ks, dm_kpts[i], hermi=hermi).real
            rhoG = tools.fft(rhoR, mesh)
            vG = coulG * rhoG
            vR[i] = tools.ifft(vG, mesh).real
    else:
        vR = cp.zeros((nset,ngrids), dtype=np.complex128)
        ao_ks = ni.eval_ao(cell, mydf.grids.coords, kpts)
        for i in range(nset):
            rhoR = ni.eval_rho(cell, ao_ks, dm_kpts[i], hermi=hermi)
            rhoG = tools.fft(rhoR, mesh)
            vG = coulG * rhoG
            vR[i] = tools.ifft(vG, mesh)

    vR *= cell.vol / ngrids
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    if is_zero(kpts_band):
        vj_kpts = cp.zeros((nset,nband,nao,nao))
    else:
        vj_kpts = cp.zeros((nset,nband,nao,nao), dtype=np.complex128)

    if input_band is not None:
        ao_ks = ni.eval_ao(cell, mydf.grids.coords, kpts_band)
    for k, ao in enumerate(ao_ks):
        for i in range(nset):
            aow = ao * vR[i,:,None]
            vj_kpts[i,k] += ao.conj().T.dot(aow)

    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point
        kpts : (nkpts, 3) ndarray

    Kwargs:
        hermi : int
            Whether K matrix is hermitian

            | 0 : not hermitian and not symmetric
            | 1 : hermitian

        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray
        or list of vj and vk if the input dm_kpts is a list of DMs
    '''
    cell = mydf.cell
    mesh = mydf.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1
    coords = mydf.grids.coords
    ngrids = coords.shape[0]

    if getattr(dm_kpts, 'mo_coeff', None) is not None:
        mo_coeff = dm_kpts.mo_coeff
        mo_occ   = dm_kpts.mo_occ
    else:
        mo_coeff = None

    ni = mydf._numint
    kpts = np.asarray(kpts)
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = 1./nkpts * (cell.vol/ngrids)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if is_zero(kpts_band) and is_zero(kpts):
        vk_kpts = cp.zeros((nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = cp.zeros((nset,nband,nao,nao), dtype=np.complex128)

    ao2_kpts = ni.eval_ao(cell, coords, kpts=kpts)
    if input_band is None:
        ao1_kpts = ao2_kpts
    else:
        ao1_kpts = ni.eval_ao(cell, coords, kpts=kpts_band)

    if mo_coeff is not None and nset == 1:
        mo2_kpts = [
            ao.dot(mo[:,occ>0] * occ[occ>0]**.5)
            for occ, mo, ao in zip(mo_occ, mo_coeff, ao2_kpts)]
        ao2_kpts = mo2_kpts
    else:
        mo2_kpts = None

    vR_dm = cp.empty((nset,nao,ngrids), dtype=vk_kpts.dtype)
    blksize = 32

    for k2, ao2 in enumerate(ao2_kpts):
        ao2T = ao2.T
        kpt2 = kpts[k2]
        naoj = ao2.shape[1]
        if mo2_kpts is None:
            ao_dms = [dms[i,k2].dot(ao2T.conj()) for i in range(nset)]
        else:
            ao_dms = [ao2T.conj()]

        for k1, ao1 in enumerate(ao1_kpts):
            ao1T = ao1.T
            kpt1 = kpts_band[k1]

            # If we have an ewald exxdiv, we add the G=0 correction near the
            # end of the function to bypass any discretization errors
            # that arise from the FFT.
            if exxdiv == 'ewald':
                coulG = tools.get_coulG(cell, kpt2-kpt1, False, mydf, mesh)
            else:
                coulG = tools.get_coulG(cell, kpt2-kpt1, exxdiv, mydf, mesh)
            if is_zero(kpt1-kpt2):
                expmikr = cp.array(1.)
            else:
                expmikr = cp.exp(-1j * coords.dot(cp.asarray(kpt2-kpt1)))

            for p0, p1 in lib.prange(0, nao, blksize):
                rho1 = contract('ig,jg->ijg', ao1T[p0:p1].conj()*expmikr, ao2T)
                vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                rho1 = None
                vG *= coulG
                vR = tools.ifft(vG, mesh).reshape(p1-p0,naoj,ngrids)
                vG = None
                if vk_kpts.dtype == np.double:
                    vR = vR.real
                for i in range(nset):
                    vR_dm[i,p0:p1] = contract('ijg,jg->ig', vR, ao_dms[i])
                vR = None
            vR_dm *= expmikr.conj()

            for i in range(nset):
                vk_kpts[i,k1] += weight * vR_dm[i].dot(ao1)

    # Function _ewald_exxdiv_for_G0 to add back in the G=0 component to vk_kpts
    # Note in the _ewald_exxdiv_for_G0 implementation, the G=0 treatments are
    # different for 1D/2D and 3D systems.  The special treatments for 1D and 2D
    # can only be used with AFTDF/GDF/MDF method.  In the FFTDF method, 1D, 2D
    # and 3D should use the ewald probe charge correction.
    if exxdiv == 'ewald':
        vk_kpts = _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

def get_jk(mydf, dm, hermi=1, kpt=np.zeros(3), kpts_band=None,
           with_j=True, with_k=True, exxdiv=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpts_band : (3,) ndarray or (*,3) ndarray
            The "outer" primary k-point at which J and K are evaluated.

    Returns:
        The function returns one J and one K matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    dm = cp.asarray(dm, order='C')
    vj = vk = None
    if with_j:
        vj = get_j(mydf, dm, hermi, kpt, kpts_band)
    if with_k:
        vk = get_k(mydf, dm, hermi, kpt, kpts_band, exxdiv)
    return vj, vk

def get_j(mydf, dm, hermi=1, kpt=np.zeros(3), kpts_band=None):
    '''Get the Coulomb (J) AO matrix for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpts_band : (3,) ndarray or (*,3) ndarray
            The "outer" primary k-point at which J and K are evaluated.

    Returns:
        The function returns one J matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    dm = cp.asarray(dm, order='C')
    nao = dm.shape[-1]
    dm_kpts = dm.reshape(-1,1,nao,nao)
    vj = get_j_kpts(mydf, dm_kpts, hermi, kpt.reshape(1,3), kpts_band)
    if kpts_band is None:
        vj = vj[:,0,:,:]
    if dm.ndim == 2:
        vj = vj[0]
    return vj


def get_k(mydf, dm, hermi=1, kpt=np.zeros(3), kpts_band=None, exxdiv=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpts_band : (3,) ndarray or (*,3) ndarray
            The "outer" primary k-point at which J and K are evaluated.

    Returns:
        The function returns one J and one K matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    dm = cp.asarray(dm, order='C')
    nao = dm.shape[-1]
    dm_kpts = dm.reshape(-1,1,nao,nao)
    vk = get_k_kpts(mydf, dm_kpts, hermi, kpt.reshape(1,3), kpts_band, exxdiv)
    if kpts_band is None:
        vk = vk[:,0,:,:]
    if dm.ndim == 2:
        vk = vk[0]
    return vk

get_j_e1_kpts = NotImplemented
get_k_e1_kpts = NotImplemented

def _ewald_exxdiv_for_G0(cell, kpts, dms, vk, kpts_band=None):
    from pyscf.pbc.tools.pbc import madelung
    s = cp.asarray(cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
    m = madelung(cell, kpts)
    if kpts is None:
        for i,dm in enumerate(dms):
            vk[i] += m * s.dot(dm).dot(s)
    elif np.shape(kpts) == (3,):
        if kpts_band is None or is_zero(kpts_band-kpts):
            for i,dm in enumerate(dms):
                vk[i] += m * s.dot(dm).dot(s)

    elif kpts_band is None or np.array_equal(kpts, kpts_band):
        for k in range(len(kpts)):
            for i,dm in enumerate(dms):
                vk[i,k] += m * s[k].dot(dm[k]).dot(s[k])
    else:
        for k, kpt in enumerate(kpts):
            for kp in member(kpt, kpts_band.reshape(-1,3)):
                for i,dm in enumerate(dms):
                    vk[i,kp] += m * s[k].dot(dm[k]).dot(s[k])
    return vk
