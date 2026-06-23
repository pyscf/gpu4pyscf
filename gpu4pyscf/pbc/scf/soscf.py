# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

'''
Second order SCF solver for PBC systems
'''

import cupy as cp
from pyscf import lib
from gpu4pyscf.lib.cupy_helper import contract, transpose_sum
from gpu4pyscf.scf import soscf as mol_soscf
from gpu4pyscf.scf.soscf import _CIAH_SOSCF

def gen_g_hop_rhf(mf, mo_coeff, mo_occ, fock_ao, h1e=None):
    nkpts, nmo = mo_occ.shape
    fock = contract('kpq,kpi->kiq', fock_ao, mo_coeff.conj())
    fock = contract('kiq,kqj->kij', fock, mo_coeff)

    omask = mo_occ > 0
    vmask = mo_occ == 0
    nocc = cp.count_nonzero(omask, axis=1).get()
    nvir = nmo - nocc
    aligned = all(nocc[0] == nocc)
    if aligned:
        orbo = mo_coeff.transpose(0,2,1)[omask].transpose(0,2,1)
        orbv = mo_coeff.transpose(0,2,1)[vmask].transpose(0,2,1)
    else:
        orbo = [mo_coeff[k][:,omask[k]] for k in range(nkpts)]
        orbv = [mo_coeff[k][:,vmask[k]] for k in range(nkpts)]

    foo = _split_matrices(fock[omask[:,:,None] & omask[:,None]], nocc)
    fvv = _split_matrices(fock[vmask[:,:,None] & vmask[:,None]], nvir)

    g = fock[vmask[:,:,None] & omask[:,None]] * 2

    diag = fock.diagonal(axis1=1, axis2=2).real
    h_diag = diag[:,:,None] - diag[:,None,:]
    h_diag = h_diag[vmask[:,:,None] & omask[:,None]] * 2

    vind = mf.gen_response(mo_coeff, mo_occ, singlet=None, hermi=1)

    def h_op(x1):
        x1 = _split_matrices(x1, nvir, nocc)
        if aligned:
            dm1 = contract('kij,qj->kiq', x1*2, orbo.conj())
            dm1 = contract('kiq,pi->kpq', dm1, orbv)
            dm1 = transpose_sum(dm1)
            v1 = vind(dm1)
            x2 = contract('kpq,kqj->kpj', v1, orbo)
            x2 = contract('kpj,kpi->kij', x2, orbv.conj()) * 2
            x2 += contract('kps,ksq->kpq', fvv, x1*2)
            x2 -= contract('kps,krp->krs', foo, x1*2)
            return x2.ravel()
        else:
            dm1 = cp.array([orbv[k].dot(x1[k]*2).dot(orbo[k].conj().T)
                            for k in range(nkpts)])
            dm1 = transpose_sum(dm1)
            v1 = vind(dm1)
            x2 = [None] * nkpts
            for k in range(nkpts):
                x2[k] = orbv[k].conj().T.dot(v1[k].dot(orbo[k])) * 2
                x2[k] += cp.einsum('ps,sq->pq', fvv[k], x1[k]) * 2
                x2[k] -= cp.einsum('ps,rp->rs', foo[k], x1[k]) * 2
            return cp.hstack([x.ravel() for x in x2])
    return g, h_op, h_diag

def gen_g_hop_uhf(mf, mo_coeff, mo_occ, fock_ao, h1e=None):
    nkpts, nao, nmo = mo_coeff.shape[1:]
    fock = contract('skpq,skpi->skiq', fock_ao, mo_coeff.conj())
    fock = contract('skiq,skqj->skij', fock, mo_coeff)

    omaska = mo_occ[0] > 0
    omaskb = mo_occ[1] > 0
    vmaska = mo_occ[0] == 0
    vmaskb = mo_occ[1] == 0
    nocca = cp.count_nonzero(omaska, axis=1).get()
    noccb = cp.count_nonzero(omaskb, axis=1).get()
    nvira = nmo - nocca
    nvirb = nmo - noccb

    aligned = all(nocca[0] == nocca) and all(noccb[0] == noccb)
    if aligned:
        orboa = mo_coeff[0].transpose(0,2,1)[omaska].transpose(0,2,1)
        orbva = mo_coeff[0].transpose(0,2,1)[vmaska].transpose(0,2,1)
        orbob = mo_coeff[1].transpose(0,2,1)[omaskb].transpose(0,2,1)
        orbvb = mo_coeff[1].transpose(0,2,1)[vmaskb].transpose(0,2,1)
    else:
        orboa = [mo_coeff[0,k][:,omaska[k]] for k in range(nkpts)]
        orbva = [mo_coeff[0,k][:,vmaska[k]] for k in range(nkpts)]
        orbob = [mo_coeff[1,k][:,omaskb[k]] for k in range(nkpts)]
        orbvb = [mo_coeff[1,k][:,vmaskb[k]] for k in range(nkpts)]

    fooa = _split_matrices(fock[0][omaska[:,:,None] & omaska[:,None]], nocca)
    fvva = _split_matrices(fock[0][vmaska[:,:,None] & vmaska[:,None]], nvira)
    foob = _split_matrices(fock[1][omaskb[:,:,None] & omaskb[:,None]], noccb)
    fvvb = _split_matrices(fock[1][vmaskb[:,:,None] & vmaskb[:,None]], nvirb)

    vo_idxa = cp.where(vmaska[:,:,None] & omaska[:,None])[0]
    vo_idxb = cp.where(vmaskb[:,:,None] & omaskb[:,None])[0]
    tot_vopair_a = len(vo_idxa)
    vo_idx = cp.append(vo_idxa, vo_idxb + tot_vopair_a)

    g = fock.ravel()[vo_idx]

    diag = fock.diagonal(axis1=2, axis2=3).real
    h_diag = diag[:,:,:,None] - diag[:,:,None,:]
    h_diag = h_diag.ravel()[vo_idx]

    vind = mf.gen_response(mo_coeff, mo_occ, hermi=1)

    def h_op(x1):
        x1a = _split_matrices(x1[:tot_vopair_a], nvira, nocca)
        x1b = _split_matrices(x1[tot_vopair_a:], nvirb, noccb)
        dm1 = cp.empty((2,nkpts,nao,nao), dtype=x1.dtype)
        if aligned:
            dm1a = contract('kij,qj->kiq', x1a*2, orboa.conj())
            dm1a = contract('kiq,pi->kpq', dm1a, orbva, out=dm1[0])
            dm1b = contract('kij,qj->kiq', x1b*2, orbob.conj())
            dm1b = contract('kiq,pi->kpq', dm1b, orbvb, out=dm1[1])
        else:
            for k in range(nkpts):
                orbva[k].dot(x1a[k]).dot(orboa[k].conj().T, out=dm1[0,k])
                orbvb[k].dot(x1b[k]).dot(orbob[k].conj().T, out=dm1[1,k])

        transpose_sum(dm1.reshape(2*nkpts,nao,nao), inplace=True)
        v1 = vind(dm1)

        if aligned:
            x2a = contract('kpq,kqj->kpj', v1[0], orboa)
            x2a = contract('kpj,kpi->kij', x2a, orbva.conj()) * 2
            x2a += contract('kps,ksq->kpq', fvva, x1a*2)
            x2a -= contract('kps,krp->krs', fooa, x1a*2)
            x2b = contract('kpq,kqj->kpj', v1[1], orbob)
            x2b = contract('kpj,kpi->kij', x2b, orbvb.conj()) * 2
            x2b += contract('kps,ksq->kpq', fvvb, x1b*2)
            x2b -= contract('kps,krp->krs', foob, x1b*2)
            x2 = cp.append(x2a.ravel(), x2b.ravel())
        else:
            x2a = [None] * nkpts
            x2b = [None] * nkpts
            for k in range(nkpts):
                x2a[k]  = orbva[k].conj().T.dot(v1[0][k].dot(orboa[k]))
                x2b[k]  = orbvb[k].conj().T.dot(v1[1][k].dot(orbob[k]))
                x2a[k] += cp.einsum('ps,sq->pq', fvva[k], x1a[k])
                x2a[k] -= cp.einsum('ps,rp->rs', fooa[k], x1a[k])
                x2b[k] += cp.einsum('ps,sq->pq', fvvb[k], x1b[k])
                x2b[k] -= cp.einsum('ps,rp->rs', foob[k], x1b[k])
            x2 = cp.hstack([x.ravel() for x in (x2a+x2b)])
        return x2
    return g, h_op, h_diag

def _split_matrices(a, dim1_list, dim2_list=None):
    if dim2_list is None:
        dim2_list = dim1_list

    n1 = dim1_list[0]
    if all(dim1_list == n1):
        return a.reshape(-1, n1, dim2_list[0])

    z = []
    p1 = 0
    for n1, n2 in zip(dim1_list, dim2_list):
        p0, p1 = p1, p1 + n1 * n2
        z.append(a[p0:p1].reshape(n1,n2))
    return z

class _SecondOrderKRHF(_CIAH_SOSCF):
    gen_g_hop = gen_g_hop_rhf

    def update_rotate_matrix(self, dx, mo_occ, u0=None, mo_coeff=None):
        nkpts, nmo = mo_occ.shape
        omask = mo_occ > 0
        vmask = mo_occ == 0
        dr = cp.zeros((nkpts, nmo, nmo), dtype=dx.dtype)
        dr[vmask[:,:,None] & omask[:,None]] = dx
        dr = dr - dr.conj().transpose(0,2,1)

        u = dr
        for k in range(nkpts):
            u[k] = mol_soscf.expm(dr[k])

        if u0 is not None:
            u = contract('kpq,kqr->kpr', u0, u)
        return u

    def rotate_mo(self, mo_coeff, u, log=None):
        return contract('kpq,kqr->kpr', mo_coeff, u)

class _SecondOrderKUHF(_CIAH_SOSCF):
    gen_g_hop = gen_g_hop_uhf

    def update_rotate_matrix(self, dx, mo_occ, u0=None, mo_coeff=None):
        nkpts, nmo = mo_occ.shape[1:]
        omaska = mo_occ[0] > 0
        omaskb = mo_occ[1] > 0
        vmaska = mo_occ[0] == 0
        vmaskb = mo_occ[1] == 0
        nocca = cp.count_nonzero(omaska, axis=1).get()
        nvira = nmo - nocca
        tot_vopair_a = nocca.dot(nvira)
        dr = cp.zeros((2, nkpts, nmo, nmo), dtype=dx.dtype)
        dr[0][vmaska[:,:,None] & omaska[:,None]] = dx[:tot_vopair_a]
        dr[1][vmaskb[:,:,None] & omaskb[:,None]] = dx[tot_vopair_a:]
        dr = dr - dr.conj().transpose(0,1,3,2)

        u = dr
        for s in range(2):
            for k in range(nkpts):
                u[s,k] = mol_soscf.expm(dr[s,k])

        if u0 is not None:
            u = contract('skpq,skqr->skpr', u0, u)
        return u

    def rotate_mo(self, mo_coeff, u, log=None):
        return contract('skpq,skqr->skpr', mo_coeff, u)

def newton(mf):
    from gpu4pyscf.pbc import scf as pscf
    if not isinstance(mf, pscf.khf.KSCF):
        # Note for single k-point other than gamma point (mf.kpt != 0) mf object,
        # orbital hessian is approximated by gamma point hessian.
        return mol_soscf.newton(mf)

    if isinstance(mf, _CIAH_SOSCF):
        return mf

    if isinstance(mf, pscf.kuhf.KUHF):
        cls = _SecondOrderKUHF
    elif isinstance(mf, pscf.khf.KRHF):
        cls = _SecondOrderKRHF
    else:
        raise NotImplementedError(f'SOSCF solver for {mf.__class__}')
    return lib.set_class(cls(mf), (cls, mf.__class__))
