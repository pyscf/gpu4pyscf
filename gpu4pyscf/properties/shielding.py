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
from pyscf.data import nist
from pyscf.scf import _vhf, jk
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import contract, take_last2d, add_sparse
from gpu4pyscf.scf import cphf

def gen_vind(mf, mo_coeff, mo_occ):
    """get the induced potential. This is the same as contract the mo1 with the kernel.

    Args:
        mf: mean field object
        mo_coeff (numpy.array): mo coefficients
        mo_occ (numpy.array): mo_coefficients

    Returns:
        fx (function): a function to calculate the induced potential with the input as the mo1.
    """
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:, mo_occ > 0]
    mvir = mo_coeff[:, mo_occ == 0]
    nocc = mocc.shape[1]
    nvir = nmo - nocc
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(
            mf.xc, spin=mf.mol.spin)

    def fx(mo1):
        mo1 = mo1.reshape(-1, nvir, nocc)  # * the saving pattern
        mo1_mo_real = contract('nai,ua->nui', mo1, mvir)
        dm1 = 2*contract('nui,vi->nuv', mo1_mo_real, mocc.conj())
        dm1 -= dm1.transpose(0, 2, 1)
        if hasattr(mf,'with_df'):
            v1 = cupy.empty((3, nao, nao))
            for i in range(3):
                v1[i] =+mf.get_jk(mf.mol, dm1[i], hermi=2, with_j=False)[1]*0.5*hyb
        else:
            v1 = np.empty((3, nao, nao))
            for i in range(3):
                v1[i] = -jk.get_jk(mf.mol, dm1[i].get(), 'ijkl,jk->il')*0.5*hyb
            v1 = cupy.array(v1)
        tmp = contract('nuv,vi->nui', v1, mocc)
        v1vo = contract('nui,ua->nai', tmp, mvir.conj())

        return v1vo

    return fx


def nr_rks(ni, mol, grids, xc_code, dms):

    xctype = ni._xc_type(xc_code)
    mo_coeff = getattr(dms, 'mo_coeff', None)
    mo_occ = getattr(dms, 'mo_occ', None)
    nao = mo_coeff.shape[1]
    
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol

    coeff = cupy.asarray(opt.coeff)
    nao, nao0 = coeff.shape
    dms = cupy.asarray(dms).reshape(-1,nao0,nao0)
    dms = take_last2d(dms, opt.ao_idx)
    mo_coeff = mo_coeff[opt.ao_idx]

    vmat = cupy.zeros((3, nao, nao))
    if xctype == 'LDA':
        ao_deriv = 0
    else:
        ao_deriv = 1

    for ao, index, weight, coords in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
        mo_coeff_mask = mo_coeff[index,:]
        rho = numint.eval_rho2(_sorted_mol, ao, mo_coeff_mask, mo_occ, None, xctype)
        vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[1]
        if xctype == 'LDA':
            wv = weight * vxc[0]
            giao = _sorted_mol.eval_gto('GTOval_ig', coords.get(), comp=3)  # (#C(0 1) g) |AO>
            giao = cupy.array(giao)
            giao_aux = giao[:,:,index]
            for idirect in range(3):
                vtmp = contract('pu,p,vp->uv', giao_aux[idirect], wv, ao)
                vtmp = cupy.ascontiguousarray(vtmp)
                add_sparse(vmat[idirect], vtmp, index)
            
        elif xctype == 'GGA':
            wv = vxc * weight
            giao = _sorted_mol.eval_gto('GTOval_ig', coords.get(), comp=3)
            giao_nabla = _sorted_mol.eval_gto('GTOval_ipig', coords.get()).reshape(3, 3, -1, nao)
            giao = cupy.array(giao)
            giao_nabla = cupy.array(giao_nabla)
            giao_aux = giao[:,:,index]
            giao_nabla_aux = giao_nabla[:,:,:,index]
            for idirect in range(3):
                # * write like the gpu4pyscf numint part, but explicitly use the einsum
                aow = contract('pn,p->pn', giao_aux[idirect], wv[0])
                aow += contract('xpn,xp->pn', giao_nabla_aux[:, idirect, :, :], wv[1:4])
                vtmp = contract('pn,mp->nm', aow, ao[0])
                vtmp = cupy.ascontiguousarray(vtmp)
                add_sparse(vmat[idirect], vtmp, index)
                aow = contract('pn,xp->xpn', giao_aux[idirect], wv[1:4])
                vtmp = contract('xpn,xmp->nm', aow, ao[1:4])
                vtmp = cupy.ascontiguousarray(vtmp)
                add_sparse(vmat[idirect], vtmp, index)

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')
        elif xctype == 'MGGA':
            raise NotImplementedError('Meta-GGA')
        elif xctype == 'HF':
            pass
        else:
            raise NotImplementedError(
                f'numint.nr_rks for functional {xc_code}')

        ao = None

    vmat = take_last2d(vmat, opt.rev_ao_idx)

    if numint.FREE_CUPY_CACHE:
        dms = None
        cupy.get_default_memory_pool().free_all_blocks()

    return (vmat - vmat.transpose(0, 2, 1))


# TODO: this can be modified into jk, not _vhf
def get_jk(mol, dm0):
    # J = Im[(i i|\mu g\nu) + (i gi|\mu \nu)] = -i (i i|\mu g\nu)
    # K = Im[(\mu gi|i \nu) + (\mu i|i g\nu)]
    #   = [-i (\mu g i|i \nu)] - h.c.   (-h.c. for anti-symm because of the factor -i)
    intor = mol._add_suffix('int2e_ig1')
    vj, vk = _vhf.direct_mapdm(intor,  # (g i,j|k,l)
                               'a4ij', ('lk->s1ij', 'jk->s1il'),
                               dm0.get(), 3,  # xyz, 3 components
                               mol._atm, mol._bas, mol._env)
    vk = vk - np.swapaxes(vk, -1, -2)
    return -cupy.array(vj), -cupy.array(vk)


def get_vxc(mf, dm0):
    vxc = nr_rks(mf._numint, mf.mol, mf.grids, mf.xc, mf.make_rdm1())
    # ! imaginary part
    vj, vk = get_jk(mf.mol, dm0)
    if not mf._numint.libxc.is_hybrid_xc(mf.xc):
        vk = None
        vxc += vj
    else:
        omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(
            mf.xc, spin=mf.mol.spin)
        vxc += vj - vk*hyb*0.5
    return vxc


def get_h1ao(mf):
    dm0 = mf.make_rdm1()
    # ! imaginary part
    h1ao = -0.5*mf.mol.intor('int1e_giao_irjxp')
    h1ao += -mf.mol.intor('int1e_igkin')
    h1ao += -mf.mol.intor('int1e_ignuc')
    h1ao = cupy.array(h1ao)
    h1ao += get_vxc(mf, dm0)

    return h1ao


def eval_shielding(mf):

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    mo_coeff = cupy.array(mo_coeff)
    mo_occ = cupy.array(mo_occ)
    mo_energy = cupy.array(mo_energy)

    nao = mo_energy.shape[0]
    idx_occ = mo_occ > 0
    idx_vir = mo_occ == 0
    fx = gen_vind(mf, mo_coeff, mo_occ)
    mocc = mo_coeff[:, idx_occ]
    mvir = mo_coeff[:, idx_vir]
    s1ao = -mf.mol.intor('int1e_igovlp')
    s1ao = cupy.array(s1ao)
    dm0 = mf.make_rdm1()
    natom = mf.mol.natm

    h1ao = get_h1ao(mf)
    tmp = contract('xuv,ua->xav', s1ao, mvir)
    s1ai = contract('xav,vi->xai', tmp, mocc)
    tmp = contract('ned,ue->nud', s1ao, dm0)
    s1dm = contract('nud,dv->nuv', tmp, dm0)*0.5
    tmp = contract('xpq,pi->xiq', s1ao, mocc)
    s1jk = -contract('xiq,qj->xij', tmp, mocc)*0.5
    tmp = contract('nai,ua->nui', s1jk, mocc)
    s1jkdm1 = contract('nui,vi->nuv', tmp, mocc.conj())*2
    s1jkdm1 -= s1jkdm1.transpose(0, 2, 1)
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(
            mf.xc, spin=mf.mol.spin)
    if hasattr(mf,'with_df'):
        vk2 = cupy.empty((3, nao, nao))
        for i in range(3):
            vk2[i] = +mf.get_jk(mf.mol, s1jkdm1[i], hermi=2, with_j=False)[1]*0.5*hyb
    
    else:
        vk2 = np.empty((3, nao, nao))
        for i in range(3):
            vk2[i] = -jk.get_jk(mf.mol, s1jkdm1[i].get(), 'ijkl,jk->il')*0.5*hyb
        vk2 = cupy.array(vk2)
    h1ao += vk2
    tmp = contract('xuv,ua->xav', h1ao, mvir)
    veff_ai = contract('xav,vi->xai', tmp, mocc)
    veff_ai -= contract('xai,i->xai', s1ai, mo_energy[idx_occ])

    mo1 = cphf.solve(fx, mo_energy, mo_occ, veff_ai, max_cycle=20, tol=1e-10)[0]

    shielding_d = cupy.empty((natom, 3, 3))
    shielding_p = cupy.empty((natom, 3, 3))

    for atm_id in range(natom):
        mf.mol.set_rinv_origin(mf.mol.atom_coord(atm_id))
        # ! imaginary part (p* | rinv cross p | )
        int_h01 = -mf.mol.intor('int1e_prinvxp')
        # ! imaginary part (g | nabla-rinv cross p |)
        int_h01_giao = mf.mol.intor('int1e_a01gp').reshape(
            3, 3, nao, nao)
        int_h11 = mf.mol.intor('int1e_giao_a11part').reshape(
            3, 3, nao, nao)  # ! (-.5 | nabla-rinv | r)
        int_h11 = cupy.array(int_h11)
        int_h01 = cupy.array(int_h01)
        int_h01_giao = cupy.array(int_h01_giao)
        int_h11_diag = int_h11[0, 0] + int_h11[1, 1] + int_h11[2, 2]
        int_h11[0, 0] -= int_h11_diag
        int_h11[1, 1] -= int_h11_diag
        int_h11[2, 2] -= int_h11_diag

        tmp = contract('ua,xai->xui', mvir, mo1)
        dm1 = contract('xui,vi->xvu', tmp, mocc)
        dm1 *= 2
        shielding_p[atm_id] = contract('yuv,xvu->xy', int_h01, dm1)*2.0
        shielding_p[atm_id] += contract('yuv,xvu->xy', int_h01, s1dm)
        shielding_d[atm_id] = contract('xyuv,vu->xy', int_h11, dm0)
        shielding_d[atm_id] += contract('xyuv,vu->xy', int_h01_giao, dm0)
    ppm = nist.ALPHA**2 * 1e6
    return shielding_d*ppm, shielding_p*ppm
