from gpu4pyscf.scf import cphf
import numpy as np
from pyscf.data import nist
import cupy
from pyscf.scf import _vhf, jk
from gpu4pyscf.dft import numint
import time
from gpu4pyscf.lib.cupy_helper import contract, release_gpu_stack


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
        mo1_mo_real = cupy.einsum('nai,ua->nui', mo1, mvir)
        dm1 = 2*cupy.einsum('nui,vi->nuv', mo1_mo_real, mocc.conj())
        dm1 -= dm1.transpose(0, 2, 1)
        if hasattr(mf,'with_df'):
            v1 = cupy.zeros((3, nao, nao))
            for i in range(3):
                v1[i] =+mf.get_jk(mf.mol, dm1[i], hermi=2, with_j=False)[1]*0.5*hyb
        else:
            v1 = np.zeros((3, nao, nao))
            for i in range(3):
                v1[i] = -jk.get_jk(mf.mol, dm1[i].get(), 'ijkl,jk->il')*0.5*hyb
            v1 = cupy.array(v1)
        tmp = cupy.einsum('nuv,vi->nui', v1, mocc)
        v1vo = cupy.einsum('nui,ua->nai', tmp, mvir.conj())

        return v1vo

    return fx


def nr_rks(ni, mol, grids, xc_code, dms):

    xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    mo_coeff = getattr(dms, 'mo_coeff', None)
    mo_occ = getattr(dms, 'mo_occ', None)

    mol = opt.mol
    coeff = cupy.asarray(opt.coeff)
    nao, nao0 = coeff.shape
    dms = cupy.asarray(dms)
    dms = coeff @ dms @ coeff.T

    if mo_coeff is not None:
        mo_coeff = coeff @ mo_coeff

    def _make_rho(ao_value, dm, xctype=None):
        if mo_coeff is not None and mo_occ is not None:
            rho = numint.eval_rho2(mol, ao, mo_coeff, mo_occ, None, xctype)
        else:
            rho = numint.eval_rho(mol, ao_value, dm, xctype=xctype, hermi=1)
        return rho

    vmat = cupy.zeros((3, nao, nao))

    release_gpu_stack()
    if xctype == 'LDA':
        ao_deriv = 0
    else:
        ao_deriv = 1
    for ao, sindex, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv):
        rho = _make_rho(ao, dms, xctype=xctype)
        vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[1]
        vxc = cupy.asarray(vxc, order='C')
        if xctype == 'LDA':
            wv = weight * vxc[0]
            if numint.USE_SPARSITY == 2:
                mask = cupy.any(cupy.abs(ao) > numint.AO_THRESHOLD, axis=[1])
                ao_mask = ao[mask, :]
                giao = mol.eval_gto('GTOval_ig', coords.get(),
                                    comp=3)  # (#C(0 1) g) |AO>
                giao = cupy.array(giao)
                for idirect in range(3):
                    giao_aux = giao[idirect, :, mask]
                    aow = numint._scale_ao(giao_aux, wv)
                    vmat[idirect][cupy.ix_(mask, mask)] -= ao_mask.dot(aow.T)
            else:
                raise NotImplementedError('Not implemented yet')
        elif xctype == 'GGA':
            wv = vxc * weight
            if numint.USE_SPARSITY == 2:
                mask = cupy.any(
                    cupy.abs(ao) > numint.AO_THRESHOLD, axis=[0, 2])
                ao_mask = ao[:, mask, :]
                giao = mol.eval_gto('GTOval_ig', coords.get(), comp=3)
                giao_nabla = mol.eval_gto(
                    'GTOval_ipig', coords.get()).reshape(3, 3, -1, nao)
                giao = cupy.array(giao)
                giao_nabla = cupy.array(giao_nabla)
                for idirect in range(3):
                    # * write like the gpu4pyscf numint part, but explicitly use the einsum
                    aow = cupy.einsum(
                        'np,p->pn', giao[idirect, :, mask], wv[0])
                    tmp = giao_nabla[:, idirect, :, :]
                    aow += cupy.einsum('xpn,xp->pn', tmp[:, :, mask], wv[1:4])
                    vmat[idirect][cupy.ix_(
                        mask, mask)] += cupy.einsum('pn,mp->nm', aow, ao_mask[0])
                    aow = cupy.einsum(
                        'np,xp->xpn', giao[idirect, :, mask], wv[1:4])
                    vmat[idirect][cupy.ix_(
                        mask, mask)] += cupy.einsum('xpn,xmp->nm', aow, ao_mask[1:4])
            else:
                raise NotImplementedError('Not implemented yet')
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

    vmat = contract('pi,npq->niq', coeff, vmat)
    vmat = contract('qj,niq->nij', coeff, vmat)

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
                               dm0, 3,  # xyz, 3 components
                               mol._atm, mol._bas, mol._env)
    vk = vk - np.swapaxes(vk, -1, -2)
    return -cupy.array(vj), -cupy.array(vk)


def get_vxc(mf, dm0):
    vxc = nr_rks(mf._numint, mf, mf.grids, mf.xc, mf.make_rdm1())
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
    t1 = time.time()
    h1ao = -0.5*mf.mol.intor('int1e_giao_irjxp')
    h1ao += -mf.mol.intor('int1e_igkin')
    h1ao += -mf.mol.intor('int1e_ignuc')
    h1ao = cupy.array(h1ao)
    t2 = time.time()
    h1ao += get_vxc(mf, dm0)
    t3 = time.time()
    print("h1ao 1e: ", t2-t1, "v2e ", t3-t2)

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
    start_time = time.time()
    mocc = mo_coeff[:, idx_occ]
    mvir = mo_coeff[:, idx_vir]
    s1ao = -mf.mol.intor('int1e_igovlp')
    t1 = time.time()
    print("s1ao ", t1-start_time)
    dm0 = mf.make_rdm1()
    natom = mf.mol.natm

    shielding_d = cupy.zeros((natom, 3, 3))
    shielding_p = cupy.zeros((natom, 3, 3))
    t2 = time.time()
    h1ao = get_h1ao(mf)
    t3 = time.time()
    print("h1ao ", t3-t2)
    tmp = cupy.einsum('xuv,ua->xav', s1ao, mvir)
    s1ai = cupy.einsum('xav,vi->xai', tmp, mocc)
    tmp = cupy.einsum('ned,ue->nud', s1ao, dm0)
    s1dm = cupy.einsum('nud,dv->nuv', tmp, dm0)*0.5
    tmp = cupy.einsum('xpq,pi->xiq', s1ao, mocc)
    s1jk = -cupy.einsum('xiq,qj->xij', tmp, mocc)*0.5
    tmp = cupy.einsum('nai,ua->nui', s1jk, mocc)
    s1jkdm1 = cupy.einsum('nui,vi->nuv', tmp, mocc.conj())*2
    t4 = time.time()
    print('sintegral cal ', t4 - t3)
    s1jkdm1 -= s1jkdm1.transpose(0, 2, 1)
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(
            mf.xc, spin=mf.mol.spin)
    if hasattr(mf,'with_df'):
        vk2 = cupy.zeros((3, nao, nao))
        for i in range(3):
            vk2[i] = +mf.get_jk(mf.mol, s1jkdm1[i], hermi=2, with_j=False)[1]*0.5*hyb
    
    else:
        vk2 = np.zeros((3, nao, nao))
        for i in range(3):
            vk2[i] = -jk.get_jk(mf.mol, s1jkdm1[i].get(), 'ijkl,jk->il')*0.5*hyb
        vk2 = cupy.array(vk2)
    t5 = time.time()
    print('vk2 cal ', t5 - t4)
    h1ao += vk2
    tmp = cupy.einsum('xuv,ua->xav', h1ao, mvir)
    Veff_ai = cupy.einsum('xav,vi->xai', tmp, mocc)
    Veff_ai -= cupy.einsum('xai,i->xai', s1ai, mo_energy[idx_occ])
    t6 = time.time()
    print('veff cal ', t6 - t5)
    Veff_ai = cupy.array(Veff_ai)
    mo1 = cphf.solve(fx, mo_energy, mo_occ, Veff_ai, max_cycle=20, tol=1e-10)[0]
    t7 = time.time()
    print("cphf-time", t7-t6)

    for atm_id in range(natom):
        mf.mol.set_rinv_origin(mf.mol.atom_coord(atm_id))
        # ! imaginary part (p* | rinv cross p | )
        int_h01 = -mf.mol.intor('int1e_prinvxp')
        # ! imaginary part (g | nabla-rinv cross p |)
        int_h01_giao = mf.mol.intor('int1e_a01gp').reshape(
            3, 3, nao, nao)
        int_h11 = mf.mol.intor('int1e_giao_a11part').reshape(
            3, 3, nao, nao)  # ! (-.5 | nabla-rinv | r)
        int_h11_diag = int_h11[0, 0] + int_h11[1, 1] + int_h11[2, 2]
        int_h11[0, 0] -= int_h11_diag
        int_h11[1, 1] -= int_h11_diag
        int_h11[2, 2] -= int_h11_diag

        tmp = cupy.einsum('ua,xai->xui', mvir, mo1)
        dm1 = cupy.einsum('xui,vi->xvu', tmp, mocc)
        dm1 *= 2
        shielding_p[atm_id] = cupy.einsum('yuv,xvu->xy', int_h01, dm1)*2.0
        shielding_p[atm_id] += cupy.einsum('yuv,xvu->xy', int_h01, s1dm)
        shielding_d[atm_id] = cupy.einsum('xyuv,vu->xy', int_h11, dm0)
        shielding_d[atm_id] += cupy.einsum('xyuv,vu->xy', int_h01_giao, dm0)
    t8 = time.time()
    print("contraction time ", t8-t7)
    ppm = nist.ALPHA**2 * 1e6
    return shielding_d*ppm, shielding_p*ppm