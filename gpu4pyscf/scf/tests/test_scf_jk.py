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

import unittest
import ctypes
import numpy as np
import numpy as cp
import pyscf
from pyscf import lib, gto
from gpu4pyscf.scf import jk
from pyscf.scf.hf import get_jk

def test_jk_hermi1():
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        C   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        basis='def2-tzvp',
        unit='B',)

    np.random.seed(9)
    nao = mol.nao
    dm = np.random.rand(nao, nao)
    dm = dm.dot(dm.T)

    vj, vk = jk.get_jk(mol, dm, hermi=1)
    vj1 = vj.get()
    vk1 = vk.get()
    ref = get_jk(mol, dm, hermi=1)
    assert abs(vj1 - ref[0]).max() < 1e-9
    assert abs(vk1 - ref[1]).max() < 1e-9
    assert abs(lib.fp(vj1) - -2327.4715195591784) < 5e-10
    assert abs(lib.fp(vk1) - -4069.3170008260583) < 5e-10

    try:
        vj = jk.get_j(mol, dm, hermi=1).get()
        assert abs(vj - ref[0]).max() < 1e-9
        assert abs(lib.fp(vj) - -2327.4715195591784) < 5e-10
    except AttributeError:
        pass

    vk = jk.get_k(mol, dm, hermi=1).get()
    assert abs(vk - ref[1]).max() < 1e-9
    assert abs(lib.fp(vk) - -4069.3170008260583) < 5e-10

    mol.omega = 0.2
    vj, vk = jk.get_jk(mol, dm, hermi=1)
    vj2 = vj.get()
    vk2 = vk.get()
    ref = get_jk(mol, dm, hermi=1)
    assert abs(vj2 - ref[0]).max() < 1e-9
    assert abs(vk2 - ref[1]).max() < 1e-9
    assert abs(lib.fp(vj2) -  1163.932604635460) < 5e-10
    assert abs(lib.fp(vk2) - -1269.969109438691) < 5e-10

    mol.omega = -0.2
    vj, vk = jk.get_jk(mol, dm, hermi=1)
    vj3 = vj.get()
    vk3 = vk.get()
    ref = get_jk(mol, dm, hermi=1)
    assert abs(vj3 - ref[0]).max() < 1e-8
    assert abs(vk3 - ref[1]).max() < 1e-9
    assert abs(lib.fp(vj3) - -3491.404124194866) < 5e-10
    assert abs(lib.fp(vk3) - -2799.347891387202) < 5e-10

    assert abs(vj2+vj3 - vj1).max() < 1e-9
    assert abs(vk2+vk3 - vk1).max() < 1e-9

def test_jk_hermi1_cart():
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        C   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        cart=True,
        basis='def2-tzvp',
        unit='B',)

    np.random.seed(9)
    nao = mol.nao
    dm = np.random.rand(nao, nao)*.1 - .03
    dm = dm.dot(dm.T)

    vj, vk = jk.get_jk(mol, dm, hermi=1)
    vj1 = vj.get()
    vk1 = vk.get()
    ref = get_jk(mol, dm, hermi=1)
    assert abs(vj1 - ref[0]).max() < 1e-10
    assert abs(vk1 - ref[1]).max() < 1e-10
    assert abs(lib.fp(vj1) - 88.88500592206657) < 1e-10
    assert abs(lib.fp(vk1) - 48.57434458906684) < 1e-10

    try:
        vj = jk.get_j(mol, dm, hermi=1).get()
        assert abs(vj - ref[0]).max() < 1e-10
    except AttributeError:
        pass

def test_jk_hermi0():
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        C   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        basis='def2-tzvp',
        unit='B',)

    np.random.seed(9)
    nao = mol.nao
    dm = np.random.rand(nao, nao)

    vj, vk = jk.get_jk(mol, dm, hermi=0)
    vj1 = vj.get()
    vk1 = vk.get()
    ref = get_jk(mol, dm, hermi=0)
    assert abs(vj1 - ref[0]).max() < 5e-10
    assert abs(vk1 - ref[1]).max() < 5e-10
    assert abs(lib.fp(vj1) - -53.489298042359046) < 5e-10
    assert abs(lib.fp(vk1) - -115.11792498085259) < 5e-10

    try:
        vj = jk.get_j(mol, dm, hermi=0).get()
        assert abs(vj - ref[0]).max() < 1e-9
        assert abs(lib.fp(vj) - -53.489298042359046) < 5e-10
    except AttributeError:
        pass

    mol.omega = 0.2
    vj, vk = jk.get_jk(mol, dm, hermi=0)
    vj2 = vj.get()
    vk2 = vk.get()
    ref = get_jk(mol, dm, hermi=0)
    assert abs(vj2 - ref[0]).max() < 1e-9
    assert abs(vk2 - ref[1]).max() < 1e-9
    assert abs(lib.fp(vj2) -  24.18519249677608) < 5e-10
    assert abs(lib.fp(vk2) - -34.15933205656134) < 5e-10

    mol.omega = -0.2
    vj, vk = jk.get_jk(mol, dm, hermi=0)
    vj3 = vj.get()
    vk3 = vk.get()
    ref = get_jk(mol, dm, hermi=0)
    assert abs(vj3 - ref[0]).max() < 1e-8
    assert abs(vk3 - ref[1]).max() < 1e-9
    assert abs(lib.fp(vj3) - -77.67449053914103) < 5e-10
    assert abs(lib.fp(vk3) - -80.95859292428769) < 5e-10

    assert abs(vj2+vj3 - vj1).max() < 1e-9
    assert abs(vk2+vk3 - vk1).max() < 1e-9

def test_jk_hermi0_l5():
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        C   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        basis={'default': 'def2-tzvp', 'O': [[5, [1., 1.]]]},
        unit='B',)

    np.random.seed(9)
    nao = mol.nao
    dm = np.random.rand(nao, nao)
    vj, vk = jk.get_jk(mol, dm, hermi=0)
    vj = vj.get()
    vk = vk.get()
    ref = get_jk(mol, dm, hermi=0)
    assert abs(vj - ref[0]).max() < 1e-9
    assert abs(vk - ref[1]).max() < 1e-9
    assert abs(lib.fp(vj) - -61.28856847097108) < 1e-9
    assert abs(lib.fp(vk) - -76.38373664249241) < 1e-9

    try:
        vj = jk.get_j(mol, dm, hermi=0).get()
        assert abs(vj - ref[0]).max() < 1e-9
        assert abs(lib.fp(vj) - -61.28856847097108) < 1e-9
    except AttributeError:
        pass

def test_k_hermi1():
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        C   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        basis=('def2-tzvp', [[4, [1, 1]]]),
        unit='B',)

    np.random.seed(9)
    nao = mol.nao
    dm = np.random.rand(nao, nao)
    dm = dm.dot(dm.T)

    ref = jk.get_jk(mol, dm, hermi=1)[1].get()
    vk = jk.get_k(mol, dm, hermi=1).get()
    assert abs(vk - ref).max() < 1e-9
    assert abs(lib.fp(vk) - 5580.092102968194) < 1e-9

    np.random.seed(9)
    nao = mol.nao
    dm = np.random.rand(2, nao, nao) - .5
    dm = cp.einsum('nij,nkj->nik', dm, dm)

    ref = jk.get_jk(mol, dm, hermi=1)[1].get()
    vk = jk.get_k(mol, dm, hermi=1).get()
    assert abs(vk - ref).max() < 1e-9
    assert abs(lib.fp(vk) - 327.9485135045478) < 1e-9

def test_general_contraction():
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        C   1.      1.    0.
        ''',
        basis=('ccpvdz', [[3, [2., 1., .5], [1., .5, 1.]]]),
        unit='B',)

    np.random.seed(9)
    nao = mol.nao
    dm = np.random.rand(nao, nao)
    dm = dm.dot(dm.T)

    vj, vk = jk.get_jk(mol, dm, hermi=1)
    vj1 = vj.get()
    vk1 = vk.get()
    ref = get_jk(mol, dm, hermi=1)
    assert abs(vj1 - ref[0]).max() < 1e-9
    assert abs(vk1 - ref[1]).max() < 1e-9

def test_vhfopt_coeff():
    from gpu4pyscf.gto.mole import group_basis
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        C   1.      1.    0.
        ''',
        basis='ccpvtz',
        unit='B',)
    vhfopt = jk._VHFOpt(mol).build()
    ref = group_basis(mol, tile=vhfopt.tile)[1]
    assert abs(vhfopt.coeff - ref).max() < 1e-12

def q_cond_reference(mol, direct_scf_tol=1e-13):
    #assert isinstance(mol, SortedMole)
    nbas = mol.nbas
    ao_loc = mol.ao_loc
    q_cond = np.empty((nbas,nbas))
    intor = mol._add_suffix('int2e')
    with mol.with_integral_screen(direct_scf_tol**2):
        jk._vhf.libcvhf.CVHFnr_int2e_q_cond(
            getattr(jk._vhf.libcvhf, intor), lib.c_null_ptr(),
            q_cond.ctypes, ao_loc.ctypes,
            mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
    q_cond = np.log(q_cond + 1e-300).astype(np.float32)

    s_estimator = None
    if mol.omega < 0:
        # CVHFnr_sr_int2e_q_cond in pyscf has bugs in upper bound estimator.
        # Use the local version of s_estimator instead

        # FIXME: To avoid changing the CUDA kernel function signature,
        # temporarily attach the extra information to the s_estimator array and
        # pass it along with s_estimator.
        # This is a workaround and should be addressed in the future.
        s_estimator = np.empty((nbas+2,nbas), dtype=np.float32)
        # The most diffuse pGTO in each shell is used to estimate the
        # asymptotic value of SR integrals. In a contracted shell, the
        # diffuse_ctr_coef for the diffuse_exps may only represent a portion
        # of the AO basis. Using this ctr_coef can introduce errors in the SR
        # integral estimation. The diffuse pGTO is normalized to approximate the
        # entire shell.
        diffuse_exps, _ = jk.extract_pgto_params(mol, 'diffuse')
        l = mol._bas[:,gto.ANG_OF]
        diffuse_ctr_coef = gto.gto_norm(l, diffuse_exps)
        diffuse_exps = diffuse_exps.astype(np.float32)
        diffuse_ctr_coef = diffuse_ctr_coef.astype(np.float32)
        s_estimator[nbas] = diffuse_exps
        s_estimator[nbas+1] = diffuse_ctr_coef
        jk.libvhf_rys.sr_eri_s_estimator(
            s_estimator.ctypes, ctypes.c_float(mol.omega),
            diffuse_exps.ctypes, diffuse_ctr_coef.ctypes,
            mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
    return q_cond, s_estimator

def test_q_cond():
    from gpu4pyscf.gto.mole import group_basis
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        H   4.      0.    3.
        H   0.      1.    .6
        C   -3.2258  -0.1262  2.6126
        H   -5.7987   0.2177  4.1423
        H   -5.8042  -1.0067  4.1503
        ''',
        basis=('def2-tzvp', [[0, [30, .2], [9.1, -.4], [5.1, -.5]], [4, [1, 1]]]),
    )

    jkopt = jk._VHFOpt(mol).build()
    sorted_mol = group_basis(mol)[0]
    qref, sref = q_cond_reference(sorted_mol)
    q_cond = jkopt.q_cond.get()
    thrd = np.log(jkopt.direct_scf_tol)
    qref[qref < thrd] = thrd
    q_cond[q_cond < thrd] = thrd
    assert abs(qref - q_cond).max() < 1e-3

    mol.omega = .25
    jkopt = jk._VHFOpt(mol).build()
    sorted_mol = group_basis(mol)[0]
    qref, sref = q_cond_reference(sorted_mol)
    q_cond = jkopt.q_cond.get()
    qref[qref < thrd] = thrd
    q_cond[q_cond < thrd] = thrd
    assert abs(qref - q_cond).max() < 1e-3

    mol.omega = -.25
    jkopt = jk._VHFOpt(mol).build()
    sorted_mol = group_basis(mol)[0]
    qref, sref = q_cond_reference(sorted_mol)
    q_cond = jkopt.q_cond.get()
    qref[qref < thrd] = thrd
    q_cond[q_cond < thrd] = thrd
    assert abs(qref - q_cond).max() < 1e-3

def test_transform_coeff():
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        H   4.      0.    3.
        H   0.      1.    .6
        C   -3.2258  -0.1262  2.6126
        H   -5.7987   0.2177  4.1423
        H   -5.8042  -1.0067  4.1503
        ''',
        basis=('def2-tzvp', [[4, [1, 1]]]),
    )
    jkopt = jk._VHFOpt(mol).build()

    coeff = np.zeros((jkopt.sorted_mol.nao, jkopt.mol.nao))
    l_max = max([l_ctr[0] for l_ctr in jkopt.uniq_l_ctr])
    if jkopt.mol.cart:
        cart2sph_per_l = [np.eye((l+1)*(l+2)//2) for l in range(l_max + 1)]
    else:
        cart2sph_per_l = [gto.mole.cart2sph(l, normalized = "sp") for l in range(l_max + 1)]
    i_spherical_offset = 0
    i_cartesian_offset = 0
    for i, l in enumerate(jkopt.uniq_l_ctr[:,0]):
        cart2sph = cart2sph_per_l[l]
        ncart, nsph = cart2sph.shape
        l_ctr_count = jkopt.l_ctr_offsets[i + 1] - jkopt.l_ctr_offsets[i]
        cart_offs = i_cartesian_offset + np.arange(l_ctr_count) * ncart
        sph_offs = i_spherical_offset + np.arange(l_ctr_count) * nsph
        cart_idx = cart_offs[:,None] + np.arange(ncart)
        sph_idx = sph_offs[:,None] + np.arange(nsph)
        coeff[cart_idx[:,:,None],sph_idx[:,None,:]] = cart2sph
        l_ctr_pad_count = jkopt.l_ctr_pad_counts[i]
        i_cartesian_offset += (l_ctr_count + l_ctr_pad_count) * ncart
        i_spherical_offset += l_ctr_count * nsph
    ref = jkopt.unsort_orbitals(coeff, axis = [1])

    dat = jkopt.coeff
    assert abs(dat - ref).max() < 1e-14

def test_jk_energy_per_atom():
    from gpu4pyscf.grad.rhf import _jk_energy_per_atom
    from gpu4pyscf.df.grad import rhf as df_rhf
    mol = pyscf.M(atom='''
    O  0.0000  0.7375 -0.0528
    O  0.0000 -0.7375 -0.1528
    ''', basis='def2-svp')
    np.random.seed(12)
    nao = mol.nao
    dm = np.random.rand(nao, nao) - .5
    dm1 = dm - dm.T

    auxmol = mol.copy()
    auxmol.basis='def2-universal-jkfit'
    auxmol.build(0, 0)

    ejk = _jk_energy_per_atom(mol, dm1, j_factor=0, k_factor=1.)
    eri1 = mol.intor('int2e_ip1')
    ref = np.einsum('xijkl,jk,li->x', eri1[:,:nao//2], dm1, dm1[:,:nao//2])
    assert abs(ejk[0] - ref).max() < 5e-12

    dm = cp.asarray(dm.dot(dm.T))
    mol.omega = -.3
    vk = jk.get_k(mol, dm, hermi=1)
    assert abs(lib.fp(vk.get()) - -1.8653967312459407) < 1e-13

    ejk = _jk_energy_per_atom(mol, dm, j_factor=0, k_factor=1.) * .5
    eri1 = mol.intor('int2e_ip1')
    ref = .5 * np.einsum('xijkl,jk,li->x', eri1[:,:nao//2], dm, dm[:,:nao//2])
    assert abs(ejk[0] - ref).max() < 5e-12
    ref = np.array([0.24806416996651, 1.11003753769514, 0.19967171093788])
    assert abs(ejk[0] - ref).max() < 1e-13
