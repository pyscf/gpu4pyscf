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

__all__ = [
    'X2C1E_GSCF',
]

import ctypes
import numpy as np
import cupy as cp
import scipy.linalg
from pyscf import lib
from pyscf.gto import mole
from pyscf.data import nist
from pyscf.x2c import x2c as x2c_cpu
from gpu4pyscf.lib.cupy_helper import block_diag, asarray, hermi_triu
from gpu4pyscf.lib.cusolver import eigh
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import hf, ghf
from gpu4pyscf.gto.mole import SortedGTO
from gpu4pyscf.df.int3c2e_bdiv import libvhf_rys, contract_int3c2e_auxvec
from gpu4pyscf import __config__
from gpu4pyscf.lib import utils

libvhf_rys.int3c2e_cart2sph.restype = ctypes.c_int

LINEAR_DEP_THRESHOLD = 1e-9

class X2CHelperBase(lib.StreamObject):
    '''2-component X2c (including spin-free and spin-dependent terms) in
    the j-adapted spinor basis.
    '''
    approx = x2c_cpu.X2CHelperBase.approx
    xuncontract = x2c_cpu.X2CHelperBase.xuncontract
    basis = x2c_cpu.X2CHelperBase.basis

    device = utils.device
    to_gpu = utils.to_gpu
    to_cpu = utils.to_cpu

    __init__ = x2c_cpu.X2CHelperBase.__init__
    dump_flags = x2c_cpu.X2CHelperBase.dump_flags
    reset = x2c_cpu.X2CHelperBase.reset

    def get_xmol(self, mol=None):
        if mol is None:
            mol = self.mol

        if self.basis is not None:
            raise NotImplementedError
        elif self.xuncontract:
            if all(mol._bas[:,mole.KAPPA_OF] == 0):
                xmol = SortedGTO.from_mol(mol, decontract=True, diffuse_cutoff=1e200)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return xmol

    get_hcore = NotImplemented
    get_xmat = NotImplemented
    _get_rmat = NotImplemented

    get_hcore = NotImplemented
    picture_change = NotImplemented

    def _picture_change(self, xmol, even_operator=(None, None), odd_operator=None):
        '''Picture change for even_operator + odd_operator

        even_operator has two terms at diagonal blocks
        [ v  0 ]
        [ 0  w ]

        odd_operator has the term at off-diagonal blocks
        [ 0    p ]
        [ p^T  0 ]

        v, w, and p can be strings (integral name) or matrices.
        '''
        c = lib.param.LIGHT_SPEED
        v_op, w_op = even_operator
        assert isinstance(xmol, SortedGTO)
        with lib.temporary_env(xmol, cart=xmol.mol.cart):
            if isinstance(v_op, str):
                v_op = xmol.intor(v_op)
            if isinstance(w_op, str):
                w_op = xmol.intor(w_op)
                w_op *= (.5/c)**2
            if isinstance(odd_operator, str):
                odd_operator = xmol.intor(odd_operator) * (.5/c)

        if v_op is not None:
            shape = v_op.shape
        elif w_op is not None:
            shape = w_op.shape
        elif odd_operator is not None:
            shape = odd_operator.shape
        else:
            raise RuntimeError('No operators provided')

        x = self.get_xmat()
        r = self._get_rmat(x)
        def transform(mat):
            nao = mat.shape[-1] // 2
            xv = mat[:nao] + x.conj().T.dot(mat[nao:])
            h = xv[:,:nao] + xv[:,nao:].dot(x)
            return r.conj().T.dot(h).dot(r)

        nao = shape[-1]
        dtype = np.result_type(v_op, w_op, odd_operator)

        if len(shape) == 2:
            mat = cp.zeros((nao*2,nao*2), dtype)
            if v_op is not None:
                mat[:nao,:nao] = asarray(v_op)
            if w_op is not None:
                mat[nao:,nao:] = asarray(w_op)
            if odd_operator is not None:
                odd_operator = asarray(odd_operator)
                mat[:nao,nao:] = odd_operator
                mat[nao:,:nao] = odd_operator.conj().T
            pc_mat = transform(mat)

        else:
            assert len(shape) == 3
            mat = cp.zeros((shape[0],nao*2,nao*2), dtype)
            if v_op is not None:
                mat[:,:nao,:nao] = asarray(v_op)
            if w_op is not None:
                mat[:,nao:,nao:] = asarray(w_op)
            if odd_operator is not None:
                odd_operator = asarray(odd_operator)
                mat[:,:nao,nao:] = odd_operator
                mat[:,nao:,:nao] = odd_operator.conj().transpose(0,2,1)
            pc_mat = cp.stack([transform(m) for m in mat])

        return pc_mat

class SpinOrbitalX2CHelper(X2CHelperBase):
    '''2-component X2c (including spin-free and spin-dependent terms) in
    the Gaussian type spin-orbital basis (as the spin-orbital basis in GHF)
    '''
    def get_hcore(self, mol=None):
        from gpu4pyscf.pbc.gto import int1e
        if mol is None: mol = self.mol
        if mol.has_ecp():
            raise NotImplementedError
        assert '1E' in self.approx.upper()

        xmol = self.get_xmol(mol)
        sort_ao = not self.xuncontract
        c = lib.param.LIGHT_SPEED
        t = int1e.int1e_kin(xmol, sort_output=sort_ao)
        s = int1e.int1e_ovlp(xmol, sort_output=sort_ao)
        nucmol = mole.fakemol_for_charges(xmol.atom_coords())
        Z = cp.asarray(xmol.atom_charges(), dtype=np.float64)
        v = contract_int3c2e_auxvec(xmol, nucmol, -Z, sort_output=sort_ao)
        with lib.temporary_env(xmol, cart=mol.cart):
            w = asarray(_sigma_dot(xmol.intor('int1e_spnucsp')))
        if not mol.cart:
            s, t, v = _orbital_pair_cart2sph(xmol, (s, t, v))
        t = _block_diag(t)
        v = _block_diag(v)
        s = _block_diag(s)

        if 'ATOM' in self.approx.upper():
            x = _atomic_1e_x(xmol)
            h1 = _get_hcore_fw(t, v, w, s, x, c)
        else:
            h1 = _x2c1e_get_hcore(t, v, w, s, c)

        nao = h1.shape[-1] // 2
        h1 = h1.view(np.float64).reshape(2,nao,2,nao,2).transpose(0,2,4,1,3)
        h1 = _recontract_matrix(xmol, h1.reshape(-1,nao,nao))
        n2c = h1.shape[-1]
        h1 = h1.reshape(2,2,2,n2c,n2c).transpose(0,3,1,4,2)
        h1 = h1.reshape(2*n2c, 2*n2c*2).view(np.complex128)
        return h1

    @lib.with_doc(X2CHelperBase.picture_change.__doc__)
    def picture_change(self, even_operator=(None, None), odd_operator=None):
        if self.basis is not None:
            raise NotImplementedError
        mol = self.mol
        xmol = self.get_xmol(mol)
        pc_mat = self._picture_change(xmol, even_operator, odd_operator)
        nao = pc_mat.shape[-1] // 2
        if pc_mat.dtype == np.complex128:
            pc_mat = pc_mat.view(np.float64).reshape(2,nao,2,nao,2).transpose(0,2,4,1,3)
            pc_mat = _recontract_matrix(xmol, pc_mat.reshape(-1,nao,nao))
            n2c = pc_mat.shape[-1]
            pc_mat = pc_mat.reshape(2,2,2,n2c,n2c).transpose(0,3,1,4,2)
            pc_mat = pc_mat.reshape(2*n2c, 2*n2c*2).view(np.complex128)
        else:
            pc_mat = pc_mat.reshape(2,nao,2,nao).transpose(0,2,1,3)
            pc_mat = _recontract_matrix(xmol, pc_mat.reshape(-1,nao,nao))
            n2c = pc_mat.shape[-1]
            pc_mat = pc_mat.reshape(2,2,n2c,n2c).transpose(0,2,1,3)
            pc_mat = pc_mat.reshape(2*n2c, 2*n2c)
        return pc_mat

    def get_xmat(self, mol=None):
        from gpu4pyscf.pbc.gto import int1e
        if mol is None:
            xmol = self.get_xmol()
        else:
            xmol = SortedGTO.from_mol(mol)
        sort_ao = not self.xuncontract
        assert '1E' in self.approx.upper()

        if 'ATOM' in self.approx.upper():
            x = _atomic_1e_x(xmol)
        else:
            c = lib.param.LIGHT_SPEED
            t = int1e.int1e_kin(xmol, sort_output=sort_ao)
            s = int1e.int1e_ovlp(xmol, sort_output=sort_ao)
            nucmol = mole.fakemol_for_charges(xmol.atom_coords())
            Z = cp.asarray(xmol.atom_charges(), dtype=np.float64)
            v = contract_int3c2e_auxvec(xmol, nucmol, -Z, sort_output=sort_ao)
            with lib.temporary_env(xmol, cart=xmol.mol.cart):
                w = asarray(_sigma_dot(xmol.intor('int1e_spnucsp')))
            if not mol.cart:
                s, t, v = _orbital_pair_cart2sph(xmol, (s, t, v))
            t = _block_diag(t)
            v = _block_diag(v)
            s = _block_diag(s)
            x = _x2c1e_xmatrix(t, v, w, s, c)
        return x

    def _get_rmat(self, x=None):
        from gpu4pyscf.pbc.gto import int1e
        xmol = self.get_xmol()
        if x is None:
            x = self.get_xmat(xmol)
        sort_ao = not self.xuncontract
        c = lib.param.LIGHT_SPEED
        s = int1e.int1e_ovlp(xmol, sort_output=sort_ao)
        t = int1e.int1e_kin(xmol, sort_output=sort_ao)
        if not xmol.mol.cart:
            s, t = _orbital_pair_cart2sph(xmol, (s, t))
        s = _block_diag(s)
        t = _block_diag(t)
        s1 = s + x.conj().T.dot(t).dot(x) * (.5/c**2)
        return _get_r(s, s1)

def x2c1e_ghf(mf):
    assert isinstance(mf, ghf.GHF)
    if isinstance(mf, _X2C_SCF):
        if mf.with_x2c is None:
            mf.with_x2c = SpinOrbitalX2CHelper(mf.mol)
        else:
            assert isinstance(mf.with_x2c, SpinOrbitalX2CHelper)
        return mf
    return lib.set_class(X2C1E_GSCF(mf), (X2C1E_GSCF, mf.__class__))

# A tag to label the derived SCF class
class _X2C_SCF:
    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        if self.with_x2c:
            self.with_x2c.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_x2c.reset(mol)
        return super().reset(mol)

class X2C1E_GSCF(_X2C_SCF):
    '''
    Attributes for spin-orbital X2C:
        with_x2c : X2C object
    '''

    __name_mixin__ = 'X2C1e'
    _keys = {'with_x2c'}

    to_gpu = utils.to_gpu
    device = utils.device

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self.with_x2c = SpinOrbitalX2CHelper(mf.mol)

    def undo_x2c(self):
        '''Remove the X2C Mixin'''
        obj = lib.view(self, lib.drop_class(self.__class__, X2C1E_GSCF))
        del obj.with_x2c
        return obj

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return self.with_x2c.get_hcore(mol)

    def dip_moment(self, mol=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   picture_change=True, **kwargs):
        r''' Dipole moment calculation with picture change correction

        Args:
             mol: an instance of :class:`Mole`
             dm : a 2D ndarrays density matrices

        Kwarg:
            picture_change (bool) : Whether to compute the dipole moment with
            picture change correction.

        Return:
            A list: the dipole moment on x, y and z component
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        dm = cp.asnumpy(dm)
        log = logger.new_logger(mol, verbose)
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        nucl_dip = np.einsum('i,ix->x', charges, coords)

        with mol.with_common_orig((0,0,0)):
            if picture_change:
                xmol = self.with_x2c.get_xmol()[0]
                nao = xmol.nao
                r = xmol.intor_symmetric('int1e_r')
                r = np.stack([_block_diag(x) for x in r])
                c1 = 0.5/lib.param.LIGHT_SPEED
                prp = xmol.intor_symmetric('int1e_sprsp').reshape(3,4,nao,nao)
                prp = np.stack([_sigma_dot(x*c1**2) for x in prp])
                ao_dip = self.with_x2c.picture_change((r, prp))
            else:
                r = mol.intor_symmetric('int1e_r')
                ao_dip = np.array([_block_diag(x) for x in r])

        el_dip = np.einsum('xij,ji->x', ao_dip, dm).real
        mol_dip = nucl_dip - el_dip

        if unit.upper() == 'DEBYE':
            mol_dip *= nist.AU2DEBYE
            log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
        else:
            log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
        return mol_dip

    def _transfer_attrs_(self, dst):
        if self.with_x2c and not hasattr(dst, 'with_x2c'):
            logger.warn(self, 'Destination object of to_hf/to_ks method is not '
                        'an X2C object. Convert dst to X2C object.')
            dst = dst.x2c()
        return hf.SCF._transfer_attrs_(self, dst)

    def to_ks(self, xc='HF'):
        raise NotImplementedError

    device = utils.device
    to_gpu = utils.to_gpu

    def to_cpu(self):
        mf = self.undo_x2c().to_cpu().x2c1e()
        return utils.to_cpu(self, mf)


def _sqrt(a, tol=1e-14):
    e, v = cp.linalg.eigh(a)
    idx = cp.where(e > tol)[0]
    v = v[:,idx]
    e = e[idx]
    return (v*cp.sqrt(e)).dot(v.conj().T)

def _invsqrt(a, tol=1e-14):
    e, v = cp.linalg.eigh(a)
    idx = cp.where(e > tol)[0]
    v = v[:,idx]
    e = e[idx]
    return (v/cp.sqrt(e)).dot(v.conj().T)

def _get_hcore_fw(t, v, w, s, x, c):
    # s1 = s + (1/2c^2)(X^{\dag}*T*X)
    s1 = s + x.conj().T.dot(t).dot(x) * (.5/c**2)
    # tx = T * X
    tx = t.dot(x)
    # h1 = (v + T*X + V^{\dag}*T^{\dag} - (X^{\dag} * T * X) + (X^{\dag} * W * X)*(1/4c^2)
    h1 = x.conj().T.dot(w).dot(x) * (.25/c**2)
    h1 += v + tx + tx.conj().T - x.conj().T.dot(tx)
    # R = S^{-1/2} * (S^{-1/2}\tilde{S}S^{-1/2})^{-1/2} * S^{1/2}
    r = _get_r(s, s1)
    # H1 = R^{\dag} * H1 * R
    h1 = r.conj().T.dot(h1).dot(r)
    return h1

def _get_r(s, snesc):
    # R^dag \tilde{S} R = S
    # R = S^{-1/2} [S^{-1/2}\tilde{S}S^{-1/2}]^{-1/2} S^{1/2}
    # Eq.(193) or (223) in 10.1080/00268971003781571
    w, v = cp.linalg.eigh(s)
    idx = cp.where(w > 1e-14)[0]
    v = v[:,idx]
    w_sqrt = cp.sqrt(w[idx])
    w_invsqrt = 1 / w_sqrt

    snesc = v.conj().T.dot(snesc).dot(v)
    r_mid = w_invsqrt[:,None] * snesc * w_invsqrt
    w1, v1 = cp.linalg.eigh(r_mid)
    idx1 = cp.where(w1 > 1e-14)[0]
    v1 = v1[:,idx1]
    r_mid = (v1/cp.sqrt(w1[idx1])).dot(v1.conj().T)
    r = w_invsqrt[:,None] * r_mid * w_sqrt
    # Back transform to AO basis
    r = v.dot(r).dot(v.conj().T)
    return r

def _x2c1e_xmatrix(t, v, w, s, c):
    nao = s.shape[0]
    n2 = nao * 2
    dtype = np.result_type(t, v, w, s)
    h = cp.zeros((n2,n2), dtype=dtype)
    m = cp.zeros((n2,n2), dtype=dtype)
    h[:nao,:nao] = v
    h[:nao,nao:] = t
    h[nao:,:nao] = t
    h[nao:,nao:] = w * (.25/c**2) - t
    m[:nao,:nao] = s
    m[nao:,nao:] = t * (.5/c**2)
    try:
        e, a = eigh(h, m)
        cl = a[:nao,nao:]
        cs = a[nao:,nao:]
        x = cp.linalg.solve(cl.T, cs.T).T  # B = XA
    except cp.linalg.LinAlgError:
        d, t = cp.linalg.eigh(m)
        idx = cp.where(d > LINEAR_DEP_THRESHOLD)[0]
        t = t[:,idx] / cp.sqrt(d[idx])
        tht = t.conj().T.dot(h).dot(t)
        e, a = cp.linalg.eigh(tht)
        a = cp.dot(t, a)
        idx = cp.where(e > -c**2)[0]
        cl = a[:nao,idx]
        cs = a[nao:,idx]
        # X = B A^{-1} = B (A^T A)^{-1} A^T
        cl_inv = cp.linalg.solve(cl.conj().T.dot(cl), cl.conj().T)
        x = cs.dot(cl_inv)
    return x

def _x2c1e_get_hcore(t, v, w, s, c):
    nao = s.shape[0]
    n2 = nao * 2
    dtype = cp.result_type(t, v, w, s)
    h = cp.zeros((n2,n2), dtype=dtype)
    m = cp.zeros((n2,n2), dtype=dtype)
    h[:nao,:nao] = v
    h[:nao,nao:] = t
    h[nao:,:nao] = t
    h[nao:,nao:] = w * (.25/c**2) - t
    m[:nao,:nao] = s
    m[nao:,nao:] = t * (.5/c**2)

    try:
        e, a = eigh(h, m)
        cl = a[:nao,nao:]
        # cs = a[nao:,nao:]
        e = e[nao:]
    except cp.linalg.LinAlgError:
        d, t = cp.linalg.eigh(m)
        idx = cp.where(d > LINEAR_DEP_THRESHOLD)[0]
        t = t[:,idx] / cp.sqrt(d[idx])
        tht = t.conj().T.dot(h).dot(t)
        e, a = cp.linalg.eigh(tht)
        a = cp.dot(t, a)
        idx = cp.where(e > -c**2)[0]
        cl = a[:nao,idx]
        # cs = a[nao:,idx]
        e = e[idx]

    w, u = cp.linalg.eigh(cl.conj().T.dot(s).dot(cl))
    idx = cp.where(w > 1e-14)[0]
    u = u[:,idx]
    # Adopt (2) here because X is not appeared in Eq (2).
    # R[A] = u w^{1/2} u^+,  so R[A]^{-1} A^+ S in Eq (2) is
    r = (u/cp.sqrt(w[idx])).dot(u.conj().T.dot(cl.conj().T).dot(s))
    h1 = (r.conj().T*e).dot(r)
    return h1


def _block_diag(mat):
    '''
    [A 0]
    [0 A]
    '''
    if isinstance(mat, cp.ndarray):
        return block_diag([mat, mat])
    else:
        return scipy.linalg.block_diag(mat, mat)

def _sigma_dot(mat):
    '''sigma dot A x B + A dot B'''
    quaternion = np.vstack([1j * lib.PauliMatrices, np.eye(2)[None,:,:]])
    if isinstance(mat, cp.ndarray):
        out = cp.einsum('sxy,spq->xpyq', asarray(quaternion), mat)
    else:
        out = np.einsum('sxy,spq->xpyq', quaternion, mat)
    nao = mat.shape[-1] * 2
    return out.reshape(nao, nao)

def _atoms_in_mole(xmol):
    mol = xmol.mol
    atoms = {}
    for i in range(mol.natm):
        symb = mol.atom_symbol(i)
        if symb not in atoms:
            atoms[symb] = atom = xmol.copy(deep=False)
            atom.cart = mol.cart
            mask = xmol._bas[:,mole.ATOM_OF] == i
            atom._bas = xmol._bas[mask]
            atom._atm = xmol._atm[i:i+1]
            atom._bas[:,mole.ATOM_OF] = 0
    return atoms

def _atomic_1e_x(xmol):
    atoms = _atoms_in_mole(xmol)
    x_conf = {}
    c = lib.param.LIGHT_SPEED
    for elem, atom in atoms.items():
        with atom.with_rinv_at_nucleus(0):
            z = -atom.atom_charge(0)
            t1 = _block_diag(atom.intor_symmetric('int1e_kin'))
            s1 = _block_diag(atom.intor_symmetric('int1e_ovlp'))
            v1 = _block_diag(z * atom.intor_symmetric('int1e_rinv'))
            w1 = _sigma_dot(z * atom.intor('int1e_sprinvsp'))
        x_conf[elem] = asarray(x2c_cpu._x2c1e_xmatrix(t1, v1, w1, s1, c))

    mol = xmol.mol
    l = xmol._bas[:,mole.ANG_OF]
    if mol.cart:
        nf = (l + 1) * (l + 2) // 2
    else:
        nf = l * 2 + 1
    ao_labels = np.repeat(xmol._bas[:,mole.ATOM_OF], nf)
    nao = nf.sum()
    n2c = nao * 2
    x = cp.zeros((n2c, n2c), dtype=np.complex128)
    for i in range(xmol.natm):
        symb = mol.atom_symbol(i)
        idx = asarray(np.where(ao_labels == i)[0])
        idx = np.append(idx, idx+nao)
        x[idx[:,None],idx] = x_conf[symb]
    return x

def _orbital_pair_cart2sph(mol, arrays, hermi=1, bas_ij_idx=None):
    '''Transforms the AO of the compressed eri3c from Cartesian to spherical basis'''
    assert isinstance(mol, SortedGTO)
    if hasattr(arrays, 'ndim') and arrays.ndim == 2:
        arrays = cp.asarray(arrays)[:,:,None]
    elif not isinstance(arrays, cp.ndarray):
        arrays = cp.stack(arrays, axis=2)
        assert arrays.ndim == 3
    else:
        assert arrays.ndim == 3
        arrays = cp.asarray(arrays.transpose(1,2,0), order='C')
    is_complex = arrays.dtype == np.complex128
    if is_complex:
        hermi = 0
        arrays = arrays.view(np.float64)

    if bas_ij_idx is None:
        bas_ij_cache = mol.generate_shl_pairs(hermi=hermi)
        bas_ij_idx = mol.aggregate_shl_pairs(bas_ij_cache)[0]
    ish, jsh = divmod(bas_ij_idx, mol.nbas)

    ao_loc = mol.ao_loc_nr(cart=True)
    nao_cart = ao_loc[-1]
    ao_loc = asarray(ao_loc)
    cart_pair_loc = ao_loc[ish] * nao_cart + ao_loc[jsh]

    ao_loc = mol.ao_loc_nr(cart=False)
    nao = ao_loc[-1]
    ao_loc = asarray(ao_loc)
    sph_pair_loc = ao_loc[ish] * nao + ao_loc[jsh]

    assert arrays.shape[0] == arrays.shape[1] == nao_cart
    rys_envs = mol.rys_envs
    naux = arrays.shape[2]
    out = cp.zeros((nao, nao, naux))
    compressed = 0
    err = libvhf_rys.int3c2e_cart2sph(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(arrays.data.ptr, ctypes.c_void_p),
        ctypes.byref(rys_envs),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(sph_pair_loc.data.ptr, ctypes.c_void_p),
        ctypes.cast(cart_pair_loc.data.ptr, ctypes.c_void_p),
        ctypes.c_int(len(bas_ij_idx)),
        ctypes.c_int(naux), ctypes.c_int(mol.nbas),
        ctypes.c_int(nao), ctypes.c_int(compressed))
    if err != 0:
        raise RuntimeError('int3c2e_cart2sph kernel failed')
    if is_complex:
        out = out.view(np.complex128)
    out = cp.asarray(out.transpose(2,0,1), order='C')
    return hermi_triu(out)

def _recontract_matrix(mol, mat):
    '''ctr_coeff.T.dot(mat).dot(ctr_coeff)'''
    assert isinstance(mol, SortedGTO)
    mat = cp.asarray(mat, order='C')
    assert mat.dtype == np.float64
    mat_ndim = mat.ndim
    if mat_ndim == 2:
        mat = mat[None]
    counts, ncol = mat.shape[:2]

    if mol.mol.cart:
        cart = 1
        c_ao_loc = cp.asarray(mol.c_ao_loc, dtype=np.int32)
        p_ao_loc = cp.asarray(mol.p_ao_loc, dtype=np.int32)
    else:
        cart = 0
        p_ao_loc = mol.ao_loc_nr(cart=False)
        assert ncol == p_ao_loc[-1], \
                'Input matrix must be transformed into spherical GTOs'
        c_ao_loc = cp.asarray(mol.c_ao_loc, dtype=np.int32)
        p_ao_loc = cp.asarray(p_ao_loc, dtype=np.int32)
    recontract_coef = cp.asarray(mol.recontract_coef)
    recontract_bas = cp.asarray(mol.recontract_bas)
    recontraction_idx = cp.asarray(mol.recontraction_idx)
    nao = mol.mol.nao
    tmp = cp.zeros((counts, nao, ncol))
    err = libvhf_rys.bra_from_sorted(
        ctypes.cast(tmp.data.ptr, ctypes.c_void_p),
        ctypes.cast(mat.data.ptr, ctypes.c_void_p),
        ctypes.cast(recontract_coef.data.ptr, ctypes.c_void_p),
        ctypes.cast(recontract_bas.data.ptr, ctypes.c_void_p),
        ctypes.cast(recontraction_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(c_ao_loc.data.ptr, ctypes.c_void_p),
        ctypes.cast(p_ao_loc.data.ptr, ctypes.c_void_p),
        ctypes.c_int(len(recontract_bas)), ctypes.c_int(mol.nbas),
        ctypes.c_int(ncol), ctypes.c_int(counts),
        ctypes.c_int(cart))
    assert err == 0

    out = cp.zeros((counts, nao, nao))
    err = libvhf_rys.ket_from_sorted(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(tmp.data.ptr, ctypes.c_void_p),
        ctypes.cast(recontract_coef.data.ptr, ctypes.c_void_p),
        ctypes.cast(recontract_bas.data.ptr, ctypes.c_void_p),
        ctypes.cast(recontraction_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(c_ao_loc.data.ptr, ctypes.c_void_p),
        ctypes.cast(p_ao_loc.data.ptr, ctypes.c_void_p),
        ctypes.c_int(len(recontract_bas)), ctypes.c_int(mol.nbas),
        ctypes.c_int(nao*counts),
        ctypes.c_int(cart))
    assert err == 0

    if mat_ndim == 2:
        out = out[0]
    return out
