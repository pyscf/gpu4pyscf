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



'''
X2C 2-component HF methods from pyscf
'''

from functools import reduce
import cupy as cp
import scipy.linalg
from pyscf import lib as pyscf_lib
from pyscf.gto import mole
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import hf, ghf
from pyscf.scf import _vhf
from pyscf import __config__
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib import utils

LINEAR_DEP_THRESHOLD = 1e-9

class X2CHelperBase(pyscf_lib.StreamObject):
    '''2-component X2c (including spin-free and spin-dependent terms) in
    the j-adapted spinor basis.
    '''
    approx = getattr(__config__, 'x2c_X2C_approx', '1e')  # 'atom1e'
    xuncontract = getattr(__config__, 'x2c_X2C_xuncontract', True)
    basis = getattr(__config__, 'x2c_X2C_basis', None)
    def __init__(self, mol):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('approx = %s',    self.approx)
        log.info('xuncontract = %d', self.xuncontract)
        if self.basis is not None:
            log.info('basis for X matrix = %s', self.basis)
        return self

    def get_xmol(self, mol=None):
        """
        Get the X2C molecule object with the specified basis,
        in order to make more accurate X2C calculations, especially for X matrix.
        """
        if mol is None:
            mol = self.mol

        if self.basis is not None:
            xmol = mol.copy(deep=False)
            xmol.build(False, False, basis=self.basis)
            return xmol, None
        elif self.xuncontract:
            if all(mol._bas[:,mole.KAPPA_OF] == 0):
                xmol, contr_coeff = _uncontract_mol(mol, self.xuncontract)
            else:
                raise NotImplementedError("X2C-GHF object cannot be initialized by uncontracting spinor basis")
            return xmol, contr_coeff
        else:
            return mol, None

    def get_hcore(self, mol=None):
        '''2-component X2c Foldy-Wouthuysen (FW) Hamiltonian (including
        spin-free and spin-dependent terms) in the j-adapted spinor basis.
        '''
        raise NotImplementedError("X2C-HelperBase does not implement get_hcore")

    def _picture_change(self, xmol, even_operator=(None, None), odd_operator=None):
        '''Picture change for property calculations
        '''
        raise NotImplementedError("Picture change for X2C is not implemented")

    def picture_change(self, even_operator=(None, None), odd_operator=None):
        raise NotImplementedError("Picture change for X2C is not implemented")

    def get_xmat(self, mol=None):
        raise NotImplementedError("X2C-HelperBase does not implement get_xmat")

    def _get_rmat(self, x=None):
        raise NotImplementedError("X2C-HelperBase does not implement _get_rmat")

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
        return self


class SpinOrbitalX2CHelper(X2CHelperBase):
    '''2-component X2c (including spin-free and spin-dependent terms) in
    the Gaussian type spin-orbital basis (as the spin-orbital basis in GHF)
    '''
    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        if mol.has_ecp():
            raise NotImplementedError

        xmol, contr_coeff = self.get_xmol(mol)
        c = pyscf_lib.param.LIGHT_SPEED
        assert ('1E' in self.approx.upper())

        t = _block_diag(xmol.intor_symmetric('int1e_kin'))
        v = _block_diag(xmol.intor_symmetric('int1e_nuc'))
        s = _block_diag(xmol.intor_symmetric('int1e_ovlp'))
        w = _sigma_dot(xmol.intor('int1e_spnucsp'))
        t = cp.asarray(t)
        v = cp.asarray(v)
        w = cp.asarray(w)
        s = cp.asarray(s)

        if 'get_xmat' in self.__dict__:
            # If the get_xmat method is overwritten by user, build the X
            # matrix with the external get_xmat method
            x = self.get_xmat(xmol)
            h1 = _get_hcore_fw(t, v, w, s, x, c)

        elif 'ATOM' in self.approx.upper():
            atom_slices = xmol.offset_nr_by_atom()
            # spin-orbital basis is twice the size of NR basis
            atom_slices[:,2:] *= 2
            nao = xmol.nao_nr() * 2
            x = cp.zeros((nao,nao))
            for ia in range(xmol.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                t1 = _block_diag(xmol.intor('int1e_kin', shls_slice=shls_slice))
                s1 = _block_diag(xmol.intor('int1e_ovlp', shls_slice=shls_slice))
                t1 = cp.asarray(t1)
                s1 = cp.asarray(s1)
                with xmol.with_rinv_at_nucleus(ia):
                    z = -xmol.atom_charge(ia)
                    v1 = _block_diag(z * xmol.intor('int1e_rinv', shls_slice=shls_slice))
                    w1 = _sigma_dot(z * xmol.intor('int1e_sprinvsp', shls_slice=shls_slice))
                    w1 = cp.asarray(w1)
                    v1 = cp.asarray(v1)
                x[p0:p1,p0:p1] = _x2c1e_xmatrix(t1, v1, w1, s1, c)
            h1 = _get_hcore_fw(t, v, w, s, x, c)

        else:
            h1 = _x2c1e_get_hcore(t, v, w, s, c)

        # Project the Hamiltonian onto the original AO basis
        if self.basis is not None:
            s22 = xmol.intor_symmetric('int1e_ovlp')
            s21 = mole.intor_cross('int1e_ovlp', xmol, mol)
            c = _block_diag(pyscf_lib.cho_solve(s22, s21))
            c = cp.asarray(c)
            h1 = reduce(cp.dot, (c.T, h1, c))
        if self.xuncontract and contr_coeff is not None:
            contr_coeff = _block_diag(contr_coeff)
            contr_coeff = cp.asarray(contr_coeff)
            h1 = reduce(cp.dot, (contr_coeff.T, h1, contr_coeff))
        return h1

    def get_xmat(self, mol=None):
        if mol is None:
            xmol = self.get_xmol(mol)[0]
        else:
            xmol = mol
        c = pyscf_lib.param.LIGHT_SPEED
        assert ('1E' in self.approx.upper())

        if 'ATOM' in self.approx.upper():
            atom_slices = xmol.offset_nr_by_atom()
            # spin-orbital basis is twice the size of NR basis
            atom_slices[:,2:] *= 2
            nao = xmol.nao_nr() * 2
            x = cp.zeros((nao,nao))
            for ia in range(xmol.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                t1 = _block_diag(xmol.intor('int1e_kin', shls_slice=shls_slice))
                s1 = _block_diag(xmol.intor('int1e_ovlp', shls_slice=shls_slice))
                with xmol.with_rinv_at_nucleus(ia):
                    z = -xmol.atom_charge(ia)
                    v1 = _block_diag(z * xmol.intor('int1e_rinv', shls_slice=shls_slice))
                    w1 = _sigma_dot(z * xmol.intor('int1e_sprinvsp', shls_slice=shls_slice))
                x[p0:p1,p0:p1] = _x2c1e_xmatrix(t1, v1, w1, s1, c)
        else:
            t = _block_diag(xmol.intor_symmetric('int1e_kin'))
            v = _block_diag(xmol.intor_symmetric('int1e_nuc'))
            s = _block_diag(xmol.intor_symmetric('int1e_ovlp'))
            w = _sigma_dot(xmol.intor('int1e_spnucsp'))
            x = _x2c1e_xmatrix(t, v, w, s, c)
        return x


make_rdm1 = hf.make_rdm1


def x2c1e_ghf(mf):
    '''
    For the given *GHF* object, generate X2C-GSCF object in GHF spin-orbital
    basis. Note the orbital basis of X2C_GSCF is different to the X2C_RHF and
    X2C_UHF objects. X2C_RHF and X2C_UHF use spinor basis.

    Args:
        mf : an GHF/GKS object

    Returns:
        An GHF/GKS object

    Examples:

    >>> mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.GHF(mol).x2c1e().run()
    '''
    assert isinstance(mf, ghf.GHF)

    if isinstance(mf, _X2C_SCF):
        if mf.with_x2c is None:
            mf.with_x2c = SpinOrbitalX2CHelper(mf.mol)
            return mf
        elif not isinstance(mf.with_x2c, SpinOrbitalX2CHelper):
            # An object associated to sfx2c1e.SpinFreeX2CHelper
            raise NotImplementedError
        else:
            return mf

    return pyscf_lib.set_class(X2C1E_GSCF(mf), (X2C1E_GSCF, mf.__class__))

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
    to_gpu = utils.to_gpu
    device = utils.device
    _keys = {'with_x2c'}

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self.with_x2c = SpinOrbitalX2CHelper(mf.mol)

    def undo_x2c(self):
        '''Remove the X2C Mixin'''
        obj = pyscf_lib.view(self, pyscf_lib.drop_class(self.__class__, X2C1E_GSCF))
        del obj.with_x2c
        return obj

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return self.with_x2c.get_hcore(mol)

    def dip_moment(self, mol=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   picture_change=True, **kwargs):
        raise NotImplementedError("dipole moment for X2C is not implemented")

    def _transfer_attrs_(self, dst):
        if self.with_x2c and not hasattr(dst, 'with_x2c'):
            logger.warn(self, 'Destination object of to_hf/to_ks method is not '
                        'an X2C object. Convert dst to X2C object.')
            dst = dst.x2c()
        return hf.SCF._transfer_attrs_(self, dst)

    def to_ks(self, xc='HF'):
        raise NotImplementedError

    def to_cpu(self):
        from pyscf.x2c.x2c import X2C1E_GSCF
        x2c1e_obj = X2C1E_GSCF(self)
        utils.to_cpu(self, out=x2c1e_obj)
        return x2c1e_obj


def _uncontract_mol(mol, xuncontract=None, exp_drop=0.2):
    '''mol._basis + uncontracted steep functions'''
    pmol, contr_coeff = mol.decontract_basis(atoms=xuncontract, aggregate=True)
    return pmol, contr_coeff


def _get_hcore_fw(t, v, w, s, x, c):
    # s1 = s + (1/2c^2)(X^{\dag}*T*X) 
    # Eq.(176) in 10.1080/00268971003781571
    s1 = s + reduce(cp.dot, (x.T.conj(), t, x)) * (.5/c**2)
    # tx = T * X 
    tx = cp.dot(t, x)
    # Eq.(176) in 10.1080/00268971003781571
    # h1 = (v + T*X + V^{\dag}*T^{\dag} - (X^{\dag} * T * X) + (X^{\dag} * W * X)*(1/4c^2)
    h1 =(v + tx + tx.T.conj() - cp.dot(x.T.conj(), tx) +
         reduce(cp.dot, (x.T.conj(), w, x)) * (.25/c**2))
    # R = S^{-1/2} * (S^{-1/2}\tilde{S}S^{-1/2})^{-1/2} * S^{1/2}
    r = _get_r(s, s1) #  R_+ 
    # H1 = R^{\dag} * H1 * R
    h1 = reduce(cp.dot, (r.T.conj(), h1, r))
    return h1

def _get_r(s, snesc):
    # R^dag \tilde{S} R = S
    # R = S^{-1/2} [S^{-1/2}\tilde{S}S^{-1/2}]^{-1/2} S^{1/2}
    # Eq.(193) or (223) in 10.1080/00268971003781571
    w, v = cp.linalg.eigh(s)
    idx = w > 1e-14
    v = v[:,idx]
    w_sqrt = cp.sqrt(w[idx])
    w_invsqrt = 1 / w_sqrt

    # eigenvectors of S as the new basis
    snesc = reduce(cp.dot, (v.conj().T, snesc, v))
    r_mid = cp.einsum('i,ij,j->ij', w_invsqrt, snesc, w_invsqrt)
    w1, v1 = cp.linalg.eigh(r_mid)
    idx1 = w1 > 1e-14
    v1 = v1[:,idx1]
    r_mid = cp.dot(v1/cp.sqrt(w1[idx1]), v1.conj().T)
    r = cp.einsum('i,ij,j->ij', w_invsqrt, r_mid, w_sqrt)
    # Back transform to AO basis
    r = reduce(cp.dot, (v, r, v.conj().T))
    return r

def _x2c1e_xmatrix(t, v, w, s, c):
    """
    Solve to get the X2C-1e matrix.
        $$hC = MC\epsilon \quad \text{where} 
        h = \begin{pmatrix} h^{LL} & h^{LS} 
                            h^{SL} & h^{SS} \end{pmatrix}, 
        M = \begin{pmatrix} S & 0 \\ 
                            0 & \frac{1}{2mc^2}T \end{pmatrix}$$
    """
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
        e, a = solve_gen_eigh_cupy(h, m)
        cl = a[:nao,nao:]
        cs = a[nao:,nao:]
        x = cp.linalg.solve(cl.T, cs.T).T  # B = XA
    except cp.linalg.LinAlgError:
        d, t = cp.linalg.eigh(m)
        idx = d>LINEAR_DEP_THRESHOLD
        t = t[:,idx] / cp.sqrt(d[idx])
        tht = reduce(cp.dot, (t.T.conj(), h, t))
        e, a = cp.linalg.eigh(tht)
        a = cp.dot(t, a)
        idx = e > -c**2
        cl = a[:nao,idx]
        cs = a[nao:,idx]
        # X = B A^{-1} = B A^T S
        x = cs.dot(cl.conj().T).dot(m)
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
        e, a = solve_gen_eigh_cupy(h, m)
        cl = a[:nao,nao:]
        # cs = a[nao:,nao:]
        e = e[nao:]
    except cp.linalg.LinAlgError:
        d, t = cp.linalg.eigh(m)
        idx = d>LINEAR_DEP_THRESHOLD
        t = t[:,idx] / cp.sqrt(d[idx])
        tht = reduce(cp.dot, (t.T.conj(), h, t))
        e, a = cp.linalg.eigh(tht)
        a = cp.dot(t, a)
        idx = e > -c**2
        cl = a[:nao,idx]
        # cs = a[nao:,idx]
        e = e[idx]

# The so obtaied X seems not numerically stable.  We changed to the
# transformed matrix
# [1 1] [ V T ] [1 0]
# [0 1] [ T W ] [1 1]
#            h[:nao,:nao] = h[:nao,nao:] = h[nao:,:nao] = h[nao:,nao:] = w * (.25/c**2)
#            m[:nao,:nao] = m[:nao,nao:] = m[nao:,:nao] = m[nao:,nao:] = t * (.5/c**2)
#            h[:nao,:nao]+= v + t
#            h[nao:,nao:]-= t
#            m[:nao,:nao]+= s
#            e, a = scipy.linalg.eigh(h, m)
#            cl = a[:nao,nao:]
#            cs = a[nao:,nao:]
#            x = cp.eye(nao) + cp.linalg.solve(cl.T, cs.T).T  # B = XA
#            h1 = _get_hcore_fw(t, v, w, s, x, c)

# Taking A matrix as basis and rewrite the FW Hcore formula, to avoid inversing matrix
#   R^dag \tilde{S} R = S
#   R = S^{-1/2} [S^{-1/2}\tilde{S}S^{-1/2}]^{-1/2} S^{1/2}
# Using A matrix as basis, the representation of R is
#   R[A] = (A^+ S A)^{1/2} = (A^+ S A)^{-1/2} A^+ S A
# Construct h = R^+ h1 R in two steps, first in basis A matrix, then back
# transformed to AO basis
#   h  = (A^+)^{-1} R[A]^+ (A^+ h1 A) R[A] A^{-1}         (0)
# Using (A^+)^{-1} = \tilde{S} A, h can be transformed to
#   h  = \tilde{S} A R[A]^+ A^+ h1 A R[A] A^+ \tilde{S}   (1)
# Using R[A] = R[A]^{-1} A^+ S A,  Eq (0) turns to
#      = S A R[A]^{-1}^+ A^+ h1 A R[A]^{-1} A^+ S
#      = S A R[A]^{-1}^+ e R[A]^{-1} A^+ S                (2)

    w, u = cp.linalg.eigh(reduce(cp.dot, (cl.T.conj(), s, cl)))
    idx = w > 1e-14
    # Adopt (2) here because X is not appeared in Eq (2).
    # R[A] = u w^{1/2} u^+,  so R[A]^{-1} A^+ S in Eq (2) is
    r = reduce(cp.dot, (u[:,idx]/cp.sqrt(w[idx]), u[:,idx].T.conj(),
                           cl.T.conj(), s))
    h1 = reduce(cp.dot, (r.T.conj()*e, r))
    return h1


def _block_diag(mat):
    '''
    [A 0]
    [0 A]
    '''
    return scipy.linalg.block_diag(mat, mat)

def _sigma_dot(mat):
    '''sigma dot A x B + A dot B'''
    quaternion = cp.vstack([1j * cp.asarray(pyscf_lib.PauliMatrices), cp.eye(2)[None,:,:]])
    nao = mat.shape[-1] * 2
    return contract('sxy,spq->xpyq', quaternion, mat).reshape(nao, nao)


def solve_gen_eigh_cupy(h, m):
    """
    Solves Hx = \lambda Mx using CuPy.
    Equivalent to numpy.linalg.eigh(h, m).
    
    Args:
        h (cp.ndarray): Hermitian matrix H
        m (cp.ndarray): Hermitian positive-definite matrix M
    
    Returns:
        tuple (e, a): eigenvalues (e) and eigenvectors (a)
    """
    
    try:
        # 1. Cholesky decomposition: M = L L^H
        L = cp.linalg.cholesky(m)
    except cp.linalg.LinAlgError as e:
        print(f"ERROR: Matrix M is not positive-definite. {e}")
        return None, None

    # 2. Transform H to C = L^{-1} H (L^H)^{-1}
    
    # K = L^{-1} H  (by solving L K = H)
    K = cp.linalg.solve(L, h)
    
    # C = K (L^H)^{-1}  (by solving L C^H = K^H, then C = (C^H)^H)
    K_H = K.T.conj()
    C_H = cp.linalg.solve(L, K_H)
    C = C_H.T.conj()
    
    # 3. Solve standard problem: C y = \lambda y
    # Symmetrize C to remove numerical noise
    C_hermitian = (C + C.T.conj()) * 0.5
    e, y = cp.linalg.eigh(C_hermitian)

    # 4. Back-transform eigenvectors: a = (L^H)^{-1} y
    # (by solving L^H a = y)
    a = cp.linalg.solve(L.T.conj(), y)
    
    return e, a