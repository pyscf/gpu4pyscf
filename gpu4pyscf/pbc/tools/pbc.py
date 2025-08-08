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

import numpy as np
import cupy as cp
import pyscf
from pyscf import lib
from pyscf.pbc.gto.cell import Cell
from gpu4pyscf.lib.cupy_helper import asarray

def fft(f, mesh):
    '''Perform the 3D FFT from real (R) to reciprocal (G) space.

    After FFT, (u, v, w) -> (j, k, l).
    (jkl) is in the index order of Gv.

    FFT normalization factor is 1., as in MH and in `numpy.fft`.

    Args:
        f : (nx*ny*nz,) ndarray
            The function to be FFT'd, flattened to a 1D array corresponding
            to the index order of :func:`cartesian_prod`.
        mesh : (3,) ndarray of ints (= nx,ny,nz)
            The number G-vectors along each direction.

    Returns:
        (nx*ny*nz,) ndarray
            The FFT 1D array in same index order as Gv (natural order of
            numpy.fft).

    '''
    if f.size == 0:
        return cp.zeros_like(f)

    f3d = cp.asarray(f).reshape(-1, *mesh)
    assert (f3d.shape[0] == 1 or f[0].size == f3d[0].size)
    g3d = cp.fft.fftn(f3d, axes=(1,2,3))
    ngrids = np.prod(mesh)
    if f.ndim == 1 or (f.ndim == 3 and f.size == ngrids):
        return g3d.ravel()
    else:
        return g3d.reshape(-1, ngrids)

def ifft(g, mesh):
    '''Perform the 3D inverse FFT from reciprocal (G) space to real (R) space.

    Inverse FFT normalization factor is 1./N, same as in `numpy.fft` but
    **different** from MH (they use 1.).

    Args:
        g : (nx*ny*nz,) ndarray
            The function to be inverse FFT'd, flattened to a 1D array
            corresponding to the index order of `span3`.
        mesh : (3,) ndarray of ints (= nx,ny,nz)
            The number G-vectors along each direction.

    Returns:
        (nx*ny*nz,) ndarray
            The inverse FFT 1D array in same index order as Gv (natural order
            of numpy.fft).

    '''
    if g.size == 0:
        return cp.zeros_like(g)

    g3d = cp.asarray(g).reshape(-1, *mesh)
    assert (g3d.shape[0] == 1 or g[0].size == g3d[0].size)
    f3d = cp.fft.ifftn(g3d, axes=(1,2,3))
    ngrids = np.prod(mesh)
    if g.ndim == 1 or (g.ndim == 3 and g.size == ngrids):
        return f3d.ravel()
    else:
        return f3d.reshape(-1, ngrids)


def fftk(f, mesh, expmikr):
    r'''Perform the 3D FFT of a real-space function which is (periodic*e^{ikr}).

    fk(k+G) = \sum_r fk(r) e^{-i(k+G)r} = \sum_r [f(k)e^{-ikr}] e^{-iGr}
    '''
    return fft(f*expmikr, mesh)


def ifftk(g, mesh, expikr):
    r'''Perform the 3D inverse FFT of f(k+G) into a function which is (periodic*e^{ikr}).

    fk(r) = (1/Ng) \sum_G fk(k+G) e^{i(k+G)r} = (1/Ng) \sum_G [fk(k+G)e^{iGr}] e^{ikr}
    '''
    return ifft(g, mesh) * expikr

def _get_Gv(cell, mesh):
    # Default, the 3D uniform grids
    rx = cp.fft.fftfreq(mesh[0], 1./mesh[0])
    ry = cp.fft.fftfreq(mesh[1], 1./mesh[1])
    rz = cp.fft.fftfreq(mesh[2], 1./mesh[2])
    b = cp.asarray(cell.reciprocal_vectors())
    #:Gv = lib.cartesian_prod(Gvbase).dot(b)
    Gv = (rx[:,None,None,None] * b[0] +
          ry[:,None,None] * b[1] +
          rz[:,None] * b[2])
    return Gv.reshape(-1, 3)

def get_coulG(cell, k=np.zeros(3), exx=False, mf=None, mesh=None, Gv=None,
              wrap_around=True, omega=None, **kwargs):
    '''Calculate the Coulomb kernel for all G-vectors, handling G=0 and exchange.

    Args:
        k : (3,) ndarray
            k-point
        exx : bool or str
            Whether this is an exchange matrix element
        mf : instance of :class:`SCF`

    Returns:
        coulG : (ngrids,) ndarray
            The Coulomb kernel.
        mesh : (3,) ndarray of ints (= nx,ny,nz)
            The number G-vectors along each direction.
        omega : float
            Enable Coulomb kernel ``erf(|omega|*r12)/r12`` if omega > 0
            and ``erfc(|omega|*r12)/r12`` if omega < 0.
            Note this parameter is slightly different to setting cell.omega for
            exxdiv='ewald' at G0. When cell.omega is configured, the Ewald probe
            charge correction will be computed using the LR or SR Coulomb
            interactions. However, when this kwarg is explicitly specified, the
            exxdiv correction is computed with the full-range Coulomb
            interaction (1/r12). This parameter should only be specified in the
            range-separated JK builder and range-separated DF (and other
            range-separated integral methods if any).
    '''
    from pyscf.pbc.tools.pbc import get_coulG, madelung
    exxdiv = exx
    if isinstance(exx, str):
        exxdiv = exx
    elif exx and mf is not None:
        exxdiv = mf.exxdiv
    if exxdiv == 'vcut_sph' or exxdiv == 'vcut_ws':
        return asarray(get_coulG(cell, k, exx, mf, mesh, Gv, wrap_around, omega, **kwargs))

    if mesh is None:
        mesh = cell.mesh
    if Gv is None:
        Gv = _get_Gv(cell, mesh)
    Gv = asarray(Gv)

    if omega is None:
        _omega = cell.omega
    else:
        _omega = omega

    if cell.dimension == 0 and cell.low_dim_ft_type != 'inf_vacuum':
        a = cell.lattice_vectors()
        assert abs(np.eye(3)*a[0,0] - a).max() < 1e-6, \
                'Must be cubic box for cell.dimension=0'
        # ensure the sphere is completely inside the box
        Rc = a[0,0] / 2
        if (_omega != 0 and
            abs(_omega) * Rc < 2.0): # typically, error of \int erf(omega r) sin (G r) < 1e-5
            raise RuntimeError(
                'In sufficient box size for the truncated range-separated '
                'Coulomb potential in 0D case')
        absG = cp.linalg.norm(Gv, axis=1)
        with np.errstate(divide='ignore',invalid='ignore'):
            coulG = 4*np.pi/absG**2
            coulG[0] = 0
        if _omega == 0:
            coulG *= 1. - cp.cos(absG*Rc)
            # G=0 term carries the charge. This special term supports the charged
            # system for dimension=0.
            coulG[0] = 2*cp.pi*Rc**2
        elif _omega > 0:
            coulG *= cp.exp(-.25/_omega**2 * absG**2) - cp.cos(absG*Rc)
            coulG[0] = 2*np.pi*Rc**2 - np.pi / _omega**2
        else:
            coulG *= 1 - cp.exp(-.25/_omega**2 * absG**2)
            coulG[0] = np.pi / _omega**2
        return coulG

    if abs(k).sum() > 1e-9:
        kG = asarray(k) + Gv
    else:
        kG = Gv

    equal2boundary = None
    if wrap_around and abs(k).sum() > 1e-9:
        equal2boundary = cp.zeros(Gv.shape[0], dtype=bool)
        # Here we 'wrap around' the high frequency k+G vectors into their lower
        # frequency counterparts.  Important if you want the gamma point and k-point
        # answers to agree
        b = cell.reciprocal_vectors()
        box_edge = np.einsum('i,ij->ij', np.asarray(mesh)//2+0.5, b)
        assert all(np.rint(np.linalg.solve(box_edge.T, k))==0)
        box_edge = asarray(box_edge)
        reduced_coords = cp.linalg.solve(box_edge.T, kG.T).T
        on_edge_p1 = abs(reduced_coords - 1) < 1e-9
        on_edge_m1 = abs(reduced_coords + 1) < 1e-9
        if cell.dimension >= 1:
            equal2boundary |= on_edge_p1[:,0]
            equal2boundary |= on_edge_m1[:,0]
            kG[reduced_coords[:,0]> 1] -= 2 * box_edge[0]
            kG[reduced_coords[:,0]<-1] += 2 * box_edge[0]
        if cell.dimension >= 2:
            equal2boundary |= on_edge_p1[:,1]
            equal2boundary |= on_edge_m1[:,1]
            kG[reduced_coords[:,1]> 1] -= 2 * box_edge[1]
            kG[reduced_coords[:,1]<-1] += 2 * box_edge[1]
        if cell.dimension == 3:
            equal2boundary |= on_edge_p1[:,2]
            equal2boundary |= on_edge_m1[:,2]
            kG[reduced_coords[:,2]> 1] -= 2 * box_edge[2]
            kG[reduced_coords[:,2]<-1] += 2 * box_edge[2]

    absG2 = cp.einsum('gi,gi->g', kG, kG)
    G0_idx = 0
    if abs(k).sum() > 1e-9:
        G0_idx = None

    # Ewald probe charge method to get the leading term of the finite size
    # error in exchange integrals

    if cell.dimension == 3 or cell.low_dim_ft_type == 'inf_vacuum':
        with np.errstate(divide='ignore'):
            coulG = 4*np.pi/absG2
            if G0_idx is not None:
                coulG[G0_idx] = 0

    elif cell.dimension == 2:
        # The following 2D analytical fourier transform is taken from:
        # R. Sundararaman and T. Arias PRB 87, 2013
        b = cell.reciprocal_vectors()
        Ld2 = np.pi/np.linalg.norm(b[2])
        Gz = kG[:,2]
        Gp = cp.linalg.norm(kG[:,:2], axis=1)
        weights = 1. - cp.cos(Gz*Ld2) * cp.exp(-Gp*Ld2)
        with np.errstate(divide='ignore', invalid='ignore'):
            coulG = weights*4*np.pi/absG2
        if G0_idx is not None:
            coulG[G0_idx] = -2*np.pi*Ld2**2 #-pi*L_z^2/2

    else:
        raise NotImplementedError(f'dimension={cell.dimension}')

    if equal2boundary is not None:
        coulG[equal2boundary] = 0

    # Scale the coulG kernel for attenuated Coulomb integrals.
    # * kwarg omega is used by RangeSeparatedJKBuilder which requires ewald probe charge
    # being evaluated with regular Coulomb interaction (1/r12).
    # * cell.omega, which affects the ewald probe charge, is often set by
    # DFT-RSH functionals to build long-range HF-exchange for erf(omega*r12)/r12
    if _omega != 0 and cell.dimension != 3:
        raise RuntimeError('The coulG kernel for range-separated Coulomb potential '
                           f'for dimension={cell.dimension} is inaccurate.')

    if _omega > 0:
        # long range part
        coulG *= cp.exp(-.25/_omega**2 * absG2)
    elif _omega < 0:
        if exxdiv == 'vcut_sph' or exxdiv == 'vcut_ws':
            raise RuntimeError(f'SR Coulomb for exxdiv={exxdiv} is not available')
        # short range part
        coulG *= (1 - cp.exp(-.25/_omega**2 * absG2))

    # For full-range Coulomb and long-range Coulomb,
    # the divergent part of periodic summation of (ii|ii) integrals in
    # Coulomb integrals were cancelled out by electron-nucleus
    # interaction. The periodic part of (ii|ii) in exchange cannot be
    # cancelled out by Coulomb integrals. Its leading term is calculated
    # using Ewald probe charge (the function madelung below)
    if cell.dimension > 0 and exxdiv == 'ewald' and G0_idx is not None:
        if hasattr(mf, 'kpts'):
            kpts = mf.kpts
            assert isinstance(kpts, np.ndarray)
            Nk = len(kpts)
        else:
            Nk = 1
        if omega is None: # Affects DFT-RSH
            coulG[G0_idx] += Nk*cell.vol*madelung(cell, kpts)
        else: # for RangeSeparatedJKBuilder
            coulG[G0_idx] += Nk*cell.vol*madelung(cell, kpts, omega=0)
    return coulG
