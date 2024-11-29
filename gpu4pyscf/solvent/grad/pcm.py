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

'''
Gradient of PCM family solvent model
'''
# pylint: disable=C0103

import numpy
import cupy
import ctypes
from cupyx import scipy
from pyscf import lib
from pyscf import gto
from pyscf.grad import rhf as rhf_grad

from gpu4pyscf.solvent.pcm import PI, switch_h, libsolvent
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import contract, load_library
from gpu4pyscf.lib import logger
from pyscf import lib as pyscf_lib

libdft = lib.load_library('libdft')

def grad_switch_h(x):
    ''' first derivative of h(x)'''
    dy = 30.0*x**2 - 60.0*x**3 + 30.0*x**4
    dy[x<0] = 0.0
    dy[x>1] = 0.0
    return dy

def gradgrad_switch_h(x):
    ''' 2nd derivative of h(x) '''
    ddy = 60.0*x - 180.0*x**2 + 120*x**3
    ddy[x<0] = 0.0
    ddy[x>1] = 0.0
    return ddy

def get_dF_dA(surface):
    '''
    J. Chem. Phys. 133, 244111 (2010), Appendix C
    '''
    atom_coords = surface['atom_coords']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    area        = surface['area']
    R_in_J      = surface['R_in_J']
    R_sw_J      = surface['R_sw_J']

    ngrids = grid_coords.shape[0]
    natom = atom_coords.shape[0]
    dF = cupy.zeros([ngrids, natom, 3])
    dA = cupy.zeros([ngrids, natom, 3])

    for ia in range(atom_coords.shape[0]):
        p0,p1 = surface['gslice_by_atom'][ia]
        coords = grid_coords[p0:p1]
        p1 = p0 + coords.shape[0]
        ri_rJ = cupy.expand_dims(coords, axis=1) - atom_coords
        riJ = cupy.linalg.norm(ri_rJ, axis=-1)
        diJ = (riJ - R_in_J) / R_sw_J
        diJ[:,ia] = 1.0
        diJ[diJ < 1e-8] = 0.0
        ri_rJ[:,ia,:] = 0.0
        ri_rJ[diJ < 1e-8] = 0.0

        fiJ = switch_h(diJ)
        dfiJ = grad_switch_h(diJ) / (fiJ * riJ * R_sw_J)
        dfiJ = cupy.expand_dims(dfiJ, axis=-1) * ri_rJ

        Fi = switch_fun[p0:p1]
        Ai = area[p0:p1]

        # grids response
        Fi = cupy.expand_dims(Fi, axis=-1)
        Ai = cupy.expand_dims(Ai, axis=-1)
        dFi_grid = cupy.sum(dfiJ, axis=1)

        dF[p0:p1,ia,:] += Fi * dFi_grid
        dA[p0:p1,ia,:] += Ai * dFi_grid

        # atom response
        Fi = cupy.expand_dims(Fi, axis=-2)
        Ai = cupy.expand_dims(Ai, axis=-2)
        dF[p0:p1,:,:] -= Fi * dfiJ
        dA[p0:p1,:,:] -= Ai * dfiJ
    dF = dF.transpose([2,0,1])
    dA = dA.transpose([2,0,1])
    return dF, dA

def get_dD_dS_slow(surface, with_S=True, with_D=False):
    '''
    derivative of D and S w.r.t grids, partial_i D_ij = -partial_j D_ij
    S is symmetric, D is not
    '''
    grid_coords = surface['grid_coords']
    exponents   = surface['charge_exp']
    norm_vec    = surface['norm_vec']

    xi_i, xi_j = cupy.meshgrid(exponents, exponents, indexing='ij')
    xi_ij = xi_i * xi_j / (xi_i**2 + xi_j**2)**0.5
    ri_rj = cupy.expand_dims(grid_coords, axis=1) - grid_coords
    rij = cupy.linalg.norm(ri_rj, axis=-1)
    xi_r_ij = xi_ij * rij
    cupy.fill_diagonal(rij, 1)
    xi_i = xi_j = None

    dS_dr = -(scipy.special.erf(xi_r_ij) - 2.0*xi_r_ij/PI**0.5*cupy.exp(-xi_r_ij**2))/rij**2
    cupy.fill_diagonal(dS_dr, 0)

    dS_dr= cupy.expand_dims(dS_dr, axis=-1)
    drij = ri_rj/cupy.expand_dims(rij, axis=-1)
    dS = dS_dr * drij

    dD = None
    if with_D:
        nj_rij = cupy.sum(ri_rj * norm_vec, axis=-1)
        dD_dri = 4.0*xi_r_ij**2 * xi_ij / PI**0.5 * cupy.exp(-xi_r_ij**2) * nj_rij / rij**3
        cupy.fill_diagonal(dD_dri, 0.0)

        rij = cupy.expand_dims(rij, axis=-1)
        nj_rij = cupy.expand_dims(nj_rij, axis=-1)
        nj = cupy.expand_dims(norm_vec, axis=0)
        dD_dri = cupy.expand_dims(dD_dri, axis=-1)

        dD = dD_dri * drij + dS_dr * (-nj/rij + 3.0*nj_rij/rij**2 * drij)
        dD_dri = None
    dD = dD.transpose([2,0,1])
    dS = dS.transpose([2,0,1])
    return dD, dS

def get_dD_dS(surface, with_S=True, with_D=False, stream=None):
    ''' Derivatives of D matrix and S matrix (offdiagonals only)
    '''
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    norm_vec    = surface['norm_vec']
    R_vdw       = surface['R_vdw']
    n = charge_exp.shape[0]
    dS = cupy.empty([3,n,n])
    dD = None
    dS_ptr = ctypes.cast(dS.data.ptr, ctypes.c_void_p)
    dD_ptr = pyscf_lib.c_null_ptr()
    if with_D:
        dD = cupy.empty([3,n,n])
        dD_ptr = ctypes.cast(dD.data.ptr, ctypes.c_void_p)
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libsolvent.pcm_dd_ds(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        dD_ptr, dS_ptr,
        ctypes.cast(grid_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(norm_vec.data.ptr, ctypes.c_void_p),
        ctypes.cast(R_vdw.data.ptr, ctypes.c_void_p),
        ctypes.cast(charge_exp.data.ptr, ctypes.c_void_p),
        ctypes.cast(switch_fun.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n)
    )
    if err != 0:
        raise RuntimeError('Failed in generating PCM dD and dS matrices.')
    return dD, dS

def get_dSii(surface, dF):
    ''' Derivative of S matrix (diagonal only)
    '''
    charge_exp  = surface['charge_exp']
    switch_fun  = surface['switch_fun']
    dSii_dF = -charge_exp * (2.0/PI)**0.5 / switch_fun**2
    dSii = dSii_dF[:,None] * dF
    return dSii

def grad_nuc(pcmobj, dm):
    mol = pcmobj.mol
    log = logger.new_logger(mol, mol.verbose)
    t1 = log.init_timer()
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is not None and cupy.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        pcmobj._get_vind(dm)

    mol = pcmobj.mol
    q_sym        = pcmobj._intermediates['q_sym'].get()
    gridslice    = pcmobj.surface['gslice_by_atom']
    grid_coords  = pcmobj.surface['grid_coords'].get()
    exponents    = pcmobj.surface['charge_exp'].get()

    atom_coords = mol.atom_coords(unit='B')
    atom_charges = numpy.asarray(mol.atom_charges(), dtype=numpy.float64)
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)
    fakemol = gto.fakemol_for_charges(grid_coords, expnt=exponents**2)

    int2c2e_ip1 = mol._add_suffix('int2c2e_ip1')

    v_ng_ip1 = gto.mole.intor_cross(int2c2e_ip1, fakemol_nuc, fakemol)

    dv_g = numpy.einsum('g,xng->nx', q_sym, v_ng_ip1)
    de = -numpy.einsum('nx,n->nx', dv_g, atom_charges)

    v_ng_ip1 = gto.mole.intor_cross(int2c2e_ip1, fakemol, fakemol_nuc)

    dv_g = numpy.einsum('n,xgn->gx', atom_charges, v_ng_ip1)
    dv_g = numpy.einsum('gx,g->gx', dv_g, q_sym)

    de -= numpy.asarray([numpy.sum(dv_g[p0:p1], axis=0) for p0,p1 in gridslice])
    t1 = log.timer_debug1('grad nuc', *t1)
    return de

def grad_qv(pcmobj, dm):
    '''
    contributions due to integrals
    '''
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is not None and cupy.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    log = logger.new_logger(mol, mol.verbose)
    t1 = log.init_timer()
    gridslice   = pcmobj.surface['gslice_by_atom']
    charge_exp  = pcmobj.surface['charge_exp']
    grid_coords = pcmobj.surface['grid_coords']
    q_sym       = pcmobj._intermediates['q_sym']

    auxmol = gto.fakemol_for_charges(grid_coords.get(), expnt=charge_exp.get()**2)
    intopt = int3c2e.VHFOpt(mol, auxmol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False)
    coeff = intopt.coeff
    dm_cart = coeff @ dm @ coeff.T

    dvj, _ = int3c2e.get_int3c2e_ip_jk(intopt, 0, 'ip1', q_sym, None, dm_cart)
    dq, _ = int3c2e.get_int3c2e_ip_jk(intopt, 0, 'ip2', q_sym, None, dm_cart)
    
    _sorted_mol = intopt._sorted_mol
    nao_cart = _sorted_mol.nao
    natm = _sorted_mol.natm
    ao2atom = numpy.zeros([nao_cart, natm])
    ao_loc = _sorted_mol.ao_loc
    for ibas, iatm in enumerate(_sorted_mol._bas[:,gto.ATOM_OF]):
        ao2atom[ao_loc[ibas]:ao_loc[ibas+1],iatm] = 1
    ao2atom = cupy.asarray(ao2atom)
    dvj = 2.0 * ao2atom.T @ dvj.T
    dq = cupy.asarray([cupy.sum(dq[:,p0:p1], axis=1) for p0,p1 in gridslice])
    de = dq + dvj
    t1 = log.timer_debug1('grad qv', *t1)
    return de.get()

def grad_solver(pcmobj, dm):
    '''
    dE = 0.5*v* d(K^-1 R) *v + q*dv
    v^T* d(K^-1 R)v = v^T*K^-1(dR - dK K^-1R)v = v^T K^-1(dR - dK q)
    '''
    mol = pcmobj.mol
    log = logger.new_logger(mol, mol.verbose)
    t1 = log.init_timer()
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is not None and cupy.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        pcmobj._get_vind(dm)

    gridslice    = pcmobj.surface['gslice_by_atom']
    v_grids      = pcmobj._intermediates['v_grids']
    A            = pcmobj._intermediates['A']
    D            = pcmobj._intermediates['D']
    S            = pcmobj._intermediates['S']
    K            = pcmobj._intermediates['K']
    q            = pcmobj._intermediates['q']

    vK_1 = cupy.linalg.solve(K.T, v_grids)
    epsilon = pcmobj.eps

    de = cupy.zeros([pcmobj.mol.natm,3])
    if pcmobj.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
        dD, dS = get_dD_dS(pcmobj.surface, with_D=False, with_S=True)
        
        # dR = 0, dK = dS
        de_dS = (vK_1 * dS.dot(q)).T                  # cupy.einsum('i,xij,j->ix', vK_1, dS, q)
        de -= cupy.asarray([cupy.sum(de_dS[p0:p1], axis=0) for p0,p1 in gridslice])
        dD = dS = None

        dF, dA = get_dF_dA(pcmobj.surface)
        dSii = get_dSii(pcmobj.surface, dF)
        de -= 0.5*contract('i,xij->jx', vK_1*q, dSii) # 0.5*cupy.einsum('i,xij,i->jx', vK_1, dSii, q)

    elif pcmobj.method.upper() in ['IEF-PCM', 'IEFPCM', 'SS(V)PE', 'SMD']:
        dF, dA = get_dF_dA(pcmobj.surface)
        dSii = get_dSii(pcmobj.surface, dF)
        dF = None

        dD, dS = get_dD_dS(pcmobj.surface, with_D=True, with_S=True)

        def contract_bra(a, B, c):
            ''' i,xij,j->jx '''
            tmp = a.dot(B)
            return (tmp*c).T

        def contract_ket(a, B, c):
            ''' i,xij,j->ix '''
            tmp = B.dot(c)
            return (a*tmp).T
        
        # IEF-PCM and SS(V)PE formally are the same in gradient calculation
        # dR = f_eps/(2*pi) * (dD*A + D*dA),
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)
        f_epsilon = (epsilon - 1.0)/(epsilon + 1.0)
        fac = f_epsilon/(2.0*PI)

        Av = A*v_grids
        de_dR  = 0.5*fac * contract_ket(vK_1, dD, Av)
        de_dR -= 0.5*fac * contract_bra(vK_1, dD, Av)
        de_dR  = cupy.asarray([cupy.sum(de_dR[p0:p1], axis=0) for p0,p1 in gridslice])

        vK_1_D = vK_1.dot(D)
        vK_1_Dv = vK_1_D * v_grids
        de_dR += 0.5*fac * contract('j,xjn->nx', vK_1_Dv, dA)

        de_dS0  = 0.5*contract_ket(vK_1, dS, q)
        de_dS0 -= 0.5*contract_bra(vK_1, dS, q)
        de_dS0  = cupy.asarray([cupy.sum(de_dS0[p0:p1], axis=0) for p0,p1 in gridslice])

        vK_1_q = vK_1 * q
        de_dS0 += 0.5*contract('i,xin->nx', vK_1_q, dSii)

        vK_1_DA = vK_1_D*A
        de_dS1  = 0.5*contract_ket(vK_1_DA, dS, q)
        de_dS1 -= 0.5*contract_bra(vK_1_DA, dS, q)
        de_dS1  = cupy.asarray([cupy.sum(de_dS1[p0:p1], axis=0) for p0,p1 in gridslice])

        vK_1_DAq = vK_1_DA*q
        de_dS1 += 0.5*contract('j,xjn->nx', vK_1_DAq, dSii)

        Sq = cupy.dot(S,q)
        ASq = A*Sq
        de_dD  = 0.5*contract_ket(vK_1, dD, ASq)
        de_dD -= 0.5*contract_bra(vK_1, dD, ASq)
        de_dD  = cupy.asarray([cupy.sum(de_dD[p0:p1], axis=0) for p0,p1 in gridslice])

        de_dA = 0.5*contract('j,xjn->nx', vK_1_D*Sq, dA)   # 0.5*cupy.einsum('j,xjn,j->nx', vK_1_D, dA, Sq)

        de_dK = de_dS0 - fac * (de_dD + de_dA + de_dS1)
        de += de_dR - de_dK
    else:
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")
    t1 = log.timer_debug1('grad solver', *t1)
    return de.get()

def make_grad_object(grad_method):
    '''For grad_method in vacuum, add nuclear gradients of solvent pcmobj'''
    if grad_method.base.with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy gradients')

    name = (grad_method.base.with_solvent.__class__.__name__
            + grad_method.__class__.__name__)
    return lib.set_class(WithSolventGrad(grad_method),
                         (WithSolventGrad, grad_method.__class__), name)

class WithSolventGrad:
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventGrad, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def to_cpu(self):
        from pyscf.solvent.grad import pcm  # type: ignore
        grad_method = self.undo_solvent().to_cpu()
        return pcm.make_grad_object(grad_method)

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        dm = kwargs.pop('dm', None)
        if dm is None:
            dm = self.base.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        self.de_solute = super().kernel(*args, **kwargs)
        self.de_solvent = grad_qv(self.base.with_solvent, dm)
        self.de_solvent+= grad_solver(self.base.with_solvent, dm)
        self.de_solvent+= grad_nuc(self.base.with_solvent, dm)
        self.de = self.de_solute + self.de_solvent

        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s (+%s) gradients ---------------',
                        self.base.__class__.__name__,
                        self.base.with_solvent.__class__.__name__)
            rhf_grad._write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')
        return self.de

    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass
