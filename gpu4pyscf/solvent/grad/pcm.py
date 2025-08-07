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
from gpu4pyscf.gto import int3c1e
from gpu4pyscf.solvent.pcm import PI, switch_h, libsolvent
from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1, int1e_grids_ip2
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib import logger
from gpu4pyscf.grad.rhf import GradientsBase
from pyscf import lib as pyscf_lib

def grad_switch_h(x):
    ''' first derivative of h(x)'''
    dy = 30.0*x**2 - 60.0*x**3 + 30.0*x**4
    dy[x<0] = 0.0
    dy[x>1] = 0.0
    return dy

def get_dF_dA(surface, with_dA = True):
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
    if with_dA:
        dA = cupy.zeros([ngrids, natom, 3])
    else:
        dA = None

    for ia in range(natom):
        p0,p1 = surface['gslice_by_atom'][ia]
        coords = grid_coords[p0:p1]
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
        if with_dA:
            Ai = area[p0:p1]

        # grids response
        Fi = cupy.expand_dims(Fi, axis=-1)
        dFi_grid = cupy.sum(dfiJ, axis=1)

        dF[p0:p1,ia,:] += Fi * dFi_grid
        if with_dA:
            Ai = cupy.expand_dims(Ai, axis=-1)
            dA[p0:p1,ia,:] += Ai * dFi_grid

        # atom response
        Fi = cupy.expand_dims(Fi, axis=-2)
        dF[p0:p1,:,:] -= Fi * dfiJ
        if with_dA:
            Ai = cupy.expand_dims(Ai, axis=-2)
            dA[p0:p1,:,:] -= Ai * dfiJ
    dF = dF.transpose([2,0,1])
    if with_dA:
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
    norm_vec    = surface['norm_vec']
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
        ctypes.cast(charge_exp.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n)
    )
    if err != 0:
        raise RuntimeError('Failed in generating PCM dD and dS matrices.')
    return dD, dS

def left_multiply_dS(surface, right_vector, stream=None):
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    n = charge_exp.shape[0]
    output = cupy.empty([3,n], order = "C")
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libsolvent.pcm_left_multiply_ds(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(output.data.ptr, ctypes.c_void_p),
        ctypes.cast(right_vector.data.ptr, ctypes.c_void_p),
        ctypes.cast(grid_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(charge_exp.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n)
    )
    if err != 0:
        raise RuntimeError('Failed in generating PCM dD and dS matrices.')
    return output.T

# Assuming S is symmetric and Sij depends only on ri and rj
def right_multiply_dS(surface, right_vector, stream=None):
    return -left_multiply_dS(surface, right_vector, stream)

def get_dSii(surface, dF):
    ''' Derivative of S matrix (diagonal only)
    '''
    charge_exp  = surface['charge_exp']
    switch_fun  = surface['switch_fun']
    dSii_dF = -charge_exp * (2.0/PI)**0.5 / switch_fun**2
    dSii = dSii_dF[:,None] * dF
    return dSii

def grad_nuc(pcmobj, dm, q_sym = None):
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
    if q_sym is None:
        q_sym = pcmobj._intermediates['q_sym'].get()
    gridslice    = pcmobj.surface['gslice_by_atom']
    grid_coords  = pcmobj.surface['grid_coords'].get()
    exponents    = pcmobj.surface['charge_exp'].get()

    if pcmobj.frozen_dm0_for_finite_difference_without_response is not None:
        # Note: The q_sym computed above actually use frozen_dm0 as input, so it's actually q_sym_right
        q_sym_left, _ = pcmobj._get_qsym(dm, with_nuc = True)
        q_sym_left = q_sym_left.get()
        q_sym += q_sym_left

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

def grad_qv(pcmobj, dm, q_sym = None):
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
    if q_sym is None:
        q_sym = pcmobj._intermediates['q_sym']

    intopt = int3c1e.VHFOpt(mol)
    intopt.build(1e-14, aosym=False)
    dvj = int1e_grids_ip1(mol, grid_coords, dm = dm, charges = q_sym,
                          direct_scf_tol = 1e-14, charge_exponents = charge_exp**2,
                          intopt=intopt)
    dq  = int1e_grids_ip2(mol, grid_coords, dm = dm, charges = q_sym,
                          direct_scf_tol = 1e-14, charge_exponents = charge_exp**2,
                          intopt=intopt)

    if pcmobj.frozen_dm0_for_finite_difference_without_response is not None:
        frozen_dm0 = pcmobj.frozen_dm0_for_finite_difference_without_response
        # Note: The q_sym computed above actually use frozen_dm0 as input, so it's actually q_sym_right
        q_sym_left, _ = pcmobj._get_qsym(dm, with_nuc = True)
        dvj += int1e_grids_ip1(mol, grid_coords, dm = frozen_dm0, charges = q_sym_left,
                              direct_scf_tol = 1e-14, charge_exponents = charge_exp**2,
                              intopt=intopt)
        dq  += int1e_grids_ip2(mol, grid_coords, dm = frozen_dm0, charges = q_sym_left,
                              direct_scf_tol = 1e-14, charge_exponents = charge_exp**2,
                              intopt=intopt)

    aoslice = mol.aoslice_by_atom()
    dvj = 2.0 * cupy.asarray([cupy.sum(dvj[:,p0:p1], axis=1) for p0,p1 in aoslice[:,2:]])
    dq = cupy.asarray([cupy.sum(dq[:,p0:p1], axis=1) for p0,p1 in gridslice])
    de = dq + dvj
    t1 = log.timer_debug1('grad qv', *t1)
    return de.get()

def grad_solver(pcmobj, dm, v_grids = None, v_grids_l = None, q = None):
    '''
    dE = 0.5*v* d(K^-1 R) *v + q*dv
    v^T* d(K^-1 R)v = v^T*K^-1(dR - dK K^-1R)v = v^T K^-1(dR v - dK q)
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
    if v_grids is None:
        v_grids = pcmobj._intermediates['v_grids']
    if v_grids_l is None:
        v_grids_l = pcmobj._intermediates['v_grids']
    if q is None:
        q = pcmobj._intermediates['q']
    f_epsilon    = pcmobj._intermediates['f_epsilon']
    if not pcmobj.if_method_in_CPCM_category:
        A = pcmobj._intermediates['A']
        D = pcmobj._intermediates['D']
        S = pcmobj._intermediates['S']

    if pcmobj.frozen_dm0_for_finite_difference_without_response is not None:
        # Note: The v_grids computed above actually use frozen_dm0 as input, so it's actually v_grids_right
        v_grids_l = pcmobj._get_vgrids(dm, with_nuc = True)[0]

        # TODO: In the case where left and right dm are not the same,
        #       we indeed need to compute the derivative of 0.5 * (K^-1 R + R^T (K^-1)^T).
        #       This is not the same as the derivative of K^-1 R, if IEFPCM or SSVPE is used.
        #       However the difference is too small, and in the use case of computing polarizability derivative,
        #       we are not able to observe the difference.
        #       If there are other use cases where the error is more significant,
        #       we probably need to fix this.
        #       This is the only term affected by this problem in energy and gradient calculation.
        #       In hessian calculation, similar problems occur in energy 2nd derivative and Fock derivative terms.

    vK_1 = pcmobj.left_solve_K(v_grids_l, K_transpose = True)

    def contract_bra(a, B, c):
        ''' i,xij,j->jx '''
        tmp = a.dot(B)
        return (tmp*c).T

    def contract_ket(a, B, c):
        ''' i,xij,j->ix '''
        tmp = B.dot(c)
        return (a*tmp).T

    de = cupy.zeros([pcmobj.mol.natm,3])
    if pcmobj.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
        # dR = 0, dK = dS
        de_dS  = 0.5 * vK_1.reshape(-1, 1) * left_multiply_dS(pcmobj.surface, q, stream=None)
        de_dS -= 0.5 * q.reshape(-1, 1) * right_multiply_dS(pcmobj.surface, vK_1, stream=None)
        de -= cupy.asarray([cupy.sum(de_dS[p0:p1], axis=0) for p0,p1 in gridslice])

        dF, _ = get_dF_dA(pcmobj.surface, with_dA = False)
        dSii = get_dSii(pcmobj.surface, dF)
        de -= 0.5*contract('i,xij->jx', vK_1*q, dSii) # 0.5*cupy.einsum('i,xij,i->jx', vK_1, dSii, q)

    elif pcmobj.method.upper() in ['IEF-PCM', 'IEFPCM', 'SMD']:
        dF, dA = get_dF_dA(pcmobj.surface)
        dSii = get_dSii(pcmobj.surface, dF)
        dF = None

        dD, dS = get_dD_dS(pcmobj.surface, with_D=True, with_S=True)

        # dR = f_eps/(2*pi) * (dD*A + D*dA),
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)
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

    elif pcmobj.method.upper() in [ 'SS(V)PE' ]:
        dF, dA = get_dF_dA(pcmobj.surface)
        dSii = get_dSii(pcmobj.surface, dF)
        dF = None

        dD, dS = get_dD_dS(pcmobj.surface, with_D=True, with_S=True)

        # dR = f_eps/(2*pi) * (dD*A + D*dA),
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)
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

        DT_q = cupy.dot(D.T, q)
        ADT_q = A * DT_q
        de_dS1_T  = 0.5*contract_ket(vK_1, dS, ADT_q)
        de_dS1_T -= 0.5*contract_bra(vK_1, dS, ADT_q)
        de_dS1_T  = cupy.asarray([cupy.sum(de_dS1_T[p0:p1], axis=0) for p0,p1 in gridslice])
        vK_1_ADT_q = vK_1 * ADT_q
        de_dS1_T += 0.5*contract('j,xjn->nx', vK_1_ADT_q, dSii)

        Sq = cupy.dot(S,q)
        ASq = A*Sq
        de_dD  = 0.5*contract_ket(vK_1, dD, ASq)
        de_dD -= 0.5*contract_bra(vK_1, dD, ASq)
        de_dD  = cupy.asarray([cupy.sum(de_dD[p0:p1], axis=0) for p0,p1 in gridslice])

        vK_1_S = cupy.dot(vK_1, S)
        vK_1_SA = vK_1_S * A
        de_dD_T  = 0.5*contract_ket(vK_1_SA, -dD.transpose(0,2,1), q)
        de_dD_T -= 0.5*contract_bra(vK_1_SA, -dD.transpose(0,2,1), q)
        de_dD_T  = cupy.asarray([cupy.sum(de_dD_T[p0:p1], axis=0) for p0,p1 in gridslice])

        de_dA = 0.5*contract('j,xjn->nx', vK_1_D*Sq, dA)   # 0.5*cupy.einsum('j,xjn,j->nx', vK_1_D, dA, Sq)

        de_dA_T = 0.5*contract('j,xjn->nx', vK_1_S*DT_q, dA)

        de_dK = de_dS0 - 0.5 * fac * (de_dD + de_dA + de_dS1 + de_dD_T + de_dA_T + de_dS1_T)
        de += de_dR - de_dK

    else:
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")

    if pcmobj.frozen_dm0_for_finite_difference_without_response is not None:
        # Refer to the comments in gpu4pyscf/solvent/pcm.py::_get_vind()
        de *= 2

    t1 = log.timer_debug1('grad solver', *t1)
    return de.get()

def make_grad_object(base_method):
    '''Create nuclear gradients object with solvent contributions for the given
    solvent-attached method based on its gradients method in vaccum
    '''
    if isinstance(base_method, GradientsBase):
        # For backward compatibility. In gpu4pyscf-1.4 and older, the input
        # argument is a gradient object.
        base_method = base_method.base

    # Must be a solvent-attached method
    with_solvent = base_method.with_solvent
    if with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy gradients')

    # create the Gradients in vacuum. Cannot call super().Gradients() here
    # because other dynamic corrections might be applied to the base_method.
    # Calling super().Gradients might discard these corrections.
    vac_grad = base_method.undo_solvent().Gradients()
    # The base method for vac_grad discards the with_solvent. Change its base to
    # the solvent-attached base method
    vac_grad.base = base_method
    name = with_solvent.__class__.__name__ + vac_grad.__class__.__name__
    return lib.set_class(WithSolventGrad(vac_grad),
                         (WithSolventGrad, vac_grad.__class__), name)

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
        return self.base.to_cpu().PCM().Gradients()

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        if dm is None:
            dm = self.base.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        logger.debug(self, 'Compute gradients from solvents')
        self.de_solvent = self.base.with_solvent.grad(dm)
        logger.debug(self, 'Compute gradients from solutes')
        self.de_solute = super().kernel(*args, **kwargs)
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
