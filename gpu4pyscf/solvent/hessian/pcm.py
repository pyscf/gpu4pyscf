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
Hessian of PCM family solvent model
'''
# pylint: disable=C0103

import numpy
import cupy
import ctypes
from pyscf import lib, gto
from gpu4pyscf import scf
from gpu4pyscf.solvent.pcm import PI, switch_h, libsolvent
from gpu4pyscf.solvent.grad.pcm import grad_qv, grad_solver, grad_nuc, get_dD_dS, get_dF_dA, get_dSii, grad_switch_h
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib import logger
from gpu4pyscf.hessian.jk import _ao2mo
from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1, int1e_grids_ip2
from gpu4pyscf.gto.int3c1e_ipip import int1e_grids_ipip1, int1e_grids_ipvip1, int1e_grids_ipip2, int1e_grids_ip1ip2
from gpu4pyscf.gto import int3c1e
from gpu4pyscf.gto.int3c1e import int1e_grids
from gpu4pyscf.hessian.rhf import HessianBase
from pyscf import lib as pyscf_lib
from gpu4pyscf.lib.cupy_helper import contract

def gradgrad_switch_h(x):
    ''' 2nd derivative of h(x) '''
    ddy = 60.0*x - 180.0*x**2 + 120.0*x**3
    ddy[x<0] = 0.0
    ddy[x>1] = 0.0
    return ddy

def get_d2F_d2A(surface):
    '''
    Notations adopted from
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
    d2F = cupy.zeros([ngrids, natom, natom, 3, 3])
    d2A = cupy.zeros([ngrids, natom, natom, 3, 3])

    for i_grid_atom in range(natom):
        p0,p1 = surface['gslice_by_atom'][i_grid_atom]
        coords = grid_coords[p0:p1]
        si_rJ = cupy.expand_dims(coords, axis=1) - atom_coords
        norm_si_rJ = cupy.linalg.norm(si_rJ, axis=-1)
        diJ = (norm_si_rJ - R_in_J) / R_sw_J
        diJ[:,i_grid_atom] = 1.0
        diJ[diJ < 1e-8] = 0.0
        si_rJ[:,i_grid_atom,:] = 0.0
        si_rJ[diJ < 1e-8] = 0.0

        fiJ = switch_h(diJ)
        dfiJ = grad_switch_h(diJ)

        fiJK = fiJ[:, :, cupy.newaxis] * fiJ[:, cupy.newaxis, :]
        dfiJK = dfiJ[:, :, cupy.newaxis] * dfiJ[:, cupy.newaxis, :]
        R_sw_JK = R_sw_J[:, cupy.newaxis] * R_sw_J[cupy.newaxis, :]
        norm_si_rJK = norm_si_rJ[:, :, cupy.newaxis] * norm_si_rJ[:, cupy.newaxis, :]
        terms_size_ngrids_natm_natm = dfiJK / (fiJK * norm_si_rJK * R_sw_JK)
        si_rJK = si_rJ[:, :, cupy.newaxis, :, cupy.newaxis] * si_rJ[:, cupy.newaxis, :, cupy.newaxis, :]
        d2fiJK_offdiagonal = terms_size_ngrids_natm_natm[:, :, :, cupy.newaxis, cupy.newaxis] * si_rJK

        d2fiJ = gradgrad_switch_h(diJ)
        terms_size_ngrids_natm = d2fiJ / (norm_si_rJ**2 * R_sw_J) - dfiJ / (norm_si_rJ**3)
        si_rJJ = si_rJ[:, :, :, cupy.newaxis] * si_rJ[:, :, cupy.newaxis, :]
        d2fiJK_diagonal = contract('qA,qAdD->qAdD', terms_size_ngrids_natm, si_rJJ)
        d2fiJK_diagonal += contract('qA,dD->qAdD', dfiJ / norm_si_rJ, cupy.eye(3))
        d2fiJK_diagonal /= (fiJ * R_sw_J)[:, :, cupy.newaxis, cupy.newaxis]

        d2fiJK = d2fiJK_offdiagonal
        for i_atom in range(natom):
            d2fiJK[:, i_atom, i_atom, :, :] = d2fiJK_diagonal[:, i_atom, :, :]

        Fi = switch_fun[p0:p1]
        Ai = area[p0:p1]

        d2F[p0:p1, :, :, :, :] += contract('q,qABdD->qABdD', Fi, d2fiJK)
        d2A[p0:p1, :, :, :, :] += contract('q,qABdD->qABdD', Ai, d2fiJK)

        d2fiJK_grid_atom_offdiagonal = -cupy.einsum('qABdD->qAdD', d2fiJK)
        d2F[p0:p1, i_grid_atom, :, :, :] = contract('q,qAdD->qAdD', Fi, d2fiJK_grid_atom_offdiagonal.transpose(0,1,3,2))
        d2F[p0:p1, :, i_grid_atom, :, :] = contract('q,qAdD->qAdD', Fi, d2fiJK_grid_atom_offdiagonal)
        d2A[p0:p1, i_grid_atom, :, :, :] = contract('q,qAdD->qAdD', Ai, d2fiJK_grid_atom_offdiagonal.transpose(0,1,3,2))
        d2A[p0:p1, :, i_grid_atom, :, :] = contract('q,qAdD->qAdD', Ai, d2fiJK_grid_atom_offdiagonal)

        d2fiJK_grid_atom_diagonal = -cupy.einsum('qAdD->qdD', d2fiJK_grid_atom_offdiagonal)
        d2F[p0:p1, i_grid_atom, i_grid_atom, :, :] = contract('q,qdD->qdD', Fi, d2fiJK_grid_atom_diagonal)
        d2A[p0:p1, i_grid_atom, i_grid_atom, :, :] = contract('q,qdD->qdD', Ai, d2fiJK_grid_atom_diagonal)

    d2F = d2F.transpose(1,2,3,4,0)
    d2A = d2A.transpose(1,2,3,4,0)
    return d2F, d2A

def get_d2Sii(surface, dF, d2F, stream=None):
    ''' Second derivative of S matrix (diagonal only)
    '''
    charge_exp  = surface['charge_exp']
    switch_fun  = surface['switch_fun']
    ngrids = switch_fun.shape[0]
    dF = dF.transpose(2,0,1)
    natm = dF.shape[0]
    assert dF.shape == (natm, 3, ngrids)

    # dF_dF = dF[:, cupy.newaxis, :, cupy.newaxis, :] * dF[cupy.newaxis, :, cupy.newaxis, :, :]
    # dF_dF_over_F3 = dF_dF * (1.0/(switch_fun**3))
    # d2F_over_F2 = d2F * (1.0/(switch_fun**2))
    # d2Sii = 2 * dF_dF_over_F3 - d2F_over_F2
    # d2Sii = (2.0/PI)**0.5 * (d2Sii * charge_exp)

    dF = dF.flatten()   # Make sure the underlying data order is the same as shape shows
    d2F = d2F.flatten() # Make sure the underlying data order is the same as shape shows
    d2Sii = cupy.empty((natm, natm, 3, 3, ngrids), dtype=cupy.float64)
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libsolvent.pcm_d2f_to_d2sii(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(switch_fun.data.ptr, ctypes.c_void_p),
        ctypes.cast(dF.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2F.data.ptr, ctypes.c_void_p),
        ctypes.cast(charge_exp.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2Sii.data.ptr, ctypes.c_void_p),
        ctypes.c_int(natm),
        ctypes.c_int(ngrids),
    )
    if err != 0:
        raise RuntimeError('Failed in converting PCM d2F to d2Sii.')
    return d2Sii

def get_d2D_d2S(surface, with_S=True, with_D=False, stream=None):
    ''' Second derivatives of D matrix and S matrix (offdiagonals only)
    '''
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    norm_vec    = surface['norm_vec']
    n = charge_exp.shape[0]
    d2S = cupy.empty([3,3,n,n])
    d2D = None
    d2S_ptr = ctypes.cast(d2S.data.ptr, ctypes.c_void_p)
    d2D_ptr = pyscf_lib.c_null_ptr()
    if with_D:
        d2D = cupy.empty([3,3,n,n])
        d2D_ptr = ctypes.cast(d2D.data.ptr, ctypes.c_void_p)
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libsolvent.pcm_d2d_d2s(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        d2D_ptr, d2S_ptr,
        ctypes.cast(grid_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(norm_vec.data.ptr, ctypes.c_void_p),
        ctypes.cast(charge_exp.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n)
    )
    if err != 0:
        raise RuntimeError('Failed in generating PCM d2D and d2S matrices.')
    return d2D, d2S

def analytical_hess_nuc(pcmobj, dm, verbose=None):
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is not None and cupy.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    log = logger.new_logger(pcmobj, verbose)
    t1 = log.init_timer()

    q_sym        = pcmobj._intermediates['q_sym'].get()
    gridslice    = pcmobj.surface['gslice_by_atom']
    grid_coords  = pcmobj.surface['grid_coords'].get()
    exponents    = pcmobj.surface['charge_exp'].get()

    ngrids = q_sym.shape[0]

    atom_coords = mol.atom_coords(unit='B')
    atom_charges = numpy.asarray(mol.atom_charges(), dtype=numpy.float64)
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)
    fakemol = gto.fakemol_for_charges(grid_coords, expnt=exponents**2)

    d2e_from_d2I = numpy.zeros([mol.natm, mol.natm, 3, 3])

    int2c2e_ip1ip2 = mol._add_suffix('int2c2e_ip1ip2')
    d2I_dAdC = gto.mole.intor_cross(int2c2e_ip1ip2, fakemol_nuc, fakemol)
    d2I_dAdC = d2I_dAdC.reshape(3, 3, mol.natm, ngrids)
    for i_atom in range(mol.natm):
        g0,g1 = gridslice[i_atom]
        d2e_from_d2I[:, i_atom, :, :] += numpy.einsum('A,dDAq,q->AdD', atom_charges, d2I_dAdC[:, :, :, g0:g1], q_sym[g0:g1])
        d2e_from_d2I[i_atom, :, :, :] += numpy.einsum('A,dDAq,q->AdD', atom_charges, d2I_dAdC[:, :, :, g0:g1], q_sym[g0:g1])

    int2c2e_ipip1 = mol._add_suffix('int2c2e_ipip1')
    # # Some explanations here:
    # # Why can we use the ip1ip2 here? Because of the translational invariance
    # # $\frac{\partial^2 I_{AC}}{\partial A^2} + \frac{\partial^2 I_{AC}}{\partial A \partial C} = 0$
    # # Why not using the ipip1 here? Because the nuclei, a point charge, is handled as a Gaussian charge with exponent = 1e16
    # # This causes severe numerical problem in function int2c2e_ip1ip2, and make the main diagonal of hessian garbage.
    # d2I_dA2 = gto.mole.intor_cross(int2c2e_ipip1, fakemol_nuc, fakemol)
    d2I_dA2 = -gto.mole.intor_cross(int2c2e_ip1ip2, fakemol_nuc, fakemol)
    d2I_dA2 = d2I_dA2 @ q_sym
    d2I_dA2 = d2I_dA2.reshape(3, 3, mol.natm)
    for i_atom in range(mol.natm):
        d2e_from_d2I[i_atom, i_atom, :, :] += atom_charges[i_atom] * d2I_dA2[:, :, i_atom]

    d2I_dC2 = gto.mole.intor_cross(int2c2e_ipip1, fakemol, fakemol_nuc)
    d2I_dC2 = d2I_dC2 @ atom_charges
    d2I_dC2 = d2I_dC2.reshape(3, 3, ngrids)
    for i_atom in range(mol.natm):
        g0,g1 = gridslice[i_atom]
        d2e_from_d2I[i_atom, i_atom, :, :] += d2I_dC2[:, :, g0:g1] @ q_sym[g0:g1]

    intopt_derivative = int3c1e.VHFOpt(mol)
    intopt_derivative.build(cutoff = 1e-14, aosym = False)

    dqdx = get_dqsym_dx(pcmobj, dm, range(mol.natm), intopt_derivative)
    dqdx = dqdx.get()

    d2e_from_dIdq = numpy.zeros([mol.natm, mol.natm, 3, 3])
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            d2e_from_dIdq[i_atom, :, i_xyz, :] = grad_nuc(pcmobj, dm, q_sym = dqdx[i_atom, i_xyz, :])

    d2e = d2e_from_d2I - d2e_from_dIdq

    t1 = log.timer_debug1('solvent hessian d(dVnuc/dx * q)/dx contribution', *t1)
    return d2e

def analytical_hess_qv(pcmobj, dm, verbose=None):
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is not None and cupy.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    log = logger.new_logger(pcmobj, verbose)
    t1 = log.init_timer()

    gridslice   = pcmobj.surface['gslice_by_atom']
    charge_exp  = pcmobj.surface['charge_exp']
    grid_coords = pcmobj.surface['grid_coords']
    q_sym       = pcmobj._intermediates['q_sym']

    aoslice = mol.aoslice_by_atom()
    aoslice = numpy.array(aoslice)

    intopt_derivative = int3c1e.VHFOpt(mol)
    intopt_derivative.build(cutoff = 1e-14, aosym = False)

    # fakemol = gto.fakemol_for_charges(grid_coords.get(), expnt=charge_exp.get()**2)
    # intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e')
    # intopt.build(1e-14, diag_block_with_triu=True, aosym=False)

    d2e_from_d2I = cupy.zeros([mol.natm, mol.natm, 3, 3])

    # d2I_dA2 = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ipip1', direct_scf_tol=1e-14)
    # d2I_dA2 = cupy.einsum('dijq,q->dij', d2I_dA2, q_sym)
    # d2I_dA2 = d2I_dA2.reshape([3, 3, nao, nao])
    d2I_dA2 = int1e_grids_ipip1(mol, grid_coords, charges = q_sym, intopt = intopt_derivative, charge_exponents = charge_exp**2)
    for i_atom in range(mol.natm):
        p0,p1 = aoslice[i_atom, 2:]
        d2e_from_d2I[i_atom, i_atom, :, :] += contract('ij,dDij->dD', dm[p0:p1, :], d2I_dA2[:, :, p0:p1, :])
        d2e_from_d2I[i_atom, i_atom, :, :] += contract('ij,dDij->dD', dm[:, p0:p1], d2I_dA2[:, :, p0:p1, :].transpose(0,1,3,2))
    d2I_dA2 = None

    # d2I_dAdB = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ipvip1', direct_scf_tol=1e-14)
    # d2I_dAdB = cupy.einsum('dijq,q->dij', d2I_dAdB, q_sym)
    # d2I_dAdB = d2I_dAdB.reshape([3, 3, nao, nao])
    d2I_dAdB = int1e_grids_ipvip1(mol, grid_coords, charges = q_sym, intopt = intopt_derivative, charge_exponents = charge_exp**2)
    for i_atom in range(mol.natm):
        pi0,pi1 = aoslice[i_atom, 2:]
        for j_atom in range(mol.natm):
            pj0,pj1 = aoslice[j_atom, 2:]
            d2e_from_d2I[i_atom, j_atom, :, :] += contract('ij,dDij->dD', dm[pi0:pi1, pj0:pj1], d2I_dAdB[:, :, pi0:pi1, pj0:pj1])
            d2e_from_d2I[i_atom, j_atom, :, :] += contract('ij,dDij->dD', dm[pj0:pj1, pi0:pi1], d2I_dAdB[:, :, pi0:pi1, pj0:pj1].transpose(0,1,3,2))
    d2I_dAdB = None

    for j_atom in range(mol.natm):
        g0,g1 = gridslice[j_atom]
        # d2I_dAdC = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ip1ip2', direct_scf_tol=1e-14)
        # d2I_dAdC = cupy.einsum('dijq,q->dij', d2I_dAdC[:, :, :, g0:g1], q_sym[g0:g1])
        # d2I_dAdC = d2I_dAdC.reshape([3, 3, nao, nao])
        d2I_dAdC = int1e_grids_ip1ip2(mol, grid_coords[g0:g1, :], charges = q_sym[g0:g1], intopt = intopt_derivative, charge_exponents = charge_exp[g0:g1]**2)

        for i_atom in range(mol.natm):
            p0,p1 = aoslice[i_atom, 2:]
            d2e_from_d2I[i_atom, j_atom, :, :] += contract('ij,dDij->dD', dm[p0:p1, :], d2I_dAdC[:, :, p0:p1, :])
            d2e_from_d2I[i_atom, j_atom, :, :] += contract('ij,dDij->dD', dm[:, p0:p1], d2I_dAdC[:, :, p0:p1, :].transpose(0,1,3,2))

            d2e_from_d2I[j_atom, i_atom, :, :] += contract('ij,dDij->dD', dm[p0:p1, :], d2I_dAdC[:, :, p0:p1, :].transpose(1,0,2,3))
            d2e_from_d2I[j_atom, i_atom, :, :] += contract('ij,dDij->dD', dm[:, p0:p1], d2I_dAdC[:, :, p0:p1, :].transpose(1,0,3,2))
    d2I_dAdC = None

    # d2I_dC2 = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ipip2', direct_scf_tol=1e-14)
    # d2I_dC2 = cupy.einsum('dijq,ij->dq', d2I_dC2, dm)
    # d2I_dC2 = d2I_dC2.reshape([3, 3, ngrids])
    d2I_dC2 = int1e_grids_ipip2(mol, grid_coords, dm = dm, intopt = intopt_derivative, charge_exponents = charge_exp**2)
    for i_atom in range(mol.natm):
        g0,g1 = gridslice[i_atom]
        d2e_from_d2I[i_atom, i_atom, :, :] += d2I_dC2[:, :, g0:g1] @ q_sym[g0:g1]
    d2I_dC2 = None

    dqdx = get_dqsym_dx(pcmobj, dm, range(mol.natm), intopt_derivative)

    d2e_from_dIdq = numpy.zeros([mol.natm, mol.natm, 3, 3])
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            d2e_from_dIdq[i_atom, :, i_xyz, :] = grad_qv(pcmobj, dm, q_sym = dqdx[i_atom, i_xyz, :])

    d2e_from_d2I = d2e_from_d2I.get()
    d2e = d2e_from_d2I + d2e_from_dIdq
    d2e *= -1

    t1 = log.timer_debug1('solvent hessian d(dI/dx * q)/dx contribution', *t1)
    return d2e

def einsum_ij_Adj_Adi_inverseK(pcmobj, Adj_term, K_transpose = False):
    nA, nd, nj = Adj_term.shape
    # return cupy.einsum('ij,Adj->Adi', cupy.linalg.inv(K), Adj_term)
    # return cupy.linalg.solve(K, Adj_term.reshape(nA * nd, nj).T).T.reshape(nA, nd, nj)
    return pcmobj.left_solve_K(Adj_term.reshape(nA * nd, nj).T, K_transpose = K_transpose).T.reshape(nA, nd, nj)
def einsum_Adi_ij_Adj_inverseK(Adi_term, pcmobj, K_transpose = False):
    # return cupy.einsum('Adi,ij->Adj', Adi_term, cupy.linalg.inv(K))
    return einsum_ij_Adj_Adi_inverseK(pcmobj, Adi_term, K_transpose = not K_transpose)

def get_dS_dot_q(dS, dSii, q, atmlst, gridslice):
    output = contract('diA,i->Adi', dSii[:,:,atmlst], q)
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        output[i_atom, :, g0:g1] += dS[:,g0:g1,:] @ q
        output[i_atom, :, :] -= dS[:,:,g0:g1] @ q[g0:g1]
    return output
def get_dST_dot_q(dS, dSii, q, atmlst, gridslice):
    # S is symmetric
    return get_dS_dot_q(dS, dSii, q, atmlst, gridslice)

def get_dA_dot_q(dA, q, atmlst):
    return contract('diA,i->Adi', dA[:,:,atmlst], q)

def get_dD_dot_q(dD, q, atmlst, gridslice, ngrids):
    output = cupy.zeros([len(atmlst), 3, ngrids])
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        output[i_atom, :, g0:g1] += dD[:,g0:g1,:] @ q
        output[i_atom, :, :] -= dD[:,:,g0:g1] @ q[g0:g1]
    return output
def get_dDT_dot_q(dD, q, atmlst, gridslice, ngrids):
    return get_dD_dot_q(-dD.transpose(0,2,1), q, atmlst, gridslice, ngrids)

def get_v_dot_d2S_dot_q(d2S, d2Sii, v_left, q_right, natom, gridslice):
    output = d2Sii @ (v_left * q_right)
    for i_atom in range(natom):
        gi0,gi1 = gridslice[i_atom]
        for j_atom in range(natom):
            gj0,gj1 = gridslice[j_atom]
            d2S_atom_ij = contract('q,dDq->dD', v_left[gi0:gi1], d2S[:,:,gi0:gi1,gj0:gj1] @ q_right[gj0:gj1])
            output[i_atom, i_atom, :, :] += d2S_atom_ij
            output[j_atom, j_atom, :, :] += d2S_atom_ij
            output[i_atom, j_atom, :, :] -= d2S_atom_ij
            output[j_atom, i_atom, :, :] -= d2S_atom_ij
    return output
def get_v_dot_d2ST_dot_q(d2S, d2Sii, v_left, q_right, natom, gridslice):
    # S is symmetric
    return get_v_dot_d2S_dot_q(d2S, d2Sii, v_left, q_right, natom, gridslice)

def get_v_dot_d2A_dot_q(d2A, v_left, q_right):
    return d2A @ (v_left * q_right)

def get_v_dot_d2D_dot_q(d2D, v_left, q_right, natom, gridslice):
    output = cupy.zeros([natom, natom, 3, 3])
    for i_atom in range(natom):
        gi0,gi1 = gridslice[i_atom]
        for j_atom in range(natom):
            gj0,gj1 = gridslice[j_atom]
            d2D_atom_ij = contract('q,dDq->dD', v_left[gi0:gi1], d2D[:,:,gi0:gi1,gj0:gj1] @ q_right[gj0:gj1])
            output[i_atom, i_atom, :, :] += d2D_atom_ij
            output[j_atom, j_atom, :, :] += d2D_atom_ij
            output[i_atom, j_atom, :, :] -= d2D_atom_ij
            output[j_atom, i_atom, :, :] -= d2D_atom_ij
    return output
def get_v_dot_d2DT_dot_q(d2D, v_left, q_right, natom, gridslice):
    return get_v_dot_d2D_dot_q(d2D.transpose(0,1,3,2), v_left, q_right, natom, gridslice)

def analytical_hess_solver(pcmobj, dm, verbose=None):
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is not None and cupy.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    log = logger.new_logger(mol, verbose)
    t1 = log.init_timer()

    natom = mol.natm
    atmlst = range(natom) # Attention: This cannot be split

    gridslice    = pcmobj.surface['gslice_by_atom']
    v_grids      = pcmobj._intermediates['v_grids']
    q            = pcmobj._intermediates['q']
    f_epsilon    = pcmobj._intermediates['f_epsilon']
    if not pcmobj.if_method_in_CPCM_category:
        A = pcmobj._intermediates['A']
        D = pcmobj._intermediates['D']
        S = pcmobj._intermediates['S']

    ngrids = q.shape[0]

    vK_1 = pcmobj.left_solve_K(v_grids, K_transpose = True)

    if pcmobj.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
        _, dS = get_dD_dS(pcmobj.surface, with_D=False, with_S=True)
        dF, _ = get_dF_dA(pcmobj.surface, with_dA = False)
        dSii = get_dSii(pcmobj.surface, dF)

        # dR = 0, dK = dS
        # d(S-1 R) = - S-1 dS S-1 R
        # d2(S-1 R) = (S-1 dS S-1 dS S-1 R) + (S-1 dS S-1 dS S-1 R) - (S-1 d2S S-1 R)
        dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)
        S_1_dSdx_dot_q = einsum_ij_Adj_Adi_inverseK(pcmobj, dSdx_dot_q)
        dSdx_dot_q = None
        VS_1_dot_dSdx = get_dST_dot_q(dS, dSii, vK_1, atmlst, gridslice)
        dS = None
        dSii = None
        d2e_from_d2KR = contract('Adi,BDi->ABdD', VS_1_dot_dSdx, S_1_dSdx_dot_q) * 2

        _, d2S = get_d2D_d2S(pcmobj.surface, with_D=False, with_S=True)
        d2F, _ = get_d2F_d2A(pcmobj.surface)
        d2Sii = get_d2Sii(pcmobj.surface, dF, d2F)
        dF = None
        d2F = None
        d2e_from_d2KR -= get_v_dot_d2S_dot_q(d2S, d2Sii, vK_1, q, natom, gridslice)
        d2S = None
        d2Sii = None

        dK_1Rv = -S_1_dSdx_dot_q
        R = -f_epsilon
        dvK_1R = -einsum_Adi_ij_Adj_inverseK(VS_1_dot_dSdx, pcmobj) * R

    elif pcmobj.method.upper() in ['IEF-PCM', 'IEFPCM', 'SMD']:
        dD, dS = get_dD_dS(pcmobj.surface, with_D=True, with_S=True)
        dF, dA = get_dF_dA(pcmobj.surface)
        dSii = get_dSii(pcmobj.surface, dF)

        # dR = f_eps/(2*pi) * (dD*A + D*dA)
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)

        # d2R = f_eps/(2*pi) * (d2D*A + dD*dA + dD*dA + D*d2A)
        # d2K = d2S - f_eps/(2*pi) * (d2D*A*S + D*d2A*S + D*A*d2S + dD*dA*S + dD*dA*S + dD*A*dS + dD*A*dS + D*dA*dS + D*dA*dS)
        # The terms showing up twice on equation above (dD*dA + dD*dA for example) refer to dD/dx * dA/dy + dD/dy * dA/dx,
        # since D is not symmetric, they are not the same.

        # d(K-1 R) = - K-1 dK K-1 R + K-1 dR
        # d2(K-1 R) = (K-1 dK K-1 dK K-1 R) + (K-1 dK K-1 dK K-1 R) - (K-1 d2K K-1 R) - (K-1 dK K-1 dR)
        #             - (K-1 dK K-1 dR) + (K-1 d2R)
        f_eps_over_2pi = f_epsilon/(2.0*PI)

        dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)
        DA = D*A
        dKdx_dot_q = dSdx_dot_q - f_eps_over_2pi * contract('ij,Adj->Adi', DA, dSdx_dot_q)
        dAdx_dot_Sq = get_dA_dot_q(dA, S @ q, atmlst)
        dKdx_dot_q -= f_eps_over_2pi * contract('ij,Adj->Adi', D, dAdx_dot_Sq)
        AS = (A * S.T).T # It's just diag(A) @ S
        ASq = AS @ q
        dDdx_dot_ASq = get_dD_dot_q(dD, ASq, atmlst, gridslice, ngrids)
        dKdx_dot_q -= f_eps_over_2pi * dDdx_dot_ASq
        dDdx_dot_ASq = None

        K_1_dot_dKdx_dot_q = einsum_ij_Adj_Adi_inverseK(pcmobj, dKdx_dot_q)
        dKdx_dot_q = None

        vK_1_dot_dSdx = get_dST_dot_q(dS, dSii, vK_1, atmlst, gridslice)
        vK_1_dot_dKdx = vK_1_dot_dSdx
        vK_1_dot_dSdx = None
        vK_1_dot_dDdx = get_dDT_dot_q(dD, vK_1, atmlst, gridslice, ngrids)
        vK_1_dot_dKdx -= f_eps_over_2pi * contract('ij,Adj->Adi', AS.T, vK_1_dot_dDdx)
        AS = None
        vK_1D = D.T @ vK_1
        vK_1D_dot_dAdx = get_dA_dot_q(dA, vK_1D, atmlst)
        vK_1_dot_dKdx -= f_eps_over_2pi * contract('ij,Adj->Adi', S.T, vK_1D_dot_dAdx)
        vK_1DA = DA.T @ vK_1
        DA = None
        vK_1DA_dot_dSdx = get_dST_dot_q(dS, dSii, vK_1DA, atmlst, gridslice)
        dS = None
        dSii = None
        vK_1_dot_dKdx -= f_eps_over_2pi * vK_1DA_dot_dSdx
        vK_1DA_dot_dSdx = None

        d2e_from_d2KR  = contract('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)
        d2e_from_d2KR += contract('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)

        d2F, d2A = get_d2F_d2A(pcmobj.surface)
        vK_1_d2K_q  = get_v_dot_d2A_dot_q(d2A, vK_1D, S @ q)
        vK_1_d2R_V  = get_v_dot_d2A_dot_q(d2A, vK_1D, v_grids)
        d2A = None
        d2Sii = get_d2Sii(pcmobj.surface, dF, d2F)
        dF = None
        d2F = None
        d2D, d2S = get_d2D_d2S(pcmobj.surface, with_D=True, with_S=True)
        vK_1_d2K_q += get_v_dot_d2D_dot_q(d2D, vK_1, ASq, natom, gridslice)
        vK_1_d2R_V += get_v_dot_d2D_dot_q(d2D, vK_1, A * v_grids, natom, gridslice)
        d2D = None
        vK_1_d2K_q += get_v_dot_d2S_dot_q(d2S, d2Sii, vK_1DA, q, natom, gridslice)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q *= -f_eps_over_2pi
        vK_1_d2K_q += get_v_dot_d2S_dot_q(d2S, d2Sii, vK_1, q, natom, gridslice)
        d2S = None
        d2Sii = None

        d2e_from_d2KR -= vK_1_d2K_q

        dAdx_dot_V = get_dA_dot_q(dA, v_grids, atmlst)
        dDdx_dot_AV = get_dD_dot_q(dD, A * v_grids, atmlst, gridslice, ngrids)
        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + contract('ij,Adj->Adi', D, dAdx_dot_V))
        dDdx_dot_AV = None

        K_1_dot_dRdx_dot_V = einsum_ij_Adj_Adi_inverseK(pcmobj, dRdx_dot_V)
        dRdx_dot_V = None

        d2e_from_d2KR -= contract('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)
        d2e_from_d2KR -= contract('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)

        vK_1_d2R_V += contract('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V += contract('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V *= f_eps_over_2pi

        d2e_from_d2KR += vK_1_d2R_V

        dK_1Rv = -K_1_dot_dKdx_dot_q + K_1_dot_dRdx_dot_V

        VK_1D_dot_dAdx = get_dA_dot_q(dA, (D.T @ vK_1).T, atmlst)
        VK_1_dot_dDdx = get_dDT_dot_q(dD, vK_1, atmlst, gridslice, ngrids)
        VK_1_dot_dRdx = f_eps_over_2pi * (VK_1D_dot_dAdx + VK_1_dot_dDdx * A)

        DA = D*A
        R = -f_epsilon * (cupy.eye(DA.shape[0]) - 1.0/(2.0*PI)*DA)
        dvK_1R = -einsum_Adi_ij_Adj_inverseK(vK_1_dot_dKdx, pcmobj) @ R + VK_1_dot_dRdx

    elif pcmobj.method.upper() in ['SS(V)PE']:
        dD, dS = get_dD_dS(pcmobj.surface, with_D=True, with_S=True)
        dF, dA = get_dF_dA(pcmobj.surface)
        dSii = get_dSii(pcmobj.surface, dF)

        # dR = f_eps/(2*pi) * (dD*A + D*dA)
        # dK = dS - f_eps/(4*pi) * (dD*A*S + D*dA*S + D*A*dS + dST*AT*DT + ST*dAT*DT + ST*AT*dDT)

        # d2R = f_eps/(2*pi) * (d2D*A + dD*dA + dD*dA + D*d2A)
        # d2K = d2S - f_eps/(4*pi) * (d2D*A*S + D*d2A*S + D*A*d2S + dD*dA*S + dD*dA*S + dD*A*dS + dD*A*dS + D*dA*dS + D*dA*dS
        #                           + d2ST*AT*DT + ST*d2AT*DT + ST*AT*d2DT + dST*dAT*DT + dST*dAT*DT + dST*AT*dDT + dST*AT*dDT + ST*dAT*dDT + ST*dAT*dDT)
        f_eps_over_2pi = f_epsilon/(2.0*PI)
        f_eps_over_4pi = f_epsilon/(4.0*PI)

        dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)
        DA = D*A
        dKdx_dot_q = dSdx_dot_q - f_eps_over_4pi * contract('ij,Adj->Adi', DA, dSdx_dot_q)
        dAdx_dot_Sq = get_dA_dot_q(dA, S @ q, atmlst)
        dKdx_dot_q -= f_eps_over_4pi * contract('ij,Adj->Adi', D, dAdx_dot_Sq)
        AS = (A * S.T).T # It's just diag(A) @ S
        ASq = AS @ q
        dDdx_dot_ASq = get_dD_dot_q(dD, ASq, atmlst, gridslice, ngrids)
        dKdx_dot_q -= f_eps_over_4pi * dDdx_dot_ASq
        dDdx_dot_ASq = None
        dDdxT_dot_q = get_dDT_dot_q(dD, q, atmlst, gridslice, ngrids)
        dKdx_dot_q -= f_eps_over_4pi * contract('ij,Adj->Adi', AS.T, dDdxT_dot_q)
        dAdxT_dot_DT_q = get_dA_dot_q(dA, D.T @ q, atmlst)
        dKdx_dot_q -= f_eps_over_4pi * contract('ij,Adj->Adi', S.T, dAdxT_dot_DT_q)
        AT_DT_q = DA.T @ q
        dSdxT_dot_AT_DT_q = get_dS_dot_q(dS, dSii, AT_DT_q, atmlst, gridslice)
        dKdx_dot_q -= f_eps_over_4pi * dSdxT_dot_AT_DT_q
        dSdxT_dot_AT_DT_q = None

        K_1_dot_dKdx_dot_q = einsum_ij_Adj_Adi_inverseK(pcmobj, dKdx_dot_q)
        dKdx_dot_q = None

        vK_1_dot_dSdx = get_dST_dot_q(dS, dSii, vK_1, atmlst, gridslice)
        vK_1_dot_dKdx = vK_1_dot_dSdx
        vK_1_dot_dSdx = None
        vK_1_dot_dDdx = get_dDT_dot_q(dD, vK_1, atmlst, gridslice, ngrids)
        vK_1_dot_dKdx -= f_eps_over_4pi * contract('ij,Adj->Adi', AS.T, vK_1_dot_dDdx)
        vK_1D_dot_dAdx = get_dA_dot_q(dA, D.T @ vK_1, atmlst)
        vK_1_dot_dKdx -= f_eps_over_4pi * contract('ij,Adj->Adi', S.T, vK_1D_dot_dAdx)
        vK_1DA = DA.T @ vK_1
        vK_1DA_dot_dSdx = get_dST_dot_q(dS, dSii, vK_1DA, atmlst, gridslice)
        vK_1_dot_dKdx -= f_eps_over_4pi * vK_1DA_dot_dSdx
        vK_1DA_dot_dSdx = None
        vK_1_dot_dSdxT = get_dS_dot_q(dS, dSii, vK_1, atmlst, gridslice)
        dS = None
        dSii = None
        vK_1_dot_dKdx -= f_eps_over_4pi * contract('ij,Adj->Adi', DA, vK_1_dot_dSdxT)
        DA = None
        vK_1_ST_dot_dAdxT = get_dA_dot_q(dA, (S @ vK_1).T, atmlst)
        vK_1_dot_dKdx -= f_eps_over_4pi * contract('ij,Adj->Adi', D, vK_1_ST_dot_dAdxT)
        vK_1_ST_AT = AS @ vK_1
        AS = None
        vK_1_ST_AT_dot_dDdxT = get_dD_dot_q(dD, vK_1_ST_AT, atmlst, gridslice, ngrids)
        vK_1_dot_dKdx -= f_eps_over_4pi * vK_1_ST_AT_dot_dDdxT
        vK_1_ST_AT_dot_dDdxT = None

        d2e_from_d2KR  = contract('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)
        d2e_from_d2KR += contract('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)

        d2F, d2A = get_d2F_d2A(pcmobj.surface)
        vK_1_d2K_q  = get_v_dot_d2A_dot_q(d2A, (D.T @ vK_1).T, S @ q)
        vK_1_d2K_q += get_v_dot_d2A_dot_q(d2A, (S @ vK_1).T, D.T @ q)
        vK_1_d2R_V  = get_v_dot_d2A_dot_q(d2A, (D.T @ vK_1).T, v_grids)
        d2A = None
        d2Sii = get_d2Sii(pcmobj.surface, dF, d2F)
        dF = None
        d2F = None
        d2D, d2S = get_d2D_d2S(pcmobj.surface, with_D=True, with_S=True)
        vK_1_d2K_q += get_v_dot_d2D_dot_q(d2D, vK_1, ASq, natom, gridslice)
        vK_1_d2K_q += get_v_dot_d2DT_dot_q(d2D, vK_1_ST_AT, q, natom, gridslice)
        vK_1_d2R_V += get_v_dot_d2D_dot_q(d2D, vK_1, A * v_grids, natom, gridslice)
        d2D = None
        vK_1_d2K_q += get_v_dot_d2S_dot_q(d2S, d2Sii, vK_1DA, q, natom, gridslice)
        vK_1_d2K_q += get_v_dot_d2ST_dot_q(d2S, d2Sii, vK_1, AT_DT_q, natom, gridslice)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1_dot_dSdxT, dAdxT_dot_DT_q)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1_dot_dSdxT * A, dDdxT_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1_ST_dot_dAdxT, dDdxT_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1_dot_dSdxT, dAdxT_dot_DT_q)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1_dot_dSdxT * A, dDdxT_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1_ST_dot_dAdxT, dDdxT_dot_q)
        vK_1_d2K_q *= -f_eps_over_4pi
        vK_1_d2K_q += get_v_dot_d2S_dot_q(d2S, d2Sii, vK_1, q, natom, gridslice)
        d2S = None
        d2Sii = None

        d2e_from_d2KR -= vK_1_d2K_q

        dAdx_dot_V = get_dA_dot_q(dA, v_grids, atmlst)
        dDdx_dot_AV = get_dD_dot_q(dD, A * v_grids, atmlst, gridslice, ngrids)
        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + contract('ij,Adj->Adi', D, dAdx_dot_V))
        dDdx_dot_AV = None

        K_1_dot_dRdx_dot_V = einsum_ij_Adj_Adi_inverseK(pcmobj, dRdx_dot_V)

        d2e_from_d2KR -= contract('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)
        d2e_from_d2KR -= contract('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)

        vK_1_d2R_V += contract('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V += contract('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V *= f_eps_over_2pi

        d2e_from_d2KR += vK_1_d2R_V

        dK_1Rv = -K_1_dot_dKdx_dot_q + K_1_dot_dRdx_dot_V

        VK_1D_dot_dAdx = get_dA_dot_q(dA, (D.T @ vK_1).T, atmlst)
        VK_1_dot_dDdx = get_dDT_dot_q(dD, vK_1, atmlst, gridslice, ngrids)
        VK_1_dot_dRdx = f_eps_over_2pi * (VK_1D_dot_dAdx + VK_1_dot_dDdx * A)

        DA = D*A
        R = -f_epsilon * (cupy.eye(DA.shape[0]) - 1.0/(2.0*PI)*DA)
        dvK_1R = -einsum_Adi_ij_Adj_inverseK(vK_1_dot_dKdx, pcmobj) @ R + VK_1_dot_dRdx

    else:
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")

    d2e = d2e_from_d2KR

    intopt_derivative = int3c1e.VHFOpt(mol)
    intopt_derivative.build(cutoff = 1e-14, aosym = False)

    dVdx = get_dvgrids(pcmobj, dm, range(mol.natm), intopt_derivative)
    d2e -= contract('Adi,BDi->BADd', dvK_1R, dVdx)
    d2e -= contract('Adi,BDi->ABdD', dVdx, dK_1Rv)

    d2e *= 0.5
    d2e = d2e.get()
    t1 = log.timer_debug1('solvent hessian d(V * dK-1R/dx * V)/dx contribution', *t1)
    return d2e

def get_dqsym_dx_fix_vgrids(pcmobj, atmlst):
    assert pcmobj._intermediates is not None

    gridslice    = pcmobj.surface['gslice_by_atom']
    v_grids      = pcmobj._intermediates['v_grids']
    q            = pcmobj._intermediates['q']
    q_sym        = pcmobj._intermediates['q_sym']
    f_epsilon    = pcmobj._intermediates['f_epsilon']
    if not pcmobj.if_method_in_CPCM_category:
        A = pcmobj._intermediates['A']
        D = pcmobj._intermediates['D']
        S = pcmobj._intermediates['S']

    ngrids = q_sym.shape[0]

    if pcmobj.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
        _, dS = get_dD_dS(pcmobj.surface, with_D=False, with_S=True)
        dF, _ = get_dF_dA(pcmobj.surface, with_dA = False)
        dSii = get_dSii(pcmobj.surface, dF)
        dF = None

        # dR = 0, dK = dS
        dSdx_dot_q = get_dS_dot_q(dS, dSii, q_sym, atmlst, gridslice)

        dqdx_fix_Vq = einsum_ij_Adj_Adi_inverseK(pcmobj, dSdx_dot_q)

    elif pcmobj.method.upper() in ['IEF-PCM', 'IEFPCM', 'SMD']:
        dF, dA = get_dF_dA(pcmobj.surface)
        dSii = get_dSii(pcmobj.surface, dF)
        dF = None

        dD, dS = get_dD_dS(pcmobj.surface, with_D=True, with_S=True)

        # dR = f_eps/(2*pi) * (dD*A + D*dA)
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)
        f_eps_over_2pi = f_epsilon/(2.0*PI)

        dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)

        DA = D*A
        dKdx_dot_q = dSdx_dot_q - f_eps_over_2pi * contract('ij,Adj->Adi', DA, dSdx_dot_q)

        dAdx_dot_Sq = get_dA_dot_q(dA, S @ q, atmlst)
        dKdx_dot_q -= f_eps_over_2pi * contract('ij,Adj->Adi', D, dAdx_dot_Sq)

        AS = (A * S.T).T # It's just diag(A) @ S
        dDdx_dot_ASq = get_dD_dot_q(dD, AS @ q, atmlst, gridslice, ngrids)
        dKdx_dot_q -= f_eps_over_2pi * dDdx_dot_ASq

        dqdx_fix_Vq = -einsum_ij_Adj_Adi_inverseK(pcmobj, dKdx_dot_q)

        dAdx_dot_V = get_dA_dot_q(dA, v_grids, atmlst)

        dDdx_dot_AV = get_dD_dot_q(dD, A * v_grids, atmlst, gridslice, ngrids)

        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + contract('ij,Adj->Adi', D, dAdx_dot_V))
        dqdx_fix_Vq += einsum_ij_Adj_Adi_inverseK(pcmobj, dRdx_dot_V)

        invKT_V = pcmobj.left_solve_K(v_grids, K_transpose = True)
        dDdxT_dot_invKT_V = get_dDT_dot_q(dD, invKT_V, atmlst, gridslice, ngrids)

        DT_invKT_V = D.T @ invKT_V
        dAdxT_dot_DT_invKT_V = get_dA_dot_q(dA, DT_invKT_V, atmlst)
        dqdx_fix_Vq += f_eps_over_2pi * (contract('i,Adi->Adi', A, dDdxT_dot_invKT_V) + dAdxT_dot_DT_invKT_V)

        dSdxT_dot_invKT_V = get_dST_dot_q(dS, dSii, invKT_V, atmlst, gridslice)
        dKdxT_dot_invKT_V = dSdxT_dot_invKT_V

        dKdxT_dot_invKT_V -= f_eps_over_2pi * contract('ij,Adj->Adi', AS.T, dDdxT_dot_invKT_V)
        dKdxT_dot_invKT_V -= f_eps_over_2pi * contract('ij,Adj->Adi', S.T, dAdxT_dot_DT_invKT_V)

        dSdxT_dot_AT_DT_invKT_V = get_dST_dot_q(dS, dSii, DA.T @ invKT_V, atmlst, gridslice)
        dKdxT_dot_invKT_V -= f_eps_over_2pi * dSdxT_dot_AT_DT_invKT_V
        invKT_dKdxT_dot_invKT_V = einsum_ij_Adj_Adi_inverseK(pcmobj, dKdxT_dot_invKT_V, K_transpose = True)

        R = -f_epsilon * (cupy.eye(DA.shape[0]) - 1.0/(2.0*PI)*DA)
        dqdx_fix_Vq += -contract('ij,Adj->Adi', R.T, invKT_dKdxT_dot_invKT_V)

        dqdx_fix_Vq *= -0.5

    elif pcmobj.method.upper() in ['SS(V)PE']:
        dF, dA = get_dF_dA(pcmobj.surface)
        dSii = get_dSii(pcmobj.surface, dF)
        dF = None

        dD, dS = get_dD_dS(pcmobj.surface, with_D=True, with_S=True)

        f_eps_over_4pi = f_epsilon/(4.0*PI)
        DA = D*A

        def dK_dot_q(q):
            dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)

            dKdx_dot_q = dSdx_dot_q - f_eps_over_4pi * contract('ij,Adj->Adi', DA, dSdx_dot_q)

            dAdx_dot_Sq = get_dA_dot_q(dA, S @ q, atmlst)
            dKdx_dot_q -= f_eps_over_4pi * contract('ij,Adj->Adi', D, dAdx_dot_Sq)

            AS = (A * S.T).T # It's just diag(A) @ S
            dDdx_dot_ASq = get_dD_dot_q(dD, AS @ q, atmlst, gridslice, ngrids)
            dKdx_dot_q -= f_eps_over_4pi * dDdx_dot_ASq

            dDdxT_dot_q = get_dDT_dot_q(dD, q, atmlst, gridslice, ngrids)
            dKdx_dot_q -= f_eps_over_4pi * contract('ij,Adj->Adi', AS.T, dDdxT_dot_q)

            dAdxT_dot_DT_q = get_dA_dot_q(dA, D.T @ q, atmlst)
            dKdx_dot_q -= f_eps_over_4pi * contract('ij,Adj->Adi', S.T, dAdxT_dot_DT_q)

            dSdxT_dot_AT_DT_q = get_dST_dot_q(dS, dSii, DA.T @ q, atmlst, gridslice)
            dKdx_dot_q -= f_eps_over_4pi * dSdxT_dot_AT_DT_q

            return dKdx_dot_q

        f_eps_over_2pi = f_epsilon/(2.0*PI)

        dKdx_dot_q = dK_dot_q(q)
        dqdx_fix_Vq = -einsum_ij_Adj_Adi_inverseK(pcmobj, dKdx_dot_q)

        dAdx_dot_V = get_dA_dot_q(dA, v_grids, atmlst)

        dDdx_dot_AV = get_dD_dot_q(dD, A * v_grids, atmlst, gridslice, ngrids)

        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + contract('ij,Adj->Adi', D, dAdx_dot_V))
        dqdx_fix_Vq += einsum_ij_Adj_Adi_inverseK(pcmobj, dRdx_dot_V)

        invKT_V = pcmobj.left_solve_K(v_grids, K_transpose = True)
        dDdxT_dot_invKT_V = get_dDT_dot_q(dD, invKT_V, atmlst, gridslice, ngrids)

        DT_invKT_V = D.T @ invKT_V
        dAdxT_dot_DT_invKT_V = get_dA_dot_q(dA, DT_invKT_V, atmlst)
        dqdx_fix_Vq += f_eps_over_2pi * (contract('i,Adi->Adi', A, dDdxT_dot_invKT_V) + dAdxT_dot_DT_invKT_V)

        dKdx_dot_invKT_V = dK_dot_q(invKT_V)
        invKT_dKdx_dot_invKT_V = einsum_ij_Adj_Adi_inverseK(pcmobj, dKdx_dot_invKT_V, K_transpose = True)

        R = -f_epsilon * (cupy.eye(DA.shape[0]) - 1.0/(2.0*PI)*DA)
        dqdx_fix_Vq += -contract('ij,Adj->Adi', R.T, invKT_dKdx_dot_invKT_V)

        dqdx_fix_Vq *= -0.5

    else:
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")

    return dqdx_fix_Vq

def get_dvgrids(pcmobj, dm, atmlst, intopt_derivative):
    assert pcmobj._intermediates is not None

    mol = pcmobj.mol
    gridslice    = pcmobj.surface['gslice_by_atom']
    charge_exp   = pcmobj.surface['charge_exp']
    grid_coords  = pcmobj.surface['grid_coords']

    atom_coords = mol.atom_coords(unit='B')
    atom_charges = numpy.asarray(mol.atom_charges(), dtype=numpy.float64)
    atom_coords = atom_coords[atmlst]
    atom_charges = atom_charges[atmlst]
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)
    fakemol = gto.fakemol_for_charges(grid_coords.get(), expnt=charge_exp.get()**2)
    int2c2e_ip1 = mol._add_suffix('int2c2e_ip1')
    v_ng_ip1 = gto.mole.intor_cross(int2c2e_ip1, fakemol_nuc, fakemol)
    v_ng_ip1 = cupy.array(v_ng_ip1)
    dV_on_charge_dx = contract('dAq,A->Adq', v_ng_ip1, atom_charges)

    v_ng_ip2 = gto.mole.intor_cross(int2c2e_ip1, fakemol, fakemol_nuc)
    v_ng_ip2 = cupy.array(v_ng_ip2)
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        dV_on_charge_dx[i_atom,:,g0:g1] += contract('dqA,A->dq', v_ng_ip2[:,g0:g1,:], atom_charges)

    dIdA = int1e_grids_ip1(mol, grid_coords, dm = dm + dm.T, intopt = intopt_derivative, charge_exponents = charge_exp**2)
    dV_on_charge_dx[atmlst,:,:] -= dIdA[atmlst,:,:]

    dIdC = int1e_grids_ip2(mol, grid_coords, intopt = intopt_derivative, dm = dm, charge_exponents = charge_exp**2)
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        dV_on_charge_dx[i_atom,:,g0:g1] -= dIdC[:,g0:g1]

    return dV_on_charge_dx

def get_dqsym_dx_fix_K_R(pcmobj, dm, atmlst, intopt_derivative):
    dV_on_charge_dx = get_dvgrids(pcmobj, dm, atmlst, intopt_derivative)

    f_epsilon = pcmobj._intermediates['f_epsilon']
    if pcmobj.if_method_in_CPCM_category:
        R = -f_epsilon
        R_dVdx = R * dV_on_charge_dx
    else:
        A = pcmobj._intermediates['A']
        D = pcmobj._intermediates['D']
        DA = D * A
        R = -f_epsilon * (cupy.eye(DA.shape[0]) - 1.0/(2.0*PI)*DA)
        R_dVdx = contract('ij,Adj->Adi', R, dV_on_charge_dx)

    K_1_R_dVdx = einsum_ij_Adj_Adi_inverseK(pcmobj, R_dVdx)
    K_1T_dVdx = einsum_ij_Adj_Adi_inverseK(pcmobj, dV_on_charge_dx, K_transpose = True)

    if pcmobj.if_method_in_CPCM_category:
        RT_K_1T_dVdx = R * K_1T_dVdx
    else:
        RT_K_1T_dVdx = contract('ij,Adj->Adi', R.T, K_1T_dVdx)

    dqdx_fix_K_R = 0.5 * (K_1_R_dVdx + RT_K_1T_dVdx)

    return dqdx_fix_K_R

def get_dqsym_dx(pcmobj, dm, atmlst, intopt_derivative):
    return get_dqsym_dx_fix_vgrids(pcmobj, atmlst) + get_dqsym_dx_fix_K_R(pcmobj, dm, atmlst, intopt_derivative)

def analytical_grad_vmat(pcmobj, dm, mo_coeff, mo_occ, atmlst=None, verbose=None):
    '''
    dv_solv / da
    '''
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is not None and cupy.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    log = logger.new_logger(pcmobj, verbose)
    t1 = log.init_timer()

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]

    if atmlst is None:
        atmlst = range(mol.natm)

    gridslice    = pcmobj.surface['gslice_by_atom']
    charge_exp   = pcmobj.surface['charge_exp']
    grid_coords  = pcmobj.surface['grid_coords']
    q_sym        = pcmobj._intermediates['q_sym']

    aoslice = mol.aoslice_by_atom()
    aoslice = numpy.array(aoslice)

    intopt_fock = int3c1e.VHFOpt(mol)
    intopt_fock.build(cutoff = 1e-14, aosym = True)
    intopt_derivative = int3c1e.VHFOpt(mol)
    intopt_derivative.build(cutoff = 1e-14, aosym = False)

    dIdx_mo = cupy.empty([len(atmlst), 3, nmo, nocc])

    dIdA = int1e_grids_ip1(mol, grid_coords, charges = q_sym, intopt = intopt_derivative, charge_exponents = charge_exp**2)
    for i_atom in atmlst:
        p0,p1 = aoslice[i_atom, 2:]
        # dIdx[i_atom, :, :, :] = 0
        # dIdx[i_atom, :, p0:p1, :] += dIdA[:, p0:p1, :]
        # dIdx[i_atom, :, :, p0:p1] += dIdA[:, p0:p1, :].transpose(0,2,1)
        dIdA_mo = dIdA[:, p0:p1, :] @ mocc
        dIdA_mo = contract('ip,dpj->dij', mo_coeff[p0:p1, :].T, dIdA_mo)
        dIdB_mo = dIdA[:, p0:p1, :].transpose(0,2,1) @ mocc[p0:p1, :]
        dIdB_mo = contract('ip,dpj->dij', mo_coeff.T, dIdB_mo)
        dIdx_mo[i_atom, :, :, :] = dIdA_mo + dIdB_mo

    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        dIdC = int1e_grids_ip2(mol, grid_coords[g0:g1,:], charges = q_sym[g0:g1],
                               intopt = intopt_derivative, charge_exponents = charge_exp[g0:g1]**2)
        dIdC_mo = dIdC @ mocc
        dIdC_mo = contract('ip,dpj->dij', mo_coeff.T, dIdC_mo)
        dIdx_mo[i_atom, :, :, :] += dIdC_mo

    dV_on_molecule_dx_mo = dIdx_mo

    dqdx = get_dqsym_dx(pcmobj, dm, atmlst, intopt_derivative)
    for i_atom in atmlst:
        for i_xyz in range(3):
            dIdx_from_dqdx = int1e_grids(mol, grid_coords, charges = dqdx[i_atom, i_xyz, :],
                                         intopt = intopt_fock, charge_exponents = charge_exp**2)
            dV_on_molecule_dx_mo[i_atom, i_xyz, :, :] += mo_coeff.T @ dIdx_from_dqdx @ mocc

    t1 = log.timer_debug1('computing solvent grad veff', *t1)
    return dV_on_molecule_dx_mo

def make_hess_object(base_method):
    '''Create nuclear hessian object with solvent contributions for the given
    solvent-attached method based on its hessian method in vaccum
    '''
    if isinstance(base_method, HessianBase):
        # For backward compatibility. In gpu4pyscf-1.4 and older, the input
        # argument is a hessian object.
        base_method = base_method.base

    # Must be a solvent-attached method
    with_solvent = base_method.with_solvent
    if with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for Hessian')

    vac_hess = base_method.undo_solvent().Hessian()
    vac_hess.base = base_method
    name = with_solvent.__class__.__name__ + vac_hess.__class__.__name__
    return lib.set_class(WithSolventHess(vac_hess),
                         (WithSolventHess, vac_hess.__class__), name)

class WithSolventHess:
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, hess_method):
        self.__dict__.update(hess_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventHess, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def to_cpu(self):
        from pyscf.solvent.hessian import pcm           # type: ignore
        hess_method = self.undo_solvent().to_cpu()
        return pcm.make_hess_object(hess_method)

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        dm = kwargs.pop('dm', None)
        if dm is None:
            dm = self.base.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        if self.base.with_solvent.frozen_dm0_for_finite_difference_without_response is not None:
            raise NotImplementedError("frozen_dm0_for_finite_difference_without_response not implemented for PCM Hessian")

        with lib.temporary_env(self.base.with_solvent, equilibrium_solvation=True):
            logger.debug(self, 'Compute hessian from solutes')
            self.de_solute = super().kernel(*args, **kwargs)
        logger.debug(self, 'Compute hessian from solvents')
        self.de_solvent = self.base.with_solvent.hess(dm)
        self.de = self.de_solute + self.de_solvent
        return self.de

    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        if atmlst is None:
            atmlst = range(self.mol.natm)
        h1ao = super().make_h1(mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
        if isinstance(self.base, scf.hf.RHF):
            dm = self.base.make_rdm1()
            dv = analytical_grad_vmat(self.base.with_solvent, dm, mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
            for i0, ia in enumerate(atmlst):
                h1ao[i0] += dv[i0]
            return h1ao
        elif isinstance(self.base, scf.uhf.UHF):
            h1aoa, h1aob = h1ao
            solvent = self.base.with_solvent
            dm = self.base.make_rdm1()
            dm = dm[0] + dm[1]
            dva = analytical_grad_vmat(solvent, dm, mo_coeff[0], mo_occ[0], atmlst=atmlst, verbose=verbose)
            dvb = analytical_grad_vmat(solvent, dm, mo_coeff[1], mo_occ[1], atmlst=atmlst, verbose=verbose)
            for i0, ia in enumerate(atmlst):
                h1aoa[i0] += dva[i0]
                h1aob[i0] += dvb[i0]
            return h1aoa, h1aob
        else:
            raise NotImplementedError('Base object is not supported')

    def get_veff_resp_mo(self, mol, dms, mo_coeff, mo_occ, hermi=1):
        v1vo = super().get_veff_resp_mo(mol, dms, mo_coeff, mo_occ, hermi=hermi)
        if not self.base.with_solvent.equilibrium_solvation:
            return v1vo
        v_solvent = self.base.with_solvent._B_dot_x(dms)
        if isinstance(self.base, scf.uhf.UHF):
            n_dm = dms.shape[1]
            mocca = mo_coeff[0][:,mo_occ[0]>0]
            moccb = mo_coeff[1][:,mo_occ[1]>0]
            moa, mob = mo_coeff
            nmoa = moa.shape[1]
            nocca = mocca.shape[1]
            v1vo_sol = v_solvent[0] + v_solvent[1]
            v1vo[:,:nmoa*nocca] += _ao2mo(v1vo_sol, mocca, moa).reshape(n_dm,-1)
            v1vo[:,nmoa*nocca:] += _ao2mo(v1vo_sol, moccb, mob).reshape(n_dm,-1)
        elif isinstance(self.base, scf.hf.RHF):
            n_dm = dms.shape[0]
            mocc = mo_coeff[:,mo_occ>0]
            v1vo += _ao2mo(v_solvent, mocc, mo_coeff).reshape(n_dm,-1)
        else:
            raise NotImplementedError('Base object is not supported')
        return v1vo

    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass


