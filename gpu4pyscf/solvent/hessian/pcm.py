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
from pyscf import lib, gto
from gpu4pyscf import scf
from gpu4pyscf.solvent.pcm import PI
from gpu4pyscf.solvent.grad.pcm import grad_qv, grad_solver, grad_nuc, get_dD_dS, get_dF_dA, get_dSii
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib import logger
from gpu4pyscf.hessian.jk import _ao2mo
from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1, int1e_grids_ip2
from gpu4pyscf.gto import int3c1e
from gpu4pyscf.gto.int3c1e import int1e_grids

def hess_nuc(pcmobj, dm):
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
    d2I_dA2 = numpy.einsum('dAq,q->dA', d2I_dA2, q_sym)
    d2I_dA2 = d2I_dA2.reshape(3, 3, mol.natm)
    for i_atom in range(mol.natm):
        d2e_from_d2I[i_atom, i_atom, :, :] += atom_charges[i_atom] * d2I_dA2[:, :, i_atom]

    d2I_dC2 = gto.mole.intor_cross(int2c2e_ipip1, fakemol, fakemol_nuc)
    d2I_dC2 = numpy.einsum('dqA,A->dq', d2I_dC2, atom_charges)
    d2I_dC2 = d2I_dC2.reshape(3, 3, ngrids)
    for i_atom in range(mol.natm):
        g0,g1 = gridslice[i_atom]
        d2e_from_d2I[i_atom, i_atom, :, :] += numpy.einsum('dDq,q->dD', d2I_dC2[:, :, g0:g1], q_sym[g0:g1])

    intopt_derivative = int3c1e.VHFOpt(mol)
    intopt_derivative.build(cutoff = 1e-14, aosym = False)

    dqdx = get_dqsym_dx(pcmobj, dm, range(mol.natm), intopt_derivative)
    dqdx = dqdx.get()

    d2e_from_dIdq = numpy.zeros([mol.natm, mol.natm, 3, 3])
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            d2e_from_dIdq[i_atom, :, i_xyz, :] = grad_nuc(pcmobj, dm, q_sym = dqdx[i_atom, i_xyz, :])

    d2e = d2e_from_d2I - d2e_from_dIdq

    return d2e

def hess_qv(pcmobj, dm, verbose=None):
    raise NotImplementedError("This implementation requires 9 * nao * nao * ngrids of GPU memory")
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

    ngrids = q_sym.shape[0]
    nao = mol.nao

    aoslice = mol.aoslice_by_atom()
    aoslice = numpy.array(aoslice)

    fakemol = gto.fakemol_for_charges(grid_coords.get(), expnt=charge_exp.get()**2)
    intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False)

    d2e_from_d2I = cupy.zeros([mol.natm, mol.natm, 3, 3])

    d2I_dA2 = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ipip1', direct_scf_tol=1e-14)
    d2I_dA2 = cupy.einsum('dijq,q->dij', d2I_dA2, q_sym)
    d2I_dA2 = d2I_dA2.reshape([3, 3, nao, nao])
    for i_atom in range(mol.natm):
        p0,p1 = aoslice[i_atom, 2:]
        d2e_from_d2I[i_atom, i_atom, :, :] += cupy.einsum('ij,dDij->dD', dm[p0:p1, :], d2I_dA2[:, :, p0:p1, :])
        d2e_from_d2I[i_atom, i_atom, :, :] += cupy.einsum('ij,dDij->dD', dm[:, p0:p1], d2I_dA2[:, :, p0:p1, :].transpose(0,1,3,2))

    d2I_dAdB = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ipvip1', direct_scf_tol=1e-14)
    d2I_dAdB = cupy.einsum('dijq,q->dij', d2I_dAdB, q_sym)
    d2I_dAdB = d2I_dAdB.reshape([3, 3, nao, nao])
    for i_atom in range(mol.natm):
        pi0,pi1 = aoslice[i_atom, 2:]
        for j_atom in range(mol.natm):
            pj0,pj1 = aoslice[j_atom, 2:]
            d2e_from_d2I[i_atom, j_atom, :, :] += cupy.einsum('ij,dDij->dD', dm[pi0:pi1, pj0:pj1], d2I_dAdB[:, :, pi0:pi1, pj0:pj1])
            d2e_from_d2I[i_atom, j_atom, :, :] += cupy.einsum('ij,dDij->dD', dm[pj0:pj1, pi0:pi1], d2I_dAdB[:, :, pi0:pi1, pj0:pj1].transpose(0,1,3,2))

    d2I_dAdC = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ip1ip2', direct_scf_tol=1e-14)
    d2I_dAdC = d2I_dAdC.reshape([3, 3, nao, nao, ngrids])
    for i_atom in range(mol.natm):
        p0,p1 = aoslice[i_atom, 2:]
        for j_atom in range(mol.natm):
            g0,g1 = gridslice[j_atom]
            d2e_from_d2I[i_atom, j_atom, :, :] += cupy.einsum('ij,dDijq,q->dD', dm[p0:p1, :], d2I_dAdC[:, :, p0:p1, :, g0:g1], q_sym[g0:g1])
            d2e_from_d2I[i_atom, j_atom, :, :] += cupy.einsum('ij,dDijq,q->dD', dm[:, p0:p1], d2I_dAdC[:, :, p0:p1, :, g0:g1].transpose(0,1,3,2,4), q_sym[g0:g1])

            d2e_from_d2I[j_atom, i_atom, :, :] += cupy.einsum('ij,dDijq,q->dD', dm[p0:p1, :], d2I_dAdC[:, :, p0:p1, :, g0:g1].transpose(1,0,2,3,4), q_sym[g0:g1])
            d2e_from_d2I[j_atom, i_atom, :, :] += cupy.einsum('ij,dDijq,q->dD', dm[:, p0:p1], d2I_dAdC[:, :, p0:p1, :, g0:g1].transpose(1,0,3,2,4), q_sym[g0:g1])

    d2I_dC2 = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ipip2', direct_scf_tol=1e-14)
    d2I_dC2 = cupy.einsum('dijq,ij->dq', d2I_dC2, dm)
    d2I_dC2 = d2I_dC2.reshape([3, 3, ngrids])
    for i_atom in range(mol.natm):
        g0,g1 = gridslice[i_atom]
        d2e_from_d2I[i_atom, i_atom, :, :] += cupy.einsum('dDq,q->dD', d2I_dC2[:, :, g0:g1], q_sym[g0:g1])

    intopt_derivative = int3c1e.VHFOpt(mol)
    intopt_derivative.build(cutoff = 1e-14, aosym = False)

    dqdx = get_dqsym_dx(pcmobj, dm, range(mol.natm), intopt_derivative)

    d2e_from_dIdq = numpy.zeros([mol.natm, mol.natm, 3, 3])
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            d2e_from_dIdq[i_atom, :, i_xyz, :] = grad_qv(pcmobj, dm, q_sym = dqdx[i_atom, i_xyz, :])

    d2e_from_d2I = d2e_from_d2I.get()
    d2e = d2e_from_d2I + d2e_from_dIdq
    d2e *= -1

    t1 = log.timer_debug1('solvent energy d(dI/dx * q)/dx contribution', *t1)
    return d2e

def hess_elec(pcmobj, dm, verbose=None):
    '''
    slow version with finite difference
    TODO: use analytical hess_nuc
    '''
    log = logger.new_logger(pcmobj, verbose)
    t1 = log.init_timer()
    pmol = pcmobj.mol.copy()
    mol = pmol.copy()
    coords = mol.atom_coords(unit='Bohr')

    def pcm_grad_scanner(mol):
        # TODO: use more analytical forms
        pcmobj.reset(mol)
        e, v = pcmobj._get_vind(dm)
        #return grad_elec(pcmobj, dm)
        pcm_grad = grad_nuc(pcmobj, dm)
        pcm_grad+= grad_solver(pcmobj, dm)
        pcm_grad+= grad_qv(pcmobj, dm)
        return pcm_grad

    mol.verbose = 0
    de = numpy.zeros([mol.natm, mol.natm, 3, 3])
    eps = 1e-3
    for ia in range(mol.natm):
        for ix in range(3):
            dv = numpy.zeros_like(coords)
            dv[ia,ix] = eps
            mol.set_geom_(coords + dv, unit='Bohr')
            g0 = pcm_grad_scanner(mol)

            mol.set_geom_(coords - dv, unit='Bohr')
            g1 = pcm_grad_scanner(mol)
            de[ia,:,ix] = (g0 - g1)/2.0/eps
    t1 = log.timer_debug1('solvent energy', *t1)
    pcmobj.reset(pmol)
    return de

def get_dqsym_dx_fix_vgrids(pcmobj, atmlst, inverse_K):
    assert pcmobj._intermediates is not None

    gridslice    = pcmobj.surface['gslice_by_atom']
    v_grids      = pcmobj._intermediates['v_grids']
    A            = pcmobj._intermediates['A']
    D            = pcmobj._intermediates['D']
    S            = pcmobj._intermediates['S']
    R            = pcmobj._intermediates['R']
    q_sym        = pcmobj._intermediates['q_sym']
    f_epsilon    = pcmobj._intermediates['f_epsilon']

    ngrids = q_sym.shape[0]

    def get_dS_dot_q(dS, dSii, q, atmlst, gridslice):
        output = cupy.einsum('diA,i->Adi', dSii[:,:,atmlst], q)
        for i_atom in atmlst:
            g0,g1 = gridslice[i_atom]
            output[i_atom, :, g0:g1] += cupy.einsum('dij,j->di', dS[:,g0:g1,:], q)
            output[i_atom, :, :] -= cupy.einsum('dij,j->di', dS[:,:,g0:g1], q[g0:g1])
        return output
    def get_dST_dot_q(dS, dSii, q, atmlst, gridslice):
        return get_dS_dot_q(-dS.transpose(0,2,1), dSii, q, atmlst, gridslice)

    def get_dA_dot_q(dA, q, atmlst, gridslice):
        return cupy.einsum('diA,i->Adi', dA[:,:,atmlst], q)

    def get_dD_dot_q(dD, q, atmlst, gridslice):
        output = cupy.zeros([len(atmlst), 3, ngrids])
        for i_atom in atmlst:
            g0,g1 = gridslice[i_atom]
            output[i_atom, :, g0:g1] += cupy.einsum('dij,j->di', dD[:,g0:g1,:], q)
            output[i_atom, :, :] -= cupy.einsum('dij,j->di', dD[:,:,g0:g1], q[g0:g1])
        return output
    def get_dDT_dot_q(dD, q, atmlst, gridslice):
        return get_dD_dot_q(-dD.transpose(0,2,1), q, atmlst, gridslice)

    if pcmobj.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
        _, dS = get_dD_dS(pcmobj.surface, with_D=False, with_S=True)
        dF, _ = get_dF_dA(pcmobj.surface)
        dSii = get_dSii(pcmobj.surface, dF)
        dF = None

        # dR = 0, dK = dS
        dSdx_dot_q = get_dS_dot_q(dS, dSii, q_sym, atmlst, gridslice)

        dqdx_fix_Vq = cupy.einsum('ij,Adj->Adi', inverse_K, dSdx_dot_q)

    elif pcmobj.method.upper() in ['IEF-PCM', 'IEFPCM', 'SMD']:
        dF, dA = get_dF_dA(pcmobj.surface)
        dSii = get_dSii(pcmobj.surface, dF)
        dF = None

        dD, dS = get_dD_dS(pcmobj.surface, with_D=True, with_S=True)

        # dR = f_eps/(2*pi) * (dD*A + D*dA)
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)
        f_eps_over_2pi = f_epsilon/(2.0*PI)

        q = inverse_K @ R @ v_grids
        dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)

        DA = D*A
        dKdx_dot_q = dSdx_dot_q - f_eps_over_2pi * cupy.einsum('ij,Adj->Adi', DA, dSdx_dot_q)

        dAdx_dot_Sq = get_dA_dot_q(dA, S @ q, atmlst, gridslice)
        dKdx_dot_q -= f_eps_over_2pi * cupy.einsum('ij,Adj->Adi', D, dAdx_dot_Sq)

        AS = (A * S.T).T # It's just diag(A) @ S
        dDdx_dot_ASq = get_dD_dot_q(dD, AS @ q, atmlst, gridslice)
        dKdx_dot_q -= f_eps_over_2pi * dDdx_dot_ASq

        dqdx_fix_Vq = -cupy.einsum('ij,Adj->Adi', inverse_K, dKdx_dot_q)

        dAdx_dot_V = get_dA_dot_q(dA, v_grids, atmlst, gridslice)

        dDdx_dot_AV = get_dD_dot_q(dD, A * v_grids, atmlst, gridslice)

        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + cupy.einsum('ij,Adj->Adi', D, dAdx_dot_V))
        dqdx_fix_Vq += cupy.einsum('ij,Adj->Adi', inverse_K, dRdx_dot_V)

        invKT_V = inverse_K.T @ v_grids
        dDdxT_dot_invKT_V = get_dDT_dot_q(dD, invKT_V, atmlst, gridslice)

        DT_invKT_V = D.T @ invKT_V
        dAdxT_dot_DT_invKT_V = get_dA_dot_q(dA, DT_invKT_V, atmlst, gridslice)
        dqdx_fix_Vq += f_eps_over_2pi * (cupy.einsum('i,Adi->Adi', A, dDdxT_dot_invKT_V) + dAdxT_dot_DT_invKT_V)

        dSdxT_dot_invKT_V = get_dST_dot_q(dS, dSii, invKT_V, atmlst, gridslice)
        dKdxT_dot_invKT_V = dSdxT_dot_invKT_V

        dKdxT_dot_invKT_V -= f_eps_over_2pi * cupy.einsum('ij,Adj->Adi', AS.T, dDdxT_dot_invKT_V)
        dKdxT_dot_invKT_V -= f_eps_over_2pi * cupy.einsum('ij,Adj->Adi', S.T, dAdxT_dot_DT_invKT_V)

        dSdxT_dot_AT_DT_invKT_V = get_dST_dot_q(dS, dSii, DA.T @ invKT_V, atmlst, gridslice)
        dKdxT_dot_invKT_V -= f_eps_over_2pi * dSdxT_dot_AT_DT_invKT_V

        dqdx_fix_Vq += -cupy.einsum('ij,Adj->Adi', R.T @ inverse_K.T, dKdxT_dot_invKT_V)

        dqdx_fix_Vq *= -0.5

    elif pcmobj.method.upper() in ['SS(V)PE']:
        dF, dA = get_dF_dA(pcmobj.surface)
        dSii = get_dSii(pcmobj.surface, dF)
        dF = None

        dD, dS = get_dD_dS(pcmobj.surface, with_D=True, with_S=True)

        f_eps_over_4pi = f_epsilon/(4.0*PI)

        def dK_dot_q(q):
            dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)

            DA = D*A
            dKdx_dot_q = dSdx_dot_q - f_eps_over_4pi * cupy.einsum('ij,Adj->Adi', DA, dSdx_dot_q)

            dAdx_dot_Sq = get_dA_dot_q(dA, S @ q, atmlst, gridslice)
            dKdx_dot_q -= f_eps_over_4pi * cupy.einsum('ij,Adj->Adi', D, dAdx_dot_Sq)

            AS = (A * S.T).T # It's just diag(A) @ S
            dDdx_dot_ASq = get_dD_dot_q(dD, AS @ q, atmlst, gridslice)
            dKdx_dot_q -= f_eps_over_4pi * dDdx_dot_ASq

            dDdxT_dot_q = get_dDT_dot_q(dD, q, atmlst, gridslice)
            dKdx_dot_q -= f_eps_over_4pi * cupy.einsum('ij,Adj->Adi', AS.T, dDdxT_dot_q)

            dAdxT_dot_DT_q = get_dA_dot_q(dA, D.T @ q, atmlst, gridslice)
            dKdx_dot_q -= f_eps_over_4pi * cupy.einsum('ij,Adj->Adi', S.T, dAdxT_dot_DT_q)

            dSdxT_dot_AT_DT_q = get_dST_dot_q(dS, dSii, DA.T @ q, atmlst, gridslice)
            dKdx_dot_q -= f_eps_over_4pi * dSdxT_dot_AT_DT_q

            return dKdx_dot_q

        f_eps_over_2pi = f_epsilon/(2.0*PI)

        q = inverse_K @ R @ v_grids
        dKdx_dot_q = dK_dot_q(q)
        dqdx_fix_Vq = -cupy.einsum('ij,Adj->Adi', inverse_K, dKdx_dot_q)

        dAdx_dot_V = get_dA_dot_q(dA, v_grids, atmlst, gridslice)

        dDdx_dot_AV = get_dD_dot_q(dD, A * v_grids, atmlst, gridslice)

        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + cupy.einsum('ij,Adj->Adi', D, dAdx_dot_V))
        dqdx_fix_Vq += cupy.einsum('ij,Adj->Adi', inverse_K, dRdx_dot_V)

        invKT_V = inverse_K.T @ v_grids
        dDdxT_dot_invKT_V = get_dDT_dot_q(dD, invKT_V, atmlst, gridslice)

        DT_invKT_V = D.T @ invKT_V
        dAdxT_dot_DT_invKT_V = get_dA_dot_q(dA, DT_invKT_V, atmlst, gridslice)
        dqdx_fix_Vq += f_eps_over_2pi * (cupy.einsum('i,Adi->Adi', A, dDdxT_dot_invKT_V) + dAdxT_dot_DT_invKT_V)

        dKdx_dot_invKT_V = dK_dot_q(invKT_V)
        dqdx_fix_Vq += -cupy.einsum('ij,Adj->Adi', R.T @ inverse_K.T, dKdx_dot_invKT_V)

        dqdx_fix_Vq *= -0.5

    else:
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")

    return dqdx_fix_Vq

def get_dqsym_dx_fix_K_R(pcmobj, dm, atmlst, inverse_K, intopt_derivative):
    assert pcmobj._intermediates is not None

    mol = pcmobj.mol
    gridslice    = pcmobj.surface['gslice_by_atom']
    charge_exp   = pcmobj.surface['charge_exp']
    grid_coords  = pcmobj.surface['grid_coords']
    R            = pcmobj._intermediates['R']

    atom_coords = mol.atom_coords(unit='B')
    atom_charges = numpy.asarray(mol.atom_charges(), dtype=numpy.float64)
    atom_coords = atom_coords[atmlst]
    atom_charges = atom_charges[atmlst]
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)
    fakemol = gto.fakemol_for_charges(grid_coords.get(), expnt=charge_exp.get()**2)
    int2c2e_ip1 = mol._add_suffix('int2c2e_ip1')
    v_ng_ip1 = gto.mole.intor_cross(int2c2e_ip1, fakemol_nuc, fakemol)
    v_ng_ip1 = cupy.array(v_ng_ip1)
    dV_on_charge_dx = cupy.einsum('dAq,A->Adq', v_ng_ip1, atom_charges)

    v_ng_ip2 = gto.mole.intor_cross(int2c2e_ip1, fakemol, fakemol_nuc)
    v_ng_ip2 = cupy.array(v_ng_ip2)
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        dV_on_charge_dx[i_atom,:,g0:g1] += cupy.einsum('dqA,A->dq', v_ng_ip2[:,g0:g1,:], atom_charges)

    dIdA = int1e_grids_ip1(mol, grid_coords, dm = dm + dm.T, intopt = intopt_derivative, charge_exponents = charge_exp**2)
    dV_on_charge_dx[atmlst,:,:] -= dIdA[atmlst,:,:]

    dIdC = int1e_grids_ip2(mol, grid_coords, intopt = intopt_derivative, dm = dm, charge_exponents = charge_exp**2)
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        dV_on_charge_dx[i_atom,:,g0:g1] -= dIdC[:,g0:g1]

    KR_symmetrized = 0.5 * (inverse_K @ R + R.T @ inverse_K.T)
    dqdx_fix_K_R = cupy.einsum('ij,Adj->Adi', KR_symmetrized, dV_on_charge_dx)

    return dqdx_fix_K_R

def get_dqsym_dx(pcmobj, dm, atmlst, intopt_derivative):
    K = pcmobj._intermediates['K']
    inverse_K = cupy.linalg.inv(K)
    return get_dqsym_dx_fix_vgrids(pcmobj, atmlst, inverse_K) + get_dqsym_dx_fix_K_R(pcmobj, dm, atmlst, inverse_K, intopt_derivative)

def analytic_grad_vmat(pcmobj, dm, mo_coeff, mo_occ, atmlst=None, verbose=None):
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
        dIdA_mo = cupy.einsum('ip,dpj->dij', mo_coeff[p0:p1, :].T, dIdA_mo)
        dIdB_mo = dIdA[:, p0:p1, :].transpose(0,2,1) @ mocc[p0:p1, :]
        dIdB_mo = cupy.einsum('ip,dpj->dij', mo_coeff.T, dIdB_mo)
        dIdx_mo[i_atom, :, :, :] = dIdA_mo + dIdB_mo

    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        dIdC = int1e_grids_ip2(mol, grid_coords[g0:g1,:], charges = q_sym[g0:g1],
                               intopt = intopt_derivative, charge_exponents = charge_exp[g0:g1]**2)
        dIdC_mo = dIdC @ mocc
        dIdC_mo = cupy.einsum('ip,dpj->dij', mo_coeff.T, dIdC_mo)
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

def make_hess_object(hess_method):
    if hess_method.base.with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy hessian')

    name = (hess_method.base.with_solvent.__class__.__name__
            + hess_method.__class__.__name__)
    return lib.set_class(WithSolventHess(hess_method),
                         (WithSolventHess, hess_method.__class__), name)

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
            dm = self.base.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        is_equilibrium = self.base.with_solvent.equilibrium_solvation
        self.base.with_solvent.equilibrium_solvation = True
        self.de_solvent = hess_elec(self.base.with_solvent, dm, verbose=self.verbose)
        #self.de_solvent+= hess_nuc(self.base.with_solvent)
        self.de_solute = super().kernel(*args, **kwargs)
        self.de = self.de_solute + self.de_solvent
        self.base.with_solvent.equilibrium_solvation = is_equilibrium
        return self.de

    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        if atmlst is None:
            atmlst = range(self.mol.natm)
        h1ao = super().make_h1(mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
        if isinstance(self.base, scf.hf.RHF):
            dm = self.base.make_rdm1(ao_repr=True)
            dv = analytic_grad_vmat(self.base.with_solvent, dm, mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
            for i0, ia in enumerate(atmlst):
                h1ao[i0] += dv[i0]
            return h1ao
        elif isinstance(self.base, scf.uhf.UHF):
            h1aoa, h1aob = h1ao
            solvent = self.base.with_solvent
            dm = self.base.make_rdm1(ao_repr=True)
            dm = dm[0] + dm[1]
            dva = analytic_grad_vmat(solvent, dm, mo_coeff[0], mo_occ[0], atmlst=atmlst, verbose=verbose)
            dvb = analytic_grad_vmat(solvent, dm, mo_coeff[1], mo_occ[1], atmlst=atmlst, verbose=verbose)
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


