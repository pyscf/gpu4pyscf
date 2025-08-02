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
from pyscf import lib
from pyscf.tdscf import uhf as tdhf_cpu
from pyscf import ao2mo
from gpu4pyscf.tdscf._lr_eig import eigh as lr_eigh, eig as lr_eig, real_eig
from gpu4pyscf import scf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.tdscf._uhf_resp_sf import gen_uhf_response_sf
from gpu4pyscf.gto.int3c1e import int1e_grids
from gpu4pyscf.tdscf import rhf as tdhf_gpu
from gpu4pyscf.dft import KohnShamDFT
from pyscf import __config__

__all__ = [
    'TDA', 'CIS', 'TDHF', 'TDUHF', 'TDBase'
]

def get_ab(td, mf, mo_energy=None, mo_coeff=None, mo_occ=None):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ai||jb)
    B[i,a,j,b] = (ai||bj)

    Spin symmetry is considered in the returned A, B lists.  List A has three
    items: (A_aaaa, A_aabb, A_bbbb). A_bbaa = A_aabb.transpose(2,3,0,1).
    B has three items: (B_aaaa, B_aabb, B_bbbb).
    B_bbaa = B_aabb.transpose(2,3,0,1).
    '''
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mo_energy = cp.asarray(mo_energy)
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    mol = mf.mol
    nao = mol.nao_nr()
    occidx_a = cp.where(mo_occ[0]==1)[0]
    viridx_a = cp.where(mo_occ[0]==0)[0]
    occidx_b = cp.where(mo_occ[1]==1)[0]
    viridx_b = cp.where(mo_occ[1]==0)[0]
    orbo_a = mo_coeff[0][:,occidx_a]
    orbv_a = mo_coeff[0][:,viridx_a]
    orbo_b = mo_coeff[1][:,occidx_b]
    orbv_b = mo_coeff[1][:,viridx_b]
    nocc_a = orbo_a.shape[1]
    nvir_a = orbv_a.shape[1]
    nocc_b = orbo_b.shape[1]
    nvir_b = orbv_b.shape[1]
    mo_a = cp.hstack((orbo_a,orbv_a))
    mo_b = cp.hstack((orbo_b,orbv_b))
    nmo_a = nocc_a + nvir_a
    nmo_b = nocc_b + nvir_b

    e_ia_a = (mo_energy[0][viridx_a,None] - mo_energy[0][occidx_a]).T
    e_ia_b = (mo_energy[1][viridx_b,None] - mo_energy[1][occidx_b]).T
    a_aa = cp.diag(e_ia_a.ravel()).reshape(nocc_a,nvir_a,nocc_a,nvir_a)
    a_bb = cp.diag(e_ia_b.ravel()).reshape(nocc_b,nvir_b,nocc_b,nvir_b)
    a_ab = cp.zeros((nocc_a,nvir_a,nocc_b,nvir_b))
    b_aa = cp.zeros_like(a_aa)
    b_ab = cp.zeros_like(a_ab)
    b_bb = cp.zeros_like(a_bb)
    a = (a_aa, a_ab, a_bb)
    b = (b_aa, b_ab, b_bb)

    def add_solvent_(a, b, pcmobj):
        charge_exp  = pcmobj.surface['charge_exp']
        grid_coords = pcmobj.surface['grid_coords']

        vmat = -int1e_grids(pcmobj.mol, grid_coords, charge_exponents = charge_exp**2, intopt = pcmobj.intopt)
        K_LU = pcmobj._intermediates['K_LU']
        K_LU_pivot = pcmobj._intermediates['K_LU_pivot']
        if not isinstance(K_LU, cp.ndarray):
            K_LU = cp.asarray(K_LU)
        if not isinstance(K_LU_pivot, cp.ndarray):
            K_LU_pivot = cp.asarray(K_LU_pivot)
        ngrid_surface = K_LU.shape[0]
        L = cp.tril(K_LU, k=-1) + cp.eye(ngrid_surface)  
        U = cp.triu(K_LU)                

        P = cp.eye(ngrid_surface)
        for i in range(ngrid_surface):
            pivot = int(K_LU_pivot[i].get())
            if K_LU_pivot[i] != i:
                P[[i, pivot]] = P[[pivot, i]]
        K = P.T @ L @ U
        Kinv = cp.linalg.inv(K)
        f_epsilon = pcmobj._intermediates['f_epsilon']
        if pcmobj.if_method_in_CPCM_category:
            R = -f_epsilon * cp.eye(K.shape[0])
        else:
            A = pcmobj._intermediates['A']
            D = pcmobj._intermediates['D']
            DA = D*A
            R = -f_epsilon * (cp.eye(K.shape[0]) - 1.0/(2.0*np.pi)*DA)
        Q = Kinv @ R
        Qs = 0.5*(Q+Q.T)
        
        q_sym = cp.einsum('gh,hkl->gkl', Qs, vmat)
        kEao = contract('gij,gkl->ijkl', vmat, q_sym)

        kEmo_aa = contract('pjkl,pi->ijkl', kEao, orbo_a.conj())
        kEmo_aa = contract('ipkl,pj->ijkl', kEmo_aa, mo_a)
        kEmo_aa = contract('ijpl,pk->ijkl', kEmo_aa, mo_a.conj())
        kEmo_aa = contract('ijkp,pl->ijkl', kEmo_aa, mo_a)

        kEmo_ab = contract('pjkl,pi->ijkl', kEao, orbo_a.conj())
        kEmo_ab = contract('ipkl,pj->ijkl', kEmo_ab, mo_a)
        kEmo_ab = contract('ijpl,pk->ijkl', kEmo_ab, mo_b.conj())
        kEmo_ab = contract('ijkp,pl->ijkl', kEmo_ab, mo_b)

        kEmo_bb = contract('pjkl,pi->ijkl', kEao, orbo_b.conj())
        kEmo_bb = contract('ipkl,pj->ijkl', kEmo_bb, mo_b)
        kEmo_bb = contract('ijpl,pk->ijkl', kEmo_bb, mo_b.conj())
        kEmo_bb = contract('ijkp,pl->ijkl', kEmo_bb, mo_b)

        kEmo_aa = kEmo_aa.reshape(nocc_a,nmo_a,nmo_a,nmo_a)
        kEmo_ab = kEmo_ab.reshape(nocc_a,nmo_a,nmo_b,nmo_b)
        kEmo_bb = kEmo_bb.reshape(nocc_b,nmo_b,nmo_b,nmo_b)
        a_aa, a_ab, a_bb = a
        b_aa, b_ab, b_bb = b

        a_aa += cp.einsum('iabj->iajb', kEmo_aa[:nocc_a,nocc_a:,nocc_a:,:nocc_a])
        b_aa += cp.einsum('iajb->iajb', kEmo_aa[:nocc_a,nocc_a:,:nocc_a,nocc_a:])

        a_bb += cp.einsum('iabj->iajb', kEmo_bb[:nocc_b,nocc_b:,nocc_b:,:nocc_b])
        b_bb += cp.einsum('iajb->iajb', kEmo_bb[:nocc_b,nocc_b:,:nocc_b,nocc_b:])

        a_ab += cp.einsum('iabj->iajb', kEmo_ab[:nocc_a,nocc_a:,nocc_b:,:nocc_b])
        b_ab += cp.einsum('iajb->iajb', kEmo_ab[:nocc_a,nocc_a:,:nocc_b,nocc_b:])

    def add_hf_(a, b, hyb=1):
        if getattr(mf, 'with_df', None):
            from gpu4pyscf.df import int3c2e
            auxmol = mf.with_df.auxmol
            naux = auxmol.nao
            int3c = int3c2e.get_int3c2e(mol, auxmol)
            int2c2e = auxmol.intor('int2c2e')
            int3c = cp.asarray(int3c)
            int2c2e = cp.asarray(int2c2e)
            df_coef = cp.linalg.solve(int2c2e, int3c.reshape(nao*nao, naux).T)
            df_coef = df_coef.reshape(naux, nao, nao)
            eri = contract('ijP,Pkl->ijkl', int3c, df_coef)
        else:
            eri = mol.intor('int2e_sph', aosym='s8')
            eri= ao2mo.restore(1, eri, nao)
            eri = cp.asarray(eri)

        eri_aa = contract('pjkl,pi->ijkl', eri, orbo_a.conj())
        eri_aa = contract('ipkl,pj->ijkl', eri_aa, mo_a)
        eri_aa = contract('ijpl,pk->ijkl', eri_aa, mo_a.conj())
        eri_aa = contract('ijkp,pl->ijkl', eri_aa, mo_a)

        eri_ab = contract('pjkl,pi->ijkl', eri, orbo_a.conj())
        eri_ab = contract('ipkl,pj->ijkl', eri_ab, mo_a)
        eri_ab = contract('ijpl,pk->ijkl', eri_ab, mo_b.conj())
        eri_ab = contract('ijkp,pl->ijkl', eri_ab, mo_b)

        eri_bb = contract('pjkl,pi->ijkl', eri, orbo_b.conj())
        eri_bb = contract('ipkl,pj->ijkl', eri_bb, mo_b)
        eri_bb = contract('ijpl,pk->ijkl', eri_bb, mo_b.conj())
        eri_bb = contract('ijkp,pl->ijkl', eri_bb, mo_b)

        eri_aa = eri_aa.reshape(nocc_a,nmo_a,nmo_a,nmo_a)
        eri_ab = eri_ab.reshape(nocc_a,nmo_a,nmo_b,nmo_b)
        eri_bb = eri_bb.reshape(nocc_b,nmo_b,nmo_b,nmo_b)
        a_aa, a_ab, a_bb = a
        b_aa, b_ab, b_bb = b

        a_aa += cp.einsum('iabj->iajb', eri_aa[:nocc_a,nocc_a:,nocc_a:,:nocc_a])
        a_aa -= cp.einsum('ijba->iajb', eri_aa[:nocc_a,:nocc_a,nocc_a:,nocc_a:]) * hyb
        b_aa += cp.einsum('iajb->iajb', eri_aa[:nocc_a,nocc_a:,:nocc_a,nocc_a:])
        b_aa -= cp.einsum('jaib->iajb', eri_aa[:nocc_a,nocc_a:,:nocc_a,nocc_a:]) * hyb

        a_bb += cp.einsum('iabj->iajb', eri_bb[:nocc_b,nocc_b:,nocc_b:,:nocc_b])
        a_bb -= cp.einsum('ijba->iajb', eri_bb[:nocc_b,:nocc_b,nocc_b:,nocc_b:]) * hyb
        b_bb += cp.einsum('iajb->iajb', eri_bb[:nocc_b,nocc_b:,:nocc_b,nocc_b:])
        b_bb -= cp.einsum('jaib->iajb', eri_bb[:nocc_b,nocc_b:,:nocc_b,nocc_b:]) * hyb

        a_ab += cp.einsum('iabj->iajb', eri_ab[:nocc_a,nocc_a:,nocc_b:,:nocc_b])
        b_ab += cp.einsum('iajb->iajb', eri_ab[:nocc_a,nocc_a:,:nocc_b,nocc_b:])

    if getattr(td, 'with_solvent', None):
        pcmobj = td.with_solvent
        add_solvent_(a, b, pcmobj)

    if isinstance(mf, scf.hf.KohnShamDFT):
        ni = mf._numint
        if mf.do_nlc():
            logger.warn(mf, 'NLC functional found in DFT object. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        add_hf_(a, b, hyb)
        if omega != 0:  # For RSH
            if getattr(mf, 'with_df', None):
                from gpu4pyscf.df import int3c2e
                auxmol = mf.with_df.auxmol
                naux = auxmol.nao
                int3c = int3c2e.get_int3c2e(mol, auxmol, omega=omega)
                with auxmol.with_range_coulomb(omega):
                    int2c2e = auxmol.intor('int2c2e')
                int3c = cp.asarray(int3c)
                int2c2e = cp.asarray(int2c2e)
                df_coef = cp.linalg.solve(int2c2e, int3c.reshape(nao*nao, naux).T)
                df_coef = df_coef.reshape(naux, nao, nao)
                eri = contract('ijP,Pkl->ijkl', int3c, df_coef)
            else:
                with mol.with_range_coulomb(omega):
                    eri = mol.intor('int2e_sph', aosym='s8')
                    eri= ao2mo.restore(1, eri, nao)
                    eri = cp.asarray(eri)

            eri_aa = contract('pjkl,pi->ijkl', eri, orbo_a.conj())
            eri_aa = contract('ipkl,pj->ijkl', eri_aa, mo_a)
            eri_aa = contract('ijpl,pk->ijkl', eri_aa, mo_a.conj())
            eri_aa = contract('ijkp,pl->ijkl', eri_aa, mo_a)

            eri_bb = contract('pjkl,pi->ijkl', eri, orbo_b.conj())
            eri_bb = contract('ipkl,pj->ijkl', eri_bb, mo_b)
            eri_bb = contract('ijpl,pk->ijkl', eri_bb, mo_b.conj())
            eri_bb = contract('ijkp,pl->ijkl', eri_bb, mo_b)

            a_aa, a_ab, a_bb = a
            b_aa, b_ab, b_bb = b
            k_fac = alpha - hyb
            a_aa -= cp.einsum('ijba->iajb', eri_aa[:nocc_a,:nocc_a,nocc_a:,nocc_a:]) * k_fac
            b_aa -= cp.einsum('jaib->iajb', eri_aa[:nocc_a,nocc_a:,:nocc_a,nocc_a:]) * k_fac
            a_bb -= cp.einsum('ijba->iajb', eri_bb[:nocc_b,:nocc_b,nocc_b:,nocc_b:]) * k_fac
            b_bb -= cp.einsum('jaib->iajb', eri_bb[:nocc_b,nocc_b:,:nocc_b,nocc_b:]) * k_fac

        xctype = ni._xc_type(mf.xc)
        opt = getattr(ni, 'gdftopt', None)
        if opt is None:
            ni.build(mol, mf.grids.coords)
            opt = ni.gdftopt
        _sorted_mol = opt._sorted_mol
        mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])
        orbo_a = opt.sort_orbitals(orbo_a, axis=[0])
        orbv_a = opt.sort_orbitals(orbv_a, axis=[0])
        orbo_b = opt.sort_orbitals(orbo_b, axis=[0])
        orbv_b = opt.sort_orbitals(orbv_b, axis=[0])

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(_sorted_mol, mf.grids, nao, ao_deriv):
                mo_coeff_mask_a = mo_coeff[0, mask]
                mo_coeff_mask_b = mo_coeff[1, mask]
                rho = cp.asarray((ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask_a,
                                               mo_occ[0], mask, xctype, with_lapl=False),
                                  ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask_b,
                                               mo_occ[1], mask, xctype, with_lapl=False)))

                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc[:,0,:,0] * weight
                orbo_a_mask = orbo_a[mask]
                orbv_a_mask = orbv_a[mask]
                orbo_b_mask = orbo_b[mask]
                orbv_b_mask = orbv_b[mask]
                rho_o_a = contract('pr,pi->ri', ao, orbo_a_mask)
                rho_v_a = contract('pr,pi->ri', ao, orbv_a_mask)
                rho_o_b = contract('pr,pi->ri', ao, orbo_b_mask)
                rho_v_b = contract('pr,pi->ri', ao, orbv_b_mask)
                rho_ov_a = contract('ri,ra->ria', rho_o_a, rho_v_a)
                rho_ov_b = contract('ri,ra->ria', rho_o_b, rho_v_b)

                w_ov = contract('ria,r->ria', rho_ov_a, wfxc[0,0])
                iajb = contract('ria,rjb->iajb', rho_ov_a, w_ov)
                a_aa += iajb
                b_aa += iajb

                w_ov = contract('ria,r->ria', rho_ov_b, wfxc[0,1])
                iajb = contract('ria,rjb->iajb', rho_ov_a, w_ov)
                a_ab += iajb
                b_ab += iajb

                w_ov = contract('ria,r->ria', rho_ov_b, wfxc[1,1])
                iajb = contract('ria,rjb->iajb', rho_ov_b, w_ov)
                a_bb += iajb
                b_bb += iajb

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(_sorted_mol, mf.grids, nao, ao_deriv):
                mo_coeff_mask_a = mo_coeff[0, mask]
                mo_coeff_mask_b = mo_coeff[1, mask]
                rho = cp.asarray((ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask_a,
                                               mo_occ[0], mask, xctype, with_lapl=False),
                                  ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask_b,
                                               mo_occ[1], mask, xctype, with_lapl=False)))
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight
                orbo_a_mask = orbo_a[mask]
                orbv_a_mask = orbv_a[mask]
                orbo_b_mask = orbo_b[mask]
                orbv_b_mask = orbv_b[mask]
                rho_o_a = contract('xpr,pi->xri', ao, orbo_a_mask)
                rho_v_a = contract('xpr,pi->xri', ao, orbv_a_mask)
                rho_o_b = contract('xpr,pi->xri', ao, orbo_b_mask)
                rho_v_b = contract('xpr,pi->xri', ao, orbv_b_mask)
                rho_ov_a = contract('xri,ra->xria', rho_o_a, rho_v_a[0])
                rho_ov_b = contract('xri,ra->xria', rho_o_b, rho_v_b[0])
                rho_ov_a[1:4] += contract('ri,xra->xria', rho_o_a[0], rho_v_a[1:4])
                rho_ov_b[1:4] += contract('ri,xra->xria', rho_o_b[0], rho_v_b[1:4])
                w_ov_aa = contract('xyr,xria->yria', wfxc[0,:,0], rho_ov_a)
                w_ov_ab = contract('xyr,xria->yria', wfxc[0,:,1], rho_ov_a)
                w_ov_bb = contract('xyr,xria->yria', wfxc[1,:,1], rho_ov_b)

                iajb = contract('xria,xrjb->iajb', w_ov_aa, rho_ov_a)
                a_aa += iajb
                b_aa += iajb

                iajb = contract('xria,xrjb->iajb', w_ov_bb, rho_ov_b)
                a_bb += iajb
                b_bb += iajb

                iajb = contract('xria,xrjb->iajb', w_ov_ab, rho_ov_b)
                a_ab += iajb
                b_ab += iajb

        elif xctype == 'HF':
            pass

        elif xctype == 'NLC':
            pass # Processed later

        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(_sorted_mol, mf.grids, nao, ao_deriv):
                mo_coeff_mask_a = mo_coeff[0, mask]
                mo_coeff_mask_b = mo_coeff[1, mask]
                rho = cp.asarray((ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask_a,
                                               mo_occ[0], mask, xctype, with_lapl=False),
                                  ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask_b,
                                               mo_occ[1], mask, xctype, with_lapl=False)))
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight
                orbo_a_mask = orbo_a[mask]
                orbv_a_mask = orbv_a[mask]
                orbo_b_mask = orbo_b[mask]
                orbv_b_mask = orbv_b[mask]
                rho_oa = contract('xpr,pi->xri', ao, orbo_a_mask)
                rho_ob = contract('xpr,pi->xri', ao, orbo_b_mask)
                rho_va = contract('xpr,pi->xri', ao, orbv_a_mask)
                rho_vb = contract('xpr,pi->xri', ao, orbv_b_mask)
                rho_ov_a = contract('xri,ra->xria', rho_oa, rho_va[0])
                rho_ov_b = contract('xri,ra->xria', rho_ob, rho_vb[0])
                rho_ov_a[1:4] += contract('ri,xra->xria', rho_oa[0], rho_va[1:4])
                rho_ov_b[1:4] += contract('ri,xra->xria', rho_ob[0], rho_vb[1:4])
                tau_ov_a = contract('xri,xra->ria', rho_oa[1:4], rho_va[1:4]) * .5
                tau_ov_b = contract('xri,xra->ria', rho_ob[1:4], rho_vb[1:4]) * .5
                rho_ov_a = cp.vstack([rho_ov_a, tau_ov_a[cp.newaxis]])
                rho_ov_b = cp.vstack([rho_ov_b, tau_ov_b[cp.newaxis]])
                w_ov_aa = contract('xyr,xria->yria', wfxc[0,:,0], rho_ov_a)
                w_ov_ab = contract('xyr,xria->yria', wfxc[0,:,1], rho_ov_a)
                w_ov_bb = contract('xyr,xria->yria', wfxc[1,:,1], rho_ov_b)

                iajb = contract('xria,xrjb->iajb', w_ov_aa, rho_ov_a)
                a_aa += iajb
                b_aa += iajb

                iajb = contract('xria,xrjb->iajb', w_ov_bb, rho_ov_b)
                a_bb += iajb
                b_bb += iajb

                iajb = contract('xria,xrjb->iajb', w_ov_ab, rho_ov_b)
                a_ab += iajb
                b_ab += iajb

        if mf.do_nlc():
            raise NotImplementedError('vv10 nlc not implemented in get_ab(). '
                                      'However the nlc contribution is small in TDDFT, '
                                      'so feel free to take the risk and comment out this line.')

    else:
        add_hf_(a, b)
    a_aa, a_ab, a_bb = a
    b_aa, b_ab, b_bb = b

    return (a_aa.get(), a_ab.get(), a_bb.get()), (b_aa.get(), b_ab.get(), b_bb.get())

REAL_EIG_THRESHOLD = tdhf_cpu.REAL_EIG_THRESHOLD

def gen_tda_operation(td, mf, fock_ao=None, wfnsym=None):
    '''A x
    '''
    assert fock_ao is None
    assert isinstance(mf, scf.hf.SCF)
    assert wfnsym is None
    if isinstance(mf.mo_coeff, (tuple, list)):
        # The to_gpu() in pyscf is not able to convert SymAdaptedUHF.mo_coeff.
        # In this case, mf.mo_coeff has the type (NPArrayWithTag, NPArrayWithTag).
        # cp.asarray() for this object leads to an error in
        # cupy._core.core._array_from_nested_sequence
        mo_coeff = cp.asarray(mf.mo_coeff[0]), cp.asarray(mf.mo_coeff[1])
    else:
        mo_coeff = cp.asarray(mf.mo_coeff)
    assert mo_coeff[0].dtype == cp.float64
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = mo_occ[0] ==0
    viridxb = mo_occ[1] ==0
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]

    e_ia_a = mo_energy[0][viridxa] - mo_energy[0][occidxa,None]
    e_ia_b = mo_energy[1][viridxb] - mo_energy[1][occidxb,None]
    e_ia = cp.hstack((e_ia_a.reshape(-1), e_ia_b.reshape(-1)))
    hdiag = e_ia
    nocca, nvira = e_ia_a.shape
    noccb, nvirb = e_ia_b.shape

    vresp = td.gen_response(hermi=0)

    def vind(zs):
        nz = len(zs)
        zs = cp.asarray(zs)
        za = zs[:,:nocca*nvira].reshape(nz,nocca,nvira)
        zb = zs[:,nocca*nvira:].reshape(nz,noccb,nvirb)
        mo1a = contract('xov,pv->xpo', za, orbva)
        dmsa = contract('xpo,qo->xpq', mo1a, orboa.conj())
        mo1b = contract('xov,pv->xpo', zb, orbvb)
        dmsb = contract('xpo,qo->xpq', mo1b, orbob.conj())
        dms = cp.asarray((dmsa, dmsb))
        dms = tag_array(dms, mo1=[mo1a,mo1b], occ_coeff=[orboa,orbob])
        v1ao = vresp(dms)
        v1a = contract('xpq,qo->xpo', v1ao[0], orboa)
        v1a = contract('xpo,pv->xov', v1a, orbva.conj())
        v1b = contract('xpq,qo->xpo', v1ao[1], orbob)
        v1b = contract('xpo,pv->xov', v1b, orbvb.conj())
        v1a += za * e_ia_a
        v1b += zb * e_ia_b
        hx = cp.hstack((v1a.reshape(nz,-1), v1b.reshape(nz,-1)))
        return hx

    return vind, hdiag


class TDBase(tdhf_gpu.TDBase):

    def gen_response(self, mo_coeff=None, mo_occ=None, hermi=0):
        '''Generate function to compute A x'''
        return self._scf.gen_response(mo_coeff=None, mo_occ=None, hermi=hermi)

    def get_ab(self, mf=None):
        if mf is None: mf = self._scf
        return get_ab(self, mf)

    def nuc_grad_method(self):
        if getattr(self._scf, 'with_df', None):
            from gpu4pyscf.df.grad import tduhf
            return tduhf.Gradients(self)
        else:
            from gpu4pyscf.grad import tduhf
            return tduhf.Gradients(self)

    def nac_method(self): 
        raise NotImplementedError("Nonadiabatic coupling vector for unrestricted case is not implemented.")

    def _contract_multipole(tdobj, ints, hermi=True, xy=None):
        if xy is None: xy = tdobj.xy
        mo_coeff = tdobj._scf.mo_coeff
        mo_occ = tdobj._scf.mo_occ
        orbo_a = mo_coeff[0][:,mo_occ[0]==1]
        orbv_a = mo_coeff[0][:,mo_occ[0]==0]
        orbo_b = mo_coeff[1][:,mo_occ[1]==1]
        orbv_b = mo_coeff[1][:,mo_occ[1]==0]
        if isinstance(orbo_a, cp.ndarray):
            orbo_a = orbo_a.get()
            orbv_a = orbv_a.get()
            orbo_b = orbo_b.get()
            orbv_b = orbv_b.get()

        ints_a = np.einsum('...pq,pi,qj->...ij', ints, orbo_a.conj(), orbv_a)
        ints_b = np.einsum('...pq,pi,qj->...ij', ints, orbo_b.conj(), orbv_b)
        pol = [(np.einsum('...ij,ij->...', ints_a, x[0]) +
                np.einsum('...ij,ij->...', ints_b, x[1])) for x,y in xy]
        pol = np.array(pol)
        y = xy[0][1]
        if isinstance(y[0], np.ndarray):
            pol_y = [(np.einsum('...ij,ij->...', ints_a, y[0]) +
                      np.einsum('...ij,ij->...', ints_b, y[1])) for x,y in xy]
            if hermi:
                pol += pol_y
            else:  # anti-Hermitian
                pol -= pol_y
        return pol


class TDA(TDBase):
    __doc__ = tdhf_gpu.TDA.__doc__

    singlet = None

    def gen_vind(self, mf=None):
        '''Generate function to compute Ax'''
        if mf is None:
            mf = self._scf
        return gen_tda_operation(self, mf)

    def init_guess(self, mf=None, nstates=None, wfnsym=None, return_symmetry=False):
        if mf is None: mf = self._scf
        if nstates is None: nstates = self.nstates
        assert wfnsym is None
        assert not return_symmetry

        mo_energy_a, mo_energy_b = mf.mo_energy
        mo_occ_a, mo_occ_b = mf.mo_occ
        if isinstance(mo_energy_a, cp.ndarray):
            mo_energy_a = mo_energy_a.get()
            mo_energy_b = mo_energy_b.get()
        if isinstance(mo_occ_a, cp.ndarray):
            mo_occ_a = mo_occ_a.get()
            mo_occ_b = mo_occ_b.get()
        occidxa = mo_occ_a >  0
        occidxb = mo_occ_b >  0
        viridxa = mo_occ_a == 0
        viridxb = mo_occ_b == 0
        e_ia_a = mo_energy_a[viridxa] - mo_energy_a[occidxa,None]
        e_ia_b = mo_energy_b[viridxb] - mo_energy_b[occidxb,None]
        nov = e_ia_a.size + e_ia_b.size
        nstates = min(nstates, nov)

        e_ia = np.append(e_ia_a.ravel(), e_ia_b.ravel())
        # Find the nstates-th lowest energy gap
        e_threshold = np.partition(e_ia, nstates-1)[nstates-1]
        e_threshold += self.deg_eia_thresh

        idx = np.where(e_ia <= e_threshold)[0]
        x0 = np.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1
        return x0

    def kernel(self, x0=None, nstates=None):
        '''TDA diagonalization solver
        '''
        log = logger.new_logger(self)
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        def pickeig(w, v, nroots, envs):
            idx = cp.where(w > self.positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx

        x0sym = None
        if x0 is None:
            x0 = self.init_guess()

        self.converged, self.e, x1 = lr_eigh(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        nmo = self._scf.mo_occ[0].size
        nocca, noccb = self._scf.nelec
        nvira = nmo - nocca
        nvirb = nmo - noccb
        self.xy = [((xi[:nocca*nvira].reshape(nocca,nvira),  # X_alpha
                     xi[nocca*nvira:].reshape(noccb,nvirb)), # X_beta
                    (0, 0))  # (Y_alpha, Y_beta)
                   for xi in x1]

        log.timer('TDA', *cpu0)
        self._finalize()
        return self.e, self.xy

CIS = TDA

class SpinFlipTDA(TDBase):
    '''
    Attributes:
        extype : int (0 or 1)
            Spin flip up: exytpe=0. Spin flip down: exytpe=1.
        collinear : str
            collinear schemes, can be
            'col': collinear, by default
            'ncol': non-collinear
            'mcol': multi-collinear
        collinear_samples : int
            Integration samples for the multi-collinear treatment
    '''

    extype = getattr(__config__, 'tdscf_uhf_SFTDA_extype', 1)
    collinear = getattr(__config__, 'tdscf_uhf_SFTDA_collinear', 'col')
    collinear_samples = getattr(__config__, 'tdscf_uhf_SFTDA_collinear_samples', 200)

    _keys = {'extype', 'collinear', 'collinear_samples'}

    def gen_vind(self):
        '''Generate function to compute A*x for spin-flip TDDFT case.
        '''
        mf = self._scf
        assert isinstance(mf, scf.hf.SCF)
        if isinstance(mf.mo_coeff, (tuple, list)):
            # The to_gpu() in pyscf is not able to convert SymAdaptedUHF.mo_coeff.
            # In this case, mf.mo_coeff has the type (NPArrayWithTag, NPArrayWithTag).
            # cp.asarray() for this object leads to an error in
            # cupy._core.core._array_from_nested_sequence
            mo_coeff = cp.asarray(mf.mo_coeff[0]), cp.asarray(mf.mo_coeff[1])
        else:
            mo_coeff = cp.asarray(mf.mo_coeff)
        assert mo_coeff[0].dtype == cp.float64
        mo_energy = cp.asarray(mf.mo_energy)
        mo_occ = cp.asarray(mf.mo_occ)
        nao, nmo = mo_coeff[0].shape

        extype = self.extype
        if extype == 0:
            occidxb = mo_occ[1] > 0
            viridxa = mo_occ[0] ==0
            orbob = mo_coeff[1][:,occidxb]
            orbva = mo_coeff[0][:,viridxa]
            orbov = (orbob, orbva)
            e_ia = mo_energy[0][viridxa] - mo_energy[1][occidxb,None]
            hdiag = e_ia.ravel()

        elif extype == 1:
            occidxa = mo_occ[0] > 0
            viridxb = mo_occ[1] ==0
            orboa = mo_coeff[0][:,occidxa]
            orbvb = mo_coeff[1][:,viridxb]
            orbov = (orboa, orbvb)
            e_ia = mo_energy[1][viridxb] - mo_energy[0][occidxa,None]
            hdiag = e_ia.ravel()

        vresp = gen_uhf_response_sf(
            mf, hermi=0, collinear=self.collinear,
            collinear_samples=self.collinear_samples)

        def vind(zs):
            zs = cp.asarray(zs).reshape(-1, *e_ia.shape)
            orbo, orbv = orbov
            mo1 = contract('xov,pv->xpo', zs, orbv)
            dms = contract('xpo,qo->xpq', mo1, orbo.conj())
            dms = tag_array(dms, mo1=mo1, occ_coeff=orbo)
            v1ao = vresp(dms)
            v1mo = contract('xpq,qo->xpo', v1ao, orbo)
            v1mo = contract('xpo,pv->xov', v1mo, orbv.conj())
            v1mo += zs * e_ia
            return v1mo.reshape(len(v1mo), -1)

        return vind, hdiag

    def _init_guess(self, mf, nstates):
        mo_energy_a, mo_energy_b = mf.mo_energy
        mo_occ_a, mo_occ_b = mf.mo_occ
        if isinstance(mo_energy_a, cp.ndarray):
            mo_energy_a = mo_energy_a.get()
            mo_energy_b = mo_energy_b.get()
        if isinstance(mo_occ_a, cp.ndarray):
            mo_occ_a = mo_occ_a.get()
            mo_occ_b = mo_occ_b.get()

        if self.extype == 0:
            occidxb = mo_occ_b > 0
            viridxa = mo_occ_a ==0
            e_ia = mo_energy_a[viridxa] - mo_energy_b[occidxb,None]

        elif self.extype == 1:
            occidxa = mo_occ_a > 0
            viridxb = mo_occ_b ==0
            e_ia = mo_energy_b[viridxb] - mo_energy_a[occidxa,None]

        e_ia = e_ia.ravel()
        nov = e_ia.size
        nstates = min(nstates, nov)
        e_threshold = np.partition(e_ia, nstates-1)[nstates-1]
        idx = np.where(e_ia <= e_threshold)[0]
        nstates = idx.size
        e = e_ia[idx]
        idx = idx[np.argsort(e)]
        x0 = np.zeros((nstates, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1
        return np.sort(e), x0.reshape(nstates, *e_ia.shape)

    def init_guess(self, mf=None, nstates=None, wfnsym=None):
        if mf is None: mf = self._scf
        if nstates is None: nstates = self.nstates
        x0 = self._init_guess(mf, nstates)[1]
        return x0.reshape(len(x0), -1)

    def dump_flags(self, verbose=None):
        TDBase.dump_flags(self, verbose)
        logger.info(self, 'extype = %s', self.extype)
        logger.info(self, 'collinear = %s', self.collinear)
        if self.collinear == 'mcol':
            logger.info(self, 'collinear_samples = %s', self.collinear_samples)
        return self

    def check_sanity(self):
        TDBase.check_sanity(self)
        assert self.extype in (0, 1)
        assert self.collinear in ('col', 'ncol', 'mcol')
        return self

    def kernel(self, x0=None, nstates=None):
        '''Spin-flip TDA diagonalization solver
        '''
        log = logger.new_logger(self)
        cpu0 = log.init_timer()
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        if self.collinear == 'col' and isinstance(self._scf, KohnShamDFT):
            mf = self._scf
            ni = mf._numint
            if not ni.libxc.is_hybrid_xc(mf.xc):
                self.converged = True
                self.e, xs = self._init_guess()
                self.xy = [(x, 0) for x in xs]
                return self.e, self.xy

        x0sym = None
        if x0 is None:
            x0 = self.init_guess()

        # Keep all eigenvalues as SF-TDDFT allows triplet to singlet
        # "dexcitation"
        def all_eigs(w, v, nroots, envs):
            return w, v, np.arange(w.size)

        vind, hdiag = self.gen_vind()
        precond = self.get_precond(hdiag)

        self.converged, self.e, x1 = lr_eigh(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=all_eigs, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        nmo = self._scf.mo_occ[0].size
        nocca, noccb = self._scf.nelec
        nvira = nmo - nocca
        nvirb = nmo - noccb

        if self.extype == 0:
            self.xy = [(xi.reshape(noccb,nvira), 0) for xi in x1]
        elif self.extype == 1:
            self.xy = [(xi.reshape(nocca,nvirb), 0) for xi in x1]
        log.timer('SpinFlipTDA', *cpu0)
        self._finalize()
        return self.e, self.xy


def gen_tdhf_operation(td, mf, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute

    [ A   B ][X]
    [-B* -A*][Y]
    '''
    assert fock_ao is None
    assert isinstance(mf, scf.hf.SCF)
    if isinstance(mf.mo_coeff, (tuple, list)):
        # The to_gpu() in pyscf is not able to convert SymAdaptedUHF.mo_coeff.
        # In this case, mf.mo_coeff has the type (NPArrayWithTag, NPArrayWithTag).
        # cp.asarray() for this object leads to an error in
        # cupy._core.core._array_from_nested_sequence
        mo_coeff = cp.asarray(mf.mo_coeff[0]), cp.asarray(mf.mo_coeff[1])
    else:
        mo_coeff = cp.asarray(mf.mo_coeff)
    assert mo_coeff[0].dtype == cp.float64
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    occidxa = mo_occ[0] >  0
    occidxb = mo_occ[1] >  0
    viridxa = mo_occ[0] == 0
    viridxb = mo_occ[1] == 0
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]

    e_ia_a = mo_energy[0][viridxa] - mo_energy[0][occidxa,None]
    e_ia_b = mo_energy[1][viridxb] - mo_energy[1][occidxb,None]
    e_ia = hdiag = cp.hstack((e_ia_a.ravel(), e_ia_b.ravel()))
    nocca, nvira = e_ia_a.shape
    noccb, nvirb = e_ia_b.shape

    vresp = td.gen_response(hermi=0)

    def vind(zs):
        nz = len(zs)
        xs, ys = zs.reshape(nz,2,-1).transpose(1,0,2)
        xs = cp.asarray(xs)
        ys = cp.asarray(ys)
        xa = xs[:,:nocca*nvira].reshape(nz,nocca,nvira)
        xb = xs[:,nocca*nvira:].reshape(nz,noccb,nvirb)
        ya = ys[:,:nocca*nvira].reshape(nz,nocca,nvira)
        yb = ys[:,nocca*nvira:].reshape(nz,noccb,nvirb)
        tmp  = contract('xov,pv->xpo', xa, orbva)
        dmsa = contract('xpo,qo->xpq', tmp, orboa.conj())
        tmp  = contract('xov,pv->xpo', xb, orbvb)
        dmsb = contract('xpo,qo->xpq', tmp, orbob.conj())
        tmp  = contract('xov,qv->xoq', ya, orbva.conj())
        dmsa+= contract('xoq,po->xpq', tmp, orboa)
        tmp  = contract('xov,qv->xoq', yb, orbvb.conj())
        dmsb+= contract('xoq,po->xpq', tmp, orbob)
        v1ao = vresp(cp.asarray((dmsa,dmsb)))
        v1a_top = contract('xpq,qo->xpo', v1ao[0], orboa)
        v1a_top = contract('xpo,pv->xov', v1a_top, orbva.conj())
        v1b_top = contract('xpq,qo->xpo', v1ao[1], orbob)
        v1b_top = contract('xpo,pv->xov', v1b_top, orbvb.conj())
        v1a_bot = contract('xpq,po->xoq', v1ao[0], orboa.conj())
        v1a_bot = contract('xoq,qv->xov', v1a_bot, orbva)
        v1b_bot = contract('xpq,po->xoq', v1ao[1], orbob.conj())
        v1b_bot = contract('xoq,qv->xov', v1b_bot, orbvb)

        v1_top = xs * e_ia
        v1_bot = ys * e_ia
        v1_top[:,:nocca*nvira] += v1a_top.reshape(nz,-1)
        v1_bot[:,:nocca*nvira] += v1a_bot.reshape(nz,-1)
        v1_top[:,nocca*nvira:] += v1b_top.reshape(nz,-1)
        v1_bot[:,nocca*nvira:] += v1b_bot.reshape(nz,-1)
        return cp.hstack([v1_top, -v1_bot])

    hdiag = cp.hstack([hdiag.ravel(), -hdiag.ravel()])
    return vind, hdiag


class TDHF(TDBase):

    singlet = None

    @lib.with_doc(gen_tdhf_operation.__doc__)
    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        return gen_tdhf_operation(self, mf, singlet=self.singlet)

    get_precond = tdhf_gpu.TDHF.get_precond

    def init_guess(self, mf=None, nstates=None, wfnsym=None, return_symmetry=False):
        assert not return_symmetry
        x0 = TDA.init_guess(self, mf, nstates, wfnsym, return_symmetry)
        y0 = np.zeros_like(x0)
        return np.hstack([x0, y0])

    def kernel(self, x0=None, nstates=None):
        '''TDHF diagonalization with non-Hermitian eigenvalue solver
        '''
        log = logger.new_logger(self)
        cpu0 = log.init_timer()
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        # handle single kpt PBC SCF
        if getattr(self._scf, 'kpt', None) is not None:
            from pyscf.pbc.lib.kpts_helper import gamma_point
            assert gamma_point(self._scf.kpt)

        x0sym = None
        if x0 is None:
            x0 = self.init_guess()

        self.converged, self.e, x1 = real_eig(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        nmo = self._scf.mo_occ[0].size
        nocca, noccb = self._scf.nelec
        nvira = nmo - nocca
        nvirb = nmo - noccb
        xy = []
        for i, z in enumerate(x1):
            x, y = z.reshape(2, -1)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            if norm < 0:
                log.warn('TDDFT amplitudes |X| smaller than |Y|')
            norm = abs(norm)**-.5
            xy.append(((x[:nocca*nvira].reshape(nocca,nvira) * norm,  # X_alpha
                        x[nocca*nvira:].reshape(noccb,nvirb) * norm), # X_beta
                       (y[:nocca*nvira].reshape(nocca,nvira) * norm,  # Y_alpha
                        y[nocca*nvira:].reshape(noccb,nvirb) * norm)))# Y_beta
        self.xy = xy

        log.timer('TDHF/TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

TDUHF = TDHF

class SpinFlipTDHF(TDBase):

    extype = SpinFlipTDA.extype
    collinear = SpinFlipTDA.collinear
    collinear_samples = SpinFlipTDA.collinear_samples

    _keys = {'extype', 'collinear', 'collinear_samples'}

    def gen_vind(self):
        '''Generate function to compute A*x for spin-flip TDDFT case.
        '''
        mf = self._scf
        assert isinstance(mf, scf.hf.SCF)
        if isinstance(mf.mo_coeff, (tuple, list)):
            # The to_gpu() in pyscf is not able to convert SymAdaptedUHF.mo_coeff.
            # In this case, mf.mo_coeff has the type (NPArrayWithTag, NPArrayWithTag).
            # cp.asarray() for this object leads to an error in
            # cupy._core.core._array_from_nested_sequence
            mo_coeff = cp.asarray(mf.mo_coeff[0]), cp.asarray(mf.mo_coeff[1])
        else:
            mo_coeff = cp.asarray(mf.mo_coeff)
        assert mo_coeff[0].dtype == cp.float64
        mo_energy = cp.asarray(mf.mo_energy)
        mo_occ = cp.asarray(mf.mo_occ)

        occidxa = mo_occ[0] > 0
        occidxb = mo_occ[1] > 0
        viridxa = mo_occ[0] ==0
        viridxb = mo_occ[1] ==0
        orboa = mo_coeff[0][:,occidxa]
        orbob = mo_coeff[1][:,occidxb]
        orbva = mo_coeff[0][:,viridxa]
        orbvb = mo_coeff[1][:,viridxb]
        e_ia_b2a = mo_energy[0][viridxa] - mo_energy[1][occidxb,None]
        e_ia_a2b = mo_energy[1][viridxb] - mo_energy[0][occidxa,None]
        nocca, nvirb = e_ia_a2b.shape
        noccb, nvira = e_ia_b2a.shape

        extype = self.extype
        if extype == 0:
            hdiag = cp.hstack([e_ia_b2a.ravel(), -e_ia_a2b.ravel()])
        else:
            hdiag = cp.hstack([e_ia_a2b.ravel(), -e_ia_b2a.ravel()])

        vresp = gen_uhf_response_sf(
            mf, hermi=0, collinear=self.collinear,
            collinear_samples=self.collinear_samples)

        def vind(zs):
            nz = len(zs)
            zs = cp.asarray(zs).reshape(nz, -1)
            if extype == 0:
                zs_b2a = zs[:,:noccb*nvira].reshape(nz,noccb,nvira)
                zs_a2b = zs[:,noccb*nvira:].reshape(nz,nocca,nvirb)
                dm_b2a = contract('xov,pv->xpo', zs_b2a, orbva)
                dm_b2a = contract('xpo,qo->xpq', dm_b2a, orbob.conj())
                dm_a2b = contract('xov,qv->xoq', zs_a2b, orbvb.conj())
                dm_a2b = contract('xoq,po->xpq', dm_a2b, orboa)
            else:
                zs_a2b = zs[:,:nocca*nvirb].reshape(nz,nocca,nvirb)
                zs_b2a = zs[:,nocca*nvirb:].reshape(nz,noccb,nvira)
                dm_b2a = contract('xov,pv->xpo', zs_b2a, orbva)
                dm_b2a = contract('xpo,qo->xpq', dm_b2a, orbob.conj())
                dm_a2b = contract('xov,qv->xoq', zs_a2b, orbvb.conj())
                dm_a2b = contract('xoq,po->xpq', dm_a2b, orboa)

            '''
            # The slow way to compute individual terms in
            # [A   B] [X]
            # [B* A*] [Y]
            dms = cp.vstack([dm_b2a, dm_a2b])
            v1ao = vresp(dms)
            v1ao_b2a, v1ao_a2b = v1ao[:nz], v1ao[nz:]
            if extype == 0:
                # A*X = (aI||Jb) * z_b2a = -(ab|IJ) * z_b2a
                v1A_b2a = contract('xpq,qo->xpo', v1ao_b2a, orbob)
                v1A_b2a = contract('xpo,pv->xov', v1A_b2a, orbva.conj())
                # (A*)*Y = (iA||Bj) * z_a2b = -(ij|BA) * z_a2b
                v1A_a2b = contract('xpq,po->xoq', v1ao_a2b, orboa.conj())
                v1A_a2b = contract('xoq,qv->xov', v1A_a2b, orbvb)
                # B*Y = (aI||Bj) * z_a2b = -(aj|BI) * z_a2b
                v1B_b2a = contract('xpq,qo->xpo', v1ao_a2b, orbob)
                v1B_b2a = contract('xpo,pv->xov', v1B_b2a, orbva.conj())
                # (B*)*X = (iA||Jb) * z_b2a = -(ib|JA) * z_b2a
                v1B_a2b = contract('xpq,po->xoq', v1ao_b2a, orboa.conj())
                v1B_a2b = contract('xoq,qv->xov', v1B_a2b, orbvb)
                # add the orbital energy difference in A matrix.
                v1_top = v1A_b2a + v1B_b2a + zs_b2a * e_ia_b2a
                v1_bot = v1B_a2b + v1A_a2b + zs_a2b * e_ia_a2b
                hx = cp.hstack([v1_top.reshape(nz,-1), -v1_bot.reshape(nz,-1)])
            else:
                # A*X = (Ai||jB) * z_a2b = -(AB|ij) * z_a2b
                v1A_a2b = contract('xpq,qo->xpo', v1ao_a2b, orboa)
                v1A_a2b = contract('xpo,pv->xov', v1A_a2b, orbvb.conj())
                # (A*)*Y = (Ia||bJ) * z_b2a = -(IJ|ba) * z_b2a
                v1A_b2a = contract('xpq,po->xoq', v1ao_b2a, orbob.conj())
                v1A_b2a = contract('xoq,qv->xov', v1A_b2a, orbva)
                # B*Y = (Ai||bJ) * z_b2a = -(AJ|bi) * z_b2a
                v1B_a2b = contract('xpq,qo->xpo', v1ao_b2a, orboa)
                v1B_a2b = contract('xpo,pv->xov', v1B_a2b, orbvb.conj())
                # (B*)*X = (Ia||jB) * z_a2b = -(IB|ja) * z_a2b
                v1B_b2a = contract('xpq,po->xoq', v1ao_a2b, orbob.conj())
                v1B_b2a = contract('xoq,qv->xov', v1B_b2a, orbva)
                # add the orbital energy difference in A matrix.
                v1_top = v1A_a2b + v1B_a2b + zs_a2b * e_ia_a2b
                v1_bot = v1B_b2a + v1A_b2a + zs_b2a * e_ia_b2a
                hx = cp.hstack([v1_top.reshape(nz,-1), -v1_bot.reshape(nz,-1)])
            '''

            # [A   B] [X]
            # [B* A*] [Y]
            # is simplified to
            dms = dm_b2a + dm_a2b
            v1ao = vresp(dms)
            if extype == 0:
                # v1_top = A*X+B*Y
                # A*X = (aI||Jb) * z_b2a = -(ab|JI) * z_b2a
                # B*Y = (aI||Bj) * z_a2b = -(aj|BI) * z_a2b
                v1_top = contract('xpq,qo->xpo', v1ao, orbob)
                v1_top = contract('xpo,pv->xov', v1_top, orbva.conj())
                # (A*)*Y = (iA||Bj) * z_a2b = -(ij|BA) * z_a2b
                # (B*)*X = (iA||Jb) * z_b2a = -(ib|JA) * z_b2a
                # v1_bot = (B*)*X + (A*)*Y
                v1_bot = contract('xpq,po->xoq', v1ao, orboa.conj())
                v1_bot = contract('xoq,qv->xov', v1_bot, orbvb)
                # add the orbital energy difference in A matrix.
                v1_top += zs_b2a * e_ia_b2a
                v1_bot += zs_a2b * e_ia_a2b
            else:
                # v1_top = A*X+B*Y
                # A*X = (Ai||jB) * z_a2b = -(AB|ji) * z_a2b
                # B*Y = (Ai||bJ) * z_b2a = -(AJ|bi) * z_b2a
                v1_top = contract('xpq,qo->xpo', v1ao, orboa)
                v1_top = contract('xpo,pv->xov', v1_top, orbvb.conj())
                # v1_bot = (B*)*X + (A*)*Y
                # (A*)*Y = (Ia||bJ) * z_b2a = -(IJ|ba) * z_b2a
                # (B*)*X = (Ia||jB) * z_a2b = -(IB|ja) * z_a2b
                v1_bot = contract('xpq,po->xoq', v1ao, orbob.conj())
                v1_bot = contract('xoq,qv->xov', v1_bot, orbva)
                # add the orbital energy difference in A matrix.
                v1_top += zs_a2b * e_ia_a2b
                v1_bot += zs_b2a * e_ia_b2a
            hx = cp.hstack([v1_top.reshape(nz,-1), -v1_bot.reshape(nz,-1)])
            return hx

        return vind, hdiag

    _init_guess = SpinFlipTDA._init_guess

    def init_guess(self, mf=None, nstates=None, wfnsym=None):
        if mf is None: mf = self._scf
        if nstates is None: nstates = self.nstates
        x0 = self._init_guess(mf, nstates)[1]
        nx = len(x0)
        nmo = mf.mo_occ[0].size
        nocca, noccb = mf.nelec
        nvira = nmo - nocca
        nvirb = nmo - noccb
        if self.extype == 0:
            y0 = np.zeros((nx, nocca*nvirb))
        else:
            y0 = np.zeros((nx, noccb*nvira))
        return np.hstack([x0.reshape(nx,-1), y0])

    dump_flags = SpinFlipTDA.dump_flags
    check_sanity = SpinFlipTDA.check_sanity

    def kernel(self, x0=None, nstates=None):
        '''Spin-flip TDA diagonalization solver
        '''
        # TODO: Enable this feature after updating the TDDFT davidson algorithm
        # in pyscf main branch
        raise RuntimeError('Numerical issues in lr_eig')
        log = logger.new_logger(self)
        cpu0 = log.init_timer()
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        if self.collinear == 'col' and isinstance(self._scf, KohnShamDFT):
            raise NotImplementedError

        x0sym = None
        if x0 is None:
            x0 = self.init_guess()

        real_system = self._scf.mo_coeff[0].dtype == np.float64
        def pickeig(w, v, nroots, envs):
            realidx = np.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > self.positive_eig_threshold))[0]
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx, real_system)

        vind, hdiag = self.gen_vind()
        precond = self.get_precond(hdiag)

        self.converged, self.e, x1 = lr_eig(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        nmo = self._scf.mo_occ[0].size
        nocca, noccb = self._scf.nelec
        nvira = nmo - nocca
        nvirb = nmo - noccb

        if self.extype == 0:
            def norm_xy(z):
                x = z[:noccb*nvira].reshape(noccb,nvira)
                y = z[noccb*nvira:].reshape(nocca,nvirb)
                norm = lib.norm(x)**2 - lib.norm(y)**2
                #assert norm > 0
                norm = abs(norm) ** -.5
                return x*norm, y*norm
        elif self.extype == 1:
            def norm_xy(z):
                x = z[:nocca*nvirb].reshape(nocca,nvirb)
                y = z[nocca*nvirb:].reshape(noccb,nvira)
                norm = lib.norm(x)**2 - lib.norm(y)**2
                #assert norm > 0
                norm = abs(norm) ** -.5
                return x*norm, y*norm

        self.xy = [norm_xy(z) for z in x1]
        log.timer('SpinFlipTDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

scf.uhf.UHF.TDA = lib.class_as_method(TDA)
scf.uhf.UHF.TDHF = lib.class_as_method(TDHF)
scf.uhf.UHF.SFTDA = lib.class_as_method(SpinFlipTDA)
scf.uhf.UHF.SFTDHF = lib.class_as_method(SpinFlipTDHF)
