# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
from gpu4pyscf.tdscf._lr_eig import eigh as lr_eigh
from gpu4pyscf.dft.rks import KohnShamDFT
from gpu4pyscf.lib.cupy_helper import contract, tag_array, transpose_sum
from gpu4pyscf.lib import logger
from gpu4pyscf.tdscf import rhf as tdhf_gpu
from gpu4pyscf.tdscf import ris
from gpu4pyscf import scf
from gpu4pyscf import dft
from gpu4pyscf.df import int3c2e

__all__ = [
    'TDA_ris', 'TDDFT_ris', 'TDRKS_ris', 'CasidaTDDFT_ris', 'TDDFTNoHybrid_ris',
]

def get_ab(td, mf, J_fit, K_fit, theta, mo_energy=None, mo_coeff=None, mo_occ=None, singlet=True):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ai||jb)
    B[i,a,j,b] = (ai||bj)

    Ref: Chem Phys Lett, 256, 454
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
    nao, nmo = mo_coeff.shape
    occidx = cp.where(mo_occ==2)[0]
    viridx = cp.where(mo_occ==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = cp.hstack((orbo,orbv))

    e_ia = mo_energy[viridx] - mo_energy[occidx,None]
    a = cp.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = cp.zeros_like(a)
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    auxmol_J = ris.get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
    if K_fit == J_fit and (omega == 0 or omega is None):
        auxmol_K = auxmol_J
    else:
        auxmol_K = ris.get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit)

    def get_erimo(auxmol_i):
        naux = auxmol_i.nao
        int3c = int3c2e.get_int3c2e(mol, auxmol_i)
        int2c2e = auxmol_i.intor('int2c2e')
        int3c = cp.asarray(int3c)
        int2c2e = cp.asarray(int2c2e)
        df_coef = cp.linalg.solve(int2c2e, int3c.reshape(nao*nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        eri = contract('ijP,Pkl->ijkl', int3c, df_coef)
        eri_mo = contract('pjkl,pi->ijkl', eri, orbo.conj())
        eri_mo = contract('ipkl,pj->ijkl', eri_mo, mo)
        eri_mo = contract('ijpl,pk->ijkl', eri_mo, mo.conj())
        eri_mo = contract('ijkp,pl->ijkl', eri_mo, mo)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        return eri_mo
    def get_erimo_omega(auxmol_i, omega):
        naux = auxmol_i.nao
        int3c = int3c2e.get_int3c2e(mol, auxmol_i, omega=omega)
        with auxmol_i.with_range_coulomb(omega):
            int2c2e = auxmol_i.intor('int2c2e')
        int3c = cp.asarray(int3c)
        int2c2e = cp.asarray(int2c2e)
        df_coef = cp.linalg.solve(int2c2e, int3c.reshape(nao*nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        eri = contract('ijP,Pkl->ijkl', int3c, df_coef)
        eri_mo = contract('pjkl,pi->ijkl', eri, orbo.conj())
        eri_mo = contract('ipkl,pj->ijkl', eri_mo, mo)
        eri_mo = contract('ijpl,pk->ijkl', eri_mo, mo.conj())
        eri_mo = contract('ijkp,pl->ijkl', eri_mo, mo)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        return eri_mo
    def add_hf_(a, b, hyb=1):
        eri_mo_J = get_erimo(auxmol_J)
        eri_mo_K = get_erimo(auxmol_K)
        if singlet:
            a += cp.einsum('iabj->iajb', eri_mo_J[:nocc,nocc:,nocc:,:nocc]) * 2
            a -= cp.einsum('ijba->iajb', eri_mo_K[:nocc,:nocc,nocc:,nocc:]) * hyb
            b += cp.einsum('iajb->iajb', eri_mo_J[:nocc,nocc:,:nocc,nocc:]) * 2
            b -= cp.einsum('jaib->iajb', eri_mo_K[:nocc,nocc:,:nocc,nocc:]) * hyb
        else:
            a -= cp.einsum('ijba->iajb', eri_mo_K[:nocc,:nocc,nocc:,nocc:]) * hyb
            b -= cp.einsum('jaib->iajb', eri_mo_K[:nocc,nocc:,:nocc,nocc:]) * hyb

    if getattr(td, 'with_solvent', None):
        raise NotImplementedError("PCM TDDFT RIS is not supported")

    if isinstance(mf, scf.hf.KohnShamDFT):
        grids = mf.grids
        ni = mf._numint
        if mf.do_nlc():
            logger.warn(mf, 'NLC functional found in DFT object. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        add_hf_(a, b, hyb)
        if omega != 0:  # For RSH
            eri_mo_K = get_erimo_omega(auxmol_K, omega)
            k_fac = alpha - hyb
            a -= cp.einsum('ijba->iajb', eri_mo_K[:nocc,:nocc,nocc:,nocc:]) * k_fac
            b -= cp.einsum('jaib->iajb', eri_mo_K[:nocc,nocc:,:nocc,nocc:]) * k_fac

        xctype = 'LDA'
        opt = getattr(ni, 'gdftopt', None)
        if opt is None:
            ni.build(mol, grids.coords)
            opt = ni.gdftopt
        _sorted_mol = opt._sorted_mol
        mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])
        orbo = opt.sort_orbitals(orbo, axis=[0])
        orbv = opt.sort_orbitals(orbv, axis=[0])
        # LDA kernel part
        ao_deriv = 0
        exchange_factor = 1.0 - hyb
        xc = f"{exchange_factor}*SLATER,VWN"
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
            mo_coeff_mask = mo_coeff[mask]
            rho = ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask,
                                mo_occ, mask, xctype, with_lapl=False)
            if singlet or singlet is None:
                fxc = ni.eval_xc_eff(xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc[0,0] * weight
            else:
                fxc = ni.eval_xc_eff(xc, cp.stack((rho, rho)) * 0.5, deriv=2, xctype=xctype)[2]
                wfxc = (fxc[0, 0, 0, 0] - fxc[1, 0, 0, 0]) * 0.5 * weight
            orbo_mask = orbo[mask]
            orbv_mask = orbv[mask]
            rho_o = contract('pr,pi->ri', ao, orbo_mask)
            rho_v = contract('pr,pi->ri', ao, orbv_mask)
            rho_ov = contract('ri,ra->ria', rho_o, rho_v)
            w_ov = contract('ria,r->ria', rho_ov, wfxc)
            iajb = contract('ria,rjb->iajb', rho_ov, w_ov) * 2
            a += iajb
            b += iajb

    else:
        add_hf_(a, b)

    return a.get(), b.get()

def gen_eri2c_eri3c(mol, auxmol, omega=0, tag='_sph'):
    '''
    Total number of contracted GTOs for the mole and auxmol object
    '''

    mol.set_range_coulomb(omega)
    auxmol.set_range_coulomb(omega)
    eri2c = auxmol.intor('int2c2e'+tag)
    pmol = mol + auxmol

    eri3c = pmol.intor('int3c2e'+tag,
                        shls_slice=(0,mol.nbas,0,mol.nbas,
                        mol.nbas,mol.nbas+auxmol.nbas))

    eri2c = cp.asarray(eri2c)
    eri3c = cp.asarray(eri3c)   

    return eri2c, eri3c


def gen_response_ris(mf, singlet=True, hermi=0, theta=0.2, J_fit = 'sp', K_fit = 's', grid_level=1):
    assert hermi != 2
    assert singlet
    auxmol_J = ris.get_auxmol(mf.mol, theta, J_fit)
    auxmol_K = ris.get_auxmol(mf.mol, theta, K_fit)
    eri2c_J, eri3c_J = gen_eri2c_eri3c(mf.mol, auxmol_J)
    eri2c_K, eri3c_K = gen_eri2c_eri3c(mf.mol, auxmol_K)
    eri2c_J_inv = cp.linalg.inv(eri2c_J)
    eri3c_2c_J = cp.einsum('uvA,AB->uvB', eri3c_J, eri2c_J_inv)
    eri2c_K_inv = cp.linalg.inv(eri2c_K)
    eri3c_2c_K = cp.einsum('uvA,AB->uvB', eri3c_K, eri2c_K_inv)
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    max_memory=None
    if isinstance(mf, dft.KohnShamDFT):
        grids = dft.gen_grid.Grids(mol)
        grids.level = grid_level
        grids.build(with_non0tab=False, sort_grids=True)

        # if grids and grids.coords is None:
        #     grids.build(mol=mol, with_non0tab=False, sort_grids=True)
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
        if singlet is None:
            # for ground state orbital hessian
            spin = 0
        else:
            spin = 1
        exchange_factor = 1.0 - hyb
        xc = f"{exchange_factor}*SLATER,VWN"
        rho0, vxc, fxc = ni.cache_xc_kernel(
            mol, grids, xc, mo_coeff, mo_occ, spin, max_memory=max_memory)
        dm0 = None
        fxc *= .5
        def vind(dm1):
            if hermi == 2:
                v1 = cp.zeros_like(dm1)
            else:
                # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                v1 = ni.nr_rks_fxc_st(mol, grids, xc, dm0, dm1, 0, True,
                                      rho0, vxc, fxc, max_memory=max_memory)
            dm1 = cp.asarray(dm1)
            tmp = cp.einsum('edB,xde->xB', eri3c_J, dm1)
            vj = cp.einsum('uvA,xA->xuv', eri3c_2c_J, tmp)
            if hybrid:
                tmp = cp.einsum('evB,xde->xdvB', eri3c_2c_K, dm1)
                vk = cp.einsum('xdvA,udA->xuv', tmp, eri3c_K)
                if hermi != 2:
                    vk *= hyb
                    v1 += vj - .5 * vk
                else:
                    v1 -= .5 * hyb *vk
            elif hermi != 2:
                v1 += vj
            return v1
    else:
        hyb=1.0

        def vind(dm1):
            dm1 = cp.asarray(dm1)
            tmp = cp.einsum('edB,xde->xB', eri3c_J, dm1)
            vj = cp.einsum('uvA,xA->xuv', eri3c_2c_J, tmp)
            tmp = cp.einsum('evB,xde->xdvB', eri3c_2c_K, dm1)
            vk = cp.einsum('xdvA,udA->xuv', tmp, eri3c_K)
            return vj - vk*hyb*.5
    return vind


def gen_tda_ris_operation(mf, fock_ao=None, singlet=True, wfnsym=None, 
        theta=0.2, J_fit = 'sp', K_fit = 's', grid_level=1):
    '''Generate function to compute A x
    '''
    assert fock_ao is None
    # assert isinstance(mf, scf.hf.SCF)
    assert wfnsym is None
    mo_coeff = mf.mo_coeff
    assert mo_coeff.dtype == cp.float64
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidx = mo_occ == 2
    viridx = mo_occ == 0
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    orbo2 = orbo * 2. # *2 for double occupancy

    e_ia = hdiag = mo_energy[viridx] - mo_energy[occidx,None]
    hdiag = hdiag.ravel()#.get()
    vresp = gen_response_ris(mf, singlet=singlet, hermi=0, theta=theta, 
        J_fit = J_fit, K_fit = K_fit, grid_level=grid_level)
    nocc, nvir = e_ia.shape

    def vind(zs):
        zs = cp.asarray(zs).reshape(-1,nocc,nvir)
        mo1 = contract('xov,pv->xpo', zs, orbv)
        dms = contract('xpo,qo->xpq', mo1, orbo2.conj())
        dms = tag_array(dms, mo1=mo1, occ_coeff=orbo)
        v1ao = vresp(dms)
        v1mo = contract('xpq,qo->xpo', v1ao, orbo)
        v1mo = contract('xpo,pv->xov', v1mo, orbv.conj())
        v1mo += zs * e_ia
        return v1mo.reshape(v1mo.shape[0],-1)#.get()

    return vind, hdiag


def gen_tdhf_ris_operation(mf, fock_ao=None, singlet=True, wfnsym=None, theta=0.2, J_fit = 'sp', K_fit = 's'):
    '''Generate function to compute

    [ A   B ][X]
    [-B* -A*][Y]
    '''
    assert fock_ao is None
    # assert isinstance(mf, scf.hf.SCF)
    mo_coeff = mf.mo_coeff
    assert mo_coeff.dtype == cp.float64
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidx = mo_occ == 2
    viridx = mo_occ == 0
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    e_ia = hdiag = mo_energy[viridx] - mo_energy[occidx,None]
    vresp = gen_response_ris(mf, singlet=singlet, hermi=0, theta=theta, J_fit = J_fit, K_fit = K_fit)
    nocc, nvir = e_ia.shape

    def vind(zs):
        nz = len(zs)
        xs, ys = zs.reshape(nz,2,nocc,nvir).transpose(1,0,2,3)
        xs = cp.asarray(xs).reshape(nz,nocc,nvir)
        ys = cp.asarray(ys).reshape(nz,nocc,nvir)
        # *2 for double occupancy
        tmp = contract('xov,pv->xpo', xs, orbv*2)
        dms = contract('xpo,qo->xpq', tmp, orbo.conj())
        tmp = contract('xov,qv->xoq', ys, orbv.conj()*2)
        dms+= contract('xoq,po->xpq', tmp, orbo)
        v1ao = vresp(dms) # = <mb||nj> Xjb + <mj||nb> Yjb
        v1_top = contract('xpq,qo->xpo', v1ao, orbo)
        v1_top = contract('xpo,pv->xov', v1_top, orbv)
        v1_bot = contract('xpq,po->xoq', v1ao, orbo)
        v1_bot = contract('xoq,qv->xov', v1_bot, orbv)
        v1_top += xs * e_ia  # AX
        v1_bot += ys * e_ia  # (A*)Y
        return cp.hstack((v1_top.reshape(nz,nocc*nvir),
                          -v1_bot.reshape(nz,nocc*nvir)))#.get()

    hdiag = cp.hstack([hdiag.ravel(), -hdiag.ravel()])
    return vind, hdiag#.get()


class TDA_ris(tdhf_gpu.TDA):

    _keys = {'theta', 'J_fit', 'K_fit', 'grid_level'}
    def __init__(self, mf):
        super().__init__(mf)
        self.theta = 2.91
        self.J_fit = 'sp'
        self.K_fit = 's'
        self.grid_level = 1

    def get_ab(self, mf=None):
        if mf is None:
            mf = self._scf
        J_fit = self.J_fit
        K_fit = self.K_fit
        theta = self.theta
        return get_ab(self, mf, J_fit, K_fit, theta, singlet=True)

    def nuc_grad_method(self):
        if getattr(self._scf, 'with_df', None):
            raise NotImplementedError("density fitting gradient is not supported.")
        else:
            from gpu4pyscf.grad import tdrks_ris
            return tdrks_ris.Gradients_ris(self)
        
    def gen_vind(self, mf=None):
        '''Generate function to compute Ax'''
        if mf is None:
            mf = self._scf
        return gen_tda_ris_operation(mf, singlet=self.singlet, theta=self.theta, 
            J_fit = self.J_fit, K_fit = self.K_fit, grid_level=self.grid_level)
    
    def NAC(self):
        if getattr(self._scf, 'with_df', None):
            raise NotImplementedError("density fitting NAC is not supported.")
        else:
            from gpu4pyscf.nac import tdrks
            return tdrks.NAC(self)

class TDDFT_ris(tdhf_gpu.TDHF):

    _keys = {'theta', 'J_fit', 'K_fit'}
    def __init__(self, mf):
        super().__init__(mf)
        self.theta = 0.2
        self.J_fit = 'sp'
        self.K_fit = 's'

    def get_ab(self, mf=None):
        if mf is None:
            mf = self._scf
        J_fit = self.J_fit
        K_fit = self.K_fit
        theta = self.theta
        return get_ab(self, mf, J_fit, K_fit, theta, singlet=True)

    def nuc_grad_method(self):
        if getattr(self._scf, 'with_df', None):
            raise NotImplementedError("density fitting gradient is not supported.")
        else:
            from gpu4pyscf.grad import tdrks_ris
            return tdrks_ris.Gradients_ris(self)
        
    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        return gen_tdhf_ris_operation(mf, singlet=self.singlet, theta=self.theta, J_fit = self.J_fit, K_fit = self.K_fit)

    def NAC(self):
        if getattr(self._scf, 'with_df', None):
            raise NotImplementedError("density fitting NAC is not supported.")
        else:
            from gpu4pyscf.nac import tdrks
            return tdrks.NAC(self)
TDRKS_ris = TDDFT_ris

class CasidaTDDFT_ris(TDDFT_ris):
    '''Solve the Casida TDDFT formula (A-B)(A+B)(X+Y) = (X+Y)w^2
    '''

    init_guess = TDA_ris.init_guess
    get_precond = TDA_ris.get_precond

    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        singlet = self.singlet
        mo_coeff = mf.mo_coeff
        assert mo_coeff.dtype == cp.double
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        occidx = mo_occ == 2
        viridx = mo_occ == 0
        orbv = mo_coeff[:,viridx]
        orbo = mo_coeff[:,occidx]

        e_ia = mo_energy[viridx] - mo_energy[occidx,None]
        d_ia = e_ia ** .5
        ed_ia = e_ia * d_ia
        hdiag = e_ia.ravel() ** 2
        vresp = gen_response_ris(mf, singlet=singlet, hermi=0, theta=self.theta, 
                                 J_fit = self.J_fit, K_fit = self.K_fit)
        nocc, nvir = e_ia.shape

        def vind(zs):
            zs = cp.asarray(zs).reshape(-1,nocc,nvir)
            # *2 for double occupancy
            mo1 = contract('xov,pv->xpo', zs*(d_ia*2), orbv)
            dms = contract('xpo,qo->xpq', mo1, orbo)
            # +cc for A+B and K_{ai,jb} in A == K_{ai,bj} in B
            dms = transpose_sum(dms)
            dms = tag_array(dms, mo1=mo1, occ_coeff=orbo)
            v1ao = vresp(dms)
            v1mo = contract('xpq,qo->xpo', v1ao, orbo)
            v1mo = contract('xpo,pv->xov', v1mo, orbv)
            v1mo += zs * ed_ia
            v1mo *= d_ia
            return v1mo.reshape(v1mo.shape[0],-1)

        return vind, hdiag

    def kernel(self, x0=None, nstates=None):
        '''TDDFT diagonalization solver
        '''
        log = logger.new_logger(self)
        cpu0 = log.init_timer()
        mf = self._scf
        if mf._numint.libxc.is_hybrid_xc(mf.xc):
            raise RuntimeError('%s cannot be used with hybrid functional'
                               % self.__class__)
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

        self.converged, w2, x1 = lr_eigh(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        mo_energy = self._scf.mo_energy
        mo_occ = self._scf.mo_occ
        occidx = mo_occ == 2
        viridx = mo_occ == 0
        e_ia = mo_energy[viridx] - mo_energy[occidx,None]
        e_ia = e_ia**.5
        if isinstance(e_ia, cp.ndarray):
            e_ia = e_ia.get()

        def norm_xy(w, z):
            zp = e_ia * z.reshape(e_ia.shape)
            zm = w/e_ia * z.reshape(e_ia.shape)
            x = (zp + zm) * .5
            y = (zp - zm) * .5
            norm = lib.norm(x)**2 - lib.norm(y)**2
            norm = abs(.5/norm)**.5  # normalize to 0.5 for alpha spin
            return (x*norm, y*norm)

        idx = np.where(w2 > self.positive_eig_threshold)[0]
        self.e = w2[idx]**.5
        self.xy = [norm_xy(self.e[i], x1[i]) for i in idx]
        log.timer('TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

TDDFTNoHybrid_ris = CasidaTDDFT_ris

def tddft_ris(mf):
    '''Driver to create TDDFT or CasidaTDDFT object'''
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        return TDDFT_ris(mf)
    else:
        return CasidaTDDFT_ris(mf)

dft.rks.RKS.TDA_ris           = lib.class_as_method(TDA_ris)
dft.rks.RKS.TDHF_ris          = None
#dft.rks.RKS.TDDFT         = lib.class_as_method(TDDFT)
dft.rks.RKS.TDDFTNoHybrid_ris = lib.class_as_method(TDDFTNoHybrid_ris)
dft.rks.RKS.CasidaTDDFT_ris   = lib.class_as_method(CasidaTDDFT_ris)
dft.rks.RKS.TDDFT_ris         = tddft_ris