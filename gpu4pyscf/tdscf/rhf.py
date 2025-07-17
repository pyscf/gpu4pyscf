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
from pyscf import lib, gto
from pyscf import ao2mo
from pyscf.tdscf import rhf as tdhf_cpu
from gpu4pyscf.tdscf._lr_eig import eigh as lr_eigh, real_eig
from gpu4pyscf import scf
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.lib import utils
from gpu4pyscf.lib import logger
from gpu4pyscf.gto.int3c1e import int1e_grids
from gpu4pyscf.scf import _response_functions # noqa
from pyscf import __config__

REAL_EIG_THRESHOLD = tdhf_cpu.REAL_EIG_THRESHOLD
#OUTPUT_THRESHOLD = tdhf_cpu.OUTPUT_THRESHOLD
OUTPUT_THRESHOLD = getattr(__config__, 'tdscf_rhf_get_nto_threshold', 0.3)

__all__ = [
    'TDA', 'CIS', 'TDHF', 'TDRHF', 'TDBase'
]


def get_ab(td, mf, mo_energy=None, mo_coeff=None, mo_occ=None, singlet=True):
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
        kEmo = contract('pjkl,pi->ijkl', kEao, orbo.conj())
        kEmo = contract('ipkl,pj->ijkl', kEmo, mo)
        kEmo = contract('ijpl,pk->ijkl', kEmo, mo.conj())
        kEmo = contract('ijkp,pl->ijkl', kEmo, mo)
        kEmo = kEmo.reshape(nocc,nmo,nmo,nmo)
        if singlet:
            a += cp.einsum('iabj->iajb', kEmo[:nocc,nocc:,nocc:,:nocc])*2
            b += cp.einsum('iajb->iajb', kEmo[:nocc,nocc:,:nocc,nocc:])*2
        else:
            raise RuntimeError("There is no solvent response for singlet-triplet excitat")


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
        eri_mo = contract('pjkl,pi->ijkl', eri, orbo.conj())
        eri_mo = contract('ipkl,pj->ijkl', eri_mo, mo)
        eri_mo = contract('ijpl,pk->ijkl', eri_mo, mo.conj())
        eri_mo = contract('ijkp,pl->ijkl', eri_mo, mo)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        if singlet:
            a += cp.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc]) * 2
            a -= cp.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:]) * hyb
            b += cp.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * 2
            b -= cp.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * hyb
        else:
            a -= cp.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:]) * hyb
            b -= cp.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * hyb

    if getattr(td, 'with_solvent', None):
        pcmobj = td.with_solvent
        add_solvent_(a, b, pcmobj)

    if isinstance(mf, scf.hf.KohnShamDFT):
        grids = mf.grids
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
            eri_mo = contract('pjkl,pi->ijkl', eri, orbo.conj())
            eri_mo = contract('ipkl,pj->ijkl', eri_mo, mo)
            eri_mo = contract('ijpl,pk->ijkl', eri_mo, mo.conj())
            eri_mo = contract('ijkp,pl->ijkl', eri_mo, mo)
            eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
            k_fac = alpha - hyb
            a -= cp.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:]) * k_fac
            b -= cp.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * k_fac

        xctype = ni._xc_type(mf.xc)
        opt = getattr(ni, 'gdftopt', None)
        if opt is None:
            ni.build(mol, grids.coords)
            opt = ni.gdftopt
        _sorted_mol = opt._sorted_mol
        mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])
        orbo = opt.sort_orbitals(orbo, axis=[0])
        orbv = opt.sort_orbitals(orbv, axis=[0])
        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
                mo_coeff_mask = mo_coeff[mask]
                rho = ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask,
                                    mo_occ, mask, xctype, with_lapl=False)
                if singlet or singlet is None:
                    fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                    wfxc = fxc[0,0] * weight
                else:
                    fxc = ni.eval_xc_eff(mf.xc, cp.stack((rho, rho)) * 0.5, deriv=2, xctype=xctype)[2]
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

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
                mo_coeff_mask = mo_coeff[mask]
                rho = ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask,
                                   mo_occ, mask, xctype, with_lapl=False)
                if singlet or singlet is None:
                    fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                    wfxc = fxc * weight
                else:
                    fxc = ni.eval_xc_eff(mf.xc, cp.stack((rho, rho)) * 0.5, deriv=2, xctype=xctype)[2]
                    wfxc = (fxc[0, :, 0, :] - fxc[1, :, 0, :]) * 0.5 * weight
                orbo_mask = orbo[mask]
                orbv_mask = orbv[mask]
                rho_o = contract('xpr,pi->xri', ao, orbo_mask)
                rho_v = contract('xpr,pi->xri', ao, orbv_mask)
                rho_ov = contract('xri,ra->xria', rho_o, rho_v[0])
                rho_ov[1:4] += contract('ri,xra->xria', rho_o[0], rho_v[1:4])
                w_ov = contract('xyr,xria->yria', wfxc, rho_ov)
                iajb = contract('xria,xrjb->iajb', w_ov, rho_ov) * 2
                a += iajb
                b += iajb

        elif xctype == 'HF':
            pass

        elif xctype == 'NLC':
            pass # Processed later

        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
                mo_coeff_mask = mo_coeff[mask]
                rho = ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask,
                                   mo_occ, mask, xctype, with_lapl=False)
                if singlet or singlet is None:
                    fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                    wfxc = fxc * weight
                else:
                    fxc = ni.eval_xc_eff(mf.xc, cp.stack((rho, rho))*0.5, deriv=2, xctype=xctype)[2]
                    wfxc = (fxc[0, :, 0, :] - fxc[1, :, 0, :]) * 0.5 * weight
                orbo_mask = orbo[mask]
                orbv_mask = orbv[mask]
                rho_o = contract('xpr,pi->xri', ao, orbo_mask)
                rho_v = contract('xpr,pi->xri', ao, orbv_mask)
                rho_ov = contract('xri,ra->xria', rho_o, rho_v[0])
                rho_ov[1:4] += contract('ri,xra->xria', rho_o[0], rho_v[1:4])
                tau_ov = contract('xri,xra->ria', rho_o[1:4], rho_v[1:4]) * .5
                rho_ov = cp.vstack([rho_ov, tau_ov[cp.newaxis]])
                w_ov = contract('xyr,xria->yria', wfxc, rho_ov)
                iajb = contract('xria,xrjb->iajb', w_ov, rho_ov) * 2
                a += iajb
                b += iajb

        if mf.do_nlc():
            raise NotImplementedError('vv10 nlc not implemented in get_ab(). '
                                      'However the nlc contribution is small in TDDFT, '
                                      'so feel free to take the risk and comment out this line.')

    else:
        add_hf_(a, b)

    return a.get(), b.get()

def gen_tda_operation(td, mf, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute A x
    '''
    assert fock_ao is None
    assert isinstance(mf, scf.hf.SCF)
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
    hdiag = hdiag.ravel()
    vresp = td.gen_response(singlet=singlet, hermi=0)
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
        return v1mo.reshape(v1mo.shape[0],-1)

    return vind, hdiag


def as_scanner(td):
    if isinstance(td, lib.SinglePointScanner):
        return td

    logger.info(td, 'Set %s as a scanner', td.__class__)
    name = td.__class__.__name__ + TD_Scanner.__name_mixin__
    return lib.set_class(TD_Scanner(td), (TD_Scanner, td.__class__), name)


class TD_Scanner(lib.SinglePointScanner):
    def __init__(self, td):
        self.__dict__.update(td.__dict__)
        self._scf = td._scf.as_scanner()

    def __call__(self, mol_or_geom, **kwargs):
        assert self.device == 'gpu'
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)

        mf_scanner = self._scf
        mf_e = mf_scanner(mol)
        self.kernel(**kwargs)
        return mf_e + self.e


class TDBase(lib.StreamObject):
    to_gpu = utils.to_gpu
    device = utils.device
    to_cpu = utils.to_cpu

    conv_tol              = tdhf_cpu.TDBase.conv_tol
    nstates               = tdhf_cpu.TDBase.nstates
    singlet               = tdhf_cpu.TDBase.singlet
    lindep                = tdhf_cpu.TDBase.lindep
    level_shift           = tdhf_cpu.TDBase.level_shift
    max_cycle             = tdhf_cpu.TDBase.max_cycle
    # threshold to filter positive eigenvalues
    positive_eig_threshold = tdhf_cpu.TDBase.positive_eig_threshold
    # threshold to determine when states are considered degenerate
    deg_eia_thresh        = tdhf_cpu.TDBase.deg_eia_thresh
    # Avoid computing NLC response in TDDFT
    exclude_nlc = True

    _keys = tdhf_cpu.TDBase._keys

    __init__ = tdhf_cpu.TDBase.__init__

    nroots = tdhf_cpu.TDBase.nroots
    e_tot = tdhf_cpu.TDBase.e_tot
    dump_flags = tdhf_cpu.TDBase.dump_flags
    check_sanity = tdhf_cpu.TDBase.check_sanity
    reset = tdhf_cpu.TDBase.reset
    _finalize = tdhf_cpu.TDBase._finalize

    gen_vind = NotImplemented

    def gen_response(self, singlet=True, hermi=0):
        '''Generate function to compute A x'''
        if (self.exclude_nlc and
            isinstance(self._scf, scf.hf.KohnShamDFT) and self._scf.do_nlc()):
            logger.warn(self, 'NLC functional found in the DFT object. Its contribution is '
                        'not included in the TDDFT response function.')
        return self._scf.gen_response(singlet=singlet, hermi=hermi,
                                      with_nlc=not self.exclude_nlc)

    def get_ab(self, mf=None):
        if mf is None:
            mf = self._scf
        return get_ab(self, mf, singlet=self.singlet)

    def get_precond(self, hdiag):
        threshold_t=1.0e-4
        def precond(x, e, *args):
            n_states = x.shape[0]
            diagd = cp.repeat(hdiag.reshape(1,-1), n_states, axis=0)
            e = e.reshape(-1,1)
            diagd = hdiag - (e-self.level_shift)
            diagd = cp.where(abs(diagd) < threshold_t, cp.sign(diagd)*threshold_t, diagd)
            a_size = x.shape[1]//2
            diagd[:,a_size:] = diagd[:,a_size:]*(-1)
            return x/diagd
        return precond

    def nuc_grad_method(self):
        if getattr(self._scf, 'with_df', None):
            from gpu4pyscf.df.grad import tdrhf
            return tdrhf.Gradients(self)
        else:
            from gpu4pyscf.grad import tdrhf
            return tdrhf.Gradients(self)

    def nac_method(self): 
        if getattr(self._scf, 'with_df', None):
            from gpu4pyscf.df.nac import tdrhf
            return tdrhf.NAC(self)
        else:
            from gpu4pyscf.nac import tdrhf
            return tdrhf.NAC(self)

    as_scanner = as_scanner

    oscillator_strength = tdhf_cpu.oscillator_strength
    transition_dipole              = tdhf_cpu.transition_dipole
    transition_quadrupole          = tdhf_cpu.transition_quadrupole
    transition_octupole            = tdhf_cpu.transition_octupole
    transition_velocity_dipole     = tdhf_cpu.transition_velocity_dipole
    transition_velocity_quadrupole = tdhf_cpu.transition_velocity_quadrupole
    transition_velocity_octupole   = tdhf_cpu.transition_velocity_octupole
    transition_magnetic_dipole     = tdhf_cpu.transition_magnetic_dipole
    transition_magnetic_quadrupole = tdhf_cpu.transition_magnetic_quadrupole

    def analyze(self, verbose=None):
        self.to_cpu().analyze(verbose)
        return self

    def get_nto(self, state=1, threshold=OUTPUT_THRESHOLD, verbose=None):
        '''
        Natural transition orbital analysis.

        Returns:
            A list (weights, NTOs).  NTOs are natural orbitals represented in AO
            basis. The first N_occ NTOs are occupied NTOs and the rest are virtual
            NTOs. weights and NTOs are all stored in nparray
        '''
        return self.to_cpu().get_nto(state, threshold, verbose)

    # needed by transition dipoles
    def _contract_multipole(tdobj, ints, hermi=True, xy=None):
        '''ints is the integral tensor of a spin-independent operator'''
        if xy is None: xy = tdobj.xy
        nstates = len(xy)
        pol_shape = ints.shape[:-2]
        nao = ints.shape[-1]

        if not tdobj.singlet:
            return np.zeros((nstates,) + pol_shape)

        mo_coeff = tdobj._scf.mo_coeff
        mo_occ = tdobj._scf.mo_occ
        orbo = mo_coeff[:,mo_occ==2]
        orbv = mo_coeff[:,mo_occ==0]
        if isinstance(orbo, cp.ndarray):
            orbo = orbo.get()
            orbv = orbv.get()

        #Incompatible to old np version
        #ints = np.einsum('...pq,pi,qj->...ij', ints, orbo.conj(), orbv)
        ints = lib.einsum('xpq,pi,qj->xij', ints.reshape(-1,nao,nao), orbo.conj(), orbv)
        pol = np.array([np.einsum('xij,ij->x', ints, x) * 2 for x,y in xy])
        if isinstance(xy[0][1], np.ndarray):
            if hermi:
                pol += [np.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
            else:  # anti-Hermitian
                pol -= [np.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
        pol = pol.reshape((nstates,)+pol_shape)
        return pol

class TDA(TDBase):
    __doc__ = tdhf_cpu.TDA.__doc__

    def get_precond(self, hdiag):
        threshold_t=1.0e-4
        def precond(x, e, *args):
            n_states = x.shape[0]
            diagd = cp.repeat(hdiag.reshape(1,-1), n_states, axis=0)
            e = e.reshape(-1,1)
            diagd = hdiag - (e-self.level_shift)
            diagd = cp.where(abs(diagd) < threshold_t, cp.sign(diagd)*threshold_t, diagd)
            return x/diagd
        return precond

    def gen_vind(self, mf=None):
        '''Generate function to compute Ax'''
        if mf is None:
            mf = self._scf
        return gen_tda_operation(self, mf, singlet=self.singlet)

    def init_guess(self, mf=None, nstates=None, wfnsym=None, return_symmetry=False):
        '''
        Generate initial guess for TDA

        Kwargs:
            nstates : int
                The number of initial guess vectors.
        '''
        if mf is None: mf = self._scf
        if nstates is None: nstates = self.nstates
        assert wfnsym is None
        assert not return_symmetry

        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        if isinstance(mo_energy, cp.ndarray):
            mo_energy = mo_energy.get()
            mo_occ = mo_occ.get()
        occidx = mo_occ == 2
        viridx = mo_occ == 0
        e_ia = (mo_energy[viridx] - mo_energy[occidx,None]).ravel()
        nov = e_ia.size
        nstates = min(nstates, nov)

        # Find the nstates-th lowest energy gap
        e_threshold = float(np.partition(e_ia, nstates-1)[nstates-1])
        e_threshold += self.deg_eia_thresh

        idx = np.where(e_ia <= e_threshold)[0]
        x0 = np.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations

        return x0

    def kernel(self, x0=None, nstates=None):
        '''TDA diagonalization solver
        '''
        log = logger.new_logger(self)
        cpu0 = log.init_timer()
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        mol = self.mol

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

        nocc = mol.nelectron // 2
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        # 1/sqrt(2) because self.x is for alpha excitation and 2(X^+*X) = 1
        self.xy = [(xi.reshape(nocc,nvir) * .5**.5, 0) for xi in x1]
        log.timer('TDA', *cpu0)
        self._finalize()
        return self.e, self.xy

CIS = TDA


def gen_tdhf_operation(td, mf, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute

    [ A   B ][X]
    [-B* -A*][Y]
    '''
    assert fock_ao is None
    assert isinstance(mf, scf.hf.SCF)
    mo_coeff = mf.mo_coeff
    assert mo_coeff.dtype == cp.float64
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidx = mo_occ == 2
    viridx = mo_occ == 0
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    e_ia = hdiag = mo_energy[viridx] - mo_energy[occidx,None]
    vresp = td.gen_response(singlet=singlet, hermi=0)
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
                          -v1_bot.reshape(nz,nocc*nvir)))

    hdiag = cp.hstack([hdiag.ravel(), -hdiag.ravel()])
    return vind, hdiag


class TDHF(TDBase):
    __doc__ = tdhf_cpu.TDHF.__doc__

    @lib.with_doc(gen_tdhf_operation.__doc__)
    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        return gen_tdhf_operation(self, mf, singlet=self.singlet)

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
        mol = self.mol

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)
        pickeig = None

        # handle single kpt PBC SCF
        if getattr(self._scf, 'kpt', None) is not None:
            from pyscf.pbc.lib.kpts_helper import gamma_point
            assert gamma_point(self._scf.kpt)

        x0sym = None
        if x0 is None:
            x0 = self.init_guess()

        self.converged, self.e, x1 = real_eig(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        nocc = mol.nelectron // 2
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        def norm_xy(z):
            x, y = z.reshape(2, -1)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            if norm < 0:
                log.warn('TDDFT amplitudes |X| smaller than |Y|')
            norm = abs(.5/norm)**.5  # normalize to 0.5 for alpha spin
            return x.reshape(nocc,nvir)*norm, y.reshape(nocc,nvir)*norm
        self.xy = [norm_xy(z) for z in x1]

        log.timer('TDHF/TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

TDRHF = TDHF

scf.hf.RHF.TDA = lib.class_as_method(TDA)
scf.hf.RHF.TDHF = lib.class_as_method(TDHF)
