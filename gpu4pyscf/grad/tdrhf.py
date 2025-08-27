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


from functools import reduce
import cupy as cp
import numpy as np
from pyscf import lib, gto
from gpu4pyscf.lib import logger
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import cphf
from pyscf import __config__
from gpu4pyscf.lib import utils
from gpu4pyscf import tdscf
from scipy.optimize import linear_sum_assignment

def match_and_reorder_mos(s12_ao, mo_coeff_b, mo_coeff, threshold=0.4):
    if mo_coeff_b.shape != mo_coeff.shape:
        raise ValueError("Mo coeff b and mo coeff must have the same shape.")
    if s12_ao.shape[0] != s12_ao.shape[1] or s12_ao.shape[0] != mo_coeff_b.shape[0]:
        raise ValueError("S12 ao must be a square matrix with the same shape as mo coeff b.")
    mo_overlap_matrix = mo_coeff_b.T @ s12_ao @ mo_coeff
    abs_mo_overlap = cp.abs(mo_overlap_matrix)
    cost_matrix = -abs_mo_overlap
    below_threshold_mask = abs_mo_overlap < threshold
    infinity_cost = mo_coeff_b.shape[1] + 1
    cost_matrix[below_threshold_mask] = infinity_cost
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix.get())

    matching_indices = col_ind
    
    mo2_reordered = mo_coeff[:, matching_indices]

    final_chosen_overlaps = abs_mo_overlap[row_ind, col_ind]
    invalid_matches_mask = final_chosen_overlaps < threshold
    
    if cp.any(invalid_matches_mask):
        num_invalid = cp.sum(invalid_matches_mask)
        print(
            f"{num_invalid} orbital below threshold {threshold}."
            "This may indicate significant changes in the properties of these orbitals between the two structures."
        )
        invalid_indices = cp.where(invalid_matches_mask)[0]
        for idx in invalid_indices:
            print(f"Warning: reference coeff #{idx}'s best match is {final_chosen_overlaps[idx]:.4f} (below threshold {threshold})")
    s_mo_new = mo_coeff_b.T @ s12_ao @ mo2_reordered
    sign_array = cp.ones(s_mo_new.shape[-1])
    for i in range(s_mo_new.shape[-1]):
        if s_mo_new[i,i] < 0.0:
            mo2_reordered[:,i] *= -1
            sign_array[i] = -1
    return mo2_reordered, matching_indices, sign_array


def grad_elec(td_grad, x_y, singlet=True, atmlst=None, verbose=logger.INFO):
    """
    Electronic part of TDA, TDHF nuclear gradients

    Args:
        td_grad : grad.tdrhf.Gradients or grad.tdrks.Gradients object.

        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    """
    if singlet is None:
        singlet = True
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.init_timer(td_grad)
    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nmo = mo_coeff.shape[1]
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    x, y = x_y
    x = cp.asarray(x)
    y = cp.asarray(y)
    xpy = (x + y).reshape(nocc, nvir).T
    xmy = (x - y).reshape(nocc, nvir).T
    orbv = mo_coeff[:, nocc:]
    orbo = mo_coeff[:, :nocc]
    dvv = contract("ai,bi->ab", xpy, xpy) + contract("ai,bi->ab", xmy, xmy)  # 2 T_{ab}
    doo = -contract("ai,aj->ij", xpy, xpy) - contract("ai,aj->ij", xmy, xmy)  # 2 T_{ij}
    dmxpy = reduce(cp.dot, (orbv, xpy, orbo.T))  # (X+Y) in ao basis
    dmxmy = reduce(cp.dot, (orbv, xmy, orbo.T))  # (X-Y) in ao basis
    dmzoo = reduce(cp.dot, (orbo, doo, orbo.T))  # T_{ij}*2 in ao basis
    dmzoo += reduce(cp.dot, (orbv, dvv, orbv.T))  # T_{ij}*2 + T_{ab}*2 in ao basis
    td_grad.dmxpy = dmxpy

    vj0, vk0 = mf.get_jk(mol, dmzoo, hermi=0)
    vj1, vk1 = mf.get_jk(mol, dmxpy + dmxpy.T, hermi=0)
    vj2, vk2 = mf.get_jk(mol, dmxmy - dmxmy.T, hermi=0)
    if not isinstance(vj0, cp.ndarray):
        vj0 = cp.asarray(vj0)
    if not isinstance(vk0, cp.ndarray):
        vk0 = cp.asarray(vk0)
    if not isinstance(vj1, cp.ndarray):
        vj1 = cp.asarray(vj1)
    if not isinstance(vk1, cp.ndarray):
        vk1 = cp.asarray(vk1)
    if not isinstance(vj2, cp.ndarray):
        vj2 = cp.asarray(vj2)
    if not isinstance(vk2, cp.ndarray):
        vk2 = cp.asarray(vk2)
    vj = cp.stack((vj0, vj1, vj2))
    vk = cp.stack((vk0, vk1, vk2))
    veff0doo = vj[0] * 2 - vk[0]  # 2 for alpha and beta
    veff0doo += td_grad.solvent_response(dmzoo)
    wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
    if singlet:
        veff = vj[1] * 2 - vk[1]
    else:
        veff = -vk[1]
    veff += td_grad.solvent_response(dmxpy + dmxpy.T)
    veff0mop = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= contract("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2  # 2 for dm + dm.T
    wvo += contract("ac,ai->ci", veff0mop[nocc:, nocc:], xpy) * 2
    veff = -vk[2]
    veff0mom = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= contract("ki,ai->ak", veff0mom[:nocc, :nocc], xmy) * 2
    wvo += contract("ac,ai->ci", veff0mom[nocc:, nocc:], xmy) * 2

    # set singlet=None, generate function for CPHF type response kernel
    vresp = td_grad.base.gen_response(singlet=None, hermi=1)

    def fvind(x):  # For singlet, closed shell ground state
        dm = reduce(cp.dot, (orbv, x.reshape(nvir, nocc) * 2, orbo.T))  # 2 for double occupancy
        v1ao = vresp(dm + dm.T)  # for the upused 2
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()

    z1 = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        wvo,
        max_cycle=td_grad.cphf_max_cycle,
        tol=td_grad.cphf_conv_tol)[0]
    z1 = z1.reshape(nvir, nocc)
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    z1ao = reduce(cp.dot, (orbv, z1, orbo.T))
    veff = vresp(z1ao + z1ao.T)

    im0 = cp.zeros((nmo, nmo))
    # in the following, all should be doubled, due to double occupancy
    # and 0.5 for i<=j and a<= b
    # but this is reduced.
    # H_{ij}^+[T] + H_{ij}^+[Z] #
    im0[:nocc, :nocc] = reduce(cp.dot, (orbo.T, veff0doo + veff, orbo))
    # H_{ij}^+[T] + H_{ij}^+[Z] + sum_{a} (X+Y)_{aj}H_{ai}^+[(X+Y)]
    im0[:nocc, :nocc] += contract("ak,ai->ki", veff0mop[nocc:, :nocc], xpy)
    # H_{ij}^+[T] + H_{ij}^+[Z] + sum_{a} (X+Y)_{aj}H_{ai}^+[(X+Y)]
    #  + sum_{a} (X-Y)_{aj}H_{ai}^-[(X-Y)]
    im0[:nocc, :nocc] += contract("ak,ai->ki", veff0mom[nocc:, :nocc], xmy)
    #  sum_{i} (X+Y)_{ci}H_{ai}^+[(X+Y)]
    im0[nocc:, nocc:] = contract("ci,ai->ac", veff0mop[nocc:, :nocc], xpy)
    #  sum_{i} (X+Y)_{ci}H_{ai}^+[(X+Y)] + sum_{i} (X-Y)_{cj}H_{ai}^-[(X-Y)]
    im0[nocc:, nocc:] += contract("ci,ai->ac", veff0mom[nocc:, :nocc], xmy)
    #  sum_{i} (X+Y)_{ki}H_{ai}^+[(X+Y)] * 2
    im0[nocc:, :nocc] = contract("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2
    #  sum_{i} (X+Y)_{ki}H_{ai}^+[(X+Y)] + sum_{i} (X-Y)_{ki}H_{ai}^-[(X-Y)] * 2
    im0[nocc:, :nocc] += contract("ki,ai->ak", veff0mom[:nocc, :nocc], xmy) * 2

    zeta = (mo_energy[:,cp.newaxis] + mo_energy)*0.5
    zeta[nocc:, :nocc] = mo_energy[:nocc]
    zeta[:nocc, nocc:] = mo_energy[nocc:]
    dm1 = cp.zeros((nmo, nmo))
    dm1[:nocc, :nocc] = doo
    dm1[nocc:, nocc:] = dvv
    dm1[nocc:, :nocc] = z1
    dm1[:nocc, :nocc] += cp.eye(nocc) * 2  # for ground state
    im0 = reduce(cp.dot, (mo_coeff, im0 + zeta * dm1, mo_coeff.T))

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td_grad.base._scf.nuc_grad_method()
    s1 = mf_grad.get_ovlp(mol)

    dmz1doo = z1ao + dmzoo  # P
    td_grad.dmz1doo = dmz1doo
    oo0 = reduce(cp.dot, (orbo, orbo.T))  # D

    if atmlst is None:
        atmlst = range(mol.natm)

    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_ground = contract("xij,ij->xi", h1, oo0 * 2)
    dh_td = contract("xij,ij->xi", h1, (dmz1doo + dmz1doo.T) * 0.5)
    ds = contract("xij,ij->xi", s1, (im0 + im0.T) * 0.5)

    dh1e_ground = int3c2e.get_dh1e(mol, oo0 * 2)  # 1/r like terms
    if mol.has_ecp():
        dh1e_ground += rhf_grad.get_dh1e_ecp(mol, oo0 * 2)  # 1/r like terms
    dh1e_td = int3c2e.get_dh1e(mol, (dmz1doo + dmz1doo.T) * 0.5)  # 1/r like terms
    if mol.has_ecp():
        dh1e_td += rhf_grad.get_dh1e_ecp(mol, (dmz1doo + dmz1doo.T) * 0.5)  # 1/r like terms
    extra_force = cp.zeros((len(atmlst), 3))

    dvhf_all = 0
    # this term contributes the ground state contribution.
    dvhf = td_grad.get_veff(mol, (dmz1doo + dmz1doo.T) * 0.5 + oo0 * 2) 
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all += dvhf
    # this term will remove the unused-part from PP density.
    dvhf = td_grad.get_veff(mol, (dmz1doo + dmz1doo.T) * 0.5)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= cp.asarray(mf_grad.extra_force(ia, locals()))
    dvhf_all -= dvhf
    if singlet:
        j_factor=1.0
        k_factor=1.0
    else:
        j_factor=0.0
        k_factor=1.0
    dvhf = td_grad.get_veff(mol, (dmxpy + dmxpy.T), j_factor, k_factor)
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals())*2)
    dvhf_all += dvhf*2
    dvhf = td_grad.get_veff(mol, (dmxmy - dmxmy.T), 0.0, k_factor, hermi=2)
    for k, ia in enumerate(atmlst):
        extra_force[k] += cp.asarray(mf_grad.extra_force(ia, locals())*2)
    dvhf_all += dvhf*2
    time1 = log.timer('2e AO integral derivatives', *time1)

    delec = 2.0 * (dh_ground + dh_td - ds)
    aoslices = mol.aoslice_by_atom()
    delec = cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])
    de = 2.0 * dvhf_all + dh1e_ground + dh1e_td + delec + extra_force

    log.timer('TDHF nuclear gradients', *time0)
    return de.get()


def as_scanner(td_grad, state=1):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    The returned solver is a function. This function requires one argument
    "mol" as input and returns energy and first order nuclear derivatives.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    nuc-grad object and SCF object (DIIS, conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.
    '''
    if isinstance(td_grad, lib.GradScanner):
        return td_grad

    if state == 0:
        return td_grad.base._scf.nuc_grad_method().as_scanner()

    logger.info(td_grad, 'Create scanner for %s', td_grad.__class__)
    name = td_grad.__class__.__name__ + TDSCF_GradScanner.__name_mixin__
    return lib.set_class(TDSCF_GradScanner(td_grad, state),
                         (TDSCF_GradScanner, td_grad.__class__), name)


def check_flip(mol0, mol1, mo_coeff0, mo_coeff1, xy0, xy1, state, nocc):

    nao = mol0.nao
    nvir = nao - nocc

    s = gto.intor_cross('int1e_ovlp', mol0, mol1)
    s = cp.asarray(s)
    mo1_reordered, matching_indices, sign_array = \
        match_and_reorder_mos(s, mo_coeff0, mo_coeff1, threshold=0.4)
    mo1_reordered = cp.asarray(mo1_reordered)
    x0 = xy0[state-1][0]
    nstates = len(xy0)
    s_state_list = []
    for istate in range(nstates):
        x1 = xy1[istate][0]
        s_state = 0
        for i in range(nocc):
            for a in range(nvir):
                for j in range(nocc):
                    for b in range(nvir):
                        mo_coeff0_tmp = mo_coeff0[:,:nocc]*1.0
                        mo_coeff1_tmp = mo1_reordered[:,:nocc]*1.0
                        mo_coeff0_tmp[:,i] = mo_coeff0[:,a+nocc]
                        mo_coeff1_tmp[:,j] = mo1_reordered[:,b+nocc]
                        s_mo = mo_coeff0_tmp.T @ s @ mo_coeff1_tmp
                        s_state += cp.linalg.det(s_mo)\
                            *x0[i,a]*x1[j,b]*2
        s_state_list.append(float(s_state))
    s_state_list = np.asarray(s_state_list)
    return np.argmax(np.abs(s_state_list))+1, s_state_list


class TDSCF_GradScanner(lib.GradScanner):
    _keys = {'e_tot'}

    def __init__(self, g, state):
        lib.GradScanner.__init__(self, g)
        if state is not None:
            self.state = state

    def __call__(self, mol_or_geom, state=None, **kwargs):
        mol0 = self.mol.copy()
        if isinstance(mol_or_geom, gto.MoleBase):
            assert mol_or_geom.__class__ == gto.Mole
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
        self.reset(mol)
        mo_coeff0 = self.base._scf.mo_coeff
        xy0 = self.base.xy

        if state is None:
            state = self.state
        else:
            self.state = state

        td_scanner = self.base
        assert td_scanner.device == 'gpu'
        assert self.device == 'gpu'
        td_scanner(mol)
        mo_coeff1 = self.base._scf.mo_coeff
        xy1 = self.base.xy
        mo_occ = cp.asarray(self.base._scf.mo_occ)
        nocc = int((mo_occ > 0).sum())

        flip_state, s_state_list = check_flip(mol0, mol, mo_coeff0, mo_coeff1, xy0, xy1, state, nocc)
        if flip_state != state:
            logger.warn(self, 'Root %d is flipped to %d', state, flip_state)
            logger.warn(self, f'S_state_list = {s_state_list}')
            self.state = flip_state
        de = self.kernel(state=flip_state, **kwargs)
        e_tot = self.e_tot[flip_state-1]
        return e_tot, de

    @property
    def converged(self):
        td_scanner = self.base
        return all((td_scanner._scf.converged,
                    td_scanner.converged[self.state-1]))


class Gradients(rhf_grad.GradientsBase):

    cphf_max_cycle = getattr(__config__, "grad_tdrhf_Gradients_cphf_max_cycle", 50)
    cphf_conv_tol = getattr(__config__, "grad_tdrhf_Gradients_cphf_conv_tol", 1e-8)

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    _keys = {
        "cphf_max_cycle",
        "cphf_conv_tol",
        "mol",
        "base",
        "chkfile",
        "state",
        "atmlst",
        "de",
        "dmz1doo",
        "dmxpy"
    }

    def __init__(self, td):
        super().__init__(td)
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.base = td
        self.chkfile = td.chkfile
        self.state = 1  # of which the gradients to be computed.
        self.atmlst = None
        self.de = None
        self.dmz1doo = None
        self.dmxpy = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("\n")
        log.info(
            "******** LR %s gradients for %s ********",
            self.base.__class__,
            self.base._scf.__class__,
        )
        log.info("cphf_conv_tol = %g", self.cphf_conv_tol)
        log.info("cphf_max_cycle = %d", self.cphf_max_cycle)
        log.info("chkfile = %s", self.chkfile)
        log.info("State ID = %d", self.state)
        log.info("\n")
        return self

    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet, atmlst=None, verbose=logger.INFO):
        return grad_elec(self, xy, singlet, atmlst, verbose)

    def kernel(self, xy=None, state=None, singlet=None, atmlst=None):
        """
        Args:
            state : int
                Excited state ID.  state = 1 means the first excited state.
        """
        if xy is None:
            if state is None:
                state = self.state
            else:
                self.state = state

            if state == 0:
                logger.warn(
                    self,
                    "state=0 found in the input. Gradients of ground state is computed.",
                )
                return self.base._scf.nuc_grad_method().kernel(atmlst=atmlst)

            xy = self.base.xy[state - 1]

        if singlet is None:
            singlet = self.base.singlet
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(xy, singlet, atmlst, verbose=self.verbose)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de

    # Calling the underlying SCF nuclear gradients because it may be modified
    # by external modules (e.g. QM/MM, solvent)
    def grad_nuc(self, mol=None, atmlst=None):
        mf_grad = self.base._scf.nuc_grad_method()
        return mf_grad.grad_nuc(mol, atmlst)

    def get_veff(self, mol=None, dm=None, j_factor=1.0, k_factor=1.0, omega=0.0, hermi=0, verbose=None):
        """
        Computes the first-order derivatives of the energy contributions from
        Veff per atom.

        NOTE: This function is incompatible to the one implemented in PySCF CPU version.
        In the CPU version, get_veff returns the first order derivatives of Veff matrix.
        """
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.base.make_rdm1()
        if omega == 0.0:
            vhfopt = self.base._scf._opt_gpu.get(None, None)
            return rhf_grad._jk_energy_per_atom(mol, dm, vhfopt, j_factor=j_factor, k_factor=k_factor, verbose=verbose)
        else:
            vhfopt = self.base._scf._opt_gpu.get(omega, None)
            with mol.with_range_coulomb(omega):
                return rhf_grad._jk_energy_per_atom(
                    mol, dm, vhfopt, j_factor=j_factor, k_factor=k_factor, verbose=verbose)

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(
                self,
                "--------- %s gradients for state %d ----------",
                self.base.__class__.__name__,
                self.state,
            )
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, "----------------------------------------------")

    def solvent_response(self, dm):
        return 0.0

    as_scanner = as_scanner

    to_gpu = lib.to_gpu


Grad = Gradients

tdscf.rhf.TDA.Gradients = tdscf.rhf.TDHF.Gradients = lib.class_as_method(Gradients)
