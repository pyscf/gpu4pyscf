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


import math
import ctypes
from functools import reduce
from collections import Counter
import numpy as np
import cupy as cp
from pyscf import lib, gto
from gpu4pyscf.lib import logger
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import contract, condense
from gpu4pyscf.scf import cphf
from pyscf import __config__
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.lib import utils
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.scf.jk import (
    LMAX, QUEUE_DEPTH, SHM_SIZE, libvhf_rys, _VHFOpt, _make_tril_pair_mappings)
from gpu4pyscf.grad.rhf import _ejk_quartets_scheme

DD_CACHE_MAX = np.array([
    256,
    2592,
    10368,
    40000,
    101250,
]) * (SHM_SIZE//48000)

def grad_elec(td_grad, x_y, singlet=True, atmlst=None, verbose=logger.INFO,
              with_solvent=False):
    """
    Electronic part of TDA, TDHF nuclear gradients

    Args:
        td_grad : grad.tdrhf.Gradients or grad.tdrks.Gradients object.

        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.

    Kwargs:
        with_solvent :
            Include the response of solvent in the gradients of the electronic
            energy.
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
    if with_solvent:
        td_grad._dmxpy = dmxpy

    vj0, vk0 = mf.get_jk(mol, dmzoo, hermi=0)
    vj1, vk1 = mf.get_jk(mol, dmxpy + dmxpy.T, hermi=0)
    vj2, vk2 = mf.get_jk(mol, dmxmy - dmxmy.T, hermi=0)
    vj0 = cp.asarray(vj0)
    vk0 = cp.asarray(vk0)
    vj1 = cp.asarray(vj1)
    vk1 = cp.asarray(vk1)
    vj2 = cp.asarray(vj2)
    vk2 = cp.asarray(vk2)
    vj = cp.stack((vj0, vj1, vj2))
    vk = cp.stack((vk0, vk1, vk2))
    veff0doo = vj[0] * 2 - vk[0]  # 2 for alpha and beta
    if with_solvent:
        veff0doo += td_grad.solvent_response(dmzoo)
    wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
    if singlet:
        veff = vj[1] * 2 - vk[1]
    else:
        veff = -vk[1]
    if with_solvent:
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
    if with_solvent:
        td_grad._dmz1doo = dmz1doo
    oo0 = reduce(cp.dot, (orbo, orbo.T))  # D
    oo0 *= 2 # *2 for double occupancy

    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_ground = rhf_grad.contract_h1e_dm(mol, h1, oo0, hermi=1)
    dh_td = rhf_grad.contract_h1e_dm(mol, h1, dmz1doo, hermi=0)
    ds = rhf_grad.contract_h1e_dm(mol, s1, im0, hermi=0)

    dh1e_ground = int3c2e.get_dh1e(mol, oo0)  # 1/r like terms
    if len(mol._ecpbas) > 0:
        dh1e_ground += rhf_grad.get_dh1e_ecp(mol, oo0)  # 1/r like terms
    dh1e_td = int3c2e.get_dh1e(mol, (dmz1doo + dmz1doo.T) * 0.5)  # 1/r like terms
    if len(mol._ecpbas) > 0:
        dh1e_td += rhf_grad.get_dh1e_ecp(mol, (dmz1doo + dmz1doo.T) * 0.5)  # 1/r like terms

    if mol._pseudo:
        raise NotImplementedError("Pseudopotential gradient not supported for molecular system yet")

    if hasattr(td_grad, 'jk_energy_per_atom'):
        # DF-TDRHF can handle multiple dms more efficiently.
        dms = cp.array([
            (dmz1doo + dmz1doo.T) * 0.5 + oo0, # ground state contribution.
            (dmz1doo + dmz1doo.T) * 0.5, # remove the unused-part from PP density.
            dmxpy + dmxpy.T,
            dmxmy - dmxmy.T])
        j_factor = [1, -1, 2,  0]
        k_factor = [1, -1, 2, -2]
        if not singlet:
            j_factor[2] = 0
        dvhf = td_grad.jk_energy_per_atom(dms, j_factor, k_factor) * .5
    else:
        # this term contributes the ground state contribution.
        dvhf = td_grad.get_veff(mol, (dmz1doo + dmz1doo.T) * 0.5 + oo0, hermi=1)
        # this term will remove the unused-part from PP density.
        dvhf -= td_grad.get_veff(mol, (dmz1doo + dmz1doo.T) * 0.5, hermi=1)
        if singlet:
            j_factor=1.0
            k_factor=1.0
        else:
            j_factor=0.0
            k_factor=1.0
        dvhf += 2 * td_grad.get_veff(mol, (dmxpy + dmxpy.T), j_factor, k_factor, hermi=1)
        dvhf -= 2 * td_grad.get_veff(mol, (dmxmy - dmxmy.T), 0.0, k_factor, hermi=2)
    time1 = log.timer('2e AO integral derivatives', *time1)

    de = dh_ground + dh_td - ds + 2 * dvhf
    de += cp.asnumpy(dh1e_ground + dh1e_td)
    if atmlst is not None:
        de = de[atmlst]
    log.timer('TDHF nuclear gradients', *time0)
    return de


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

def _jk_energies_per_atom(mol, dm_pairs, vhfopt=None,
                          j_factor=None, k_factor=None, verbose=None):
    '''
    Computes first-order derivatives of J/K contributions for multiple density
    matrices, analogous to _jk_energy_per_atom.

    This function can evaluatie multiple sets of energy derivatives in a
    single call. Additionally, for each set, the two density matrices for the
    four-index Coulomb integrals can be different.

    This function only supports closed shell (RHF-type) density matrices.

    Args:
        dm_pairs:
            A list of density-matrix-pairs [[dm, dm], [dm, dm], ...].
            Each element corresponds to one set of energy derivative.
        j_factor:
            A list of factors for Coulomb (J) term
        k_factor:
            A list of factors for Coulomb (K) term
    '''
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()
    if vhfopt is None:
        vhfopt = _VHFOpt(mol, tile=1).build()
    assert vhfopt.tile == 1

    mol = vhfopt.sorted_mol
    nao_orig = vhfopt.mol.nao

    n_dm = len(dm_pairs)
    dm1 = cp.empty((n_dm, nao_orig, nao_orig))
    dm2 = cp.empty((n_dm, nao_orig, nao_orig))
    for i, dm1_dm2 in enumerate(dm_pairs):
        if dm1_dm2.ndim == 2:
            dm1[i] = dm2[i] = dm1_dm2
        else:
            assert dm1_dm2.shape == (2, nao_orig, nao_orig)
            dm1[i] = dm1_dm2[0]
            dm2[i] = dm1_dm2[1]
    dm1 = vhfopt.apply_coeff_C_mat_CT(dm1)
    dm2 = vhfopt.apply_coeff_C_mat_CT(dm2)
    nao = dm1.shape[-1]

    assert j_factor is None or len(j_factor) == n_dm
    assert k_factor is None or len(k_factor) == n_dm
    if j_factor is None:
        j_factor = np.zeros(n_dm)
    if k_factor is None:
        k_factor = np.zeros(n_dm)
    do_j = 1 if any(x != 0 for x in j_factor) else 0
    do_k = 1 if any(x != 0 for x in k_factor) else 0

    ao_loc = mol.ao_loc
    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    lmax = uniq_l.max()
    l_ctr_bas_loc = vhfopt.l_ctr_offsets
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    assert uniq_l.max() <= LMAX

    n_groups = len(uniq_l_ctr)
    tasks = ((i, j, k, l)
             for i in range(n_groups)
             for j in range(i+1)
             for k in range(i+1)
             for l in range(k+1))

    def proc():
        device_id = cp.cuda.device.get_device_id()
        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()

        timing_counter = Counter()
        kern_counts = 0
        kern = libvhf_rys.RYS_per_atom_jk_ip1_multidm

        _dm1 = cp.asarray(dm1, order='C')
        _dm2 = cp.asarray(dm2, order='C')
        s_ptr = lib.c_null_ptr()
        if mol.omega < 0:
            s_ptr = ctypes.cast(vhfopt.s_estimator.data.ptr, ctypes.c_void_p)

        ejk = cp.zeros((n_dm, mol.natm, 3))
        _j_factor = cp.asarray(j_factor, dtype=np.float64)
        _k_factor = cp.asarray(k_factor, dtype=np.float64)

        dms = cp.vstack([_dm1, _dm2])
        dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
        dms = None
        q_cond = cp.asarray(vhfopt.q_cond)
        log_max_dm = float(dm_cond.max())
        log_cutoff = math.log(vhfopt.direct_scf_tol)
        pair_mappings = _make_tril_pair_mappings(
            l_ctr_bas_loc, q_cond, log_cutoff-log_max_dm, tile=6)
        rys_envs = vhfopt.rys_envs
        workers = gpu_specs['multiProcessorCount']
        # An additional integer to count for the proccessed pair_ijs
        pool = cp.empty(workers*QUEUE_DEPTH+1, dtype=np.int32)
        dd_cache_maxsize = DD_CACHE_MAX[lmax] * n_dm
        dd_pool = cp.empty((workers, dd_cache_maxsize), dtype=np.float64)
        t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *cput0)

        for i, j, k, l in tasks:
            shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
            llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
            pair_ij_mapping = pair_mappings[i,j]
            pair_kl_mapping = pair_mappings[k,l]
            npairs_ij = pair_ij_mapping.size
            npairs_kl = pair_kl_mapping.size
            if npairs_ij == 0 or npairs_kl == 0:
                continue
            scheme = _ejk_quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
            for pair_kl0, pair_kl1 in lib.prange(0, npairs_kl, QUEUE_DEPTH):
                _pair_kl_mapping = pair_kl_mapping[pair_kl0:]
                _npairs_kl = pair_kl1 - pair_kl0
                err = kern(
                    ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                    ctypes.c_double(do_j), ctypes.c_double(do_k),
                    ctypes.cast(_j_factor.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_k_factor.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_dm1.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_dm2.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao),
                    rys_envs, (ctypes.c_int*2)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(_npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
                    s_ptr,
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dd_pool.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(dd_cache_maxsize),
                    mol._atm.ctypes, ctypes.c_int(mol.natm),
                    mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
                if err != 0:
                    raise RuntimeError(f'RYS_per_atom_jk_ip1 kernel for {llll} failed')
            if log.verbose >= logger.DEBUG1:
                ntasks = npairs_ij * npairs_kl
                msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                t1, t1p = log.timer_debug1(msg, *t1), t1
                timing_counter[llll] += t1[1] - t1p[1]
                kern_counts += 1
        return ejk, kern_counts, timing_counter

    results = multi_gpu.run(proc, non_blocking=True)

    kern_counts = 0
    timing_collection = Counter()
    ejk_dist = []
    for ejk, counts, counter in results:
        kern_counts += counts
        timing_collection += counter
        ejk_dist.append(ejk)
    ejk = multi_gpu.array_reduce(ejk_dist, inplace=True)

    if log.verbose >= logger.DEBUG1:
        log.debug1('kernel launches %d', kern_counts)
        for llll, t in timing_collection.items():
            log.debug1('%s wall time %.2f', llll, t)

    log.timer_debug1('grad jk energy', *cput0)
    return ejk.get()


class TDSCF_GradScanner(lib.GradScanner):
    _keys = {'e_tot'}

    def __init__(self, g, state):
        lib.GradScanner.__init__(self, g)
        if state is not None:
            self.state = state

    def __call__(self, mol_or_geom, state=None, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            assert mol_or_geom.__class__ == gto.Mole
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
        self.reset(mol)

        if state is None:
            state = self.state
        else:
            self.state = state

        td_scanner = self.base
        assert td_scanner.device == 'gpu'
        assert self.device == 'gpu'
        td_scanner(mol)
        # TODO: Check root flip.  Maybe avoid the initial guess in TDHF otherwise
        # large error may be found in the excited states amplitudes
        de = self.kernel(state=state, **kwargs)
        e_tot = self.e_tot[state-1]
        return e_tot, de

    @property
    def converged(self):
        td_scanner = self.base
        return all((td_scanner._scf.converged,
                    td_scanner.converged[self.state]))


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
        "state",
        "atmlst",
        "de",
    }

    def __init__(self, td):
        super().__init__(td)
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.base = td
        self.state = 1  # of which the gradients to be computed.
        self.atmlst = None
        self.de = None

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
        log.info("State ID = %d", self.state)
        log.info("\n")
        return self

    grad_elec = grad_elec

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

    def get_veff(self, mol=None, dm=None, j_factor=1.0, k_factor=1.0, omega=0.0,
                 hermi=0, verbose=None):
        """
        Computes the first-order derivatives of the energy contributions from
        Veff per atom.

        NOTE: This function is incompatible to the one implemented in PySCF CPU version.
        In the CPU version, get_veff returns the first order derivatives of Veff matrix.
        """
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        if hermi == 2:
            j_factor = 0
        with mol.with_range_coulomb(omega):
            vhfopt = self.base._scf._opt_gpu.get(omega, None)
            return rhf_grad._jk_energy_per_atom(
                mol, dm, vhfopt, j_factor=j_factor, k_factor=k_factor,
                verbose=verbose) * .5

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

    @classmethod
    def from_cpu(cls, method):
        td = method.base.to_gpu()
        out = cls(td)
        out.cphf_max_cycle = method.cphf_max_cycle
        out.cphf_conv_tol = method.cphf_conv_tol
        out.state = method.state
        out.de = method.de
        return out

Grad = Gradients
