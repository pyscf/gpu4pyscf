import numpy as np
import cupy as cp
import ctypes
from pyscf import lib
from gpu4pyscf.grad.rhf import GradientsBase
from gpu4pyscf.sem.lib import libsem
from gpu4pyscf.sem.integral import hcore2c1e, eri_2c2e
from gpu4pyscf.sem.integral.fock import _LOCAL_ROW_IDX, _LOCAL_COL_IDX
from gpu4pyscf.lib.cupy_helper import asarray


def as_scanner(mf_grad):
    
    if isinstance(mf_grad, lib.GradScanner):
        return mf_grad

    from gpu4pyscf.lib import logger
    logger.info(mf_grad, 'Create scanner for %s', mf_grad.__class__)
    name = mf_grad.__class__.__name__ + SCF_GradScanner.__name_mixin__
    return lib.set_class(SCF_GradScanner(mf_grad),
                         (SCF_GradScanner, mf_grad.__class__), name)


class SCF_GradScanner(lib.GradScanner):
    def __init__(self, g):
        lib.GradScanner.__init__(self, g)

    def __call__(self, mol_or_geom, **kwargs):
        from gpu4pyscf.sem.gto.mole import Mole
        if isinstance(mol_or_geom, Mole):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)
        mf_scanner = self.base
        e_tot = mf_scanner(mol)

        de = self.kernel(**kwargs)
        return e_tot, de


class Gradients(GradientsBase):
    
    def __init__(self, mf, h=1.0E-5):
        super().__init__(mf)
        self.mol = mf.mol
        self.h = h

    as_scanner = as_scanner

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self.base.reset(mol)
        return self
        
    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None, h=None):
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if h is None: h = self.h
        
        dm = self.base.make_rdm1(mo_coeff, mo_occ)
        if isinstance(dm, np.ndarray):
            dm = cp.asarray(dm)
            
        grad = self._grad_elec_nuc_batched(dm, h)
        self.de = asarray(grad).get()
        return self.de
        
    def _grad_elec_nuc_batched(self, dm, h=1.0E-5):
        """
        We evaluate the pair energy E_AB for 6 displacements per pair.
        """
        mol = self.mol
        n_pairs = mol.npairs
        if n_pairs == 0:
            return cp.zeros((mol.natm, 3))
            
        pair_i = cp.asarray(mol.pair_i, dtype=cp.int32)
        pair_j = cp.asarray(mol.pair_j, dtype=cp.int32)
        
        aoslice = cp.asarray(mol._aoslice, dtype=cp.int32)
        natorb = cp.asarray(mol.topology.norbitals_per_atom, dtype=cp.int32)
        
        offset_i = aoslice[pair_i, 0]
        offset_j = aoslice[pair_j, 0]
        n_i = natorb[pair_i]
        n_j = natorb[pair_j]
        
        grid_r = cp.arange(9, dtype=cp.int32)
        grid_c = cp.arange(9, dtype=cp.int32)
        
        idx_i = cp.clip(offset_i[:, None] + grid_r, 0, mol.nao - 1)
        idx_j = cp.clip(offset_j[:, None] + grid_c, 0, mol.nao - 1)
        
        P_AA = dm[idx_i[:, :, None], idx_i[:, None, :]]
        P_BB = dm[idx_j[:, :, None], idx_j[:, None, :]]
        P_AB = dm[idx_i[:, :, None], idx_j[:, None, :]]
        
        mask_i = grid_r[None, :] < n_i[:, None]
        mask_j = grid_c[None, :] < n_j[:, None]
        
        P_AA *= (mask_i[:, :, None] & mask_i[:, None, :])
        P_BB *= (mask_j[:, :, None] & mask_j[:, None, :])
        P_AB *= (mask_i[:, :, None] & mask_j[:, None, :])
        
        # Pack P_AA and P_BB into 45-element arrays
        # with factor 2 for off-diagonal elements to match the energy contraction
        local_row = _LOCAL_ROW_IDX
        local_col = _LOCAL_COL_IDX
        
        P_AA_packed = P_AA[:, local_row, local_col].copy()
        P_BB_packed = P_BB[:, local_row, local_col].copy()
        
        off_diag_mask = local_row != local_col
        P_AA_packed[:, off_diag_mask] *= 2.0
        P_BB_packed[:, off_diag_mask] *= 2.0
        
        coords = cp.asarray(mol._coords, dtype=cp.float64)
        coords_i = coords[pair_i]
        coords_j = coords[pair_j]
        
        # Displacements: +x, -x, +y, -y, +z, -z
        disp_vec = cp.array([
            [h, 0, 0], [-h, 0, 0],
            [0, h, 0], [0, -h, 0],
            [0, 0, h], [0, 0, -h]
        ], dtype=cp.float64)
        
        na_pairs = mol.topology.principal_quantum_numbers[pair_i]
        nb_pairs = mol.topology.principal_quantum_numbers[pair_j]
        za_pairs = mol.topology.eta_1e[pair_i]
        zb_pairs = mol.topology.eta_1e[pair_j]

        beta = cp.asarray(mol.beta, dtype=cp.float64)
        beta_expanded = cp.zeros((mol.natm, 9), dtype=cp.float64)
        beta_expanded[:, 0]   = beta[:, 0]
        beta_expanded[:, 1:4] = beta[:, 1][:, None]
        beta_expanded[:, 4:9] = beta[:, 2][:, None]
        b_i = beta_expanded[pair_i]
        b_j = beta_expanded[pair_j]
        beta_sum = 0.5 * (b_i[:, :, None] + b_j[:, None, :])

        ele_id = cp.asarray(mol._atom_ids, dtype=cp.int32)
        task_arrays = (
            eri_2c2e.TASK_ACTION_GPU, eri_2c2e.TASK_TARGET_GPU, eri_2c2e.TASK_IJ_GPU, eri_2c2e.TASK_KL_GPU,
            eri_2c2e.TASK_LI_GPU, eri_2c2e.TASK_LJ_GPU, eri_2c2e.TASK_LK_GPU, eri_2c2e.TASK_LL_GPU
        )
        ch = cp.asarray(mol.params.get_parameter('multipole_angular_factors', to_gpu=True), dtype=cp.float64)

        tore = cp.asarray(mol.topology.core_charges, dtype=cp.float64)
        natorb_gpu = cp.asarray(mol.topology.norbitals_per_atom, dtype=cp.int32)
        guess1 = cp.asarray(mol.nuclear_params.guess1, dtype=cp.float64)
        guess2 = cp.asarray(mol.nuclear_params.guess2, dtype=cp.float64)
        guess3 = cp.asarray(mol.nuclear_params.guess3, dtype=cp.float64)
        v_par6 = cp.asarray(mol.nuclear_params.v_par6, dtype=cp.float64)
        xfac = cp.asarray(mol.nuclear_params.xfac, dtype=cp.float64)
        alpb = cp.asarray(mol.nuclear_params.alpb, dtype=cp.float64)

        ii_arr = natorb_gpu[pair_i]
        kk_arr = natorb_gpu[pair_j]
        block_sizes = (ii_arr * (ii_arr + 1) // 2) * (kk_arr * (kk_arr + 1) // 2)
        kr_offsets_full = cp.zeros(n_pairs + 1, dtype=cp.int32)
        kr_offsets_full[1:] = cp.cumsum(block_sizes)
        total_w_size = int(kr_offsets_full[-1].get())
        kr_offsets = kr_offsets_full[:-1]
        
        ind2_arr = cp.ascontiguousarray(eri_2c2e.IND2.ravel(), dtype=cp.int32)

        # TODO: following lines use many memory, consider to optimize it (e.g., batch?)
        # Because the atom coords is shifted, thus we need to save 2 batch of coords.
        # for pair p we point to even index for i and odd index for j
        # This will not conflict whether atom i shifted or not
        fake_pair_i = cp.arange(0, n_pairs * 2, 2, dtype=cp.int32)
        fake_pair_j = cp.arange(1, n_pairs * 2, 2, dtype=cp.int32)
        
        fake_ele_id = cp.empty(n_pairs * 2, dtype=cp.int32)
        fake_ele_id[0::2] = ele_id[pair_i]
        fake_ele_id[1::2] = ele_id[pair_j]
        
        fake_natorb = cp.empty(n_pairs * 2, dtype=cp.int32)
        fake_natorb[0::2] = natorb_gpu[pair_i]
        fake_natorb[1::2] = natorb_gpu[pair_j]
        
        fake_tore = cp.empty(n_pairs * 2, dtype=cp.float64)
        fake_tore[0::2] = tore[pair_i]
        fake_tore[1::2] = tore[pair_j]
        
        # guess parameters in PM6 have multiple terms (natm, 4), so fake arrays must match the 2nd dimension
        fake_guess1 = cp.empty((n_pairs * 2, guess1.shape[1]), dtype=cp.float64)
        fake_guess1[0::2] = guess1[pair_i]
        fake_guess1[1::2] = guess1[pair_j]
        
        fake_guess2 = cp.empty((n_pairs * 2, guess2.shape[1]), dtype=cp.float64)
        fake_guess2[0::2] = guess2[pair_i]
        fake_guess2[1::2] = guess2[pair_j]
        
        fake_guess3 = cp.empty((n_pairs * 2, guess3.shape[1]), dtype=cp.float64)
        fake_guess3[0::2] = guess3[pair_i]
        fake_guess3[1::2] = guess3[pair_j]
        
        def _ptr(arr):
            return ctypes.cast(arr.data.ptr, ctypes.c_void_p)

        # Array to store energies for the 6 displacements
        E_total_disp = cp.zeros((n_pairs, 6), dtype=cp.float64)

        # Loop over 6 displacements to evaluate E_AB incrementally
        for d_idx in range(6):
            coords_i_disp = coords_i + disp_vec[d_idx]
            coords_j_disp = coords_j

            rij_vec = coords_j_disp - coords_i_disp
            r_dist = cp.linalg.norm(rij_vec, axis=1)

            S_local = hcore2c1e.calc_local_overlap(
                na_pairs, nb_pairs, za_pairs, zb_pairs, r_dist)
            C_tensor = hcore2c1e.get_direction_cosines(rij_vec)
            S_global = hcore2c1e.rotation_transform(S_local, C_tensor)
            H_blocks = S_global * beta_sum

            rep_out, core_out, gab_out = eri_2c2e.calc_local_rep_core(
                pair_i, pair_j, ele_id, r_dist,
                mol.two_center_integral_params.am, mol.two_center_integral_params.ad, 
                mol.two_center_integral_params.aq, mol.two_center_integral_params.dd, mol.two_center_integral_params.qq, 
                mol.two_center_integral_params.po_tensor, mol.two_center_integral_params.ddp_tensor, 
                mol.two_center_integral_params.core_rho, ch, 
                mol.topology.core_charges, mol.topology.norbitals_per_atom, mol.topology.has_d_orbitals, 
                task_arrays, 
                HATREE2EV=mol.HARTREE2EV
            )

            # Global transform
            fake_coords = cp.zeros((n_pairs * 2, 3), dtype=cp.float64)
            fake_coords[0::2] = coords_i_disp
            fake_coords[1::2] = coords_j_disp

            w_out = cp.zeros(total_w_size, dtype=cp.float64)
            e1b_out = cp.zeros((n_pairs, 45), dtype=cp.float64)
            e2a_out = cp.zeros((n_pairs, 45), dtype=cp.float64)
            enuc_out = cp.zeros(n_pairs, dtype=cp.float64)

            err = libsem.launch_global_transform_kernel_c(
                ctypes.c_int(n_pairs),
                _ptr(fake_pair_i), _ptr(fake_pair_j), _ptr(fake_ele_id),
                _ptr(fake_coords), _ptr(rep_out), _ptr(core_out), _ptr(gab_out),
                _ptr(ind2_arr), _ptr(fake_natorb), _ptr(kr_offsets),
                _ptr(fake_tore), _ptr(xfac), _ptr(alpb),
                _ptr(fake_guess1), _ptr(fake_guess2), _ptr(fake_guess3),
                _ptr(v_par6), ctypes.c_double(mol.BOHR),
                _ptr(w_out), _ptr(e1b_out), _ptr(e2a_out), _ptr(enuc_out)
            )
            if err != 0:
                raise RuntimeError("Failed in global transform for gradients")

            # Evaluate E_AB for the current displacement
            E_hcore = cp.sum(P_AB * H_blocks, axis=(1, 2)) * 2.0
            E_e1b = cp.sum(P_AA_packed * e1b_out, axis=1)
            E_e2a = cp.sum(P_BB_packed * e2a_out, axis=1)

            # Direct scalar E_2e calculation
            E_2e_out = cp.zeros(n_pairs, dtype=cp.float64)
            err = libsem.launch_calc_pair_e2e_c(
                _ptr(w_out), _ptr(P_AA), _ptr(P_BB), _ptr(P_AB),
                _ptr(fake_pair_i), _ptr(fake_pair_j), _ptr(fake_natorb), _ptr(kr_offsets),
                _ptr(E_2e_out), ctypes.c_int(n_pairs)
            )
            if err != 0:
                raise RuntimeError("Failed in pairwise E_2e calculation")

            # Final energy for this displacement
            E_total_disp[:, d_idx] = E_hcore + E_e1b + E_e2a + E_2e_out + enuc_out

        # Compute gradient
        grad_pair = (E_total_disp[:, 0::2] - E_total_disp[:, 1::2]) / (2.0 * h)
        
        # Convert energy gradient from eV/Bohr to Hartree/Bohr
        grad_pair /= mol.HARTREE2EV
        
        # Accumulate forces to atoms
        grad = cp.zeros((mol.natm, 3), dtype=cp.float64)
        
        from cupyx import scatter_add
        cp.add.at(grad, pair_i, grad_pair)
        cp.add.at(grad, pair_j, -grad_pair) # due to symmetry of the gradient for one pair
        
        return grad