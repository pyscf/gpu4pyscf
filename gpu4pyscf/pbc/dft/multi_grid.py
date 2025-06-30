import ctypes

import numpy as np
import cupy as cp
import cupyx.scipy.fft as fft
import scipy

import pyscf.pbc.gto as gto

import pyscf.pbc.dft.multigrid as multigrid_parent
from pyscf.pbc.dft.multigrid import multigrid

from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.lib.kpts_helper import is_gamma_point
from pyscf.pbc.tools import madelung
from pyscf.lib import prange
from pyscf.pbc.scf import hf as hf_cpu

import gpu4pyscf.pbc.df.fft as fftdf_module
import gpu4pyscf.pbc.df.fft_jk as fft_jk
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc import tools
import gpu4pyscf.pbc.dft.multigrid as multigrid_qiming
import gpu4pyscf.lib.cupy_helper as cupy_helper
import gpu4pyscf.mpi as mpi
from gpu4pyscf.lib.cupy_helper import return_cupy_array

libgpbc = cupy_helper.load_library("libgpbc")


def complex_type(dtype):
    if dtype == cp.float32:
        return cp.complex64
    elif dtype == cp.float64:
        return cp.complex128
    else:
        raise ValueError("Invalid dtype")


def cast_to_pointer(array):
    if isinstance(array, cp.ndarray):
        return ctypes.cast(array.data.ptr, ctypes.c_void_p)
    elif isinstance(array, np.ndarray):
        return array.ctypes.data_as(ctypes.c_void_p)
    else:
        raise ValueError("Invalid array type")


def fft_in_place(x):
    return fft.fftn(x, axes=(-3, -2, -1), overwrite_x=True)


def ifft_in_place(x):
    return fft.ifftn(x, axes=(-3, -2, -1), overwrite_x=True)


def image_pair_to_difference(
    vectors_to_neighboring_images_cpu,
    lattice_vectors_cpu,
    reciprocal_lattice_vectors_cpu,
):
    translation_vectors = np.asarray(
        np.round(vectors_to_neighboring_images_cpu @ reciprocal_lattice_vectors_cpu.T),
        dtype=np.int32,
    )
    image_difference_full = (
        translation_vectors[:, None, :] - translation_vectors[None, :, :]
    ).reshape(-1, 3)
    difference_images, inverse = np.unique(
        image_difference_full, axis=0, return_inverse=True
    )

    difference_images = translation_vectors @ lattice_vectors_cpu

    index = np.arange(len(difference_images))[inverse]
    return cp.asarray(difference_images), cp.asarray(index, dtype=cp.int32)


def count_non_trivial_pairs(
    i_angular,
    j_angular,
    i_shells,
    j_shells,
    vectors_to_neighboring_images,
    mesh,
    atm,
    bas,
    env,
    threshold_in_log,
):
    n_i_shells = len(i_shells)
    n_j_shells = len(j_shells)
    n_images = len(vectors_to_neighboring_images)
    n_pairs = cp.zeros(1, dtype=cp.int32)
    libgpbc.count_non_trivial_pairs(
        cast_to_pointer(n_pairs),
        ctypes.c_int(i_angular),
        ctypes.c_int(j_angular),
        cast_to_pointer(i_shells),
        ctypes.c_int(n_i_shells),
        cast_to_pointer(j_shells),
        ctypes.c_int(n_j_shells),
        cast_to_pointer(vectors_to_neighboring_images),
        ctypes.c_int(n_images),
        (ctypes.c_int * 3)(*mesh),
        cast_to_pointer(atm),
        cast_to_pointer(bas),
        cast_to_pointer(env),
        ctypes.c_double(threshold_in_log),
    )
    return int(n_pairs[0])


def screen_gaussian_pairs(
    i_angular,
    j_angular,
    i_shells,
    j_shells,
    vectors_to_neighboring_images,
    mesh,
    atm,
    bas,
    env,
    threshold_in_log,
):
    n_i_shells = len(i_shells)
    n_j_shells = len(j_shells)
    n_images = len(vectors_to_neighboring_images)
    n_pairs = count_non_trivial_pairs(
        i_angular,
        j_angular,
        i_shells,
        j_shells,
        vectors_to_neighboring_images,
        mesh,
        atm,
        bas,
        env,
        threshold_in_log,
    )
    screened_shell_pairs = cp.full(n_pairs, -1, dtype=cp.int32)
    image_indices = cp.full(n_pairs, -1, dtype=cp.int32)
    pairs_to_blocks_begin = cp.full((3, n_pairs), -1, dtype=cp.int32)
    pairs_to_blocks_end = cp.full((3, n_pairs), -1, dtype=cp.int32)
    libgpbc.screen_gaussian_pairs(
        cast_to_pointer(screened_shell_pairs),
        cast_to_pointer(image_indices),
        cast_to_pointer(pairs_to_blocks_begin),
        cast_to_pointer(pairs_to_blocks_end),
        ctypes.c_int(i_angular),
        ctypes.c_int(j_angular),
        cast_to_pointer(i_shells),
        ctypes.c_int(n_i_shells),
        cast_to_pointer(j_shells),
        ctypes.c_int(n_j_shells),
        ctypes.c_int(n_pairs),
        cast_to_pointer(vectors_to_neighboring_images),
        ctypes.c_int(n_images),
        (ctypes.c_int * 3)(*mesh),
        cast_to_pointer(atm),
        cast_to_pointer(bas),
        cast_to_pointer(env),
        ctypes.c_double(threshold_in_log),
    )
    return (
        screened_shell_pairs,
        image_indices,
        pairs_to_blocks_begin,
        pairs_to_blocks_end,
    )


def assign_pairs_to_blocks(
    pairs_to_blocks_begin, pairs_to_blocks_end, n_blocks_abc, n_pairs, n_indices
):
    n_blocks = np.prod(n_blocks_abc)
    n_pairs_on_blocks = cp.full(n_blocks + 1, 0, dtype=cp.int32)
    libgpbc.count_pairs_on_blocks(
        cast_to_pointer(n_pairs_on_blocks),
        cast_to_pointer(pairs_to_blocks_begin),
        cast_to_pointer(pairs_to_blocks_end),
        cast_to_pointer(n_blocks_abc),
        ctypes.c_int(n_pairs),
    )

    n_contributing_blocks = int(n_pairs_on_blocks[-1])
    n_pairs_on_blocks = n_pairs_on_blocks[:-1]
    sorted_block_index = cp.asarray(cp.argsort(-n_pairs_on_blocks), dtype=cp.int32)
    accumulated_n_pairs_per_block = cp.zeros(n_blocks + 1, dtype=cp.int32)
    accumulated_n_pairs_per_block[1:] = cp.cumsum(n_pairs_on_blocks, dtype=cp.int32)
    sorted_block_index = sorted_block_index[:n_contributing_blocks]
    pairs_on_blocks = cp.full(n_indices, -1, dtype=cp.int32)
    libgpbc.put_pairs_on_blocks(
        cast_to_pointer(pairs_on_blocks),
        cast_to_pointer(accumulated_n_pairs_per_block),
        cast_to_pointer(sorted_block_index),
        cast_to_pointer(pairs_to_blocks_begin),
        cast_to_pointer(pairs_to_blocks_end),
        cast_to_pointer(n_blocks_abc),
        ctypes.c_int(n_contributing_blocks),
        ctypes.c_int(n_pairs),
    )

    if mpi.comm.size > 1:
        sorted_block_index = sorted_block_index[
            cp.arange(mpi.comm.rank, n_contributing_blocks, mpi.comm.size)
        ]

    return (
        pairs_on_blocks,
        accumulated_n_pairs_per_block,
        sorted_block_index,
    )


def sort_gaussian_pairs(mydf, xc_type="LDA"):
    log = logger.new_logger(mydf, mydf.verbose)
    t0 = log.init_timer()
    cell = mydf.cell
    vol = cell.vol
    block_size = np.array([4, 4, 4])
    lattice_vectors = cp.asarray(cell.lattice_vectors())
    off_diagonal = lattice_vectors - cp.diag(lattice_vectors.diagonal())
    is_non_orthogonal = cp.any(cp.abs(off_diagonal) > 1e-10)
    if is_non_orthogonal:
        is_non_orthogonal = 1
    else:
        is_non_orthogonal = 0
    reciprocal_lattice_vectors = cp.asarray(cp.linalg.inv(lattice_vectors).T, order="C")

    reciprocal_norms = cp.linalg.norm(reciprocal_lattice_vectors, axis=1)
    libgpbc.update_lattice_vectors(
        ctypes.cast(lattice_vectors.data.ptr, ctypes.c_void_p),
        ctypes.cast(reciprocal_lattice_vectors.data.ptr, ctypes.c_void_p),
        ctypes.cast(reciprocal_norms.data.ptr, ctypes.c_void_p),
    )

    tasks = getattr(mydf, "tasks", None)
    if tasks is None:
        tasks = multigrid.multi_grids_tasks(cell, mydf.mesh, log)
        mydf.tasks = tasks

    t0 = log.timer("task generation", *t0)
    t1 = t0
    pairs = []
    for grids_localized, grids_diffused in tasks:
        subcell_in_localized_region = grids_localized.cell
        # the original grids_localized.mesh has dtype=np.int64, which can cause
        # misalignment when the pointer is passed to the C code.
        mesh = np.asarray(grids_localized.mesh, dtype=np.int32)

        fft_grid = list(
            map(
                lambda n_mesh_points: cp.fft.fftfreq(
                    n_mesh_points, 1.0 / n_mesh_points
                ).astype(cp.int32),
                mesh,
            )
        )

        dxyz_dabc = cp.asarray((lattice_vectors.T / cp.asarray(mesh)).T)
        n_blocks_abc = np.asarray(np.ceil(mesh / block_size), dtype=cp.int32)
        equivalent_cell_in_localized, coeff_in_localized = (
            subcell_in_localized_region.decontract_basis(to_cart=True, aggregate=True)
        )

        n_primitive_gtos_in_localized = multigrid._pgto_shells(
            subcell_in_localized_region
        )

        # theoretically we can use the rcut defined in localized cell to reduce the
        # number of images, but somehow it can introduce some error when the lattice
        # is super small, for example primitive diamond cell. Using the rcut defined
        # in the global cell can fix this.
        vectors_to_neighboring_images = cp.asarray(gto.eval_gto.get_lattice_Ls(cell))
        n_images = len(vectors_to_neighboring_images)

        if is_gamma_point(mydf.kpts):
            phase_diff_among_images = cp.asarray([[1.0]])
            image_pair_difference_index = cp.zeros((n_images, n_images), dtype=cp.int32)
        else:
            difference_images, image_pair_difference_index = image_pair_to_difference(
                vectors_to_neighboring_images.get(),
                lattice_vectors.get(),
                reciprocal_lattice_vectors.get(),
            )
            phase_diff_among_images = cp.exp(
                1j * cp.asarray(mydf.kpts.reshape(-1, 3)).dot(difference_images.T)
            )
        if grids_diffused is None:
            grouped_cell = equivalent_cell_in_localized
            concatenated_coeff = scipy.linalg.block_diag(coeff_in_localized)
        else:
            subcell_in_diffused_region = grids_diffused.cell
            equivalent_cell_in_diffused, coeff_in_diffused = (
                subcell_in_diffused_region.decontract_basis(
                    to_cart=True, aggregate=True
                )
            )

            grouped_cell = equivalent_cell_in_localized + equivalent_cell_in_diffused

            grouped_cell._bas[n_primitive_gtos_in_localized:, 0] -= len(
                subcell_in_localized_region._atm
            )

            concatenated_coeff = scipy.linalg.block_diag(
                coeff_in_localized, coeff_in_diffused
            )
        concatenated_coeff = cp.asarray(concatenated_coeff)

        n_primitive_gtos_in_two_regions = multigrid._pgto_shells(grouped_cell)
        weight_penalty = np.prod(grouped_cell.mesh) / vol
        minimum_exponent = np.hstack(grouped_cell.bas_exps()).min()
        theta_ij = minimum_exponent / 2
        lattice_summation_factor = max(2 * np.pi * cell.rcut / (vol * theta_ij), 1)

        precision = cell.precision / weight_penalty / lattice_summation_factor
        if xc_type != "LDA":
            precision *= 0.1
        threshold_in_log = np.log(precision * multigrid.EXTRA_PREC)

        shell_to_ao_indices = cp.asarray(
            gto.moleintor.make_loc(grouped_cell._bas, "cart"), dtype=cp.int32
        )
        ao_indices_in_localized = cp.asarray(grids_localized.ao_idx, dtype=cp.int32)
        if grids_diffused is None:
            ao_indices_in_diffused = cp.array([], dtype=cp.int32)
        else:
            ao_indices_in_diffused = cp.asarray(grids_diffused.ao_idx, dtype=cp.int32)

        concatenated_ao_indices = cp.concatenate(
            (ao_indices_in_localized, ao_indices_in_diffused)
        )
        coeff_in_localized = cp.asarray(coeff_in_localized)
        per_angular_pairs = []

        i_angulars = grouped_cell._bas[:n_primitive_gtos_in_localized, multigrid.ANG_OF]
        i_angulars_unique = np.unique(i_angulars)
        sorted_i_shells = []
        for l in i_angulars_unique:
            i_shells = cp.asarray(np.where(i_angulars == l)[0], dtype=cp.int32)
            sorted_i_shells.append(i_shells)

        j_angulars = grouped_cell._bas[
            :n_primitive_gtos_in_two_regions, multigrid.ANG_OF
        ]
        j_angulars_unique = np.unique(j_angulars)
        sorted_j_shells = []
        for l in j_angulars_unique:
            j_shells = cp.asarray(np.where(j_angulars == l)[0], dtype=cp.int32)
            sorted_j_shells.append(j_shells)

        atm = cp.asarray(grouped_cell._atm, dtype=cp.int32)
        bas = cp.asarray(grouped_cell._bas, dtype=cp.int32)
        env = cp.asarray(grouped_cell._env)

        t1 = log.timer_debug2("routines before screening", *t1)
        for i_angular, i_shells in zip(i_angulars_unique, sorted_i_shells):
            for j_angular, j_shells in zip(j_angulars_unique, sorted_j_shells):
                (
                    screened_shell_pairs,
                    image_indices,
                    pairs_to_blocks_begin,
                    pairs_to_blocks_end,
                ) = screen_gaussian_pairs(
                    i_angular,
                    j_angular,
                    i_shells,
                    j_shells,
                    vectors_to_neighboring_images,
                    mesh,
                    atm,
                    bas,
                    env,
                    threshold_in_log,
                )
                t1 = log.timer_debug2(
                    "screening in angular pair" + str((i_angular, j_angular)), *t1
                )
                contributing_block_ranges = (
                    pairs_to_blocks_end - pairs_to_blocks_begin + 1
                )
                n_contributing_blocks_per_pair = cp.prod(
                    contributing_block_ranges, axis=0
                )
                n_indices = int(cp.sum(n_contributing_blocks_per_pair))
                n_pairs = len(screened_shell_pairs)
                (
                    gaussian_pair_indices,
                    accumulated_counts,
                    sorted_contributing_blocks,
                ) = assign_pairs_to_blocks(
                    pairs_to_blocks_begin,
                    pairs_to_blocks_end,
                    n_blocks_abc,
                    n_pairs,
                    n_indices,
                )
                t1 = log.timer_debug2(
                    "assigning pairs to blocks in angular pair"
                    + str((i_angular, j_angular)),
                    *t1
                )
                per_angular_pairs.append(
                    {
                        "angular": (i_angular, j_angular),
                        "screened_shell_pairs": screened_shell_pairs,
                        "pair_indices_per_block": gaussian_pair_indices,
                        "accumulated_counts_per_block": accumulated_counts,
                        "sorted_block_index": sorted_contributing_blocks,
                        "image_indices": image_indices,
                        "i_shells": i_shells,
                        "j_shells": j_shells,
                        "shell_to_ao_indices": shell_to_ao_indices,
                    }
                )

        pairs.append(
            {
                "per_angular_pairs": per_angular_pairs,
                "neighboring_images": vectors_to_neighboring_images,
                "grouped_cell": grouped_cell,
                "mesh": mesh,  # this one is on cpu memory
                "fft_grid": fft_grid,
                "ao_indices_in_localized": ao_indices_in_localized,
                "ao_indices_in_diffused": ao_indices_in_diffused,
                "concatenated_ao_indices": concatenated_ao_indices,
                "coeff_in_localized": coeff_in_localized,
                "concatenated_coeff": concatenated_coeff,
                "atm": atm,
                "bas": bas,
                "env": env,
                "phase_diff_among_images": phase_diff_among_images,
                "image_pair_difference_index": image_pair_difference_index,
                "dxyz_dabc": dxyz_dabc,
                "is_non_orthogonal": is_non_orthogonal,
            }
        )

    mydf.sorted_gaussian_pairs = pairs

    t0 = log.timer("sort_gaussian_pairs", *t0)


def evaluate_density_wrapper(pairs_info, dm_slice, ignore_imag=True):
    c_driver = libgpbc.evaluate_density_driver
    n_images = pairs_info["neighboring_images"].shape[0]
    n_k_points, n_difference_images = pairs_info["phase_diff_among_images"].shape
    if n_k_points == 1:
        density_matrix_with_translation = dm_slice
    else:
        density_matrix_with_translation = cp.einsum(
            "kt, ikpq->itpq", pairs_info["phase_diff_among_images"], dm_slice
        )

    n_channels, _, n_i_functions, n_j_functions = density_matrix_with_translation.shape

    if ignore_imag is False:
        raise NotImplementedError
    density_matrix_with_translation_real_part = cp.asarray(
        density_matrix_with_translation.real, order="C"
    )

    if dm_slice.dtype == cp.float32:
        use_float_precision = ctypes.c_int(1)
    else:
        use_float_precision = ctypes.c_int(0)

    density = cp.zeros((n_channels,) + tuple(pairs_info["mesh"]), dtype=dm_slice.dtype)

    for gaussians_per_angular_pair in pairs_info["per_angular_pairs"]:
        (i_angular, j_angular) = gaussians_per_angular_pair["angular"]

        c_driver(
            cast_to_pointer(density),
            cast_to_pointer(density_matrix_with_translation_real_part),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
            cast_to_pointer(gaussians_per_angular_pair["screened_shell_pairs"]),
            cast_to_pointer(gaussians_per_angular_pair["i_shells"]),
            cast_to_pointer(gaussians_per_angular_pair["j_shells"]),
            ctypes.c_int(len(gaussians_per_angular_pair["j_shells"])),
            cast_to_pointer(gaussians_per_angular_pair["shell_to_ao_indices"]),
            ctypes.c_int(n_i_functions),
            ctypes.c_int(n_j_functions),
            cast_to_pointer(gaussians_per_angular_pair["pair_indices_per_block"]),
            cast_to_pointer(gaussians_per_angular_pair["accumulated_counts_per_block"]),
            cast_to_pointer(gaussians_per_angular_pair["sorted_block_index"]),
            ctypes.c_int(len(gaussians_per_angular_pair["sorted_block_index"])),
            cast_to_pointer(gaussians_per_angular_pair["image_indices"]),
            cast_to_pointer(pairs_info["neighboring_images"]),
            ctypes.c_int(n_images),
            cast_to_pointer(pairs_info["image_pair_difference_index"]),
            ctypes.c_int(n_difference_images),
            (ctypes.c_int * 3)(*pairs_info["mesh"]),
            cast_to_pointer(pairs_info["atm"]),
            cast_to_pointer(pairs_info["bas"]),
            cast_to_pointer(pairs_info["env"]),
            ctypes.c_int(n_channels),
            ctypes.c_int(pairs_info["is_non_orthogonal"]),
            use_float_precision,
        )

    mpi.comm.reduce(density, in_place=True)

    return density


def evaluate_density_on_g_mesh(mydf, dm_kpts, kpts=np.zeros((1, 3)), deriv=0):
    dm_kpts = cp.asarray(dm_kpts, order="C")
    dms = fft_jk._format_dms(dm_kpts, kpts)
    n_channels, n_k_points = dms.shape[:2]

    tasks = getattr(mydf, "tasks", None)
    if tasks is None:
        raise NotImplementedError

    density_slices = 1

    if deriv == 1:
        density_slices += 3

    if deriv > 1:
        raise NotImplementedError

    nx, ny, nz = mydf.mesh
    density_on_g_mesh = cp.zeros(
        (n_channels, density_slices, nx, ny, nz), dtype=cp.complex128
    )
    for pairs in mydf.sorted_gaussian_pairs:

        mesh = pairs["mesh"]

        n_grid_points = np.prod(mesh)
        weight_per_grid_point = 1.0 / n_k_points * mydf.cell.vol / n_grid_points

        density_matrix_with_rows_in_localized = dms[
            :,
            :,
            pairs["ao_indices_in_localized"][:, None],
            pairs["concatenated_ao_indices"],
        ]

        density_matrix_with_rows_in_diffused = dms[
            :,
            :,
            pairs["ao_indices_in_diffused"][:, None],
            pairs["ao_indices_in_localized"],
        ]

        n_ao_in_localized = density_matrix_with_rows_in_diffused.shape[3]
        density_matrix_with_rows_in_localized[
            :, :, :, n_ao_in_localized:
        ] += density_matrix_with_rows_in_diffused.transpose(0, 1, 3, 2).conj()

        coeff_sandwiched_density_matrix = cp.einsum(
            "nkij,pi->nkpj",
            density_matrix_with_rows_in_localized,
            pairs["coeff_in_localized"],
        )

        coeff_sandwiched_density_matrix = cp.einsum(
            "nkpj, qj -> nkpq",
            coeff_sandwiched_density_matrix,
            pairs["concatenated_coeff"],
        )

        libgpbc.update_dxyz_dabc(
            ctypes.cast(pairs["dxyz_dabc"].data.ptr, ctypes.c_void_p)
        )

        density = (
            evaluate_density_wrapper(
                pairs, cp.asarray(coeff_sandwiched_density_matrix, dtype=mydf.dtype)
            )
            * weight_per_grid_point
        )

        density = fft_in_place(density)

        density_on_g_mesh[
            :,
            0,
            pairs["fft_grid"][0][:, None, None],
            pairs["fft_grid"][1][:, None],
            pairs["fft_grid"][2],
        ] += density

    density_on_g_mesh = density_on_g_mesh.reshape([n_channels, density_slices, -1])
    if density_slices == 4:
        density_on_g_mesh[:, 1:] = (
            cp.einsum(
                "np, xp -> nxp", density_on_g_mesh[:, 0], mydf.gradient_vector_on_g_mesh
            )
            * 1j
        )

    return cp.asarray(density_on_g_mesh)


def evaluate_xc_wrapper(pairs_info, xc_weights):
    c_driver = libgpbc.evaluate_xc_driver
    n_i_functions = len(pairs_info["coeff_in_localized"])
    n_j_functions = len(pairs_info["concatenated_coeff"])

    n_channels = xc_weights.shape[0]
    n_k_points, n_difference_images = pairs_info["phase_diff_among_images"].shape
    n_images = pairs_info["neighboring_images"].shape[0]

    fock = cp.zeros(
        (n_channels, n_difference_images, n_i_functions, n_j_functions),
        dtype=xc_weights.dtype,
    )
    if xc_weights.dtype == cp.float32:
        use_float_precision = ctypes.c_int(1)
    else:
        use_float_precision = ctypes.c_int(0)

    for gaussians_per_angular_pair in pairs_info["per_angular_pairs"]:
        fock_slice = cp.zeros(
            (n_channels, n_difference_images, n_i_functions, n_j_functions),
            dtype=xc_weights.dtype,
        )
        (i_angular, j_angular) = gaussians_per_angular_pair["angular"]
        c_driver(
            cast_to_pointer(fock_slice),
            cast_to_pointer(xc_weights),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
            cast_to_pointer(gaussians_per_angular_pair["screened_shell_pairs"]),
            cast_to_pointer(gaussians_per_angular_pair["i_shells"]),
            cast_to_pointer(gaussians_per_angular_pair["j_shells"]),
            ctypes.c_int(len(gaussians_per_angular_pair["j_shells"])),
            cast_to_pointer(gaussians_per_angular_pair["shell_to_ao_indices"]),
            ctypes.c_int(n_i_functions),
            ctypes.c_int(n_j_functions),
            cast_to_pointer(gaussians_per_angular_pair["pair_indices_per_block"]),
            cast_to_pointer(gaussians_per_angular_pair["accumulated_counts_per_block"]),
            cast_to_pointer(gaussians_per_angular_pair["sorted_block_index"]),
            ctypes.c_int(len(gaussians_per_angular_pair["sorted_block_index"])),
            cast_to_pointer(gaussians_per_angular_pair["image_indices"]),
            cast_to_pointer(pairs_info["neighboring_images"]),
            ctypes.c_int(n_images),
            cast_to_pointer(pairs_info["image_pair_difference_index"]),
            ctypes.c_int(n_difference_images),
            cast_to_pointer(pairs_info["mesh"]),
            cast_to_pointer(pairs_info["atm"]),
            cast_to_pointer(pairs_info["bas"]),
            cast_to_pointer(pairs_info["env"]),
            ctypes.c_int(n_channels),
            ctypes.c_int(pairs_info["is_non_orthogonal"]),
            use_float_precision,
        )
        fock += fock_slice

    if n_k_points > 1:
        return cp.einsum(
            "kt, ntij -> nkij", pairs_info["phase_diff_among_images"], fock
        )
    else:
        return fock


def convert_xc_on_g_mesh_to_fock(
    mydf,
    xc_on_g_mesh,
    hermi=1,
    kpts=np.zeros((1, 3)),
):
    cell = mydf.cell
    n_k_points = len(kpts)
    nao = cell.nao_nr()

    xc_on_g_mesh = xc_on_g_mesh.reshape(-1, *mydf.mesh)
    n_channels = xc_on_g_mesh.shape[0]
    at_gamma_point = multigrid.gamma_point(kpts)

    if hermi != 1:
        raise NotImplementedError

    data_type = cp.float64
    if not at_gamma_point:
        data_type = complex_type(cp.float64)

    fock = cp.zeros((n_channels, n_k_points, nao, nao), dtype=data_type)

    for pairs in mydf.sorted_gaussian_pairs:
        interpolated_xc = xc_on_g_mesh[
            :,
            pairs["fft_grid"][0][:, None, None],
            pairs["fft_grid"][1][:, None],
            pairs["fft_grid"][2],
        ]
        interpolated_xc = cp.asarray(
            ifft_in_place(interpolated_xc).real, order="C", dtype=mydf.dtype
        )

        n_ao_in_localized = len(pairs["ao_indices_in_localized"])
        libgpbc.update_dxyz_dabc(
            ctypes.cast(pairs["dxyz_dabc"].data.ptr, ctypes.c_void_p)
        )
        fock_slice = evaluate_xc_wrapper(pairs, interpolated_xc)
        fock_slice = cp.einsum("nkpq,pi->nkiq", fock_slice, pairs["coeff_in_localized"])
        fock_slice = cp.einsum("nkiq,qj->nkij", fock_slice, pairs["concatenated_coeff"])

        # While mathematically it is correct to have concatenated
        # ao indices in the addition, but it is possible that the ao
        # indices overlap between localized gaussians and diffused gaussians
        # (imagine two gaussians within a single shell, say, C2s).
        # In this case, the addition to the same place requires atomic
        # operation, while I guess in the cupy code it is assumed that
        # the indices do not overlap, and hence no atomic guard.
        # Anyway, the numerical result will be wrong if we use
        # concatenated ao indices.
        fock[
            :,
            :,
            pairs["ao_indices_in_localized"][:, None],
            pairs["ao_indices_in_localized"],
        ] += fock_slice[:, :, :, :n_ao_in_localized]
        fock[
            :,
            :,
            pairs["ao_indices_in_localized"][:, None],
            pairs["ao_indices_in_diffused"],
        ] += fock_slice[:, :, :, n_ao_in_localized:]
        if hermi == 1:
            fock[
                :,
                :,
                pairs["ao_indices_in_diffused"][:, None],
                pairs["ao_indices_in_localized"],
            ] += (
                fock_slice[:, :, :, n_ao_in_localized:].transpose(0, 1, 3, 2).conj()
            )
        else:
            raise NotImplementedError

    mpi.comm.reduce(fock, in_place=True)

    return fock


def get_ovlp(
    mydf,
    kpts=np.zeros((1, 3)),
):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t0 = log.init_timer()
    cell = mydf.cell
    n_k_points = len(kpts)
    nao = cell.nao_nr()

    if is_gamma_point(kpts):
        dtype = cp.float64
    else:
        dtype = complex_type(cp.float64)

    overlap = cp.zeros((n_k_points, nao, nao), dtype=dtype)

    for pairs in mydf.sorted_gaussian_pairs:
        mesh = pairs["mesh"]
        n_grid_points = np.prod(mesh)
        weight = cell.vol / n_grid_points

        overlap_weight = cp.ones((1, n_grid_points), dtype=mydf.dtype)
        n_ao_in_localized = len(pairs["ao_indices_in_localized"])
        libgpbc.update_dxyz_dabc(
            ctypes.cast(pairs["dxyz_dabc"].data.ptr, ctypes.c_void_p)
        )
        overlap_slice = evaluate_xc_wrapper(pairs, overlap_weight)
        overlap_slice = overlap_slice.reshape(overlap_slice.shape[-3:]) * weight
        overlap_slice = cp.einsum(
            "kpq,pi->kiq", overlap_slice, pairs["coeff_in_localized"]
        )
        overlap_slice = cp.einsum(
            "kiq,qj->kij", overlap_slice, pairs["concatenated_coeff"]
        )

        # While mathematically it is correct to have concatenated
        # ao indices in the addition, but it is possible that the ao
        # indices overlap between localized gaussians and diffused gaussians
        # (imagine two gaussians within a single shell, say, C2s).
        # In this case, the addition to the same place requires atomic
        # operation, while I guess in the cupy code it is assumed that
        # the indices do not overlap, and hence no atomic guard.
        # Anyway, the numerical result will be wrong if we use
        # concatenated ao indices.
        overlap[
            :,
            pairs["ao_indices_in_localized"][:, None],
            pairs["ao_indices_in_localized"],
        ] += overlap_slice[:, :, :n_ao_in_localized]
        overlap[
            :,
            pairs["ao_indices_in_localized"][:, None],
            pairs["ao_indices_in_diffused"],
        ] += overlap_slice[:, :, n_ao_in_localized:]
        overlap[
            :,
            pairs["ao_indices_in_diffused"][:, None],
            pairs["ao_indices_in_localized"],
        ] += (
            overlap_slice[:, :, n_ao_in_localized:].transpose(0, 2, 1).conj()
        )

    mpi.comm.reduce(overlap, in_place=True)

    t0 = log.timer("ovlp", *t0)

    return overlap


def evaluate_xc_gradient_wrapper(
    gradient, pairs_info, xc_weights, dm_slice, ignore_imag=True
):
    c_driver = libgpbc.evaluate_xc_gradient_driver

    assert gradient.dtype == xc_weights.dtype
    assert gradient.dtype == dm_slice.dtype

    if gradient.dtype == cp.float32:
        use_float_precision = ctypes.c_int(1)
    else:
        use_float_precision = ctypes.c_int(0)

    n_images = pairs_info["neighboring_images"].shape[0]
    n_k_points, n_difference_images = pairs_info["phase_diff_among_images"].shape

    if n_k_points == 1:
        density_matrix_with_translation = dm_slice
    else:
        density_matrix_with_translation = cp.einsum(
            "kt, ikpq -> itpq", pairs_info["phase_diff_among_images"], dm_slice
        )

    n_channels, _, n_i_functions, n_j_functions = density_matrix_with_translation.shape
    if ignore_imag is False:
        raise NotImplementedError

    density_matrix_with_translation_real_part = cp.asarray(
        density_matrix_with_translation.real, order="C"
    )

    for gaussians_per_angular_pair in pairs_info["per_angular_pairs"]:
        (i_angular, j_angular) = gaussians_per_angular_pair["angular"]
        c_driver(
            cast_to_pointer(gradient),
            cast_to_pointer(xc_weights),
            cast_to_pointer(density_matrix_with_translation_real_part),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
            cast_to_pointer(gaussians_per_angular_pair["screened_shell_pairs"]),
            cast_to_pointer(gaussians_per_angular_pair["i_shells"]),
            cast_to_pointer(gaussians_per_angular_pair["j_shells"]),
            ctypes.c_int(len(gaussians_per_angular_pair["j_shells"])),
            cast_to_pointer(gaussians_per_angular_pair["shell_to_ao_indices"]),
            ctypes.c_int(n_i_functions),
            ctypes.c_int(n_j_functions),
            cast_to_pointer(gaussians_per_angular_pair["pair_indices_per_block"]),
            cast_to_pointer(gaussians_per_angular_pair["accumulated_counts_per_block"]),
            cast_to_pointer(gaussians_per_angular_pair["sorted_block_index"]),
            ctypes.c_int(len(gaussians_per_angular_pair["sorted_block_index"])),
            cast_to_pointer(gaussians_per_angular_pair["image_indices"]),
            cast_to_pointer(pairs_info["neighboring_images"]),
            ctypes.c_int(n_images),
            cast_to_pointer(pairs_info["image_pair_difference_index"]),
            ctypes.c_int(n_difference_images),
            cast_to_pointer(pairs_info["mesh"]),
            cast_to_pointer(pairs_info["atm"]),
            cast_to_pointer(pairs_info["bas"]),
            cast_to_pointer(pairs_info["env"]),
            ctypes.c_int(n_channels),
            ctypes.c_int(pairs_info["is_non_orthogonal"]),
            use_float_precision,
        )


def convert_xc_on_g_mesh_to_fock_gradient(
    mydf,
    xc_on_g_mesh,
    dm_kpts,
    hermi=1,
    kpts=np.zeros((1, 3)),
):
    cell = mydf.cell
    dm_kpts = cp.asarray(dm_kpts, order="C")
    dms = fft_jk._format_dms(dm_kpts, kpts)
    n_atoms = cell.natm

    xc_on_g_mesh = xc_on_g_mesh.reshape(-1, *mydf.mesh)

    if hermi != 1:
        raise NotImplementedError

    gradient = cp.zeros((n_atoms, 3), dtype=mydf.dtype)

    for pairs in mydf.sorted_gaussian_pairs:
        interpolated_xc = xc_on_g_mesh[
            :,
            pairs["fft_grid"][0][:, None, None],
            pairs["fft_grid"][1][:, None],
            pairs["fft_grid"][2],
        ]

        interpolated_xc = cp.asarray(
            ifft_in_place(interpolated_xc).real, order="C", dtype=mydf.dtype
        )

        density_matrix_slice = dms[
            :,
            :,
            pairs["ao_indices_in_localized"][:, None],
            pairs["concatenated_ao_indices"],
        ]
        density_matrix_with_rows_in_diffused = dms[
            :,
            :,
            pairs["ao_indices_in_diffused"][:, None],
            pairs["ao_indices_in_localized"],
        ]

        n_ao_in_localized = density_matrix_slice.shape[2]
        density_matrix_slice[
            :, :, :, n_ao_in_localized:
        ] += density_matrix_with_rows_in_diffused.transpose(0, 1, 3, 2).conj()

        coeff_sandwiched_density_matrix = cp.einsum(
            "nkij,pi->nkpj",
            density_matrix_slice,
            pairs["coeff_in_localized"],
        )

        coeff_sandwiched_density_matrix = cp.einsum(
            "nkpj, qj -> nkpq",
            coeff_sandwiched_density_matrix,
            pairs["concatenated_coeff"],
        )

        libgpbc.update_dxyz_dabc(cast_to_pointer(pairs["dxyz_dabc"]))

        evaluate_xc_gradient_wrapper(
            gradient,
            pairs,
            interpolated_xc,
            cp.asarray(coeff_sandwiched_density_matrix, dtype=mydf.dtype),
            ignore_imag=True,
        )

    mpi.comm.reduce(gradient, in_place=True)

    return gradient


def get_nuc(mydf, kpts=None):
    kpts, is_single_kpt = fftdf_module._check_kpts(mydf, kpts)
    cell = mydf.cell
    mesh = mydf.mesh
    charge = cp.asarray(-cell.atom_charges())
    Gv = cell.get_Gv(mesh)
    SI = cp.asarray(cell.get_SI(Gv))
    rhoG = cp.dot(charge, SI)

    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
    vneG = rhoG * coulG
    hermi = 1
    vne = convert_xc_on_g_mesh_to_fock(mydf, vneG, hermi, kpts)[0]

    if is_single_kpt:
        vne = vne[0]
    return vne


def get_pp(mydf, kpts=None):
    """Get the periodic pseudopotential nuc-el AO matrix, with G=0 removed."""

    assert kpts is None or all(kpts == 0)
    is_single_kpt = False
    if kpts is None or kpts.ndim == 1:
        is_single_kpt = True
    kpts = np.zeros((1, 3))

    cell = mydf.cell
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    mesh = mydf.mesh
    # Compute the vpplocG as
    # -einsum('ij,ij->j', pseudo.get_vlocG(cell, Gv), cell.get_SI(Gv))
    vpplocG, vpplocG_part1 = multigrid_qiming.eval_vpplocG(cell, mesh, cache_part1=True)
    mydf.vpplocG_part1 = vpplocG_part1
    vpp = convert_xc_on_g_mesh_to_fock(mydf, vpplocG, hermi=1, kpts=kpts)[0]
    t1 = log.timer_debug1("vpploc", *t0)

    vppnl = pp_int.get_pp_nl(cell, kpts)
    for k, kpt in enumerate(kpts):
        if is_single_kpt:
            vpp[k] += cp.asarray(vppnl[k].real)
        else:
            vpp[k] += cp.asarray(vppnl[k])

    if is_single_kpt:
        vpp = vpp[0]
    log.timer_debug1("vppnl", *t1)
    log.timer("get_pp", *t0)
    return vpp


def get_veff(
    mydf,
    xc_code,
    dm_kpts,
    hermi=1,
    kpts=None,
    kpts_band=None,
    with_j=True,
    return_j=False,
    verbose=None,
):
    if kpts is None:
        kpts = mydf.kpts
    log = logger.new_logger(mydf, verbose)
    t0 = log.init_timer()
    cell = mydf.cell
    dm_kpts = cp.asarray(dm_kpts, order="C")
    dms = fft_jk._format_dms(dm_kpts, kpts)
    nset = dms.shape[0]

    kpts_band = _format_kpts_band(kpts_band, kpts)

    numerical_integrator = mydf._numint
    xc_type = numerical_integrator._xc_type(xc_code)

    if xc_type == "LDA":
        derivative_order = 0
    elif xc_type == "GGA":
        derivative_order = 1
    else:
        raise NotImplementedError

    mesh = mydf.mesh
    ngrids = np.prod(mesh)

    density = evaluate_density_on_g_mesh(mydf, dm_kpts, kpts, derivative_order)

    coulomb_on_g_mesh = cp.einsum(
        "ng, g -> g", density[:, 0], mydf.coulomb_kernel_on_g_mesh
    )

    coulomb_energy = 0.5 * cp.einsum(
        "ng,g->", density[:, 0].real, coulomb_on_g_mesh.real
    )
    coulomb_energy += 0.5 * cp.einsum(
        "ng,g->", density[:, 0].imag, coulomb_on_g_mesh.imag
    )
    coulomb_energy /= cell.vol

    log.debug("Multigrid Coulomb energy %s", coulomb_energy)
    t0 = log.timer("coulomb", *t0)
    weight = cell.vol / ngrids

    density = density.reshape(-1, *mesh) / weight
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    density = cp.asarray(
        ifft_in_place(density).real.reshape(nset, -1, ngrids),
        order="C",
        dtype=mydf.dtype,
    )

    n_electrons = density[:, 0].sum() * weight

    if nset == 1:
        xc_for_energy, xc_for_fock = numerical_integrator.eval_xc_eff(
            xc_code, density[0], deriv=1, xctype=xc_type
        )[:2]
    else:
        xc_for_energy, xc_for_fock = numerical_integrator.eval_xc_eff(
            xc_code, density, deriv=1, xctype=xc_type
        )[:2]

    xc_energy_sum = (density[:, 0] * xc_for_energy.flatten()).sum() * weight

    xc_for_fock = xc_for_fock.reshape(-1, *mesh) * weight

    # To reduce the memory usage, we reuse the xc_for_fock name.
    # Now xc_for_fock represents xc on G space
    xc_for_fock = fft_in_place(xc_for_fock).reshape(nset, -1, ngrids)

    log.debug("Multigrid exc %s  nelec %s", xc_energy_sum, n_electrons)

    kpts_band = _format_kpts_band(kpts_band, kpts)

    if xc_type == "GGA":
        xc_for_fock = (
            xc_for_fock[:, 0]
            - cp.einsum(
                "ngp, gp -> np",
                xc_for_fock[:, 1:],
                mydf.gradient_vector_on_g_mesh,
            )
            * 1j
        )
        xc_for_fock = xc_for_fock.reshape((nset, -1, ngrids))

    if with_j:
        xc_for_fock[:, 0] += coulomb_on_g_mesh

    xc_for_fock = convert_xc_on_g_mesh_to_fock(mydf, xc_for_fock, hermi, kpts_band)

    xc_for_fock = xc_for_fock.reshape(dm_kpts.shape)
    t0 = log.timer("xc", *t0)

    xc_for_fock = cupy_helper.tag_array(
        xc_for_fock,
        ecoul=coulomb_energy,
        exc=xc_energy_sum,
        vj=None,
        vk=None,
    )

    return n_electrons, xc_energy_sum, xc_for_fock


def get_k_kpts(
    df_object,
    dm_kpts,
    hermi=1,
    kpts=np.zeros((1, 3)),
    kpts_band=None,
    exxdiv=None,
    p_slice=4,
    q_slice=32,
):
    """Get the Coulomb (J) and exchange (K) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point
        kpts : (nkpts, 3) ndarray

    Kwargs:
        hermi : int
            Whether K matrix is hermitian

            | 0 : not hermitian and not symmetric
            | 1 : hermitian

        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray
        or list of vj and vk if the input dm_kpts is a list of DMs
    """
    log = logger.new_logger(df_object, df_object.verbose)
    t0 = log.init_timer()
    cell = df_object.cell
    mesh = df_object.mesh
    assert cell.low_dim_ft_type != "inf_vacuum"
    assert cell.dimension > 1

    formatted_density_matrices = fft_jk._format_dms(dm_kpts, kpts)
    n_k_points = len(kpts)
    if len(dm_kpts.shape) == 3:
        n_channels, n_ao = dm_kpts.shape[:2]
    else:
        n_channels = 1
        n_ao = dm_kpts.shape[0]

    occupied_mo_coeff = dm_kpts.__dict__["occ_coeff"].reshape(
        n_channels, n_k_points, n_ao, -1
    )
    overlap = df_object._overlap.reshape(n_k_points, n_ao, n_ao)

    data_type = overlap.dtype
    occupation_numbers = dm_kpts.__dict__["mo_occ"]
    is_occupied = occupation_numbers > 0
    occupied_occupation_numbers = occupation_numbers[is_occupied]
    n_occupied = len(occupied_occupation_numbers)
    mo_to_ao = cp.einsum("kpq, ikqm -> ikpm", overlap, occupied_mo_coeff)

    weight = 1.0 / n_k_points * (cell.vol / df_object.n_grid_points)

    vk = cp.zeros((n_channels, n_k_points, n_ao, n_ao), dtype=data_type)
    if not is_gamma_point(kpts):
        raise NotImplementedError

    for k2_index, k2 in enumerate(kpts):
        occupied_mo_at_k2 = df_object.get_mo_values(occupied_mo_coeff[:, k2_index], k2)

        t1 = log.timer_debug1("occupied_mo_at_k2", *t0)

        coulomb_weighted_density_dot_ao_at_k2 = cp.zeros(
            (n_channels, n_occupied, df_object.n_grid_points), dtype=data_type
        )

        for k1_index, k1 in enumerate(kpts):
            if is_gamma_point(kpts):
                occupied_mo_at_k1 = occupied_mo_at_k2
            else:
                e_ikr = cp.exp(df_object.grids.coords @ cp.asarray(1j * (k1 - k2)))

                occupied_mo_at_k1 = df_object.get_mo_values(
                    occupied_mo_coeff[:, k1_index], k1
                )

            for index, ao_range in enumerate(prange(0, n_occupied, p_slice)):
                p0, p1 = ao_range
                if index % mpi.comm.size != mpi.comm.rank:
                    continue

                for i in range(n_channels):

                    for contracted_mo_range in prange(0, n_occupied, q_slice):
                        q0, q1 = contracted_mo_range
                        if is_gamma_point(kpts):
                            coulomb_in_mo_pair = cp.einsum(
                                "pn, qn -> pqn",
                                occupied_mo_at_k2[i, p0:p1],
                                occupied_mo_at_k2[i, q0:q1],
                            ).reshape(p1 - p0, q1 - q0, *mesh)
                        else:
                            coulomb_in_mo_pair = cp.einsum(
                                "pn, qn, n -> pqn",
                                occupied_mo_at_k1[i, p0:p1],
                                occupied_mo_at_k2[i, q0:q1],
                                e_ikr,
                            ).reshape(p1 - p0, q1 - q0, *mesh)

                        t1 = log.timer_debug1("mo pair", *t1)

                        coulomb_in_mo_pair = fft_in_place(coulomb_in_mo_pair)

                        coulomb_in_mo_pair *= (
                            df_object.coulomb_kernel_on_g_mesh.reshape(*mesh)
                        )

                        coulomb_in_mo_pair = ifft_in_place(coulomb_in_mo_pair).reshape(
                            p1 - p0, q1 - q0, -1
                        )

                        if is_gamma_point(kpts):
                            coulomb_in_mo_pair = coulomb_in_mo_pair.real

                        coulomb_weighted_density_dot_ao_at_k2[i, p0:p1] += cp.einsum(
                            "pqn, qn , q -> pn",
                            coulomb_in_mo_pair,
                            occupied_mo_at_k2[i, q0:q1],
                            occupied_occupation_numbers[q0:q1],
                        )
                        if not is_gamma_point(kpts):
                            coulomb_weighted_density_dot_ao_at_k2[
                                i, p0:p1
                            ] *= e_ikr.conj()

            mpi.comm.reduce(coulomb_weighted_density_dot_ao_at_k2, in_place=True)

            fock_slice_in_occupied = df_object.contract_mo_values_to_fock(
                coulomb_weighted_density_dot_ao_at_k2, k1
            )

            fock_slice_in_occupied *= weight

            fock_slice = mo_to_ao[:, k1_index] @ fock_slice_in_occupied
            for i in range(n_channels):
                vk[i, k1_index] += (
                    fock_slice[i]
                    + fock_slice[i].conj().T
                    - fock_slice[i]
                    @ occupied_mo_coeff[i, k1_index]
                    @ mo_to_ao[i, k1_index].conj().T
                )
            t1 = log.timer_debug1("fock_slice", *t1)

    mpi.comm.reduce(vk, in_place=True)
    if exxdiv == "ewald":
        for i in range(n_channels):
            vk[i] += df_object.madelung * cp.einsum(
                "kpq, kqr, krs -> kps",
                df_object.overlap,
                formatted_density_matrices[i],
                df_object.overlap,
            )
    log.timer("get_k_kpts", *t0)
    return fft_jk._format_jks(vk, dm_kpts, None, kpts)


def get_veff_ip1(
    mydf,
    xc_code,
    dm_kpts,
    hermi=1,
    kpts=None,
    kpts_band=None,
    with_j=True,
    verbose=None,
):
    if kpts is None:
        kpts = mydf.kpts
    log = logger.new_logger(mydf, verbose)
    t0 = log.init_timer()
    cell = mydf.cell
    dm_kpts = cp.asarray(dm_kpts, order="C")
    dms = fft_jk._format_dms(dm_kpts, kpts)
    nset = dms.shape[0]
    kpts_band = _format_kpts_band(kpts_band, kpts)

    numerical_integrator = mydf._numint
    xc_type = numerical_integrator._xc_type(xc_code)

    if xc_type == "LDA":
        derivative_order = 0
    elif xc_type == "GGA":
        derivative_order = 1
    else:
        raise NotImplementedError

    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    density = evaluate_density_on_g_mesh(mydf, dm_kpts, kpts, derivative_order)
    mydf.rhoG = density.get()
    coulomb_on_g_mesh = cp.einsum(
        "ng, g -> g", density[:, 0], mydf.coulomb_kernel_on_g_mesh
    )

    weight = cell.vol / ngrids

    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    density = (
        cp.asarray(
            ifft_in_place(density.reshape(nset, -1, *mesh)).real,
            order="C",
            dtype=mydf.dtype,
        ).reshape(nset, -1, ngrids)
        / weight
    )

    if nset == 1:
        xc_for_fock = numerical_integrator.eval_xc_eff(
            xc_code, density[0], deriv=1, xctype=xc_type
        )[1]
    else:
        xc_for_fock = numerical_integrator.eval_xc_eff(
            xc_code, density, deriv=1, xctype=xc_type
        )[1]

    xc_for_fock = xc_for_fock.reshape(nset, -1, *mesh) * weight
    xc_for_fock = fft_in_place(xc_for_fock).reshape(nset, -1, ngrids)

    if xc_type == "GGA":
        xc_for_fock = (
            xc_for_fock[:, 0]
            - cp.einsum(
                "ngp, gp -> np", xc_for_fock[:, 1:], mydf.gradient_vector_on_g_mesh
            )
            * 1j
        ).reshape((nset, -1, ngrids))
    if with_j:
        xc_for_fock[:, 0] += coulomb_on_g_mesh

    if mydf.vpplocG_part1 is not None:
        xc_for_fock[:, 0] += mydf.vpplocG_part1

    veff_gradient = convert_xc_on_g_mesh_to_fock_gradient(
        mydf, xc_for_fock, dm_kpts, hermi, kpts_band
    )

    t0 = log.timer("veff_gradient", *t0)

    return veff_gradient


def get_ovlp_ip1(
    mydf,
    dme0,
    kpts=np.zeros((1, 3)),
):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t0 = log.init_timer()
    cell = mydf.cell
    dme0_kpts = fft_jk._format_dms(dme0, kpts)
    n_atoms = cell.natm
    n_set = dme0_kpts.shape[0]

    gradient = cp.zeros((n_atoms, 3), dtype=mydf.dtype)

    for pairs in mydf.sorted_gaussian_pairs:
        mesh = pairs["mesh"]
        n_grid_points = np.prod(mesh)
        weight = cell.vol / n_grid_points

        overlap_weight = cp.full((n_set, n_grid_points), weight, dtype=mydf.dtype)

        dme0_slice = dme0_kpts[
            :,
            :,
            pairs["ao_indices_in_localized"][:, None],
            pairs["concatenated_ao_indices"],
        ]

        dme0_slice_in_transposed_part = dme0_kpts[
            :,
            :,
            pairs["ao_indices_in_diffused"][:, None],
            pairs["ao_indices_in_localized"],
        ]

        n_ao_in_localized = len(pairs["ao_indices_in_localized"])
        dme0_slice[
            :, :, :, n_ao_in_localized:
        ] += dme0_slice_in_transposed_part.transpose(0, 1, 3, 2).conj()

        coeff_sandwiched_dme0 = cp.einsum(
            "nkij,pi->nkpj",
            dme0_slice,
            pairs["coeff_in_localized"],
        )

        coeff_sandwiched_dme0 = cp.einsum(
            "nkpj, qj -> nkpq",
            coeff_sandwiched_dme0,
            pairs["concatenated_coeff"],
        )

        libgpbc.update_dxyz_dabc(
            ctypes.cast(pairs["dxyz_dabc"].data.ptr, ctypes.c_void_p)
        )

        evaluate_xc_gradient_wrapper(
            gradient,
            pairs,
            overlap_weight,
            coeff_sandwiched_dme0,
            ignore_imag=True,
        )

    mpi.comm.reduce(gradient, in_place=True)

    t0 = log.timer("ovlp_ip1", *t0)

    return gradient


class FFTDF(fftdf_module.FFTDF, multigrid.MultiGridFFTDF):
    def __init__(
        self,
        cell,
        kpts=np.zeros((1, 3)),
        xc="LDA",
        p_slice=2,
        q_slice=32,
        use_float_precision=False,
    ):

        fftdf_module.FFTDF.__init__(self, cell, kpts)

        xc_type = self._numint._xc_type(xc)

        if use_float_precision:
            self.dtype = cp.float32
        else:
            self.dtype = cp.float64
        self.sorted_gaussian_pairs = None
        sort_gaussian_pairs(self, xc_type)
        self.gradient_vector_on_g_mesh = None
        if xc_type == "GGA":
            self.gradient_vector_on_g_mesh = cp.asarray(cell.get_Gv(self.mesh)).T

        self.coulomb_kernel_on_g_mesh = tools.get_coulG(cell, mesh=self.mesh)
        self._overlap = None
        self.p_slice = p_slice
        self.q_slice = q_slice
        self.vpplocG_part1 = None
        self.rhoG = None
        self._overlap = get_ovlp(self, kpts)

        if is_gamma_point(kpts):
            self._overlap = self._overlap.reshape(self._overlap.shape[-2:])

    def get_ovlp(self, cell=None, kpt=None):
        return self._overlap

    def get_mo_values(self, mo_coeff, kpt):
        n_grid_points = len(self.grids.coords)
        n_grid_points_per_process = int(np.ceil(n_grid_points / mpi.comm.size))
        grid_slice_start = mpi.comm.rank * n_grid_points_per_process
        grid_slice_end = min(
            grid_slice_start + n_grid_points_per_process, n_grid_points
        )

        ao_values_slice = self._numint.eval_ao(
            self.cell,
            self.grids.coords[grid_slice_start:grid_slice_end],
            kpt.reshape(1, 3),
        )[0]

        n_set, _, n_occupied = mo_coeff.shape
        grid_slice_size = grid_slice_end - grid_slice_start

        padded_mo_values = cp.zeros(
            (n_grid_points_per_process, n_occupied, n_set), dtype=mo_coeff.dtype
        )

        padded_mo_values[:grid_slice_size] = cp.einsum(
            "ipm, gp -> gmi", mo_coeff, ao_values_slice
        )

        gathered_mo_values = mpi.comm.gather(padded_mo_values)

        return gathered_mo_values[:n_grid_points].T

    def contract_mo_values_to_fock(self, mo_values, kpt):
        n_grid_points = len(self.grids.coords)
        n_grid_points_per_process = int(np.ceil(n_grid_points / mpi.comm.size))
        grid_slice_start = mpi.comm.rank * n_grid_points_per_process
        grid_slice_end = min(
            grid_slice_start + n_grid_points_per_process, n_grid_points
        )

        n_set = mo_values.shape[0]

        ao_values_slice = self._numint.eval_ao(
            self.cell,
            self.grids.coords[grid_slice_start:grid_slice_end],
            kpt.reshape(1, 3),
        )[0]

        fock = cp.asarray(
            [
                mo_values[i, :, grid_slice_start:grid_slice_end] @ ao_values_slice
                for i in range(n_set)
            ]
        )

        mpi.comm.reduce(fock, in_place=True)

        return fock

    def get_k(self, dm_kpts, hermi=1, kpt=np.zeros(3), kpts_band=None, exxdiv=None):
        self.n_grid_points = np.prod(self.mesh)
        self.madelung = madelung(self.cell, self.mesh)

        return get_k_kpts(
            self,
            dm_kpts,
            hermi,
            kpt.reshape(1, 3),
            kpts_band,
            exxdiv,
            self.p_slice,
            self.q_slice,
        )

    get_nuc = get_nuc
    get_pp = get_pp

    def get_veff_ip1(
        self, dm, xc_code=None, hermi=1, kpts=None, kpts_band=None, with_j=True
    ):
        return get_veff_ip1(self, xc_code, dm, hermi, kpts, kpts_band, with_j=with_j)

    def get_ovlp_ip1(self, dme0, kpts=None):
        return get_ovlp_ip1(self, dme0, kpts=kpts)

    vpploc_part1_nuc_grad = return_cupy_array(
        multigrid_parent.MultiGridFFTDF2.vpploc_part1_nuc_grad
    )


def fftdf(mf, use_float_precision=False):
    mf.with_df, old_df = (
        FFTDF(mf.cell, kpts=mf.kpts, xc=mf.xc, use_float_precision=use_float_precision),
        mf.with_df,
    )
    mf.with_df.__dict__.update(old_df.__dict__)
    mf.get_ovlp = mf.with_df.get_ovlp

    return mf
