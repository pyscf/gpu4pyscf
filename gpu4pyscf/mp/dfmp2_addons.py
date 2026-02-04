# # Addons for GPU MP2

# This is addon modules for GPU MP2.
#
# This will
# - implement some algorithms, which will perform on CPU or single-GPU.
# - not implement multi-GPU related algorithms.
# - not implement strong-correlated functions (must involve instance of DFMP2).

# +
import pyscf
import gpu4pyscf
import numpy as np
import scipy
import cupy
import cupy as cp

import cupyx.scipy.linalg

from pyscf import __config__
from gpu4pyscf.lib.cupy_helper import ndarray, contract

# +
CONFIG_USE_SCF_WITH_DF = getattr(__config__, 'gpu_mp_dfmp2_use_scf_with_df', False)
""" Flag for using cderi from SCF object (not implemented).

This option will be overrided if auxiliary basis set is explicitly specified.

- True: Use cderi from SCF object. This will override user-specified auxiliary basis.
- False: Always generate cderi.
"""

CONFIG_WITH_T2 = getattr(__config__, 'gpu_mp_dfmp2_with_t2', False)
""" Flag for computing T2 amplitude.

In many cases, this is not recommended, except for debugging.
Energy (or possibly gradient in future) can be computed without T2 amplitude.
"""

CONFIG_WITH_CDERI_OVL = getattr(__config__, 'gpu_mp_dfmp2_with_cderi_ovl', False)
""" Flag for save Cholesky decomposed 3c-2e ERI (occ-vir part). """

CONFIG_FP_TYPE = getattr(__config__, 'gpu_mp_dfmp2_fp_type', 'FP64')
""" Floating point type for MP2 calculation.

Currently only FP64 and FP32 are supported.
To use TF32, set environment variable ``CUPY_TF32=1`` before running python / importing cupy, and set ``FP32`` for this option.
Use TF32 with caution for RI-MP2. TF32 is not recommended when performing LT-OS-MP2.

- FP64: Double precision
- FP32: Single precision
"""

CONFIG_FP_TYPE_DECOMP = getattr(__config__, 'gpu_mp_dfmp2_same_fp_type_decomp', None)
""" Flag for using the same floating point type for decomposition.

Note that ERI is always generated in FP64. This only affects the decomposition.

- None: Use the same floating point type as the MP2 calculation.
- FP64: Double precision
- FP32: Single precision
"""

CONFIG_CDERI_ON_GPU = getattr(__config__, 'gpu_mp_dfmp2_cderi_on_gpu', True)
""" Flag for storing cderi (MO part) on GPU.

- None: (not implemented) Automatically choose based on the available GPU memory.
- True: Always storing cderi on GPU DRAM.
- False: Always storing cderi on CPU DRAM.
"""

CONFIG_J2C_ALG = getattr(__config__, 'gpu_mp_dfmp2_j2c_alg', 'cd')
""" Algorithm for j2c decomposition.

- "cd": Cholesky decomposition
- "eig": Eigen decomposition
"""

CONFIG_THRESH_LINDEP = getattr(__config__, 'mp_dfmp2_thresh_lindep', 1e-10)
""" Threshold for linear dependence detection of j2c. """
# -

MIN_BATCH_AUX_CPU = 32
MIN_BATCH_AUX_GPU = 32
BLKSIZE_AO = 128
CUTOFF_J3C = 1e-10


def balanced_split(a, n):
    v, r = divmod(a, n)
    lst = [v] * n
    for i in range(r):
        lst[i] += 1
    assert sum(lst) == a
    return lst


def wrapper_device(idx_device, func, *args, **kwargs):
    with cupy.cuda.Device(idx_device):
        return func(*args, **kwargs)


def get_avail_mem_devices(device_list=None):
    """Get available memory (in Byte) for all devices."""
    if device_list is None:
        device_list = [i for i in range(cupy.cuda.runtime.getDeviceCount())]
    avail_mem = []
    for device_id in device_list:
        with cupy.cuda.Device(device_id):
            avail_mem.append(gpu4pyscf.lib.cupy_helper.get_avail_mem())
    return avail_mem


def get_frozen_mask_restricted(mp, frozen=None, mo_occ=None):
    """Get boolean mask for the restricted reference orbitals.

    This will return numpy object, instead of cupy object.

    Args:
        mp: pyscf.lib.StreamObject

        frozen: int or list(int) or None

            - int: number of frozen occupied orbitals
            - list: frozen orbital indices

        mo_occ: np.ndarray
            Molecular occupation numbers

    Returns:
        moidx: np.ndarray
            Mask array of frozen (true) and active (false) orbitals.

    See also:
        pyscf.mp.mp2.get_frozen_mask
    """
    mo_occ = mp.mo_occ if mo_occ is None else mo_occ
    frozen = mp.frozen if frozen is None else frozen

    moidx = np.ones(mo_occ.size, dtype=bool)
    if hasattr(mp, '_nmo') and mp._nmo is not None:
        # frozen virtual orbitals by number
        moidx[mp._nmo :] = False
    if frozen is None:
        pass
    elif isinstance(frozen, (int, np.integer, cp.integer)):
        # frozen occupied orbitals by number
        moidx[: int(frozen)] = False
    elif len(frozen) > 0:
        # frozen orbitals by index list
        moidx[list(frozen)] = False
    else:
        raise NotImplementedError
    return moidx


def mo_splitter_restricted(mp, frozen=None, mo_occ=None):
    """Active orbital masks for the restricted reference orbitals.

    Args:
        mp: pyscf.lib.StreamObject

        frozen: int or list(int) or None

            - int: number of frozen occupied orbitals
            - list: frozen orbital indices

        mo_occ: np.ndarray
            Molecular occupation numbers.

    Returns:
        masks: list(np.ndarray)

            - occupied frozen
            - occupied active
            - virtual active
            - virtual frozen
    """
    mo_occ = mp.mo_occ if mo_occ is None else mo_occ
    frozen = mp.frozen if frozen is None else frozen
    if isinstance(mo_occ, cp.ndarray):
        mo_occ = mo_occ.get()
    mask_act = get_frozen_mask_restricted(mp, mo_occ=mo_occ, frozen=frozen)
    mask_occ = mo_occ > 1e-6
    masks = [
        mask_occ & ~mask_act,  # frz occ
        mask_occ & mask_act,  # act occ
        ~mask_occ & mask_act,  # act vir
        ~mask_occ & ~mask_act,  # frz vir
    ]
    return masks


def split_mo_coeff_restricted(mp, mo_coeff=None, frozen=None, mo_occ=None):
    mo_coeff = mp.mo_coeff if mo_coeff is None else mo_coeff
    masks = mo_splitter_restricted(mp, frozen=frozen, mo_occ=mo_occ)
    return [mo_coeff[:, mask] for mask in masks]


def split_mo_energy_restricted(mp, mo_energy=None, frozen=None, mo_occ=None):
    mo_energy = mp.mo_energy if mo_energy is None else mo_energy
    masks = mo_splitter_restricted(mp, frozen=frozen, mo_occ=mo_occ)
    return [mo_energy[mask] for mask in masks]


def get_dtype(type_token, is_gpu):
    if type_token.upper() == 'FP64':
        return cp.float64 if is_gpu else np.float64
    elif type_token.upper() == 'FP32':
        return cp.float32 if is_gpu else np.float32
    else:
        raise ValueError(f'Unknown type {type_token}')


def get_j2c_decomp_cpu(streamobj, j2c, alg=CONFIG_J2C_ALG, thresh_lindep=CONFIG_THRESH_LINDEP, verbose=None):
    r"""Get j2c decomposition in GPU.

    Given 2c-2e ERI (j2c) :math:`J_{PQ}`, decomposed j2c :math:`L_{PQ}` is defined as

    .. math::
        \sum_{R} L_{PR} L_{QR} = J_{PQ}

    This decomposition can be obtained by Cholesky decomposition or eigen decomposition.

    Args:
        streamobj: pyscf.lib.StreamObject

        j2c: np.ndarray
            2c-2e ERI, could be obtained from ``mol.intor("int2c2e")`` or other equilvants.

        alg: str
            Algorithm for decomposition.
            - "cd": Cholesky decomposition by default, eigen decomposition when scipy raises error
            - "eig": Eigen decomposition

        thresh_lindep: float
            Threshold for linear dependence detection of j2c.

        verbose: int

    Returns:
        dict

        j2c_l: np.ndarray
            Decomposed j2c. Shape (naux, naux).

        j2c_l_inv: np.ndarray
            Matrix inverse of ``j2c_l``. Only computed when algorithm is ``eig``.

        tag: str
            Algorithm for decomposition.

            - "cd": Cholesky decomposition
            - "eig": Eigen decomposition

    See also:
        get_j2c_decomp_cpu
    """
    log = pyscf.lib.logger.new_logger(streamobj, verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()

    # Cholesky decomposition
    if alg.lower().startswith('cd'):
        log.debug('j2c decomposition by Cholesky decomposition')
        j2c_l = scipy.linalg.cholesky(j2c, lower=True)
        if not np.isnan(j2c_l[0, 0]):
            # cupy does not raise error, but will give nan lower triangular on return
            log.timer('get_j2c_decomp by cd', *t0)
            return {
                'j2c_l': j2c_l,
                'tag': 'cd',
            }
        else:
            log.warn('j2c decomposition by Cholesky failed. Switching to eigen decomposition.')
            alg = 'eig'

    # Eigen decomposition
    if alg.lower().startswith('eig'):
        log.debug('j2c decomposition by eigen')
        e, u = scipy.linalg.eigh(j2c)
        cond = abs(e).max() / abs(e).min()
        keep = e > thresh_lindep
        rkeep = e < -thresh_lindep
        if rkeep.sum() > 0:
            log.warn(f'Some {rkeep.sum()} j2c eigenvalues are much smaller than zero, which is unexpected.')
        log.debug(f'cond(j2c) = {cond}')
        log.debug(f'keep {keep.sum()}/{keep.size} j2c vectors')
        e = e[keep]
        u = u[:, keep]
        j2c_l = u * e**0.5 @ u.T.conj()
        j2c_l_inv = u * e**-0.5 @ u.T.conj()
        log.timer('get_j2c_decomp by eig', *t0)
        return {
            'j2c_l': j2c_l,
            'j2c_l_inv': j2c_l_inv,
            'tag': 'eig',
        }
    else:
        raise ValueError(f'Unknown j2c decomposition algorithm: {alg}')


def get_j2c_decomp_gpu(streamobj, j2c, alg=CONFIG_J2C_ALG, thresh_lindep=CONFIG_THRESH_LINDEP, verbose=None):
    r"""Get j2c decomposition in GPU.

    Given 2c-2e ERI (j2c) :math:`J_{PQ}`, decomposed j2c :math:`L_{PQ}` is defined as

    .. math::
        \sum_{R} L_{PR} L_{QR} = J_{PQ}

    This decomposition can be obtained by Cholesky decomposition or eigen decomposition.

    Args:
        streamobj: pyscf.lib.StreamObject

        j2c: cp.ndarray
            2c-2e ERI, could be obtained from ``mol.intor("int2c2e")`` or other equilvants.

        alg: str
            Algorithm for decomposition.
            - "cd": Cholesky decomposition by default, eigen decomposition when scipy raises error
            - "eig": Eigen decomposition

        thresh_lindep: float
            Threshold for linear dependence detection of j2c.

        verbose: int

    Returns:
        dict

        j2c_l: cp.ndarray
            Decomposed j2c. Shape (naux, naux).

        j2c_l_inv: cp.ndarray
            Matrix inverse of ``j2c_l``. Only computed when algorithm is ``eig``.

        tag: str
            Algorithm for decomposition.

            - "cd": Cholesky decomposition
            - "eig": Eigen decomposition

    See also:
        get_j2c_decomp_cpu
    """
    log = pyscf.lib.logger.new_logger(streamobj, verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()

    # Cholesky decomposition
    if alg.lower().startswith('cd'):
        log.debug('j2c decomposition by Cholesky decomposition')
        j2c_l = cp.linalg.cholesky(j2c)
        if not cp.isnan(j2c_l[0, 0]):
            # cupy does not raise error, but will give nan lower triangular on return
            log.timer('get_j2c_decomp by cd', *t0)
            return {
                'j2c_l': j2c_l,
                'tag': 'cd',
            }
        else:
            log.warn('j2c decomposition by Cholesky failed. Switching to eigen decomposition.')
            alg = 'eig'

    # Eigen decomposition
    if alg.lower().startswith('eig'):
        log.debug('j2c decomposition by eigen')
        e, u = cp.linalg.eigh(j2c)
        cond = abs(e).max() / abs(e).min()
        keep = e > thresh_lindep
        rkeep = e < -thresh_lindep
        if rkeep.sum() > 0:
            log.warn(f'Some {rkeep.sum()} j2c eigenvalues are much smaller than zero, which is unexpected.')
        log.debug(f'cond(j2c) = {cond}')
        log.debug(f'keep {keep.sum()}/{keep.size} j2c vectors')
        e = e[keep]
        u = u[:, keep]
        j2c_l = u * e**0.5 @ u.T.conj()
        j2c_l_inv = u * e**-0.5 @ u.T.conj()
        log.timer('get_j2c_decomp by eig', *t0)
        return {
            'j2c_l': j2c_l,
            'j2c_l_inv': j2c_l_inv,
            'tag': 'eig',
        }
    else:
        raise ValueError(f'Unknown j2c decomposition algorithm: {alg}')


def get_j3c_by_shls_cpu(mol, aux, aux_slice=None, omega=None, out=None):
    """Generator of int3c2e on CPU.

    Please note that this will return lower-triangular packed int3c2e, with shape (naux, nao_tp).
    The returned matrix is c-contiguous.

    Args:
        mol: pyscf.gto.Mole
            Molecule object with normal basis set.

        aux: pyscf.gto.Mole
            Molecule object with auxiliary basis set.

        aux_slice: list(int) or None
            Shell slices to be computed at auxiliary basis.

        omega: float or None
            Range separate parameter.

        out: cp.ndarray

    Returns:
        np.ndarray
    """
    mol_concat = mol + aux
    nbas = mol.nbas
    nao = mol.nao
    naux = aux.nao
    nao_tp = nao * (nao + 1) // 2
    shlP0, shlP1 = aux_slice if aux_slice else (0, aux.nbas)
    shls_slice = (0, nbas, 0, nbas, nbas + shlP0, nbas + shlP1)
    if out is None:
        out = np.empty((nao_tp, naux), order='F')
    else:
        out = out.T
    with mol_concat.with_range_coulomb(omega):
        mol_concat.intor('int3c2e', aosym='s2ij', shls_slice=shls_slice, out=out)
    return out.T


def get_j3c_ovl_cart_gpu(intopt, occ_coeff_set, vir_coeff_set, j3c_ovl_cart_set, aux_batch_size, log=None):
    """
    Args:
        mol: pyscf.gto.Mole
        intopt: gpu4pyscf.df.int3c2e.VHFOpt
        occ_coeff_set: list[cp.ndarray]
        vir_coeff_set: list[cp.ndarray]
        j3c_ovl_cart_set: list[np.ndarray or cp.ndarray]
        aux_batch_size: int
        log: pyscf.lib.logger.Logger
    """
    mol = intopt.mol.mol

    # determine the number of tasks (spins/properties)
    nset = len(j3c_ovl_cart_set)
    assert len(occ_coeff_set) == len(vir_coeff_set) == nset

    nao_cart = mol.nao_cart()
    cache1 = cp.empty(nao_cart * nao_cart * aux_batch_size)
    cache2 = cp.empty(nao_cart * nao_cart * aux_batch_size)

    int3c2e_gen, aux_sorting, ao_pair_offsets, aux_offsets = intopt.int3c2e_evaluator(
        cart=True, reorder_aux=True, aux_batch_size=aux_batch_size
    )
    assert len(ao_pair_offsets) == 2, 'AO pair should not be sliced.'

    rows, cols = divmod(intopt.pair_and_diag_indices(cart=True, original_ao_order=False)[0], nao_cart)
    aux_resorting = np.argsort(aux_sorting)

    occ_coeff_cart_set = []
    vir_coeff_cart_set = []
    for iset in range(nset):
        occ_coeff_cart_set.append(intopt.mol.C_dot_mat(occ_coeff_set[iset]))
        vir_coeff_cart_set.append(intopt.mol.C_dot_mat(vir_coeff_set[iset]))

    for ibatch_aux, (p0, p1) in enumerate(zip(aux_offsets[:-1], aux_offsets[1:])):
        naux_batch = p1 - p0
        # step 1
        j3c_raw = int3c2e_gen(aux_batch_id=ibatch_aux, out=cache1)
        # step 2
        j3c_expand_pair = ndarray([nao_cart, nao_cart, naux_batch], buffer=cache2)
        j3c_expand_pair[:] = 0.0
        j3c_expand_pair[rows, cols] = j3c_raw
        j3c_expand_pair[cols, rows] = j3c_raw
        j3c_raw = None
        for iset in range(nset):
            occ_coeff_cart = occ_coeff_cart_set[iset]
            vir_coeff_cart = vir_coeff_cart_set[iset]
            nocc = occ_coeff_cart.shape[1]
            nvir = vir_coeff_cart.shape[1]
            # step 3
            j3c_obx_sorted = ndarray([nocc, nao_cart, naux_batch], buffer=cache1)
            contract('uvP, ui -> ivP', j3c_expand_pair, occ_coeff_cart, out=j3c_obx_sorted)
            # step 4
            j3c_ovl_sorted = ndarray([nocc, nvir, naux_batch], buffer=cache2)
            contract('ivP, va -> iaP', j3c_obx_sorted, vir_coeff_cart, out=j3c_ovl_sorted)
            j3c_obx_sorted = None
            # step 5
            j3c_ovl_cart_set[iset][:, :, aux_resorting[p0:p1]] = j3c_ovl_sorted
            j3c_ovl_sorted = None


def sph2cart_j3c_ovl(intopt, j3c_ovl_cart_set, batch_ov_size, j3c_ovl_set=None):
    aux = intopt.auxmol.mol
    naux = aux.nao
    cache = cp.empty([batch_ov_size, naux])

    # create j3c_ovl_set if not given
    if j3c_ovl_set is not None:
        assert len(j3c_ovl_set) == len(j3c_ovl_cart_set)
    else:
        j3c_ovl_set = []
        for j3c_ovl_cart in j3c_ovl_cart_set:
            nocc, nvir, _ = j3c_ovl_cart.shape
            j3c_ovl = ndarray([nocc, nvir, naux], buffer=j3c_ovl_cart)
            j3c_ovl_set.append(j3c_ovl)

    for j3c_ovl_cart, j3c_ovl in zip(j3c_ovl_cart_set, j3c_ovl_set):
        assert j3c_ovl_cart.ndim == 3
        nocc, nvir, naux_cart = j3c_ovl_cart.shape
        assert naux_cart == aux.nao_cart(), f'Auxiliary basis inconsistent: {naux_cart} != {aux.nao_cart()}.'
        j3c_ovl_cart = j3c_ovl_cart.reshape(nocc * nvir, naux_cart)
        j3c_ovl = j3c_ovl.reshape(nocc * nvir, naux)
        for idx_ov in range(0, nocc * nvir, batch_ov_size):
            nbatch_ov = min(nocc * nvir - idx_ov, batch_ov_size)
            slc = range(idx_ov, idx_ov + nbatch_ov)
            cache[:] = 0.0
            cache_ovl = intopt.auxmol.mat_dot_C(j3c_ovl_cart[slc], buffer=cache)
            j3c_ovl[slc] = cache_ovl
            cache_ovl = None
    return j3c_ovl_set


def get_j3c_ovl_gpu_bdiv(intopt, occ_coeff_set, vir_coeff_set, j3c_ovl_cart_set, aux_batch_size, log=None):
    get_j3c_ovl_cart_gpu(intopt, occ_coeff_set, vir_coeff_set, j3c_ovl_cart_set, aux_batch_size, log=log)
    return sph2cart_j3c_ovl(intopt, j3c_ovl_cart_set, 256)


def get_j3c_by_aux_id_gpu(mol, intopt, idx_k, omega=None, out=None):
    """Generator of int3c2e on GPU.

    This function only give 3-dimension ``int3c2e`` (k, j, i) in c-contiguous array.
    Currently, other integrals are not available.

    Args:
        mol: pyscf.gto.Mole
            Molecule object with normal basis set.

        intopt: gpu4pyscf.df.int3c2e.VHFOpt
            Integral optimizer object for 3c-2e ERI on GPU.

        idx_k: int
            Index of third index in 3c-2e ERI (most cases auxiliary basis).

        omega: float or None
            Range separate parameter.

        out: cp.ndarray

    Returns:
        cp.ndarray

    Example:
        Return value of function is dependent on how ``intopt`` is optimized, specifically size of ``group_size_aux``.

        .. code-block::
            mol = pyscf.gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="def2-TZVP", max_memory=6000).build()
            auxmol = pyscf.df.make_auxmol(mol, "def2-TZVP-ri")
            intopt = gpu4pyscf.df.int3c2e.VHFOpt(mol, auxmol, "int2e")
            intopt.build(diag_block_with_triu=True, aosym=True, group_size_aux=32)
            for idx_k in range(len(intopt.aux_log_qs)):
                j3c_batched = get_j3c_by_aux_id(mol, intopt, idx_k)
                print(j3c_batched.shape, j3c_batched.strides)
    """
    nao = mol.nao
    k0, k1 = intopt.aux_ao_loc[idx_k], intopt.aux_ao_loc[idx_k + 1]

    if out is None:
        out = cp.zeros([k1 - k0, nao, nao], order='C')

    for idx_ij, _ in enumerate(intopt.log_qs):
        cpi = intopt.cp_idx[idx_ij]
        cpj = intopt.cp_jdx[idx_ij]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]
        i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi + 1]
        j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj + 1]

        int3c_slice = cp.zeros([k1 - k0, j1 - j0, i1 - i0], order='C')
        int3c_slice = gpu4pyscf.df.int3c2e.get_j3c_slice(intopt, idx_ij, idx_k, out=int3c_slice, omega=omega)
        if not mol.cart:
            int3c_slice = gpu4pyscf.lib.cupy_helper.cart2sph(int3c_slice, axis=1, ang=lj)
            int3c_slice = gpu4pyscf.lib.cupy_helper.cart2sph(int3c_slice, axis=2, ang=li)
        i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi + 1]
        j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj + 1]
        out[:, j0:j1, i0:i1] = int3c_slice
    row, col = np.tril_indices(nao)
    out[:, row, col] = out[:, col, row]

    return out


def get_j3c_ovl_gpu(mol, intopt, occ_coeff, vir_coeff, j3c_ovl, log=None):
    """Inner function for generate and transform 3c-2e ERI to MO basis

    Args:
        mol: pyscf.gto.Mole
        intopt: gpu4pyscf.df.int3c2e.VHFOpt
        occ_coeff: list[cp.ndarray]
        vir_coeff: list[cp.ndarray]
        j3c_ovl: list[np.ndarray or cp.ndarray]
        log: pyscf.lib.logger.Logger
    """
    if log is None:
        log = pyscf.lib.logger.new_logger(mol, verbose=mol.verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()
    idx_device = cupy.cuda.get_device_id()

    nset = len(occ_coeff)
    assert len(occ_coeff) == len(vir_coeff) == len(j3c_ovl)
    j3c_on_gpu = isinstance(j3c_ovl[0], cp.ndarray)
    occ_coeff_sorted = [intopt.sort_orbitals(occ_coeff[iset], axis=[0]) for iset in range(nset)]
    vir_coeff_sorted = [intopt.sort_orbitals(vir_coeff[iset], axis=[0]) for iset in range(nset)]

    dtype = j3c_ovl[0].dtype

    for idx_p in range(len(intopt.aux_log_qs)):
        log.debug(
            f'processing auxiliary part {idx_p}/{len(intopt.aux_log_qs)} at device {idx_device}, len {len(intopt.aux_log_qs[idx_p])}'
        )
        if not mol.cart:
            p0, p1 = intopt.sph_aux_loc[idx_p], intopt.sph_aux_loc[idx_p + 1]
        else:
            p0, p1 = intopt.cart_aux_loc[idx_p], intopt.cart_aux_loc[idx_p + 1]
        # obtained j3c is (nbatch_aux, nao, nao)
        t1 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()
        j3c = get_j3c_by_aux_id_gpu(mol, intopt, idx_p)
        t1 = log.timer(f'get_j3c_by_aux_id_gpu at device {idx_device}', *t1)
        nbatch_aux, nao, _ = j3c.shape
        for iset in range(nset):
            co = cp.asarray(occ_coeff_sorted[iset])
            cv = cp.asarray(vir_coeff_sorted[iset])
            nocc, nvir = co.shape[1], cv.shape[1]
            # Puv, vi -> iPu
            j3c_half = co.T @ j3c.reshape(nbatch_aux * nao, nao).T
            j3c_half.shape = (nocc, nbatch_aux, nao)
            # iPu, ua -> iaP
            if j3c_on_gpu:
                for i in range(nocc):
                    j3c_ovl[iset][i, :, p0:p1] = cv.T @ j3c_half[i].T
            else:
                for i in range(nocc):
                    j3c_ovl[iset][i, :, p0:p1] = (cv.T @ j3c_half[i].T).astype(dtype).get(blocking=False)
            co = cv = j3c_half = None
        t1 = log.timer(f'in get_j3c_ovl_gpu, ao2mo at device {idx_device}', *t1)
        j3c = None
    log.timer(f'get_j3c_ovl_gpu at device {idx_device}', *t0)


def decompose_j3c(mol, j2c_decomp, j3c, log=None):
    """Inner function for decompose 3c-2e ERI (occ-vir part)

    Args:
        j2c_decomp: dict
        j3c: list[np.ndarray or cp.ndarray]
        log: pyscf.lib.logger.Logger
    """
    if log is None:
        log = pyscf.lib.logger.new_logger(mol, verbose=mol.verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()
    idx_device = cupy.cuda.get_device_id()

    j3c_on_gpu = isinstance(j3c[0], cp.ndarray)
    nset = len(j3c)
    naux = j3c[0].shape[2]
    dtype = j3c[0].dtype

    if j3c_on_gpu:
        # directly perform decomposition
        if j2c_decomp['tag'] == 'cd':
            j2c_l = cp.asarray(j2c_decomp['j2c_l'], dtype=dtype, order='F')
            # probably memory copy occurs due to c-contiguous array?
            for iset in range(nset):
                shape = j3c[iset].shape
                j3c[iset] = cupyx.scipy.linalg.solve_triangular(
                    j2c_l, j3c[iset].reshape((-1, naux)).T, lower=True, overwrite_b=True
                ).T.reshape(shape)
        elif j2c_decomp['tag'] == 'eig':
            j2c_l_inv = cp.asarray(j2c_decomp['j2c_l_inv'], dtype=dtype, order='C')
            for iset in range(nset):
                shape = j3c[iset].shape
                j3c[iset] = (j3c[iset].reshape((-1, naux)) @ j2c_l_inv).reshape(shape)
        else:
            raise ValueError(f'Unknown j2c decomposition tag: {j2c_decomp["tag"]}')
    else:
        cp.get_default_memory_pool().free_all_blocks()
        gpu_mem_avail = get_avail_mem_devices(device_list=[idx_device])[0]
        log.debug(f'Available GPU memory: {gpu_mem_avail / 1024**3:.6f} GB')
        fp_avail = 0.7 * gpu_mem_avail / min(j3c[0].strides)
        if j2c_decomp['tag'] == 'cd':
            j2c_l = cp.asarray(j2c_decomp['j2c_l'], dtype=dtype, order='F')
            for iset in range(nset):
                shape = j3c[iset].shape
                j3c[iset].shape = (-1, naux)
                n_ov = j3c[iset].shape[0]
                batch_ov = int(fp_avail / (4 * naux))
                log.debug(f'number of batched non-auxiliary indices: {batch_ov}')
                for i_ov in range(0, n_ov, batch_ov):
                    log.debug(f'load non-auxiliary index: {i_ov}/{n_ov}')
                    nbatch_ov = min(batch_ov, n_ov - i_ov)
                    j3c_batched = cp.asarray(j3c[iset][i_ov : i_ov + nbatch_ov])
                    j3c_batched = cupyx.scipy.linalg.solve_triangular(
                        j2c_l, j3c_batched.T, lower=True, overwrite_b=True
                    ).T
                    j3c_batched.get(out=j3c[iset][i_ov : i_ov + nbatch_ov], blocking=False)
                    j3c_batched = None
                j3c[iset].shape = shape
        elif j2c_decomp['tag'] == 'eig':
            j2c_l_inv = cp.asarray(j2c_decomp['j2c_l_inv'], dtype=dtype, order='C')
            for iset in range(nset):
                shape = j3c[iset].shape
                j3c[iset].shape = (naux, -1)
                n_ov = j3c[iset].shape[1]
                batch_ov = int(fp_avail / (4 * naux))
                for i_ov in range(0, n_ov, batch_ov):
                    nbatch_ov = min(batch_ov, n_ov - i_ov)
                    j3c_batched = cp.asarray(j3c[iset][i_ov : i_ov + nbatch_ov])
                    (j3c_batched @ j2c_l_inv).get(out=j3c[iset][i_ov : i_ov + nbatch_ov], blocking=False)
                    j3c_batched = None
                j3c[iset].shape = shape
        else:
            raise ValueError(f'Unknown j2c decomposition tag: {j2c_decomp["tag"]}')
    cupy.cuda.get_current_stream().synchronize()
    log.timer(f'decompose_j3c at device {idx_device}', *t0)


def handle_cderi_gpu(mol, intopt, j2c_decomp, occ_coeff, vir_coeff, j3c_gpu, j3c_cpu, log=None):
    if log is None:
        log = pyscf.lib.logger.new_logger(mol, verbose=mol.verbose)
    if j3c_gpu is None:
        get_j3c_ovl_gpu(mol, intopt, occ_coeff, vir_coeff, j3c_cpu, log=log)
        decompose_j3c(mol, j2c_decomp, j3c_cpu, log=log)
    else:
        get_j3c_ovl_gpu(mol, intopt, occ_coeff, vir_coeff, j3c_gpu, log=log)
        decompose_j3c(mol, j2c_decomp, j3c_gpu, log=log)
        # store j3c_gpu to j3c_cpu
        for j3c_gpu_item, j3c_cpu_item in zip(j3c_gpu, j3c_cpu):
            j3c_gpu_item.get(out=j3c_cpu_item, blocking=False)


def get_dfmp2_energy_pair_intra(mol, cderi_ovl, occ_energy, vir_energy, log=None):
    if log is None:
        log = pyscf.lib.logger.new_logger(mol, verbose=mol.verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()
    idx_device = cupy.cuda.get_device_id()

    if not isinstance(cderi_ovl, cp.ndarray):
        cderi_ovl = cp.asarray(cderi_ovl)

    occ_energy = cp.asarray(occ_energy, dtype=cp.float32)
    vir_energy = cp.asarray(vir_energy, dtype=cp.float32)
    nocc = len(occ_energy)
    nvir = len(vir_energy)
    naux = cderi_ovl.shape[2]
    assert cderi_ovl.shape == (nocc, nvir, naux)

    d_vv_gpu = -vir_energy[:, None] - vir_energy[None, :]
    eng_pair_bi1 = np.zeros([nocc, nocc])
    eng_pair_bi2 = np.zeros([nocc, nocc])
    for i in range(0, nocc):
        for j in range(0, i + 1):
            g_ab = cderi_ovl[i] @ cderi_ovl[j].T
            d_ab = occ_energy[i] + occ_energy[j] + d_vv_gpu
            t_ab = g_ab / d_ab
            e_bi1 = (t_ab * g_ab).sum()
            e_bi2 = (t_ab.T * g_ab).sum()
            eng_pair_bi1[i, j] = eng_pair_bi1[j, i] = float(e_bi1)
            eng_pair_bi2[i, j] = eng_pair_bi2[j, i] = float(e_bi2)
    log.timer(f'get_dfmp2_energy_pair_intra at device {idx_device}', *t0)
    return eng_pair_bi1, eng_pair_bi2


def get_dfmp2_energy_pair_inter(
    mol,
    cderi_ovl: cp.ndarray,
    occ_energy: np.ndarray,
    vir_energy: np.ndarray,
    cderi_ovl_host_list,  # list(np.ndarray)
    occ_energy_host_list,  # list(np.ndarray)
    eval_mode_list,  # list(bool or None)
    log=None,
):
    if log is None:
        log = pyscf.lib.logger.new_logger(mol, verbose=mol.verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()
    idx_device = cupy.cuda.get_device_id()

    if not isinstance(cderi_ovl, cp.ndarray):
        cderi_ovl = cp.asarray(cderi_ovl)

    # arrange molecular energy and dimensionality
    occ_energy = cp.asarray(occ_energy, dtype=cp.float32)
    vir_energy = cp.asarray(vir_energy, dtype=cp.float32)
    nocc = len(occ_energy)
    nvir = len(vir_energy)
    naux = cderi_ovl.shape[2]
    dtype = cderi_ovl.dtype

    # handle tasks for host->device
    # | T | F |
    # | F | T |
    cderi_ovl_host_view_list = []
    occ_energy_host_view_list = []
    occ_idx_host_list = []
    occ_idx_device_list = []
    nocc_device_split = nocc // 2
    nocc_full = 0
    for cderi_ovl_host, occ_energy_host, eval_mode in zip(cderi_ovl_host_list, occ_energy_host_list, eval_mode_list):
        nocc_host = occ_energy_host.shape[0]
        if eval_mode is not None:
            nocc_host = occ_energy_host.shape[0]
            nocc_split = nocc_host // 2
            cderi_ovl_host_view_list.append(cderi_ovl_host[:nocc_split])
            occ_energy_host_view_list.append(occ_energy_host[:nocc_split])
            occ_idx_host_list.append([nocc_full + i for i in range(nocc_split)])
            cderi_ovl_host_view_list.append(cderi_ovl_host[nocc_split:])
            occ_energy_host_view_list.append(occ_energy_host[nocc_split:])
            occ_idx_host_list.append([nocc_full + i for i in range(nocc_split, nocc_host)])
            if eval_mode is True:
                occ_idx_device_list.append([i for i in range(nocc_device_split)])
                occ_idx_device_list.append([i for i in range(nocc_device_split, nocc)])
            elif eval_mode is False:
                occ_idx_device_list.append([i for i in range(nocc_device_split, nocc)])
                occ_idx_device_list.append([i for i in range(nocc_device_split)])
        nocc_full += nocc_host
    ntask = len(cderi_ovl_host_view_list)

    d_vv_gpu = -vir_energy[:, None] - vir_energy[None, :]
    eng_pair_bi1 = np.zeros([nocc, nocc_full])
    eng_pair_bi2 = np.zeros([nocc, nocc_full])
    stream_task = cupy.cuda.stream.Stream(non_blocking=True)
    cderi_ovl_task = None
    with stream_task:
        cderi_ovl_next = cp.empty(cderi_ovl_host_view_list[0].shape, dtype=dtype)
        cderi_ovl_next.set(cderi_ovl_host_view_list[0], stream=stream_task)
    for itask in range(ntask):
        stream_task.synchronize()
        occ_energy_task = occ_energy_host_view_list[itask]
        occ_idx_task = occ_idx_host_list[itask]
        occ_device_task = occ_idx_device_list[itask]
        cderi_ovl_task = cderi_ovl_next
        with stream_task:
            if itask < ntask - 1:
                cderi_ovl_next = cp.empty(cderi_ovl_host_view_list[itask + 1].shape, dtype=dtype)
                cderi_ovl_next.set(cderi_ovl_host_view_list[itask + 1], stream=stream_task)
        for i in occ_device_task:
            for j, j_task in enumerate(occ_idx_task):
                g_ab = cderi_ovl[i] @ cderi_ovl_task[j].T
                d_ab = occ_energy[i] + occ_energy_task[j] + d_vv_gpu
                t_ab = g_ab / d_ab
                e_bi1 = (t_ab * g_ab).sum()
                e_bi2 = (t_ab.T * g_ab).sum()
                eng_pair_bi1[i, j_task] = float(e_bi1)
                eng_pair_bi2[i, j_task] = float(e_bi2)
    log.timer(f'get_dfmp2_energy_pair_inter at device {idx_device}', *t0)
    return eng_pair_bi1, eng_pair_bi2
