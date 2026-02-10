"""
Addons for GPU MP2.

This will
- implement some algorithms, which will perform on CPU or single-GPU.
- not implement multi-GPU related algorithms.
- does not involve classes, only algorithms are implemented as functions.
- configuration options for DF-MP2 are defined here.
"""

import pyscf
import gpu4pyscf
import numpy as np
import scipy
import cupy
import cupy as cp

import cupyx.scipy.linalg

from pyscf import __config__
from gpu4pyscf.lib.cupy_helper import ndarray, contract

# region configurations

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

This option only affects the tensor contraction step (the bottleneck of energy evaluation).
This option does not affect integral accuracy, j3c storage and cholesky decomposition.

Currently only FP64 and FP32 are supported.

In most cases, FP32 is sufficiently accurate for energy evaluation (< 0.1 kcal/mol). However, we still use FP64 as default.

To use TF32, set bash environment variable ``CUPY_TF32=1`` before running python / importing cupy, and set ``FP32`` for this option.
Use TF32 with caution for RI-MP2. TF32 is not recommended when performing LT-OS-MP2.

- 'FP64': Double precision
- 'FP32': Single precision
"""

CONFIG_FP_TYPE_DECOMP = getattr(__config__, 'gpu_mp_dfmp2_same_fp_type_decomp', 'FP64')
""" Flag for using the same floating point type for decomposition.

Note that ERI is always generated in FP64. This only affects the decomposition.

- None: Use the same floating point type as the MP2 calculation.
- 'FP64': Double precision
- 'FP32': Single precision
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

MIN_BATCH_AUX_CPU = 32
MIN_BATCH_AUX_GPU = 32
BLKSIZE_AO = 128
CUTOFF_J3C = 1e-10

# endregion configurations

# region Utility functions


def balanced_split(a, n):
    """Split integer `a` into `n` balanced integers.

    Parameters
    ----------
    a : int
    n : int

    Returns
    -------
    list of int

    Examples
    --------
    >>> balanced_split(10, 3)
    [4, 3, 3]
    """
    v, r = divmod(a, n)
    lst = [v] * n
    for i in range(r):
        lst[i] += 1
    assert sum(lst) == a
    return lst


def wrapper_device(idx_device, func, *args, **kwargs):
    """Wrapper to run function on specified device.

    This function is mostly used for submit job to `ThreadPoolExecutor`, where it only accepts function but not closure
    (local variables are diffcult to be passed into `ThreadPoolExecutor`).

    Parameters
    ----------
    idx_device : int
        GPU device index.
    func : callable
        Function to run on the specified device.
    """
    with cupy.cuda.Device(idx_device):
        return func(*args, **kwargs)


def get_avail_mem_devices(device_list=None):
    """Get available memory (in Byte) for all devices.

    Parameters
    ----------
    device_list : list of int, optional
        List of device indices to query. If None, all available devices are queried.

    Returns
    -------
    list of int
        Available memory (in Byte) for each device.
    """
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

    Parameters
    ----------
    mp : pyscf.lib.StreamObject
        Any object (usually Moller-Plesset object) that has ``mo_occ`` and ``frozen`` attributes.
    frozen : int | list of int | None, optional
        - int: number of frozen occupied orbitals.
        - list of int: frozen orbital indices.
        - None: no frozen orbitals.
        - by default use ``mp.frozen`` if defined.
    mo_occ : np.ndarray, optional
        Molecular occupation list, by default use ``mp.mo_occ`` if defined.

    Returns
    -------
    np.ndarray
        Boolean mask for active orbitals (True for frozen, False for active).

    See also
    --------
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

    Parameters see also ``get_frozen_mask_restricted``.

    Parameters
    ----------
    mp : pyscf.lib.StreamObject
    frozen : int | list of int | None, optional
    mo_occ : np.ndarray, optional

    Returns
    -------
    list of np.ndarray
        List of boolean masks for the following orbital groups:
        - frozen occupied
        - active occupied
        - active virtual
        - frozen virtual
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
    """Split molecular orbital coefficients for the restricted reference orbitals.

    Parameters
    ----------
    mp : pyscf.lib.StreamObject
    mo_coeff : np.ndarray, optional
        Molecular orbital coefficients, by default use ``mp.mo_coeff`` if defined.
        This must be of shape (nao, nmo).
    frozen : int | list of int | None, optional
    mo_occ : np.ndarray, optional

    Returns
    -------
    list of np.ndarray
        List of molecular orbital coefficients for the following orbital groups:
        - frozen occupied
        - active occupied
        - active virtual
        - frozen virtual
    """
    mo_coeff = mp.mo_coeff if mo_coeff is None else mo_coeff
    masks = mo_splitter_restricted(mp, frozen=frozen, mo_occ=mo_occ)
    return [mo_coeff[:, mask] for mask in masks]


def split_mo_energy_restricted(mp, mo_energy=None, frozen=None, mo_occ=None):
    """Split molecular orbital energies for the restricted reference orbitals.

    Parameters
    ----------
    mp : pyscf.lib.StreamObject
    mo_energy : np.ndarray, optional
        Molecular orbital energies, by default use ``mp.mo_energy`` if defined.
    frozen : int | list of int | None, optional
    mo_occ : np.ndarray, optional

    Returns
    -------
    list of np.ndarray
        List of molecular orbital energies for the following orbital groups:
        - frozen occupied
        - active occupied
        - active virtual
        - frozen virtual
    """
    mo_energy = mp.mo_energy if mo_energy is None else mo_energy
    masks = mo_splitter_restricted(mp, frozen=frozen, mo_occ=mo_occ)
    return [mo_energy[mask] for mask in masks]


def get_dtype(type_token):
    """Get numpy dtype from type token.

    Parameters
    ----------
    type_token : str
        Type token, could be 'FP64' or 'FP32'.
    """
    if type_token.upper() == 'FP64':
        return np.float64
    elif type_token.upper() == 'FP32':
        return np.float32
    else:
        raise ValueError(f'Unknown type {type_token}')


# endregion Utility functions

# region j2c and decomp


def get_j2c_vhfopt(vhfopt):
    """Get sorted 2c-2e ERI (j2c) from VHFOpt object.

    The orbital sequence is different to CPU's ``intor`` convention, sorted along with j3c creation conventional using VHFOpt.

    Parameters
    ----------
    vhfopt : gpu4pyscf.df.int3c2e.VHFOpt
        VHFOpt object to generate 2c-2e ERI, which also contains orbital-sorting information.

    Returns
    -------
    cp.ndarray
        2c-2e ERI on GPU.
    """
    mol, aux = vhfopt.mol, vhfopt.auxmol
    j2c = pyscf.df.incore.fill_2c2e(mol, aux)
    j2c = vhfopt.sort_orbitals(j2c, aux_axis=[0, 1])
    j2c = cp.asarray(j2c, order='C')
    return j2c


def get_j2c_bdiv(intopt):
    """Get 2c-2e ERI (j2c) from Int3c2eOpt object.

    The orbital sequence is the same to CPU's ``intor`` convention.

    Parameters
    ----------
    intopt : gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt
        Int3c2eOpt object to generate 2c-2e ERI.

    Returns
    -------
    cp.ndarray
        2c-2e ERI on GPU.
    """
    aux = intopt.auxmol.mol
    return gpu4pyscf.df.int3c2e_bdiv.int2c2e(aux)


def get_j2c_decomp_cpu(streamobj, j2c, alg=CONFIG_J2C_ALG, thresh_lindep=CONFIG_THRESH_LINDEP, log=None):
    """Get j2c decomposition in CPU (scipy implementation of ``get_j2c_decomp``)."""
    if log is None:
        log = pyscf.lib.logger.new_logger(streamobj, verbose=streamobj.verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()

    # Cholesky decomposition
    # SciPy will raise error when j2c is not positive definite
    if alg.lower().startswith('cd'):
        log.debug('j2c decomposition by Cholesky decomposition')
        try:
            j2c_l = scipy.linalg.cholesky(j2c, lower=True)
            log.timer('get_j2c_decomp by cd', *t0)
            return {
                'j2c_l': j2c_l,
                'tag': 'cd',
            }
        except np.linalg.LinAlgError:
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


def get_j2c_decomp_gpu(streamobj, j2c, alg=CONFIG_J2C_ALG, thresh_lindep=CONFIG_THRESH_LINDEP, log=None):
    """Get j2c decomposition in GPU (cupy implementation of ``get_j2c_decomp``)."""
    if log is None:
        log = pyscf.lib.logger.new_logger(streamobj, verbose=streamobj.verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()

    # Cholesky decomposition
    # cupy does not raise error, but will give nan
    if alg.lower().startswith('cd'):
        log.debug('j2c decomposition by Cholesky decomposition')
        j2c_l = cp.linalg.cholesky(j2c)
        if not cp.isnan(j2c_l).any():
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


def get_j2c_decomp(streamobj, j2c, alg=CONFIG_J2C_ALG, thresh_lindep=CONFIG_THRESH_LINDEP, log=None):
    r"""Get j2c decomposition.

    Given 2c-2e ERI (j2c) :math:`J_{PQ}`, decomposed j2c :math:`L_{PQ}` is defined as

    .. math::
        \sum_{R} L_{PR} L_{QR} = J_{PQ}

    This decomposition can be obtained by Cholesky decomposition or eigen decomposition.

    Parameters
    ----------
    streamobj : pyscf.lib.StreamObject
        Any stream object for logging.
    j2c : np.ndarray | cp.ndarray
        2c-2e ERI, could be obtained from ``mol.intor("int2c2e")`` or other equilvants.
    alg : str, optional
        Algorithm for decomposition.
        - "cd": Cholesky decomposition by default, eigen decomposition when scipy raises error
        - "eig": Eigen decomposition
    thresh_lindep : float, optional
        Threshold for linear dependence detection of j2c.
    log : pyscf.lib.logger.Logger, optional
        Logger. If None, a new logger will be created with verbosity level from ``streamobj.verbose``.

    Returns
    -------
    dict
        Dictionary containing the decomposition results.
        - j2c_l : np.ndarray | cp.ndarray
            Decomposed j2c. Shape (naux, naux).
        - j2c_l_inv : np.ndarray | cp.ndarray
            Matrix inverse of ``j2c_l``. Only computed when algorithm is ``"eig"``. Shape (naux, naux).
        - tag : str
            Algorithm for decomposition.
            - "cd": Cholesky decomposition
            - "eig": Eigen decomposition

    See also
    --------
    get_j2c_decomp_cpu
    get_j2c_decomp_gpu

    Examples
    --------
    >>> # assumes j2c is a numpy array
    >>> j2c_decomp = dfmp2_addons.get_j2c_decomp(mol, j2c, "cd", thresh_lindep=1e-15)
    >>> j2c_l = j2c_decomp['j2c_l']
    >>> j2c_rebuild = j2c_l @ j2c_l.T
    >>> np.allclose(j2c, j2c_rebuild)
    True
    """
    if isinstance(j2c, cp.ndarray):
        return get_j2c_decomp_gpu(streamobj, j2c, alg=alg, thresh_lindep=thresh_lindep, log=log)
    else:
        return get_j2c_decomp_cpu(streamobj, j2c, alg=alg, thresh_lindep=thresh_lindep, log=log)


def decompose_j3c_gpu(streamobj, j2c_decomp, j3c, log=None):
    """3c-2e ERI decomposition on GPU inplace.

    Parameters
    ----------
    streamobj : pyscf.lib.StreamObject
        Any stream object for logging.
    j2c_decomp : dict
        Decomposition results of j2c, obtained from ``get_j2c_decomp``.
    j3c : list of np.ndarray | list of cp.ndarray
        3c-2e ERI, could be obtained from ``mol.intor("int3c2e")`` or other equilvants.
        This function requires auxiliary index to be the last index, in C-contiguous order.
    log : pyscf.lib.logger.Logger, optional
        Logger. If None, a new logger will be created with verbosity level from ``streamobj.verbose``.
    """
    if log is None:
        log = pyscf.lib.logger.new_logger(streamobj, verbose=streamobj.verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()
    idx_device = cupy.cuda.get_device_id()

    # check strides
    j3c_strides = j3c[0].strides
    if j3c_strides[-1] != min(j3c_strides):
        raise ValueError('The last index of j3c must be the auxiliary index (C-contiguous order).')

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
                j3c[iset] = cupyx.scipy.linalg.solve_triangular(j2c_l, j3c[iset].reshape((-1, naux)).T, lower=True, overwrite_b=True).T.reshape(shape)
        elif j2c_decomp['tag'] == 'eig':
            j2c_l_inv = cp.asarray(j2c_decomp['j2c_l_inv'], dtype=dtype, order='C')
            for iset in range(nset):
                shape = j3c[iset].shape
                j3c[iset] = (j3c[iset].reshape((-1, naux)) @ j2c_l_inv).reshape(shape)
        else:
            raise ValueError(f'Unknown j2c decomposition tag: {j2c_decomp["tag"]}')
    else:
        cp.get_default_memory_pool().free_all_blocks()
        gpu_mem_avail = gpu4pyscf.lib.cupy_helper.get_avail_mem()
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
                    j3c_batched = cupyx.scipy.linalg.solve_triangular(j2c_l, j3c_batched.T, lower=True, overwrite_b=True).T
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


# endregion j2c and decomp

# region j3c (bdiv-kernel)


def get_j3c_by_shls_cpu(mol, aux, aux_slice=None, omega=None, out=None):
    """Get 3c-2e ERI (j3c) in CPU by specified shell slices.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object with normal basis set.
    aux : pyscf.gto.Mole
        Molecule object with auxiliary basis set.
    aux_slice : list of int | None, optional
        Shell slices to be computed at auxiliary basis.
    omega : float | None, optional
        Range separate parameter.
    out : np.ndarray | None, optional
        Output array to store the results. If None, a new array will be created.

    Returns
    -------
    np.ndarray
        3c-2e ERI matrix in lower-triangular packed form with shape (naux, nao_tp) in C-contiguous order,
        where nao_tp refers to number of triangular-packed AO pair.
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


def estimate_j3c_batch(streamobj, nao_cart, naux, mem_avail=None, prefactor=0.8, log=None):
    """Estimate the batch size for j3c computation based on available memory.

    Parameters
    ----------
    streamobj : pyscf.lib.StreamObject
        Any stream object for logging.
    nao_cart : int
        Number of the number of atomic orbitals in cartesian.
    mem_avail : float
        Available memory in MB.
    prefactor : float, optional
        Prefactor for memory usage estimation, by default 0.8.

    Returns
    -------
    aux_batch_size : int
        Estimated batch size for auxiliary basis (used in ``get_j3c_ovl_cart_bdiv_gpu``).
    batch_ov_size : int
        Estimated batch size for occupied-virtual pair (used in ``sph2cart_j3c_ovl``).
    """
    if log is None:
        log = pyscf.lib.logger.new_logger(streamobj, verbose=streamobj.verbose)

    if mem_avail is None:
        mem_avail = gpu4pyscf.lib.cupy_helper.get_avail_mem() / 1024**2  # in MB

    # get_j3c_ovl_cart_bdiv_gpu
    # cache1: (nao_cart * nao_cart * aux_batch_size) in FP64
    # cache2: (nao_cart * nao_cart * aux_batch_size) in FP64

    # available floats in FP64
    nflop_avail = mem_avail * 1024**2 / 8 * prefactor
    # batch size for auxiliary basis
    aux_batch_size = int(nflop_avail // (2 * nao_cart**2))
    if aux_batch_size < MIN_BATCH_AUX_GPU:
        log.warn(
            f'Estimated batch size for auxiliary basis is {aux_batch_size}, which is smaller than the minimum {MIN_BATCH_AUX_GPU}. '
            'This may lead to out-of-memory error.'
        )
    aux_batch_size = max(aux_batch_size, MIN_BATCH_AUX_GPU)

    # sph2cart_j3c_ovl
    # cache: (batch_ov_size * naux) in FP64
    batch_ov_size = int(nflop_avail // naux)
    if batch_ov_size < MIN_BATCH_AUX_GPU:
        log.warn(
            f'Estimated batch size for occupied-virtual pair is {batch_ov_size}, which is smaller than the minimum {MIN_BATCH_AUX_GPU}. '
            'This may lead to out-of-memory error.'
        )
    batch_ov_size = max(batch_ov_size, MIN_BATCH_AUX_GPU)
    return aux_batch_size, batch_ov_size


def get_j3c_ovl_cart_bdiv_gpu(intopt, occ_coeff_set, vir_coeff_set, j3c_ovl_cart_set, aux_batch_size, log=None):
    """Get 3-center overlap integrals in Cartesian basis on GPU by batch of auxiliary basis.

    Parameters
    ----------
    intopt : gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt
    occ_coeff_set : list of cupy.ndarray | list of numpy.ndarray
    vir_coeff_set : list of cupy.ndarray | list of numpy.ndarray
    j3c_ovl_cart_set : list of cupy.ndarray | list of numpy.ndarray
    aux_batch_size : int | None
    log : pyscf.lib.logger.Logger, optional

    See also
    --------
    get_j3c_ovl_gpu_bdiv
    """
    mol = intopt.mol.mol
    aux = intopt.auxmol.mol
    on_gpu = isinstance(j3c_ovl_cart_set[0], cp.ndarray)
    if log is None:
        log = pyscf.lib.logger.new_logger(mol, verbose=mol.verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()
    t1 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()

    # determine the number of tasks (spins/properties)
    nset = len(j3c_ovl_cart_set)
    assert len(occ_coeff_set) == len(vir_coeff_set) == nset

    # check dimensionality of input arrays
    for occ_coeff, vir_coeff, j3c_ovl_cart in zip(occ_coeff_set, vir_coeff_set, j3c_ovl_cart_set):
        nao = mol.nao
        naux_cart = aux.nao_cart()
        nocc, nvir, _ = j3c_ovl_cart.shape
        assert occ_coeff.shape == (nao, nocc)
        assert vir_coeff.shape == (nao, nvir)
        assert j3c_ovl_cart.shape[2] == naux_cart

    # allocate temporary buffers
    nao_cart = mol.nao_cart()
    cache1 = cp.empty(nao_cart * nao_cart * aux_batch_size, dtype=np.float64)
    cache2 = cp.empty(nao_cart * nao_cart * aux_batch_size, dtype=np.float64)

    # if all auxiliary basis can fit into memory, we should not use a finite value to split batch
    aux_batch_size_evaluator = aux_batch_size if aux_batch_size < aux.nao_cart() else None
    int3c2e_gen, aux_sorting, ao_pair_offsets, aux_offsets = intopt.int3c2e_evaluator(cart=True, reorder_aux=True, aux_batch_size=aux_batch_size_evaluator)
    assert len(ao_pair_offsets) == 2, 'AO pair should not be sliced.'

    rows, cols = divmod(intopt.pair_and_diag_indices(cart=True, original_ao_order=False)[0], nao_cart)
    aux_resorting = np.argsort(aux_sorting)
    if not on_gpu:
        aux_resorting = aux_resorting.get()

    # transform MO coefficients to cartesian basis (also apply AO sorting)
    occ_coeff_cart_set = []
    vir_coeff_cart_set = []
    for iset in range(nset):
        occ_coeff_cart_set.append(intopt.mol.C_dot_mat(occ_coeff_set[iset]))
        vir_coeff_cart_set.append(intopt.mol.C_dot_mat(vir_coeff_set[iset]))

    t1 = log.timer_debug1('prepare for j3c_ovl_cart_bdiv', *t1)

    nbatch_aux = len(aux_offsets) - 1
    for ibatch_aux, (p0, p1) in enumerate(zip(aux_offsets[:-1], aux_offsets[1:])):
        naux_batch = p1 - p0
        # step 1: evaluate compressed 3c-2e ERI
        t1 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()
        j3c_raw = int3c2e_gen(aux_batch_id=ibatch_aux, out=cache1)
        cupy.cuda.stream.get_current_stream().synchronize()
        t1 = log.timer_debug1(f'compute int3c2e for aux batch {ibatch_aux}/{nbatch_aux}', *t1)
        # step 2: decompress to full 3c-2e ERI (3-dimension tensor)
        j3c_expand_pair = ndarray([nao_cart, nao_cart, naux_batch], dtype=np.float64, buffer=cache2)
        j3c_expand_pair[:] = 0.0
        j3c_expand_pair[rows, cols] = j3c_raw
        j3c_expand_pair[cols, rows] = j3c_raw
        j3c_raw = None
        cupy.cuda.stream.get_current_stream().synchronize()
        t1 = log.timer_debug1(f'decompress int3c2e for aux batch {ibatch_aux}/{nbatch_aux}', *t1)
        for iset in range(nset):
            occ_coeff_cart = occ_coeff_cart_set[iset]
            vir_coeff_cart = vir_coeff_cart_set[iset]
            nocc = occ_coeff_cart.shape[1]
            nvir = vir_coeff_cart.shape[1]
            # step 3: contract over first AO index (nao_cart -> nocc)
            j3c_obx_sorted = ndarray([nocc, nao_cart, naux_batch], dtype=np.float64, buffer=cache1)
            contract('uvP, ui -> ivP', j3c_expand_pair, occ_coeff_cart, out=j3c_obx_sorted)
            # step 4: contract over second AO index (nao_cart -> nvir)
            j3c_ovl_sorted = ndarray([nocc, nvir, naux_batch], dtype=np.float64, buffer=cache2)
            contract('ivP, va -> iaP', j3c_obx_sorted, vir_coeff_cart, out=j3c_ovl_sorted)
            j3c_obx_sorted = None
            # step 5: inplace store the result with aux resorting
            if on_gpu:
                j3c_ovl_cart_set[iset][:, :, aux_resorting[p0:p1]] = j3c_ovl_sorted
            else:
                j3c_ovl_cart_set[iset][:, :, aux_resorting[p0:p1]] = j3c_ovl_sorted.get(blocking=False)
            j3c_ovl_sorted = None
            cupy.cuda.stream.get_current_stream().synchronize()
            t1 = log.timer_debug1(f'contract for aux batch {ibatch_aux}/{nbatch_aux}, set {iset}/{nset}', *t1)
    t0 = log.timer('get_j3c_ovl_cart_bdiv_gpu', *t0)


def sph2cart_j3c_ovl_bdiv(intopt, j3c_ovl_cart_set, batch_ov_size, j3c_ovl_set=None, log=None):
    """Apply AO transformation (sorting, sph2cart) for j3c_ovl_cart_set in batch of occupied-virtual pairs.

    Parameters
    ----------
    intopt : gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt
    j3c_ovl_cart_set : list of cupy.ndarray | list of numpy.ndarray
    batch_ov_size : int
    j3c_ovl_set : list of cupy.ndarray | list of numpy.ndarray, optional
    log : pyscf.lib.logger.Logger, optional

    Returns
    -------
    list of cupy.ndarray | list of numpy.ndarray

    See also
    --------
    get_j3c_ovl_gpu_bdiv
    """
    mol = intopt.mol.mol
    aux = intopt.auxmol.mol
    if log is None:
        log = pyscf.lib.logger.new_logger(mol, verbose=mol.verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()

    naux = aux.nao
    cache = cp.empty([batch_ov_size, naux], dtype=np.float64)
    on_gpu = isinstance(j3c_ovl_set[0], cp.ndarray) if j3c_ovl_set is not None else isinstance(j3c_ovl_cart_set[0], cp.ndarray)

    # create j3c_ovl_set if not given
    if j3c_ovl_set is not None:
        assert len(j3c_ovl_set) == len(j3c_ovl_cart_set)
    else:
        j3c_ovl_set = []
        for j3c_ovl_cart in j3c_ovl_cart_set:
            nocc, nvir, _ = j3c_ovl_cart.shape
            if on_gpu:
                j3c_ovl = ndarray([nocc, nvir, naux], dtype=j3c_ovl_cart.dtype, buffer=j3c_ovl_cart)
            else:
                j3c_ovl = np.ndarray(shape=[nocc, nvir, naux], dtype=j3c_ovl_cart.dtype, buffer=j3c_ovl_cart)
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
            if on_gpu:
                j3c_ovl[slc] = cache_ovl
            else:
                j3c_ovl[slc] = cache_ovl.get(blocking=False)
            cache_ovl = None
    cupy.cuda.stream.get_current_stream().synchronize()
    t0 = log.timer('sph2cart_j3c_ovl_bdiv', *t0)
    return j3c_ovl_set


def get_j3c_ovl_gpu_bdiv(
    intopt,
    occ_coeff_set,
    vir_coeff_set,
    j3c_ovl_cart_set,
    j3c_ovl_set=None,
    aux_batch_size=None,
    batch_ov_size=None,
    log=None,
):
    """Generate 3c-2e ERI (j3c) using block-divergent kernel.

    The generated j3c will be in (nocc, nvir, naux) shape, with the same convention to CPU's j3c (no orbital resorting).

    However, API caller should preallocate the buffer ``j3c_ovl_cart_set`` for the intermediate 3c-2e ERI in Cartesian basis, which will be used for the
    block-divergent kernel.

    Parameters
    ----------
    intopt : gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt
        Integral optimizer handler for 3c-2e ERI on GPU.
    occ_coeff_set : list of cupy.ndarray | list of numpy.ndarray
        List of occupied molecular orbital coefficients.
    vir_coeff_set : list of cupy.ndarray | list of numpy.ndarray
        List of virtual molecular orbital coefficients.
    j3c_ovl_cart_set : list of cupy.ndarray | list of numpy.ndarray
        List of 3-center overlap integrals in Cartesian basis, of shape (nocc, nvir, naux_cart).
    j3c_ovl_set : list of cupy.ndarray | list of numpy.ndarray, optional
        List of 3-center overlap integrals, of shape (nocc, nvir, naux).
        This buffer will be output; if None, it will reuse the buffer of ``j3c_ovl_cart_set`` to save memory.
    aux_batch_size : int | None
        Auxiliary basis batch size. If None, use all auxiliary basis at once.
    batch_ov_size : int | None, optional
        Batch size for occupied-virtual pairs. If None, this will try to estimate an optimal size.
    log : pyscf.lib.logger.Logger, optional
        Logger object for logging. If None, a new logger will be created with verbosity level from `intopt.mol.verbose`.

    Returns
    -------
    j3c_ovl_set : list of cupy.ndarray | list of numpy.ndarray
        List of 3-center overlap integrals in spherical basis, of shape (nocc, nvir, naux).

    Notes on Signature
    ------------------
    - Number of list (``nset``) determines the number of tasks (spins/properties).
      ``occ_coeff_set``, ``vir_coeff_set``, and ``j3c_ovl_cart_set`` should have the same length of ``nset``.
    - Though ``j3c_ovl_cart_set`` is purely output, this parameter is required to determine the data type (numpy or cupy, FP64/FP32).
      It should be pre-allocated before calling this function.
    """
    mol = intopt.mol.mol
    aux = intopt.auxmol.mol
    if log is None:
        log = pyscf.lib.logger.new_logger(mol, verbose=mol.verbose)

    nao_cart = mol.nao_cart()
    naux = aux.nao
    aux_batch_size_estimate, batch_ov_size_estimate = estimate_j3c_batch(mol, nao_cart, naux, log=log)
    aux_batch_size = aux_batch_size or aux_batch_size_estimate
    batch_ov_size = batch_ov_size or batch_ov_size_estimate
    log.debug(f'Estimation for j3c batch: aux_batch_size={aux_batch_size_estimate}, batch_ov_size={batch_ov_size_estimate}')
    get_j3c_ovl_cart_bdiv_gpu(intopt, occ_coeff_set, vir_coeff_set, j3c_ovl_cart_set, aux_batch_size, log=log)
    j3c_ovl_set = sph2cart_j3c_ovl_bdiv(intopt, j3c_ovl_cart_set, batch_ov_size, j3c_ovl_set=j3c_ovl_set, log=log)
    return j3c_ovl_set


# endregion j3c (bdiv-kernel)

# region j3c (old-kernel)


def get_j3c_by_aux_id_gpu(vhfopt, idx_k, omega=None, out=None):
    """Generator of int3c2e on GPU.

    Parameters
    ----------
    streamobj : pyscf.lib.StreamObject
        Any stream object for logging.
    vhfopt : gpu4pyscf.df.int3c2e.VHFOpt
        Integral optimizer handler for 3c-2e ERI on GPU.
    idx_k : int
        Index of the auxiliary basis batch.
    omega : float, optional
        Frequency parameter for frequency-dependent integrals, by default None.
    out : cupy.ndarray, optional
        Output buffer for the integrals, by default None.

    Returns
    -------
    cupy.ndarray
        3c-2e ERI matrix with shape (naux_batch, nao, nao) in C-contiguous order.

    Examples
    --------
    >>> mol = pyscf.gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="def2-TZVP", max_memory=6000).build()
    >>> auxmol = pyscf.df.make_auxmol(mol, "def2-TZVP-ri")
    >>> intopt = gpu4pyscf.df.int3c2e.VHFOpt(mol, auxmol, "int2e")
    >>> intopt.build(diag_block_with_triu=True, aosym=True, group_size_aux=32)
    >>> for idx_k in range(len(intopt.aux_log_qs)):
    >>>     j3c_batched = get_j3c_by_aux_id_gpu(intopt, idx_k)
    >>>     print(j3c_batched.shape, j3c_batched.strides)
    (16, 43, 43) (14792, 344, 8)
    (24, 43, 43) (14792, 344, 8)
    (6, 43, 43) (14792, 344, 8)
    (15, 43, 43) (14792, 344, 8)
    (15, 43, 43) (14792, 344, 8)
    (21, 43, 43) (14792, 344, 8)
    (9, 43, 43) (14792, 344, 8)
    """
    mol = vhfopt.mol
    nao = mol.nao
    k0, k1 = vhfopt.aux_ao_loc[idx_k], vhfopt.aux_ao_loc[idx_k + 1]

    if out is None:
        out = cp.zeros([k1 - k0, nao, nao], order='C')

    for idx_ij, _ in enumerate(vhfopt.log_qs):
        cpi = vhfopt.cp_idx[idx_ij]
        cpj = vhfopt.cp_jdx[idx_ij]
        li = vhfopt.angular[cpi]
        lj = vhfopt.angular[cpj]
        i0, i1 = vhfopt.cart_ao_loc[cpi], vhfopt.cart_ao_loc[cpi + 1]
        j0, j1 = vhfopt.cart_ao_loc[cpj], vhfopt.cart_ao_loc[cpj + 1]

        int3c_slice = cp.zeros([k1 - k0, j1 - j0, i1 - i0], order='C')
        int3c_slice = gpu4pyscf.df.int3c2e.get_int3c2e_slice(vhfopt, idx_ij, idx_k, out=int3c_slice, omega=omega)
        if not mol.cart:
            int3c_slice = gpu4pyscf.lib.cupy_helper.cart2sph(int3c_slice, axis=1, ang=lj)
            int3c_slice = gpu4pyscf.lib.cupy_helper.cart2sph(int3c_slice, axis=2, ang=li)
        i0, i1 = vhfopt.ao_loc[cpi], vhfopt.ao_loc[cpi + 1]
        j0, j1 = vhfopt.ao_loc[cpj], vhfopt.ao_loc[cpj + 1]
        out[:, j0:j1, i0:i1] = int3c_slice
    row, col = np.tril_indices(nao)
    out[:, row, col] = out[:, col, row]

    return out


def get_j3c_ovl_gpu_vhfopt(streamobj, vhfopt, occ_coeff, vir_coeff, j3c_ovl, log=None):
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
        log = pyscf.lib.logger.new_logger(streamobj, verbose=streamobj.verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()
    idx_device = cupy.cuda.get_device_id()

    mol = vhfopt.mol
    nset = len(occ_coeff)
    assert len(occ_coeff) == len(vir_coeff) == len(j3c_ovl)
    j3c_on_gpu = isinstance(j3c_ovl[0], cp.ndarray)
    occ_coeff_sorted = [vhfopt.sort_orbitals(occ_coeff[iset], axis=[0]) for iset in range(nset)]
    vir_coeff_sorted = [vhfopt.sort_orbitals(vir_coeff[iset], axis=[0]) for iset in range(nset)]

    dtype = j3c_ovl[0].dtype

    for idx_p in range(len(vhfopt.aux_log_qs)):
        log.debug(f'processing auxiliary part {idx_p}/{len(vhfopt.aux_log_qs)} at device {idx_device}, len {len(vhfopt.aux_log_qs[idx_p])}')
        if not mol.cart:
            p0, p1 = vhfopt.sph_aux_loc[idx_p], vhfopt.sph_aux_loc[idx_p + 1]
        else:
            p0, p1 = vhfopt.cart_aux_loc[idx_p], vhfopt.cart_aux_loc[idx_p + 1]
        # obtained j3c is (nbatch_aux, nao, nao)
        t1 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()
        j3c = get_j3c_by_aux_id_gpu(vhfopt, idx_p)
        t1 = log.timer(f'get_j3c_by_aux_id_gpu at device {idx_device}', *t1)
        nbatch_aux, nao, _ = j3c.shape
        for iset in range(nset):
            co = cp.asarray(occ_coeff_sorted[iset])
            cv = cp.asarray(vir_coeff_sorted[iset])
            nocc = co.shape[1]
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


def handle_cderi_gpu_vhfopt(streamobj, intopt, j2c_decomp, occ_coeff, vir_coeff, j3c_gpu, j3c_cpu, log=None):
    if log is None:
        log = pyscf.lib.logger.new_logger(streamobj, verbose=streamobj.verbose)
    if j3c_gpu is None:
        get_j3c_ovl_gpu_vhfopt(streamobj, intopt, occ_coeff, vir_coeff, j3c_cpu, log=log)
        decompose_j3c_gpu(streamobj, j2c_decomp, j3c_cpu, log=log)
    else:
        get_j3c_ovl_gpu_vhfopt(streamobj, intopt, occ_coeff, vir_coeff, j3c_gpu, log=log)
        decompose_j3c_gpu(streamobj, j2c_decomp, j3c_gpu, log=log)
        # store j3c_gpu to j3c_cpu
        for j3c_gpu_item, j3c_cpu_item in zip(j3c_gpu, j3c_cpu):
            j3c_gpu_item.get(out=j3c_cpu_item, blocking=False)


# endregion j3c (old-kernel)

# region mp2 energy pair


def get_dfmp2_energy_pair_intra(streamobj, cderi_ovl, occ_energy, vir_energy, log=None):
    r"""Obtain MP2 occupied orbital pair energies (intra GPU device).

    This function only handles one component (``nset=1``).
    To handle multiple components, the caller should call this function multiple times and arrange the results accordingly.

    Parameters
    ----------
    streamobj : pyscf.lib.StreamObject
        Any stream object for logging.
    cderi_ovl : cp.ndarray | np.ndarray
        Cholesky-decomposed 3c-2e ERI in MO basis, of shape (nocc, nvir, naux).
    occ_energy : cp.ndarray | np.ndarray
        Occupied orbital energies, of shape (nocc,).
    vir_energy : cp.ndarray | np.ndarray
        Virtual orbital energies, of shape (nvir,).
    log : pyscf.lib.Logger, optional
        Logger object for logging, by default None.

    Returns
    -------
    eng_pair_bi1 : np.ndarray
        Bi-orthogonal pair energies for the first term, of shape (nocc, nocc).

        .. math::
            E_{ij}^\textrm{bi1} = \sum_{ab} t_{ij}^{ab} g_{ij}^{ab} / D_{ij}^{ab}

    eng_pair_bi2 : np.ndarray
        Bi-orthogonal pair energies for the second term, of shape (nocc, nocc).

        .. math::
            E_{ij}^\textrm{bi2} = \sum_{ab} t_{ij}^{ba} g_{ij}^{ab} / D_{ij}^{ab}
    """
    if log is None:
        log = pyscf.lib.logger.new_logger(streamobj, verbose=streamobj.verbose)
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
    r"""Obtain MP2 occupied orbital pair energies (one occ-index in GPU, another occ-index in CPU by list).

    This function evaluates $E_{ij}^\textrm{bi1}$ and $E_{ij}^\textrm{bi2}$.
    However, it should be noted that index $i$ is in GPU (``cderi_ovl``), while index $j$ is in CPU (``cderi_ovl_host_list``).
    """
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


# endregion mp2 energy pair
