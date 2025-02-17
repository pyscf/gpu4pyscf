import ctypes
import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import load_library
from gpu4pyscf.df.int3c2e import sort_mol, basis_seg_contraction

libecp = load_library('libgecp')

libecp.ECPtype1_cart.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int
]

libecp.ECPtype2_cart.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int
]


def sort_ecp(mol0, cart=True, log=None):
    '''
    # Sort basis according to angular momentum and contraction patterns so
    # as to group the basis functions to blocks in GPU kernel.
    '''
    if log is None:
        log = logger.new_logger(mol0, mol0.verbose)
    mol = mol0.copy(deep=True)
    l_ctrs = mol._ecpbas[:,[gto.ANG_OF, gto.NPRIM_OF]]
    uniq_l_ctr, _, inv_idx, l_ctr_counts = np.unique(
        l_ctrs, return_index=True, return_inverse=True, return_counts=True, axis=0)

    if mol.verbose >= logger.DEBUG:
        log.debug1('Number of shells for each [l, nctr] group')
        for l_ctr, n in zip(uniq_l_ctr, l_ctr_counts):
            log.debug('    %s : %s', l_ctr, n)

    sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)

    # Sort basis inplace
    mol._ecpbas = mol._ecpbas[sorted_idx]
    return mol, sorted_idx, uniq_l_ctr, l_ctr_counts

def get_ecp(mol):

    _mol = basis_seg_contraction(mol, allow_replica=True)[0]
    _sorted_mol, sorted_idx, uniq_l_ctr, l_ctr_counts = sort_mol(_mol)
    _sorted_mol, sorted_idx, uniq_lecp_ctr, lecp_ctr_counts = sort_ecp(_sorted_mol)

    print(uniq_l_ctr)
    print(l_ctr_counts)

    atm = cp.asarray(_sorted_mol._atm)
    bas = cp.asarray(_sorted_mol._bas)
    env = cp.asarray(_sorted_mol._env)

    ecpbas = cp.asarray(_sorted_mol._ecpbas)

    for li, lj, lecp in zip(uniq_l_ctr[:,0], uniq_l_ctr[:,0], uniq_lecp_ctr[:,0]):
        #tasks = [[i,j,k]
        #tasks = cp.asarray(tasks, dtype=np.int32)
        #ntasks =
        libecp.ECPtype1_cart(
            mat1.data.ptr, tasks.data.ptr, ntasks,
            ecpbas.data.ptr, ecploc.data.ptr,
            atm.data.ptr, bas.data.ptr, env.data.ptr,
            li, lj)

        libecp.ECPtype2_cart(
            mat1.data.ptr, tasks.data.ptr, ntasks,
            ecpbas.data.ptr, ecploc.data.ptr,
            atm.data.ptr, bas.data.ptr, env.data.ptr,
            li, lj)
