from pyscf import lib
libcupy_helper = lib.load_library('libcupy_helper')

# for i > j of 2d mat, mat[j,i] = mat[i,j]
def hermi_triu(mat, hermi=1, inplace=True):
    '''Use the elements of the lower triangular part to fill the upper triangular part.
    See also pyscf.lib.hermi_triu
    '''
    if not inplace:
        mat = mat.copy('C')
    assert mat.flags.c_contiguous

    if mat.ndim == 2:
        n = mat.shape[0]
        counts = 1
    elif mat.ndim == 3:
        counts, n = mat.shape[:2]
    else:
        raise ValueError(f'dimension not supported {mat.ndim}')

    libcupy_helper.CPdsymm_triu(
        ctypes.cast(mat.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n), ctypes.c_int(counts))
    return mat
