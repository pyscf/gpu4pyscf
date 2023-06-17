import numpy
import cupy
from functools import reduce

def _get_err_vec_orig(s, d, f):
    s = cupy.asarray(s)
    d = cupy.asarray(d)
    f = cupy.asarray(f)

    '''error vector = SDF - FDS'''
    if isinstance(f, cupy.ndarray) and f.ndim == 2:
        sdf = reduce(cupy.dot, (s,d,f))
        errvec = (sdf.conj().T - sdf).ravel()

    elif isinstance(f, cupy.ndarray) and f.ndim == 3 and s.ndim == 3:
        errvec = []
        for i in range(f.shape[0]):
            sdf = reduce(cupy.dot, (s[i], d[i], f[i]))
            errvec.append((sdf.conj().T - sdf).ravel())
        errvec = cupy.hstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        #probably want to avoid multiple conversion between numpy and cupy
        errvec = cupy.hstack([
            _get_err_vec_orig(s, d[0], f[0]).ravel(),
            _get_err_vec_orig(s, d[1], f[1]).ravel()])
    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return cupy.asnumpy(errvec)

def _get_err_vec_orth(s, d, f, Corth):
    '''error vector in orthonormal basis = C.T.conj() (SDF - FDS) C'''
    # Symmetry information to reduce numerical error in DIIS (issue #1524)
    print("gpu_replaced get err vec")
    orbsym = getattr(Corth, 'orbsym', None)
    if orbsym is not None:
        sym_forbid = orbsym[:,None] != orbsym

    s = cupy.asarray(s)
    d = cupy.asarray(d)
    f = cupy.asarray(f)
    Corth = cupy.asarray(Corth)

    if isinstance(f, cupy.ndarray) and f.ndim == 2:
        sdf = reduce(cupy.dot, (Corth.conj().T, s, d, f, Corth))
        if orbsym is not None:
            sdf[sym_forbid] = 0
        errvec = (sdf.conj().T - sdf).ravel()

    elif isinstance(f, cupy.ndarray) and f.ndim == 3 and s.ndim == 3:
        errvec = []
        for i in range(f.shape[0]):
            sdf = reduce(cupy.dot, (Corth[i].conj().T, s[i], d[i], f[i], Corth[i]))
            if orbsym is not None:
                sdf[sym_forbid] = 0
            errvec.append((sdf.conj().T - sdf).ravel())
        errvec = cupy.vstack(errvec).ravel()

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = cupy.hstack([
            cupy.asarray(_get_err_vec_orth(s, d[0], f[0], Corth[0]).ravel()),
            cupy.asarray(_get_err_vec_orth(s, d[1], f[1], Corth[1]).ravel())])
    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return cupy.asnumpy(errvec)


