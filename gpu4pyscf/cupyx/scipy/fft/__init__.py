# cupyx/scipy/fft/__init__.py
"""
Minimal shim mapping cupyx.scipy.fft -> dpnp.fft
Exposes only: fftn, ifftn, fftfreq

`overwrite_x=True` must behave the way CuPy behaves: for a complex
in-place-capable transform CuPy writes the result into the input buffer and
returns that same array.  Callers rely on this *semantically*, not only as a
memory optimization -- e.g. gpu4pyscf/pbc/dft/multigrid_v3.py's
`_vxc_to_reciprocal_space` does

    fft_in_place(work)      # return value discarded
    out += work             # expects `work` to now hold fft(work)

dpnp.fft always allocates a new output array, so without the write-back below
that code adds the *untransformed* real-space potential to the reciprocal-space
potential (observed: PBE0 + multigrid total energy off by ~3e-3 Ha).
"""

import dpnp as _xp

_xp_fft = _xp.fft


def _overwrite(a, res):
    """Emulate CuPy's in-place transform: copy `res` back into `a`.

    Only done when it is actually possible (same shape and dtype, `a` is a
    dpnp array).  Otherwise the freshly allocated result is returned, which
    matches CuPy's behaviour when it cannot transform in place either.
    """
    if res is a:
        return res
    if not isinstance(a, _xp.ndarray):
        return res
    if a.shape != res.shape or a.dtype != res.dtype:
        return res
    try:
        a[...] = res
    except Exception:
        return res
    # The temporary `res` buffer was just read by a device copy kernel; make
    # sure its release is queue-ordered (deferred-free layer) rather than the
    # eager sycl::free dpctl would otherwise do when it is collected.
    try:
        from gpu4pyscf.cupy.cuda import _register_deferred_free
        _register_deferred_free(res)
    except Exception:
        pass
    return a


def fftn(a, s=None, axes=None, norm=None, overwrite_x=False, workers=None, plan=None):
    """CuPy/SciPy-compatible fftn; `workers`/`plan` are accepted but ignored."""
    res = _xp_fft.fftn(a, s=s, axes=axes, norm=norm)
    if overwrite_x:
        return _overwrite(a, res)
    return res


def ifftn(a, s=None, axes=None, norm=None, overwrite_x=False, workers=None, plan=None):
    """CuPy/SciPy-compatible ifftn; `workers`/`plan` are accepted but ignored."""
    res = _xp_fft.ifftn(a, s=s, axes=axes, norm=norm)
    if overwrite_x:
        return _overwrite(a, res)
    return res


def fftfreq(n, d=1.0):
    """CuPy/SciPy-compatible fftfreq."""
    return _xp_fft.fftfreq(n, d=d)


__all__ = ["fftn", "ifftn", "fftfreq"]
