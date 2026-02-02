# cupyx/scipy/fft/__init__.py
"""
Minimal shim mapping cupyx.scipy.fft -> dpnp.fft
Exposes only: fftn, ifftn, fftfreq
"""

import dpnp as _xp
_xp_fft = _xp.fft


def fftn(a, s=None, axes=None, norm=None, overwrite_x=False, workers=None, plan=None):
    """CuPy/SciPy-compatible fftn; extra args are accepted but ignored."""
    return _xp_fft.fftn(a, s=s, axes=axes, norm=norm)


def ifftn(a, s=None, axes=None, norm=None, overwrite_x=False, workers=None, plan=None):
    """CuPy/SciPy-compatible ifftn; extra args are accepted but ignored."""
    return _xp_fft.ifftn(a, s=s, axes=axes, norm=norm)


def fftfreq(n, d=1.0):
    """CuPy/SciPy-compatible fftfreq."""
    return _xp_fft.fftfreq(n, d=d)

__all__ = ["fftn", "ifftn", "fftfreq"]
