# cupyx/scipy/__init__.py

from . import fft      # expose cupyx.scipy.fft
from . import sparse   # expose cupyx.scipy.sparse
from . import special  # expose cupyx.scipy.special

__all__ = ["fft", "sparse", "special"]
