# cupyx/scipy/sparse/__init__.py
# Fake cupyx.scipy.sparse — forwards to dpnp.scipy.sparse
from . import linalg
__all__ = ["linalg"]
