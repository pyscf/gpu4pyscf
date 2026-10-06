# cupyx/scipy/special/__init__.py
# Shim: re-export dpnp.scipy.special under the cupyx namespace.
from dpnp.scipy import special as _dp_special
from dpnp.scipy.special import *  # noqa: F401,F403
__all__ = [n for n in dir(_dp_special) if not n.startswith("_")]
