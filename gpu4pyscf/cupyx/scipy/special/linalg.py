# cupyx/scipy/sparse/linalg.py
# Shim: re-export dpnp equivalents under the cupyx namespace.
from dpnp.scipy.sparse.linalg import (
        LinearOperator,
        minres,
        cg,
        gmres,
    )

__all__ = ["LinearOperator", "minres", "cg", "gmres"]
