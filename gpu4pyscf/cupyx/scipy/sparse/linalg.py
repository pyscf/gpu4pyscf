# gpu4pyscf/gpu4pyscf/cupyx/scipy/sparse/linalg.py
#
# Shim: re-export dpnp equivalents under the cupyx namespace, with a
# tol -> rtol translation for minres since cupyx uses `tol` but dpnp
# (matching modern scipy) uses `rtol`.

from dpnp.scipy.sparse.linalg import (
    LinearOperator,
    cg,
    gmres,
    minres as _dpnp_minres,
)



def minres(A, b, x0=None, *, shift=0.0, tol=1e-5, maxiter=None,
           M=None, callback=None, check=False):
    """cupyx-style minres. Translates tol -> rtol for dpnp."""
    return _dpnp_minres(
        A, b, x0,
        rtol=tol,
        shift=shift,
        maxiter=maxiter,
        M=M,
        callback=callback,
        check=check,
    )


__all__ = ["LinearOperator", "minres", "cg", "gmres"]
