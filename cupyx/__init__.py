"""Top-level ``cupyx`` for a gpu4pyscf checkout, backend-agnostic.

Mirrors ``cupy/__init__.py``: on a SYCL venv, load the dpnp-backed
``gpu4pyscf/cupyx/__init__.py`` by file path under the name ``cupyx``; on a
CUDA venv, defer to the real ``cupyx`` shipped with CuPy. See the ``cupy``
docstring for why ``<repo>/gpu4pyscf`` must stay off ``sys.path``.
"""
import importlib.machinery
import importlib.util
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dpnp_available():
    try:
        return importlib.util.find_spec("dpnp") is not None
    except Exception:
        return False


def _load_real(name):
    paths = [p for p in sys.path
             if p and os.path.realpath(p) != os.path.realpath(_REPO)]
    spec = importlib.machinery.PathFinder.find_spec(name, paths)
    if spec is None or spec.loader is None:
        return False
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return True


def _load_shim(name):
    pkg_dir = os.path.join(_REPO, "gpu4pyscf", name)
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    sys.modules["gpu4pyscf." + name] = sys.modules[name]
    _parent = sys.modules.get("gpu4pyscf")
    if _parent is not None:
        setattr(_parent, name, sys.modules[name])


if not _dpnp_available():
    if not _load_real("cupyx"):
        raise ImportError(
            "neither dpnp (SYCL) nor real cupyx (CUDA) is installed in this venv")
else:
    _load_shim("cupyx")
