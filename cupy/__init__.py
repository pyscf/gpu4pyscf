"""Top-level ``cupy`` for a gpu4pyscf checkout, backend-agnostic.

With only the repo root on ``sys.path`` (``PYTHONPATH=<repo>``), ``import
cupy`` lands here:

* SYCL venv (dpnp installed): load the dpnp-backed facade from
  ``gpu4pyscf/cupy/__init__.py`` directly under the name ``cupy`` (the facade
  registers itself and its ``cuda``/``fft``/``testing`` submodules in
  ``sys.modules``). Loading by file path -- not via ``import gpu4pyscf.cupy``
  -- avoids a circular import through ``gpu4pyscf/__init__``.
* CUDA venv (no dpnp): defer to the real CuPy installed in site-packages,
  so one checkout coexists with both backends and no extra PYTHONPATH
  entries are needed.

IMPORTANT: ``<repo>/gpu4pyscf`` must NOT be on ``sys.path``. With pytest's
``consider_namespace_packages`` (pytest.toml) that entry makes test modules
under ``gpu4pyscf/<pkg>/tests/`` import as ``<pkg>.tests.test_x`` (in either
PYTHONPATH order), executing the package ``__init__`` a second time as a
top-level package -- for dft this rebinds ``scf.hf.KohnShamDFT`` via
dft/rks.py and ``gen_response`` silently drops fxc. Keeping only ``<repo>``
on the path avoids that for both CUPY and DPNP runs.
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
    """Load a module installed outside this repo (e.g. site-packages)."""
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
    """Load gpu4pyscf/<name>/__init__.py under the top-level name `name`."""
    pkg_dir = os.path.join(_REPO, "gpu4pyscf", name)
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # in place before exec: recursive imports see it
    spec.loader.exec_module(mod)
    # gpu4pyscf/cupy's tail re-registers sys.modules['cupy'] to its facade;
    # keep whatever the shim installed, else keep the module we loaded.
    sys.modules["gpu4pyscf." + name] = sys.modules[name]
    _parent = sys.modules.get("gpu4pyscf")
    if _parent is not None:
        setattr(_parent, name, sys.modules[name])


if not _dpnp_available():
    if not _load_real("cupy"):
        raise ImportError(
            "neither dpnp (SYCL) nor real cupy (CUDA) is installed in this venv")
else:
    _load_shim("cupy")
