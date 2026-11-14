"""Top-level ``cupy`` for a gpu4pyscf checkout. Exists for the SYCL backend only.

With the repo root on ``sys.path`` (``PYTHONPATH=<repo>``, running from the
repo root, or pytest's rootdir conftest), ``import cupy`` lands here:

* CUDA venv (real CuPy installed): this module turns itself into the real
  CuPy in place and nothing here is used. Real CuPy always wins, even if dpnp
  is also installed. The module object is kept rather than swapped in
  ``sys.modules``, because CuPy imports ``cupyx`` through importlib's
  LazyLoader, which raises if a lazily loaded module replaces itself.
* SYCL venv (no real CuPy, dpnp installed): load the dpnp-backed facade from
  ``gpu4pyscf/cupy/__init__.py`` directly under the name ``cupy`` (the facade
  registers itself and its ``cuda``/``fft``/``testing`` submodules in
  ``sys.modules``). Loading by file path -- not via ``import gpu4pyscf.cupy``
  -- avoids a circular import through ``gpu4pyscf/__init__``.

These forwarders are never installed (setup.py excludes them).

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
# sys.path entries through which this forwarder or the dpnp shim is reachable.
_SHADOW_DIRS = {os.path.realpath(_REPO),
                os.path.realpath(os.path.join(_REPO, "gpu4pyscf"))}


def _dpnp_available():
    try:
        return importlib.util.find_spec("dpnp") is not None
    except Exception:
        return False


def _find_real(name):
    """Spec of `name` installed outside this checkout (e.g. site-packages)."""
    paths = [p for p in sys.path
             if p and os.path.realpath(p) not in _SHADOW_DIRS]
    spec = importlib.machinery.PathFinder.find_spec(name, paths)
    if spec is None or spec.loader is None:
        return None
    return spec


def _become_real(spec):
    """Run the real package's __init__ in this module object.

    The current __spec__ object is updated rather than replaced: when CuPy
    loads us lazily, the LazyLoader keeps its state on that object and reads
    it again on any attribute access made while the real __init__ runs.
    """
    code = spec.loader.get_code(spec.name)
    g = globals()
    for k in list(g):
        if k not in ('__name__', '__builtins__', '__spec__', '__loader__'):
            del g[k]
    own = g['__spec__']
    own.origin = spec.origin
    own.submodule_search_locations = list(spec.submodule_search_locations)
    own.cached = spec.cached
    g.update(__file__=spec.origin, __path__=own.submodule_search_locations,
             __package__=spec.parent, __cached__=spec.cached, __doc__=None)
    exec(code, g)


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


_real_spec = _find_real(__name__)
if _real_spec is not None:
    _become_real(_real_spec)
elif _dpnp_available():
    _load_shim(__name__)
else:
    raise ImportError(
        "neither real cupy (CUDA) nor dpnp (SYCL) is installed in this venv")
