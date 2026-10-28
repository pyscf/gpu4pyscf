try:
    from pyscf.dispersion import __version__
except ModuleNotFoundError as err:
    raise ImportError(
        'Dispersion corrections require pyscf-dispersion. '
        'Install it with pip install pyscf-dispersion, or gpu4pyscf-cuda??x[dispersion].') from err
