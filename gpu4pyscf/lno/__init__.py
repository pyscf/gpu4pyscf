"""GPU fragment restricted CCSD solver and molecular LNO integration.

Imports are lazy to permit `python -m gpu_lno.solver` without runpy warnings.
"""
__all__ = ('LNOCCSDGPUBridge','run_live_fragment_ccsd','RestrictedCCSD','read_fixed_las')


def __getattr__(name):
    if name in ('LNOCCSDGPUBridge','run_live_fragment_ccsd'):
        from . import bridge
        return getattr(bridge,name)
    if name in ('RestrictedCCSD','read_fixed_las'):
        from . import solver
        return getattr(solver,name)
    raise AttributeError(name)
