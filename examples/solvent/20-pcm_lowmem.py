'''
Run solvent models on large systems with limited GPU memory.

By default, the cavity surface and intermediates of PCM (and its derived
implicit solvent models) are stored in GPU memory. For large molecular
systems, these require a significant amount of GPU memory.

The "lowmem_intermediate_storage" flag can enable the lowmem mode. The lowmem
mode reconstructs the solvent intermediates on the fly and solves the
electrostatic equations using sparse linear algebra, substantially reducing GPU
memory usage. This mode is slower than the default dense implementation but
allows larger systems to be treated on memory-limited GPUs.

This option is independent of the DFT low-memory mode and can be enabled
regardless of the DFT algorithm.
'''

import pyscf

mol = pyscf.M(
    atom="""
O  0.0000   0.7375  -0.0528
O  0.0000  -0.7375  -0.1528
H  0.8190   0.8170   0.4220
H -0.8190  -0.8170   0.4220
""",
    basis="def2-svp",
)

mf = mol.to_gpu().RKS(xc="b3lyp")

mf = mf.PCM()
mf.with_solvent.eps = 35.9
mf.with_solvent.verbose = 0
mf.with_solvent.lebedev_order = 17
mf.with_solvent.method = "C-PCM"  # Also supports IEF-PCM and SS(V)PE.

# Enalbe "lowmem" mode, reconstructing solvent intermediates on the fly.
mf.with_solvent.lowmem_intermediate_storage = True

mf.run()
