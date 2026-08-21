'''
Run DFT calculations on very large molecules with reduced GPU memory usage.

When handling very large molecular systems, for example, over 20,000 basis
functions, the default HF/KS implementation may require more GPU memory than is
available.

The lowmem mode provides an option to run the SCF iterations with reduces GPU
memory usage at the cost of some additional computation:
- Constructing overlap matrices and density matrices on the fly.
- Storing only the lower-triangular part of symmetric matrices such as the
  Fock matrix.
- Storing DIIS intermediates in host memory.

If an implicit solvent model is also used for these large systems, the default
solvent implementation may become another source of insufficient GPU memory.
In that case, you can enable the low-memory mode for solvent. See example
solvent/20-pcm_lowmem.py for additonal settings for solvent models.
'''

import pyscf
from gpu4pyscf.dft import rks_lowmem
mol = pyscf.M(atom='''
   C          3.21659       -1.41022       -0.26053
   C          2.16708       -0.35258       -0.59607
   N          1.21359       -0.16703        0.41640
   C          0.11616        0.82394        0.50964
   C         -1.19613        0.03585        0.74226
   N         -2.18193       -0.02502       -0.18081
   C         -3.43891       -0.74663        0.01614
   O          2.19596        0.25708       -1.63440
   C          0.11486        1.96253       -0.53088
   O         -1.29658       -0.59392        1.85462
   H          3.25195       -2.14283       -1.08721
   H          3.06369       -1.95423        0.67666
   H          4.20892       -0.93714       -0.22851
   H          1.24786       -0.78278        1.21013
   H          0.25990        1.31404        1.47973
   H         -2.02230        0.38818       -1.10143
   H         -3.60706       -1.48647       -0.76756
   H         -4.29549       -0.06423        0.04327
   H         -3.36801       -1.25875        0.98106
   H         -0.68664        2.66864       -0.27269
   H          0.01029        1.65112       -1.56461
   H          1.06461        2.50818       -0.45885''',
basis='def2-tzvp')

mf = rks_lowmem.RKS(mol, xc='b3lyp')
mf.run()
