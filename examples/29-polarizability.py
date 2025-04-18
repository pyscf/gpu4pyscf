import pyscf
from gpu4pyscf.dft import rks
from gpu4pyscf.properties import polarizability

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='631g'
grids_level = 6

mol = pyscf.M(atom=atom, basis=bas, max_memory=32000)
mol.build()

mf = rks.RKS(mol, xc='b3lyp')
mf.grids.level = grids_level
e_gpu = mf.kernel() # -76.3849465946694
polar_gpu = polarizability.eval_polarizability(mf)
print('------------------- Polarizability -----------------------------')
print(polar_gpu)
"""
[[ 6.96412939e+00  8.89901195e-16  1.41475771e-13]
 [ 8.89901195e-16  1.48264173e+00 -2.84030606e-14]
 [ 1.41475771e-13 -2.84030606e-14  4.81230456e+00]]
"""