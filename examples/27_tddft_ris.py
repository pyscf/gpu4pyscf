import pyscf
from gpu4pyscf.dft import rks
import gpu4pyscf.tdscf.ris as ris

atom ='''
C         -4.89126        3.29770        0.00029
H         -5.28213        3.05494       -1.01161
O         -3.49307        3.28429       -0.00328
H         -5.28213        2.58374        0.75736
H         -5.23998        4.31540        0.27138
H         -3.22959        2.35981       -0.24953
'''

mol = pyscf.M(atom=atom, basis='def2-svp',verbose=3)
mf = rks.RKS(mol, xc='wb97x').density_fit()

e_dft = mf.kernel()  
print(f"total energy = {e_dft}")



''' TDDFT-ris'''
td = ris.TDDFT(mf=mf.to_gpu(), nstates=10) 
td.kernel()
# energies, X, Y, oscillator_strength, rotatory_strength = td.kernel()

energies = td.energies
# X = td.X
# Y = td.Y
oscillator_strength = td.oscillator_strength
rotatory_strength = td.rotatory_strength

print("TDDFT-ris ex energies", energies)
print("TDDFT-ris oscillator_strength", oscillator_strength)

''' TDA-ris'''
td = ris.TDA(mf=mf.to_gpu(), nstates=10) 
td.kernel()
# energies, X, oscillator_strength, rotatory_strength = td.kernel()

energies = td.energies
# X = td.X
oscillator_strength = td.oscillator_strength
rotatory_strength = td.rotatory_strength

print("TDA-ris ex energies", energies)
print("TDA-ris oscillator_strength", oscillator_strength)



