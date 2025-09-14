import numpy as np
from fssh import FSSH
import pyscf
import gpu4pyscf

# will change to Wigner distribution in later implementation
def generate_velocities(masses, temperature=300.0):
    k_b = 8.314462618e-3 
    N = len(masses)
    masses_kg = masses * 1e-3
    std_dev = np.sqrt(k_b * temperature / masses_kg[:, np.newaxis]) 
    velocities = np.random.normal(0, std_dev, (N, 3))
    total_momentum = np.sum(velocities * masses_kg[:, np.newaxis], axis=0)
    velocities -= total_momentum / np.sum(masses_kg)
    kinetic_energy = 0.5 * np.sum(masses_kg * np.sum(velocities**2, axis=1)) 
    target_energy = 1.5 * N * k_b * temperature  
    velocities *= np.sqrt(target_energy / kinetic_energy)
    velocities *= 0.01
    return velocities


atom ='''
C   -1.302   0.206   0.000  
H   -0.768  -0.722   0.000  
H   -2.372   0.206   0.000  
C   -0.626   1.381   0.000  
H   -1.159   2.309   0.000  
H    0.444   1.381   0.000  
'''

coord = np.array([[-1.302,  0.206,  0.000,], 
[-0.768, -0.722,  0.000,],
[-2.372,  0.206,  0.000,],  
[-0.626,  1.381,  0.000,],  
[-1.159,  2.309,  0.000,],  
[ 0.444,  1.381,  0.000,],]) / 0.53
atm_type = ['C','H','H','C','H','H']
atom = '\n'.join([f'{atm_type[i]}  {coord[i][0]}   {coord[i][1]}   {coord[i][2]}' for i in range(len(coord))])

mol = pyscf.M(atom=atom, basis='6-31G', unit='bohr')
xc = 'b3lyp'
mf = gpu4pyscf.dft.RKS(mol, xc=xc)
td = mf.TDA().set(nstates=3)

fssh = FSSH(td, [1,2])
fssh.cur_state = 2
fssh.nsteps = 200
fssh.kernel(None,vel,np.array([0.0,1.0]))
