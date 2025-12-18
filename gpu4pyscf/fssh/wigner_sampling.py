import numpy as np
import random
from scipy.special import laguerre

def wignerfunc(mu, temp):
    max_pop, max_nlevel = 0.9999, 150
    ex = mu / (0.69503 * temp)  # vibrational temperature: ex=h*c*mu/(kb*T), 0.69503 convert cm-1 to K
    pop = []

    while np.sum(pop) < max_pop and len(pop) < max_nlevel:
        _pop = float(np.exp(-1 * ex * len(pop)) * (1 - np.exp(-1 * ex)))
        pop.append(_pop)
    pop.append(1 - np.sum(pop))

    while True:
        random_state = np.random.choice(len(pop), p=pop)  # random generate a state
        q = np.random.uniform(0, 1) * 10.0 - 5.0
        p = np.random.uniform(0, 1) * 10.0 - 5.0
        rho2 = 2 * (q ** 2 + p ** 2)
        w = (-1) ** random_state * laguerre(random_state)(rho2) * np.exp(-0.5 * rho2)
        if np.random.uniform(0, 1) < w < 1:
            break
    return float(q), float(p)


def wigner(temp, freqs, xyz, vib):

    nfreq = len(freqs)
    natom = len(xyz)

    mu_to_hartree = 4.55633518e-6  # 1 cm-1  = h*c/Eh = 4.55633518e-6 au
    ma_to_amu = 1822.88852  # 1 g/mol = 1/Na*me*1000 = 1822.88852 amu
    bohr_to_angstrom = 0.529177249  # 1 Bohr  = 0.529177249 Angstrom

    q_p = np.array([wignerfunc(i, temp) for i in freqs])  # generates update coordinates and momenta pairs Q and P

    q = q_p[:, 0].reshape((nfreq, 1))  # first column is Q

    q *= 1 / np.sqrt(freqs * mu_to_hartree * ma_to_amu)  # convert coordinates from m to Bohr
    qvib = np.array([np.ones((natom, 3)) * i for i in q])  # generate identity array to expand Q
    qvib = np.sum(vib * qvib, axis=0)  # sum sampled structure over all modes
    newc = (xyz + qvib) * bohr_to_angstrom  # cartesian coordinates in Angstrom

    p = q_p[:, 1].reshape((nfreq, 1))  # second column is P
    p *= np.sqrt(freqs * mu_to_hartree / ma_to_amu)  # convert velocity from m/s to Bohr/au
    pvib = np.array([np.ones((natom, 3)) * i for i in p])  # generate identity array to expand P
    velo = np.sum(vib * pvib, axis=0)  # sum sampled velocity over all modes in Bohr/au
 
    initcond = np.concatenate((newc, velo), axis=1)

    return initcond

import numpy as np
from pyscf import gto, scf, hessian
from pyscf.geomopt import geometric_solver
from pyscf.hessian import thermo
from pyscf.data.nist import HARTREE2EV

mol = gto.Mole()
data = np.loadtxt('/mlx_devbox/users/fancheng.99/playground/arnold_workspace_root/fssh/azomethane.xyz',skiprows=2,dtype=str)
atm = data[:,0]
crd = data[:,1:].astype(float)

mol.atom = '\n'.join([f'{atm[i]}  {crd[i][0]}   {crd[i][1]}   {crd[i][2]}' for i in range(len(crd))])
mol.basis = 'sto-3g'
mol.build()
mf = scf.RKS(mol)
mol_eq = geometric_solver.optimize(mf)
print("Equilibrium coordinates (Angstrom):")
print(mol_eq.atom_coords(unit='a'))
mf_eq = scf.RKS(mol_eq).run()
h = mf_eq.Hessian().kernel()
thermo_data = thermo.harmonic_analysis(mol_eq, h)

valid = []
while len(valid) < 50:
    initcond = wigner(300, thermo_data['freq_wavenumber'].reshape(-1,1), mol.atom_coords(), thermo_data['norm_mode'])
    dis = np.linalg.norm(initcond[:,0:3][None] - initcond[:,0:3][:,None],axis=-1)
    if np.where(dis<1e-5, 5e4, dis).min() < 0.7:
        continue
    else:
        valid.append(initcond)

valid= np.array(valid)