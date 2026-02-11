# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import re
import numpy as np
import cupy as cp
import pyscf
from pyscf import lib
from gpu4pyscf.md.wigner_sampling import wigner_samples
from gpu4pyscf.md.fssh_tddft import FSSH
from gpu4pyscf.md.fssh import h5_to_xyz
from gpu4pyscf.md.distributions import maxwell_boltzmann_velocities

def extract_energies(filename):
    h5_to_xyz(filename, 'from_h5.xyz')
    pattern = re.compile(r"Energy\s+([-+]?\d*\.\d+|\d+)")
    energies = []
    with open('from_h5.xyz', 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                energies.append(float(match.group(1)))
    return np.array(energies)

class KnownValues(unittest.TestCase):
    def test_wigner_sampling(self):
        mol = pyscf.M(
            atom='''
C   -1.302   0.206   0.000
H   -0.768  -0.722   0.000
H   -2.372   0.206   0.000
C   -0.626   1.381   0.000
H   -1.159   2.309   0.000
H    0.444   1.381   0.000
''', basis='sto-3g')
        freqs = np.array([ 881.56, 1059.94, 1065.64, 1209.12, 1427.12, 1517.68,
                          1672.09, 1864.10, 3725.89, 3763.67, 3931.81, 3941.81])
        norm_mode = np.array(
        [[[ 3.19879886e-02, -1.84850587e-02,  6.08824409e-10],
          [-4.09333141e-01, -2.67382350e-01, -2.30507240e-09],
          [ 2.73978931e-02,  4.88816759e-01, -3.04002001e-09],
          [ 3.22275148e-02, -1.86255485e-02, -1.17063530e-10],
          [-4.10884419e-01, -2.68162739e-01,  1.12618091e-10],
          [ 2.76486229e-02,  4.88926250e-01, -6.27188354e-10],],
         [[-1.48082132e-10,  1.41899199e-10,  6.60563451e-02],
          [ 1.40601809e-09,  9.89969448e-10, -4.20922615e-01],
          [-1.46912087e-10, -1.70451525e-09, -4.24014722e-01],
          [-1.99422778e-11,  4.50642357e-13,  8.80026616e-02],
          [ 7.45877724e-10,  4.12302096e-10, -4.94146670e-01],
          [-2.85940489e-12, -1.39395093e-09, -4.96632985e-01],],
         [[ 1.19863710e-11,  1.52497588e-10,  1.27215012e-01],
          [ 1.90354296e-09,  1.43540385e-09, -4.39841504e-01],
          [ 1.47940806e-10, -5.10572037e-10, -4.43623902e-01],
          [-2.24534616e-10, -4.13074453e-11, -1.13163719e-01],
          [ 6.23901802e-10,  5.10209327e-10,  3.60864382e-01],
          [-1.42729960e-10, -2.75994672e-09,  3.55170384e-01],],
         [[-1.67066474e-10,  4.02483549e-11, -7.80152550e-04],
          [-1.59906183e-10,  1.78314498e-10,  5.01192215e-01],
          [-1.91443565e-10, -1.70617689e-09, -4.94587161e-01],
          [ 1.68082908e-10,  5.55507423e-11,  2.81891272e-04],
          [ 2.09742641e-10, -3.24980847e-11,  4.97784981e-01],
          [ 1.29495960e-10,  4.18849263e-10, -4.98452916e-01],],
         [[-1.03621549e-01,  6.10514952e-02,  3.08291324e-12],
          [ 2.86901425e-01,  2.87616667e-01, -4.71765249e-10],
          [-1.04630062e-01, -3.85599970e-01,  7.42769508e-10],
          [ 1.03671985e-01, -6.08334279e-02, -1.13913107e-10],
          [-2.87576737e-01, -2.87463465e-01, -8.21166067e-11],
          [ 1.04704392e-01,  3.82848350e-01,  1.13172861e-09],],
         [[-5.87213875e-02, -9.97810765e-02,  2.74100606e-10],
          [-3.08781055e-01, -2.63797352e-01, -1.60882262e-09],
          [-7.58609667e-02, -4.06712688e-01, -4.12569453e-10],
          [ 5.87265661e-02,  9.97323618e-02, -1.99084911e-10],
          [ 3.08668527e-01,  2.63432125e-01, -3.28274684e-11],
          [ 7.59117894e-02,  4.07658383e-01,  1.16035772e-09],],
         [[-3.29573096e-02, -5.66115768e-02,  2.56772073e-11],
          [ 4.24309373e-01,  2.07053494e-01,  1.66220352e-10],
          [-3.28154708e-02,  4.69865387e-01, -4.53016778e-10],
          [-3.27153211e-02, -5.69954667e-02,  1.84064746e-11],
          [ 4.23549649e-01,  2.05650610e-01,  1.90486333e-10],
          [-3.25098537e-02,  4.71135072e-01, -4.28976723e-10],],
         [[ 7.98096112e-02,  1.38535980e-01,  7.11144798e-11],
          [-3.04381338e-01, -5.78287185e-02, -4.28680539e-10],
          [ 1.01196541e-01, -2.92512255e-01, -8.40297097e-11],
          [-7.97631383e-02, -1.38488541e-01, -5.61120767e-11],
          [ 3.03708424e-01,  5.71738471e-02, -4.93771447e-12],
          [-1.01077383e-01,  2.92601864e-01,  3.38884451e-10],],
         [[ 2.04140152e-02,  3.73942475e-02,  2.61860473e-11],
          [ 2.54781862e-01, -4.41976113e-01, -9.39086970e-11],
          [-4.95830677e-01,  2.34719839e-04, -7.86785301e-11],
          [ 1.94889387e-02,  3.41171978e-02, -2.68156017e-11],
          [ 2.36254163e-01, -4.10589367e-01,  8.26415476e-11],
          [-4.70675961e-01,  2.23647708e-04,  9.74472462e-11],],
         [[-2.41892878e-02, -4.42181650e-02, -2.22679109e-12],
          [-2.37487623e-01,  4.09773115e-01, -6.64817820e-11],
          [ 4.57979347e-01, -1.33123904e-03,  7.26613142e-11],
          [ 2.62750400e-02,  4.60680717e-02,  5.50365682e-12],
          [ 2.49918258e-01, -4.31822893e-01, -9.22529761e-11],
          [-4.95263128e-01,  1.33813189e-03,  4.70270084e-11],],
         [[-6.57752313e-02,  3.70926583e-02,  3.51436260e-12],
          [ 2.54374047e-01, -4.40833046e-01,  3.54866480e-11],
          [ 5.26313346e-01, -3.77191983e-04, -5.86391427e-11],
          [ 5.26190095e-02, -3.01462031e-02, -3.59878579e-12],
          [-2.06846816e-01,  3.57087923e-01,  5.91932775e-11],
          [-4.17075319e-01,  1.35061523e-03, -3.50344109e-11],],
         [[ 5.22357216e-02, -2.94624458e-02, -2.29932069e-12],
          [-2.00273660e-01,  3.56268280e-01, -1.17783096e-11],
          [-4.22088193e-01, -4.34223984e-03,  2.23387685e-11],
          [ 6.49208127e-02, -3.74889822e-02,  4.09905451e-12],
          [-2.54030641e-01,  4.49984557e-01, -3.30974991e-11],
          [-5.19606648e-01, -4.13916721e-03,  1.09231884e-12],]])

        initcond = wigner_samples(300, freqs, mol.atom_coords(), norm_mode, 20, seed=4)
        self.assertAlmostEqual(lib.fp(initcond), -5.570450505881205, 8)

    def test_fssh_tdrhf(self):
        mol = pyscf.M(
            atom='''
C   -1.302   0.206   0.000
H   -0.768  -0.722   0.000
H   -2.372   0.206   0.000
C   -0.626   1.381   0.000
H   -1.159   2.309   0.000
H    0.444   1.381   0.000
''', basis='6-31g')
        mf = mol.RHF().to_gpu().density_fit().set(conv_tol=1e-14)
        td = mf.TDA().set(nstates=3, conv_tol=1e-6)
        np.random.seed(5)

        mass = mol.atom_mass_list(True)
        vel = maxwell_boltzmann_velocities(mass, temperature=300)

        fssh = FSSH(td, [1,2])
        fssh.cur_state = 2
        fssh.nsteps = 2
        fssh.seed = 1201
        fssh.kernel(None,vel,np.array([0.0,1.0]))

        ref = np.array([-77.64845562, -77.65034498, -77.65310929])
        energies = extract_energies(fssh.filename)
        assert abs(ref - energies).max() < 1e-8

    def test_fssh_restart(self):
        mol = pyscf.M(
            atom='''
C   -1.302   0.206   0.000
H   -0.768  -0.722   0.000
H   -2.372   0.206   0.000
C   -0.626   1.381   0.000
H   -1.159   2.309   0.000
H    0.444   1.381   0.000
''', basis='6-31g')
        mf = mol.RHF().to_gpu().density_fit().set(conv_tol=1e-14)
        td = mf.TDA().set(nstates=3, conv_tol=1e-6)
        np.random.seed(5)
        mass = mol.atom_mass_list(True)
        vel = maxwell_boltzmann_velocities(mass, temperature=300, force_temp=True)

        fssh = FSSH(td, [1,2])
        fssh.cur_state = 2
        fssh.timestep_fs = 2.2
        fssh.nsteps = 3
        fssh.seed = 1201
        fssh.kernel(None,vel,np.array([0.0,1.0]))

        ref = np.array([-77.64845562, -77.65848913, -77.65135950, -77.68274985])
        energies = extract_energies(fssh.filename)
        assert abs(ref - energies).max() < 1e-8

        # Normally, restart should be able to reproduce a non-terminated
        # calculation, except that the hopping happens right before termination.
        # Here, hopping at step 2; force was generated by state 2, while
        # cur_state points to state 1. When restarting, force will be evaluated
        # for state 1, not identical to the state before termination.
        fssh = FSSH(td, [1,2])
        fssh.cur_state = 2
        fssh.timestep_fs = 2.2
        fssh.nsteps = 2
        fssh.seed = 1201
        fssh.kernel(None,vel,np.array([0.0,1.0]))
        fssh.restore(fssh.filename)
        fssh.nsteps = 3
        fssh.kernel()

        ref = np.array([-77.64845562, -77.65848913, -77.65135950, -77.67088619])
        energies = extract_energies(fssh.filename)
        assert abs(ref - energies).max() < 1e-8

    def test_fssh_td_ris(self):
        from gpu4pyscf.tdscf.ris import TDA
        mol = pyscf.M(
            atom='''
C   -1.302   0.206   0.000
H   -0.768  -0.722   0.000
H   -2.372   0.206   0.000
C   -0.626   1.381   0.000
H   -1.159   2.309   0.000
H    0.444   1.381   0.000
''', basis='6-31g')
        mf = mol.RKS(xc='pbe0').to_gpu().density_fit().set(conv_tol=1e-14)
        td = TDA(mf, Ktrunc=0).set(nstates=3, conv_tol=1e-5)
        np.random.seed(5)
        mass = mol.atom_mass_list(True)
        vel = maxwell_boltzmann_velocities(mass, temperature=300, force_temp=True)

        fssh = FSSH(td, [1,2])
        fssh.cur_state = 2
        fssh.nsteps = 2
        fssh.seed = 1201
        fssh.tdnac.ris_zvector_solver = True
        fssh.kernel(None,vel,np.array([0.0,1.0]))

        ref = np.array([-78.11999512, -78.12371063, -78.12418365])
        energies = extract_energies(fssh.filename)
        assert abs(ref - energies).max() < 1e-8

    def test_fssh_kTDC(self):
        mol = pyscf.M(
            atom='''
C   -1.302   0.206   0.000
H   -0.768  -0.722   0.000
H   -2.372   0.206   0.000
C   -0.626   1.381   0.000
H   -1.159   2.309   0.000
H    0.444   1.381   0.000
''', basis='6-31g')
        mf = mol.RHF().to_gpu().density_fit().set(conv_tol=1e-14)
        td = mf.TDA().set(nstates=3, conv_tol=1e-6)
        np.random.seed(5)
        vel = maxwell_boltzmann_velocities(mol.atom_mass_list(True), temperature=300, force_temp=True)

        fssh = FSSH(td, [1,2])
        fssh.cur_state = 2
        fssh.coupling_method = 'ktdc'
        fssh.nsteps = 3
        fssh.seed = 1201
        fssh.kernel(None,vel,np.array([0.0,1.0]))

        ref = np.array([-77.65031018, -77.65306016, -77.65595287, -77.65827718])
        energies = extract_energies(fssh.filename)
        assert abs(ref - energies).max() < 1e-8

        fssh = FSSH(td, [1,2])
        fssh.cur_state = 2
        fssh.coupling_method = 'ktdc'
        fssh.nsteps = 2
        fssh.seed = 1201
        fssh.kernel(None,vel,np.array([0.0,1.0]))
        fssh.restore(fssh.filename)
        fssh.nsteps = 3
        fssh.kernel()

        energies = extract_energies(fssh.filename)
        assert abs(ref - energies).max() < 1e-8

if __name__ == "__main__":
    print("Full Tests for FSSH")
    unittest.main()
