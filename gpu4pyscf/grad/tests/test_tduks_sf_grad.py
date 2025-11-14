# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import pyscf
import numpy as np
import unittest
import pytest
from pyscf import scf, dft
import gpu4pyscf
from packaging import version
from gpu4pyscf.lib.multi_gpu import num_devices

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

bas0 = "ccpvdz"

def setUpModule():
    global mol, mol1
    mol = pyscf.M(
        atom=atom,
        basis=bas0,
        max_memory=32000,
        charge=1,
        spin=1,
        output="/dev/null",
        verbose=1,
    )
    mol1 = pyscf.M(
        atom=atom,
        basis=bas0,
        max_memory=32000,
        charge=0,
        spin=0,
        output="/dev/null",
        verbose=1,
    )


def tearDownModule():
    global mol, mol1
    mol.stdout.close()
    del mol
    mol1.stdout.close()
    del mol1


def benchmark_with_cpu(mol, xc, nstates=3, lindep=1.0e-12, tda=False, extype=0):
    mf = dft.UKS(mol, xc=xc).to_gpu().run()
    tdsf = mf.SFTDA()
    tdsf.extype = extype
    tdsf.collinear = 'mcol'
    tdsf.nstates=5
    tdsf.collinear_samples=10
    tdsf.kernel()

    g = tdsf.Gradients()
    g.kernel()

    return mf.e_tot, tdsf.e, g.de


def _check_grad(mol, xc, tol=1e-5, lindep=1.0e-12, disp=None, tda=True, method="cpu", extype=0):
    if not tda:
        raise NotImplementedError("spin-flip TDDFT gradients is not implemented")
    if method == "cpu":
        etot, e, grad_gpu = benchmark_with_cpu(mol, xc, nstates=5, lindep=lindep, tda=tda, extype=extype)
    else:
        raise NotImplementedError("Only compared with CPU")
        
    return etot, e, grad_gpu


class KnownValues(unittest.TestCase):
    def test_grad_b3lyp_tda_spinflip_up_cpu(self):
        etot, e, grad_gpu = _check_grad(mol, xc="b3lyp", tol=5e-10, method="cpu")
        # ref from pyscf-forge
        assert abs(etot - -75.9674347270528) < 1e-8
        assert abs(e - np.array([0.46618494, 0.53438998, 0.60047275, 0.65786033, 0.92091718])).max() < 1e-5
        ref = np.array([[ 8.79547051e-16,  8.63728537e-14,  1.87755267e-01],
                        [-4.31890391e-16,  2.15026042e-01, -9.38746716e-02],
                        [-4.50003252e-16, -2.15026042e-01, -9.38746716e-02]])
        assert abs(grad_gpu - ref).max() < 1e-5
        
    def test_grad_b3lyp_tda_spinflip_down_cpu(self):
        etot, e, grad_gpu = _check_grad(mol, xc="b3lyp", tol=5e-10, method="cpu", extype=1)
        # ref from pyscf-forge
        assert abs(etot - -75.96743472705282) < 1e-8
        assert abs(e - np.array([0.0034149,  0.08157731, 0.23027453, 0.50644857, 0.51065628])).max() < 1e-5
        ref = np.array([[-3.01640558e-16,  1.52982216e-13,  5.10689029e-02],
                        [ 1.36165869e-16,  4.52872857e-02, -2.55387304e-02],
                        [-3.08111636e-17, -4.52872857e-02, -2.55387304e-02],])
        assert abs(grad_gpu - ref).max() < 1e-5

    def test_grad_svwn_tda_spinflip_down_cpu(self):
        etot, e, grad_gpu = _check_grad(mol, xc="svwn", tol=5e-10, method="cpu", extype=1)
        # ref from pyscf-forge
        assert abs(etot - -75.39033965461661) < 1e-8
        assert abs(e - np.array([0.00210504, 0.07530215, 0.22255285, 0.50300732, 0.50382963])).max() < 1e-5
        ref = np.array([[-8.15030724e-16, -6.13885762e-14,  6.41681368e-02],
                        [ 1.12931062e-16,  5.34632826e-02, -3.20887796e-02],
                        [ 7.97399496e-17, -5.34632826e-02, -3.20887796e-02],])
        assert abs(grad_gpu - ref).max() < 1e-5

    def test_grad_camb3lyp_tda_spinflip_down_cpu(self):
        etot, e, grad_gpu = _check_grad(mol, xc="camb3lyp", tol=5e-10, method="cpu", extype=1)
        # ref from pyscf-forge
        assert abs(etot - -75.93920847775132) < 1e-8
        assert abs(e - np.array([0.00335301, 0.07772481, 0.2267033, 0.50960632, 0.5133939])).max() < 1e-5
        ref = np.array([[-7.43754261e-18, -1.56347842e-13,  4.99263503e-02],
                        [-1.84572351e-17,  4.52908126e-02, -2.49673842e-02],
                        [ 2.40683934e-17, -4.52908126e-02, -2.49673842e-02],])
        assert abs(grad_gpu - ref).max() < 1e-5

    def test_grad_b3lyp_tda_spinflip_up_cpu_closed(self):
        etot, e, grad_gpu = _check_grad(mol1, xc="b3lyp", tol=5e-10, method="cpu")
        # ref from pyscf-forge
        assert abs(etot - -76.42037833354925) < 1e-8
        assert abs(e - np.array([0.25433265, 0.33124974, 0.3313682, 0.40247177, 0.47307456])).max() < 1e-5
        ref = np.array([[ 1.29088518e-16,  6.98423827e-14,  1.25014262e-01],
                        [-1.36624149e-16,  8.37484153e-02, -6.25098673e-02],
                        [ 1.80012190e-16, -8.37484153e-02, -6.25098673e-02]])
        assert abs(grad_gpu - ref).max() < 1e-5
        
    def test_grad_b3lyp_tda_spinflip_down_cpu_closed(self):
        etot, e, grad_gpu = _check_grad(mol1, xc="b3lyp", tol=5e-10, method="cpu", extype=1)
        # ref from pyscf-forge
        assert abs(etot - -76.42037833354925) < 1e-8
        assert abs(e - np.array([0.2543327, 0.33124974, 0.3313685, 0.40247202, 0.4730746])).max() < 1e-5
        ref = np.array([[-5.16805682e-16,  7.28823057e-14,  1.25014068e-01],
                        [ 1.94935391e-16,  8.37484121e-02, -6.25097703e-02],
                        [ 1.20139074e-17, -8.37484121e-02, -6.25097703e-02],])
        assert abs(grad_gpu - ref).max() < 1e-5

    def test_grad_svwn_tda_spinflip_down_cpu_closed(self):
        etot, e, grad_gpu = _check_grad(mol1, xc="svwn", tol=5e-10, method="cpu", extype=1)
        # ref from pyscf-forge
        assert abs(etot - -75.85470242125601) < 1e-8
        assert abs(e - np.array([0.25020513, 0.32400566, 0.32879602, 0.39954396, 0.47440403])).max() < 1e-5
        ref = np.array([[-1.04007210e-16,  2.76349222e-15,  1.40334993e-01],
                        [ 4.57442221e-17,  9.05506406e-02, -7.01720839e-02],
                        [ 1.95402062e-16, -9.05506406e-02, -7.01720839e-02],])
        assert abs(grad_gpu - ref).max() < 1e-5

    def test_grad_camb3lyp_tda_spinflip_down_cpu_closed(self):
        etot, e, grad_gpu = _check_grad(mol1, xc="camb3lyp", tol=5e-10, method="cpu", extype=1)
        # ref from pyscf-forge
        assert abs(etot - -76.39180300401368) < 1e-8
        assert abs(e - np.array([0.25653358, 0.33449489, 0.33602869, 0.40788379, 0.47369817])).max() < 1e-5
        ref = np.array([[ 8.58453733e-14, -2.06289065e-13,  1.21090859e-01],
                        [-4.13696957e-14,  8.17477776e-02, -6.05484440e-02],
                        [-4.39523378e-14, -8.17477776e-02, -6.05484440e-02],])
        assert abs(grad_gpu - ref).max() < 1e-5
    

if __name__ == "__main__":
    print("Full Tests for spin-flip TD-UKS Gradient")
    unittest.main()
