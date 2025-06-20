# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
import cupy as cp
from gpu4pyscf.tdscf._krylov_family import krylov_solver 


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set random seed for reproducibility
        cp.random.seed(42)
        cls.A_size = 2000
        cls.n_vec = 5
        cls.n_states = 5
        cls.scaling = 50

        # Generate symmetric matrix A
        A = cp.random.rand(cls.A_size, cls.A_size) * 0.01
        A = A + A.T
        cp.fill_diagonal(A, (cp.random.rand(cls.A_size) + 2) * cls.scaling)
        cls.A = A
        cls.hdiag = cp.diag(A)

        # Generate random right-hand side and omega shift
        cls.rhs = cp.random.rand(cls.n_vec, cls.A_size) * cls.scaling
        cls.omega_shift = (cp.random.rand(cls.n_vec) + 2) * cls.scaling /2

        @staticmethod
        def matrix_vector_product(x):
            return x.dot(cls.A)
        
        cls.matrix_vector_product = matrix_vector_product

        # Reference eigenvalues and eigenvectors
        ref_eigvals, ref_eigvecs = cp.linalg.eigh(cls.A)
        cls.ref_eigenvalues = cp.asarray(ref_eigvals)[:cls.n_states]
        cls.ref_eigenvectors = ref_eigvecs[:,:cls.n_states].T
        # Reference solutions for linear system
        cls.ref_solution_vectors = cp.linalg.solve(cls.A, cls.rhs.T).T
        # Reference solutions for shifted linear system
        cls.ref_solution_vectors_shifted = cp.zeros_like(cls.ref_solution_vectors)
        for i in range(cls.n_vec):
            shifted_A = cls.A - cls.omega_shift[i] * cp.eye(cls.A_size)
            cls.ref_solution_vectors_shifted[i,:] = cp.linalg.solve(shifted_A, cls.rhs[i])

    @classmethod
    def tearDownClass(cls):
        # Clean up CuPy memory
        cp.get_default_memory_pool().free_all_blocks()

    def test_krylov_eigenvalue(self):
        """Test Krylov solver for eigenvalue problem"""
        eigenvalues, eigenvectors = krylov_solver(
            matrix_vector_product=self.matrix_vector_product,
            hdiag=self.hdiag,
            problem_type='eigenvalue',
            n_states=self.n_states,
            conv_tol=1e-8,
            max_iter=35,
            gram_schmidt=True,
            verbose=4,
            single=False
        )

        print('eigenvectors.shape', eigenvectors.shape)
        # Compare eigenvalues
        self.assertAlmostEqual(
            float(cp.linalg.norm(eigenvalues - self.ref_eigenvalues)), 0, places=5,
            msg="Eigenvalues do not match reference within tolerance"
        )
        self.assertAlmostEqual(
            float(cp.linalg.norm(cp.abs(eigenvectors) - cp.abs(self.ref_eigenvectors))), 0, places=5,
            msg="Eigenvectors do not match reference within tolerance"
        )


    def test_krylov_linear(self):
        """Test Krylov solver for linear system"""
        solution_vectors = krylov_solver(
            matrix_vector_product=self.matrix_vector_product,
            hdiag=self.hdiag,
            problem_type='linear',
            rhs=self.rhs,
            conv_tol=1e-8,
            max_iter=35,
            gram_schmidt=True,
            verbose=4,
            single=False
        )

        # Compare solutions
        self.assertAlmostEqual(
            float(cp.linalg.norm(solution_vectors - self.ref_solution_vectors)), 0, places=5,
            msg="Linear system solutions do not match reference within tolerance"
        )

    def test_krylov_shifted_linear(self):
        """Test Krylov solver for shifted linear system"""
        solution_vectors_shifted = krylov_solver(
            matrix_vector_product=self.matrix_vector_product,
            hdiag=self.hdiag,
            problem_type='shifted_linear',
            rhs=self.rhs,
            omega_shift=self.omega_shift,
            conv_tol=1e-12,
            max_iter=35,
            gram_schmidt=True,
            verbose=4,
            single=False
        )

        # Compare solutions
        self.assertAlmostEqual(
            float(cp.linalg.norm(solution_vectors_shifted - self.ref_solution_vectors_shifted)), 0, places=5,
            msg="Shifted linear system solutions do not match reference within tolerance"
        )
if __name__ == "__main__":
    print("Running tests for Krylov solver with eigenvalue, linear, and shifted linear problems")
    unittest.main()