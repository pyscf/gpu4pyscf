/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ double c_lattice_vectors[9];
__constant__ double c_reciprocal_lattice_vectors[9];
__constant__ double c_dxyz_dabc[9];

extern "C" {
void update_lattice_vectors(double *lattice_vectors,
                            double *reciprocal_lattice_vectors,
                            double *reciprocal_norm)
{
    cudaMemcpyToSymbol(c_lattice_vectors, lattice_vectors, 9 * sizeof(double));
    cudaMemcpyToSymbol(c_reciprocal_lattice_vectors, reciprocal_lattice_vectors, 9 * sizeof(double));
}

void update_dxyz_dabc(double *dxyz_dabc) {
    cudaMemcpyToSymbol(c_dxyz_dabc, dxyz_dabc, 9 * sizeof(double));
}
}
