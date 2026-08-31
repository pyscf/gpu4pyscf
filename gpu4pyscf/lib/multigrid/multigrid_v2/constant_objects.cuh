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

#pragma once
namespace gpu4pyscf::gpbc::multi_grid {

extern __constant__ double lattice_vectors[9];
extern __constant__ double reciprocal_lattice_vectors[9];
extern __constant__ double dxyz_dabc[9];
extern __constant__ double reciprocal_norm[3];

} // namespace gpu4pyscf::gpbc::multi_grid
