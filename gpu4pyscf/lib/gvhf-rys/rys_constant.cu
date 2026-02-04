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

#include <cuda.h>
#include <cuda_runtime.h>
#include <gvhf-rys/vhf.cuh>

__constant__ int _c_cartesian_lexical_xyz[] = {
    // s
    0, 0, 0,
    // p
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
    // d
    2, 0, 0,
    1, 1, 0,
    1, 0, 1,
    0, 2, 0,
    0, 1, 1,
    0, 0, 2,
    // f
    3, 0, 0,
    2, 1, 0,
    2, 0, 1,
    1, 2, 0,
    1, 1, 1,
    1, 0, 2,
    0, 3, 0,
    0, 2, 1,
    0, 1, 2,
    0, 0, 3,
    // g
    4, 0, 0,
    3, 1, 0,
    3, 0, 1,
    2, 2, 0,
    2, 1, 1,
    2, 0, 2,
    1, 3, 0,
    1, 2, 1,
    1, 1, 2,
    1, 0, 3,
    0, 4, 0,
    0, 3, 1,
    0, 2, 2,
    0, 1, 3,
    0, 0, 4,
    // h
    5, 0, 0,
    4, 1, 0,
    4, 0, 1,
    3, 2, 0,
    3, 1, 1,
    3, 0, 2,
    2, 3, 0,
    2, 2, 1,
    2, 1, 2,
    2, 0, 3,
    1, 4, 0,
    1, 3, 1,
    1, 2, 2,
    1, 1, 3,
    1, 0, 4,
    0, 5, 0,
    0, 4, 1,
    0, 3, 2,
    0, 2, 3,
    0, 1, 4,
    0, 0, 5,
    // j
    6, 0, 0,
    5, 1, 0,
    5, 0, 1,
    4, 2, 0,
    4, 1, 1,
    4, 0, 2,
    3, 3, 0,
    3, 2, 1,
    3, 1, 2,
    3, 0, 3,
    2, 4, 0,
    2, 3, 1,
    2, 2, 2,
    2, 1, 3,
    2, 0, 4,
    1, 5, 0,
    1, 4, 1,
    1, 3, 2,
    1, 2, 3,
    1, 1, 4,
    1, 0, 5,
    0, 6, 0,
    0, 5, 1,
    0, 4, 2,
    0, 3, 3,
    0, 2, 4,
    0, 1, 5,
    0, 0, 6,
};

__constant__ GXYZOffset c_gxyz_offset[625];
