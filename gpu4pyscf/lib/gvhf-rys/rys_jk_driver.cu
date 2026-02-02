/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "vhf.cuh"

__constant__ int c_nf[] = {
    1,
    3,
    6,
    10,
    15,
    21,
    28,
    36,
    45,
};

__constant__ float c_div_nf[] = {
    1.f,
    0.333334f,
    0.166667f,
    0.100001f,
    0.066667f,
    0.047620f,
    0.035715f,
    0.027778f,
    0.022223f,
};

extern "C" {
int RYS_init_constant()
{
    return 0;
}

int cuda_version()
{
    return CUDA_VERSION;
}
}
