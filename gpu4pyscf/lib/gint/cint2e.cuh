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

#pragma once

#include "gint.h"

//extern __constant__ GINTEnvVars c_envs;
extern __constant__ BasisProdCache c_bpcache;
//extern __constant__ int16_t c_idx4c[NFffff*3];

extern __constant__ int c_idx[TOT_NF*3];
extern __constant__ int c_l_locs[GPU_LMAX+2];
