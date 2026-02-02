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

typedef struct {
    int natm;
    int nbas;
    int *bas_atom;
    int *bas_exp;
    int *bas_coeff;
    double *env;
    double *atom_coordx;
} GTOValEnvVars;

typedef struct {
    int ngrids;
    int nbas;
    int nao;
    int bas_off;
    int nprim;
    int *ao_loc;
    int *bas_indices;
    double fac;
    double *gridx;
    double *data;
} BasOffsets;

#define C_ATOM          0
#define C_EXP           1
#define C_COEFF         2
#define C_BAS_SLOTS     3
#define NBAS_MAX        6000

