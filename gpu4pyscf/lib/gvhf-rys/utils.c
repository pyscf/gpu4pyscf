/*
 * Copyright 2026 The PySCF Developers. All Rights Reserved.
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

int ao_pair_indices(int *ao_pair_idx, int *bas_ij_idx, int *ao_loc,
                    int npairs, int nbas, int nao)
{
    int n = 0;
    for (int pair_id = 0; pair_id < npairs; pair_id++) {
        int bas_ij = bas_ij_idx[pair_id];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int i1 = ao_loc[ish+1];
        int j1 = ao_loc[jsh+1];
        for (int i = i0; i < i1; i++) {
        for (int j = j0; j < j1; j++) {
            ao_pair_idx[n] = i * nao + j;
            n++;
        } }
    }
    return n;
}
