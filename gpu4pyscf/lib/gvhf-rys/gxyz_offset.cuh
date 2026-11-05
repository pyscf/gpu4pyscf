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

#pragma once

#include <assert.h>
#include "gvhf-rys/vhf.cuh"

constexpr int RYS_gxyz_offset_table_size()
{
    int size = 0;
    for (int li = 0; li <= LMAX; ++li) {
    for (int lj = 0; lj <= LMAX; ++lj) {
    for (int lk = 0; lk <= LMAX; ++lk) {
    for (int ll = 0; ll <= LMAX; ++ll) {
        int ni = ((li+1)*(li+2)/2 + 2) / 3;
        int nj = ((lj+1)*(lj+2)/2 + 2) / 3;
        int nk = ((lk+1)*(lk+2)/2 + 2) / 3;
        int nl = ((ll+1)*(ll+2)/2 + 2) / 3;
        // The tile count prod((nf+2)/3) reaches 625 at (4,4,4,4)
        int n = ni * nj * nk * nl;
        size += n < 256 ? 256 : n;
    } } } }
    return size;
}

struct RYSGXYZOffsetTable {
    // RYS_get_gxyz_offset constructs one static instance on first use.
    // Concurrent callers wait for initialization; subsequent calls only read it.
    GXYZOffset data[RYS_gxyz_offset_table_size()];
    uint32_t offsets[LMAX1][LMAX1][LMAX1][LMAX1];

    RYSGXYZOffsetTable()
    {
        int offset = 0;
        for (int li = 0; li <= LMAX; ++li) {
        for (int lj = 0; lj <= LMAX; ++lj) {
        for (int lk = 0; lk <= LMAX; ++lk) {
        for (int ll = 0; ll <= LMAX; ++ll) {
            int nfi = (li + 1) * (li + 2) / 2;
            int nfj = (lj + 1) * (lj + 2) / 2;
            int nfk = (lk + 1) * (lk + 2) / 2;
            int nfl = (ll + 1) * (ll + 2) / 2;
            offsets[li][lj][lk][ll] = offset;
            GXYZOffset *table = data + offset;
            int nf = 0;
            for (int i = 0; i < nfi; i += 3) {
            for (int j = 0; j < nfj; j += 3) {
            for (int k = 0; k < nfk; k += 3) {
            for (int l = 0; l < nfl; l += 3) {
                table[nf++] = {int8_t(i), int8_t(j), int8_t(k), int8_t(l)};
            } } } }
            // Small quartets repeat their offsets to fill the first 256 tiles.
            for (int n = nf; n < 256; ++n) {
                table[n] = table[n % nf];
            }
            offset += nf < 256 ? 256 : nf;
        } } } }
    }
};

inline const GXYZOffset *RYS_get_gxyz_offset(int li, int lj, int lk, int ll)
{
    assert(li >= 0 && li <= LMAX && lj >= 0 && lj <= LMAX);
    assert(lk >= 0 && lk <= LMAX && ll >= 0 && ll <= LMAX);
    // Thread-safe initialization; immutable host storage survives graph replay.
    static const RYSGXYZOffsetTable table;
    return table.data + table.offsets[li][lj][lk][ll];
}
