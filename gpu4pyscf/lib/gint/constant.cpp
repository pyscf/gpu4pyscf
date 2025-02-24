/* Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "gint.h"
#include "sycl_device.hpp"
#include "constant.hpp"

//SYCL_EXTERNAL sycl_device_global< GINTEnvVars c_envs;
// SYCL_EXTERNAL sycl_device_global<BasisProdCache> c_bpcache;
// SYCL_EXTERNAL sycl_device_global<int16_t[NFffff*3]> c_idx4c;
// SYCL_EXTERNAL sycl_device_global<int[TOT_NF*3]> c_idx; 
// SYCL_EXTERNAL sycl_device_global<int[GPU_LMAX+1]> c_l_locs;
