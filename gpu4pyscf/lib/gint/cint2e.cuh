/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
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

#pragma once

#include "gint.h"

//extern __constant__ GINTEnvVars c_envs;
extern __constant__ BasisProdCache c_bpcache;
extern __constant__ int16_t c_idx4c[NFffff*3];

/*
__constant__ GINTEnvVars c_envs;
__constant__ BasisProdCache c_bpcache;
__constant__ int16_t c_idx4c[NFffff*3];
*/