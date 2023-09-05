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

#include <cuda_runtime.h>
#include <cutensor.h>

extern "C" {
__host__
int create_plan_cache(cutensorHandle_t *handle, int numCachelines){
    const size_t sizeCache = numCachelines * sizeof(cutensorPlanCacheline_t);
    cutensorPlanCacheline_t* cachelines = (cutensorPlanCacheline_t*) malloc(sizeCache);
    const auto err = cutensorHandleAttachPlanCachelines(handle, cachelines, numCachelines);
    if( err != CUTENSOR_STATUS_SUCCESS ){
        printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__);
        return 1;

    }
    return 0;
}

__host__
int delete_plan_cache(cutensorHandle_t *handle){
    const auto err = cutensorHandleDetachPlanCachelines(handle);
    if( err != CUTENSOR_STATUS_SUCCESS ){
        printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__);
        return 1;
    }
    return 0;
}
}
