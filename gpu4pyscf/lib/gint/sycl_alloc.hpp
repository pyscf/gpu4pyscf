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

#include "sycl_device.hpp"

// sycl::queue q;

// Function to check SYCL errors
template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "SYCL error at " << file << ":" << line << " code=" << result << " \"" << func << "\" \n";
        std::exit(EXIT_FAILURE);
    }
}
#define checkSyclErrors(val) check((val), #val, __FILE__, __LINE__)


#define MALLOC(type, var, size) \
    type *var = sycl::malloc_device<type>(size, q); \
    if (var == nullptr) { \
        std::cerr << "Memory allocation failed for " #var " at " __FILE__ ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }
    
#define FREE(var) \
    sycl::free(var, *(sycl_get_queue()))

#define MEMSET(addr, val, size) \
    { \
        q.submit([&](sycl::handler& cgh) { \
            cgh.memset(addr, val, size); \
        }).wait(); \
    }

#define DEVICE_INIT(type, dst, src, size) \
    MALLOC(type, dst, size); \
    { \
        q.submit([&](sycl::handler& cgh) { \
            cgh.memcpy(dst, src, sizeof(type) * (size)); \
        }).wait(); \
    }

