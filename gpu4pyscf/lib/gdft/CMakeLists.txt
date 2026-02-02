# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# set(GPU_SRCS
#   nr_eval_gto.cu
#   contract_rho.cu
#   gen_grids.cu
#   nr_numint_sparse.cu
#   vv10.cu
#   #libxc.cu
# )

# add_library(gdft SHARED ${GPU_SRCS})

# if (USE_SYCL)
#   # ------------------------------------------------------------------
#   # Provide a minimal SYCL::SYCL if nothing defines it for us.
#   # Do this BEFORE find_package(ExchCXX) because ExchCXX may depend on it.
#   # ------------------------------------------------------------------
#   if(NOT TARGET SYCL::SYCL)
#     add_library(SYCL::SYCL INTERFACE IMPORTED)
#     target_compile_options(SYCL::SYCL INTERFACE -fsycl)
#     target_link_options(SYCL::SYCL INTERFACE -fsycl)
#     # Optional: expose include dirs env if helpful
#     # target_include_directories(SYCL::SYCL INTERFACE $ENV{ONEAPI_ROOT}/compiler/latest/linux/include/sycl)
#   endif()

#   find_package(ExchCXX REQUIRED CONFIG)

#   # Build the shim code once; link its objects into both DSOs
#   add_library(exchcxx_obj OBJECT
#     ${CMAKE_CURRENT_SOURCE_DIR}/exchcxx.cpp
#   )
#   set_target_properties(exchcxx_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

#   # libgdft gets the shim objects
#   target_sources(gdft PRIVATE $<TARGET_OBJECTS:exchcxx_obj>)
#   target_link_libraries(gdft PRIVATE ExchCXX::ExchCXX SYCL::SYCL)

#   # libexchcxx.so for ctypes
#   add_library(exchcxx_shim SHARED $<TARGET_OBJECTS:exchcxx_obj>)
#   set_target_properties(exchcxx_shim PROPERTIES OUTPUT_NAME exchcxx)
#   target_link_libraries(exchcxx_shim PRIVATE ExchCXX::ExchCXX SYCL::SYCL)
# else()
#   # CUDA path
#   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
# endif()

# set_target_properties(gdft PROPERTIES
#   LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
# )

# set(GPU_SRCS
#   nr_eval_gto.cu
#   contract_rho.cu
#   gen_grids.cu
#   nr_numint_sparse.cu
#   vv10.cu
# )

# if (USE_SYCL)
#   list(APPEND ${GPU_SRCS} exchcxx.cpp)
#   list(APPEND ${CUH_HEADERS} exchcxx.h)
#   add_library(gdft SHARED ${GPU_SRCS})

#   find_package(ExchCXX REQUIRED CONFIG)
#   if(NOT TARGET SYCL::SYCL)
#     add_library(SYCL::SYCL INTERFACE IMPORTED)
#     target_compile_options(SYCL::SYCL INTERFACE -fsycl)
#     target_link_options(SYCL::SYCL INTERFACE -fsycl)
#   endif()

#   file(GLOB CUH_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")
#   set_source_files_properties(${CUH_HEADERS} ${GPU_SRCS} PROPERTIES LANGUAGE CXX)
#   target_compile_options(gdft PRIVATE -x c++ -nocudainc -nocudalib)


#   # Link static libexchcxx.a into libgdft.so
#   # (path below is your install; adjust as needed)
#   target_link_libraries(gdft PRIVATE
#     SYCL::SYCL
#     ExchCXX::ExchCXX                    # gives include paths / transitive deps
#     /home/abagusetty/gpu4pyscf-testing/ExchCXX/install_aot_static_perkernel/lib64/libexchcxx.a
#   )

#   # If the linker still drops needed objects from the static archive, force it:
#   # target_link_options(gdft PRIVATE
#   #   "-Wl,--whole-archive,/home/.../libexchcxx.a,--no-whole-archive")

#   # Ensure symbols are exported (default on, but be explicit if you used -fvisibility=hidden elsewhere)
#   target_compile_options(gdft PRIVATE -fvisibility=default)
# else(USE_SYCL)
#   list(APPEND ${GPU_SRCS} libxc.cu)
#   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")
# endif (USE_SYCL)

# set_target_properties(gdft PROPERTIES
#   LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GPU_SRCS
  nr_eval_gto.cu
  contract_rho.cu
  gen_grids.cu
  nr_numint_sparse.cu
  vv10.cu
)

if (USE_SYCL)
  list(APPEND GPU_SRCS exchcxx.cpp)

  file(GLOB CUH_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")
  list(APPEND CUH_HEADERS exchcxx.h)

  add_library(gdft SHARED ${GPU_SRCS} ${CUH_HEADERS})

  find_package(ExchCXX REQUIRED CONFIG)
  if(NOT TARGET SYCL::SYCL)
    add_library(SYCL::SYCL INTERFACE IMPORTED)
    target_compile_options(SYCL::SYCL INTERFACE -fsycl)
    target_link_options(SYCL::SYCL INTERFACE -fsycl)
  endif()

  set_source_files_properties(${GPU_SRCS} PROPERTIES LANGUAGE CXX)
  target_compile_options(gdft PRIVATE -x c++ -nocudainc -nocudalib)

  target_link_libraries(gdft PRIVATE
    SYCL::SYCL
    ExchCXX::ExchCXX
  )
  target_compile_options(gdft PRIVATE -fvisibility=default)
else()
  list(APPEND GPU_SRCS libxc.cu)
  add_library(gdft SHARED ${GPU_SRCS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
endif()

set_target_properties(gdft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
