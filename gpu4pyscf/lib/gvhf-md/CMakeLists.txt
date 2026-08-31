set(GPU_SRCS
  md_contract_j.cu unrolled_md_j.cu unrolled_md_j_4dm.cu
  md_indices.cu md_j_driver.cu md_pairdata.c 
  contract_int3c2e.cu
  pbc_md_contract_j.cu
  contract_Et.cu
)

add_library(gvhf_md SHARED ${GPU_SRCS})

set_target_properties(gvhf_md PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

if (USE_SYCL)
  set_source_files_properties(${GPU_SRCS} PROPERTIES LANGUAGE CXX)

  # Under CUDA, md_indices.cu is a translation unit: it defines the
  # __device__/__constant__ index tables that every other TU declares
  # extern. SYCL has no cross-TU __device__ linkage, so md_j.cuh
  # #includes md_indices.cu directly and the tables become
  # `inline constexpr`. Compiling it as a TU as well would be redundant.
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/md_indices.cu
    PROPERTIES HEADER_FILE_ONLY ON)

  target_compile_options(gvhf_md PRIVATE -x c++ -nocudainc -nocudalib)
  target_link_libraries(gvhf_md PRIVATE sycl_compat gsycl)
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
  set_target_properties(gvhf_md PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_link_libraries(gvhf_md PRIVATE OpenMP::OpenMP_C)
