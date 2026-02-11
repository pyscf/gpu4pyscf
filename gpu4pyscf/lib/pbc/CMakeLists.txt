set(GPU_SRCS
  #pbc_driver.cu
  ft_ao.cu ft_ao_ip1.cu
  fill_int3c2e.cu contract_int3c2e.cu ejk_int3c2e_ip1.cu
  fill_int2c2e.cu fill_int2c2e_ip1.cu
  overlap.cu
  estimator.cu supmol_sr_estimator.cu
  rys_roots_dat.cu
  rys_contract_k.cu rys_contract_jk_ip1.cu
  nr_eval_gto.cu
  sorting.c
  decompress.cu fill_triu.cu
  )

if (USE_SYCL)
  set(C_SRCS sorting.c)
  add_library(pbc SHARED ${GPU_SRCS})

  set_source_files_properties(${C_SRCS} PROPERTIES LANGUAGE C)
  set_source_files_properties(${GPU_SRCS} PROPERTIES LANGUAGE CXX)
  set_target_properties(pbc PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
  target_compile_options(pbc PRIVATE -x c++ -nocudainc -nocudalib)
else()
  list(APPEND GPU_SRCS sorting.c)
  add_library(pbc SHARED ${GPU_SRCS})

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

  set_target_properties(pbc PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    CUDA_SEPARABLE_COMPILATION ON)
endif()

target_link_libraries(pbc OpenMP::OpenMP_C)
