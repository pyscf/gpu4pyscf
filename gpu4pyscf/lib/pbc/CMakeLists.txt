set(GPU_SRCS
  pbc_driver.cu ft_ao.cu unrolled_ft_ao.cu
  fill_int3c2e.cu unrolled_int3c2e.cu
  estimator.cu
  rys_roots_dat.cu
  sorting.c
  )

add_library(pbc SHARED ${GPU_SRCS})
  
if (USE_SYCL)
  set_source_files_properties(${GPU_SRCS} PROPERTIES LANGUAGE CXX)
  set_target_properties(pbc PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
  target_compile_options(pbc PRIVATE -x c++ -nocudainc -nocudalib)
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

  set_target_properties(pbc PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    CUDA_SEPARABLE_COMPILATION ON)
endif()
  
#target_link_libraries(ft_ao OpenMP::OpenMP_C)
