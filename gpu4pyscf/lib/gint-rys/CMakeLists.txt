set(GPU_SRCS
  gint_driver.cu fill_int3c2e.cu unrolled_int3c2e.cu
  fill_int3c2e_bdiv.cu
  rys_roots_dat.cu
  )


if (USE_SYCL)
  add_library(gint_rys SHARED ${GPU_SRCS})
  
  file(GLOB CUH_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")
  file(GLOB ALL_GPU_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

  set_source_files_properties(${ALL_GPU_SRCS} PROPERTIES LANGUAGE CXX)
  set_target_properties(gint_rys PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
  target_compile_options(gint_rys PRIVATE -x c++ -nocudainc -nocudalib)
else()
  list(APPEND GPU_SRCS rys_constant.cu)
  add_library(gint_rys SHARED ${GPU_SRCS})
  
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")
  set_target_properties(gint_rys PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    CUDA_SEPARABLE_COMPILATION ON)
endif()

#target_link_libraries(ft_ao OpenMP::OpenMP_C)
