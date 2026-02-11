set(GPU_SRCS
  rys_jk_driver.cu rys_roots_dat.cu
  nr_sr_estimator.c nr_sr_estimator.cu
  #rys_contract_j.cu unrolled_rys_j.cu cart2xyz.c
  rys_contract_k.cu unrolled_rys_k.cu
  rys_contract_jk.cu unrolled_rys_jk.cu
  rys_contract_jk_ip1.cu unrolled_rys_jk_ip1.cu unrolled_ejk_ip1.cu
  rys_contract_jk_ip2.cu unrolled_ejk_ip2_type12.cu unrolled_ejk_ip2_type3.cu
  contract_int3c2e.cu
  mole_helper.cu
  ejk_int3c2e_ip1.cu fill_int3c2e.cu
)

if (USE_SYCL)
  add_library(gvhf_rys SHARED ${GPU_SRCS})

  set_source_files_properties(${GPU_SRCS} PROPERTIES LANGUAGE CXX)

  # Note: this is needed in SYCL to prevent compiling it as a
  # a seperate TU and cause multi-definition error from all the
  # `#inlcudes <>` of this file
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/rys_roots_for_k.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/rys_roots.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/rys_contract_k.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/vhf.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/rys_roots.cuh
    PROPERTIES HEADER_FILE_ONLY ON
  )

  set_target_properties(gvhf_rys PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
  target_compile_options(gvhf_rys PRIVATE -x c++ -nocudainc -nocudalib)
else()
  list(APPEND GPU_SRCS rys_roots_dat.cu rys_constant.cu)
  add_library(gvhf_rys SHARED ${GPU_SRCS})

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

  set_target_properties(gvhf_rys PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    CUDA_SEPARABLE_COMPILATION ON)
endif()

#option(BUILD_SHARED_LIBS "build shared libraries" 1)
#option(ENABLE_STATIC "Enforce static library build" 0)
#if(ENABLE_STATIC)
#  set(BUILD_SHARED_LIBS 0)
#endif()

target_link_libraries(gvhf_rys OpenMP::OpenMP_C)
