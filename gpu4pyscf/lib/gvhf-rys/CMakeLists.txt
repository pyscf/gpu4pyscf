set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

add_library(gvhf_rys SHARED
  rys_contract_jk.cu rys_jk_driver.cu rys_roots_dat.cu
  unrolled_os.cu unrolled_rys.cu
  nr_sr_estimator.c
  rys_contract_j.cu cart2xyz.c unrolled_rys_j.cu
  count_tasks.cu
  rys_contract_jk_ip1.cu unrolled_rys_ip1.cu unrolled_ejk_ip1.cu
  rys_contract_jk_ip2.cu unrolled_ejk_ip2_type12.cu unrolled_ejk_ip2_type3.cu
)

#option(BUILD_SHARED_LIBS "build shared libraries" 1)
#option(ENABLE_STATIC "Enforce static library build" 0)
#if(ENABLE_STATIC)
#  set(BUILD_SHARED_LIBS 0)
#endif()

set_target_properties(gvhf_rys PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(gvhf_rys OpenMP::OpenMP_C)
