#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")
#
#add_library(mgrid SHARED
#  mg_driver.cu estimator.cu
#  eval_rho.cu
#  eval_mat_lda.cu eval_mat_gga.cu
#  eval_tau.cu eval_mat_tau.cu
#)
#
#set_target_properties(mgrid PROPERTIES
#  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
#  CUDA_SEPARABLE_COMPILATION ON)

set(GPU_SRCS_v2
  multigrid_v2/drivers.cu
  multigrid_v2/screen.cu
  multigrid_v2/eval_xc.cu
  multigrid_v2/eval_xc_grad.cu
)

set(GPU_SRCS_v3
  multigrid_v3/screen.cu
  multigrid_v3/drivers.cu
  multigrid_v3/eval_density.cu
  multigrid_v3/eval_tau.cu
  #multigrid_v3/eval_lda_mat.cu
  #multigrid_v3/eval_mgga_mat.cu
  multigrid_v3/eval_lda_mat_v2.cu
  multigrid_v3/eval_mgga_mat_v2.cu
  multigrid_v3/eval_lda_strain_grad.cu
  multigrid_v3/eval_mgga_strain_grad.cu
  multigrid_v3/aft_eval_density.cu
  multigrid_v3/aft_eval_tau.cu
  multigrid_v3/aft_eval_lda_mat.cu
  multigrid_v3/aft_eval_mgga_mat.cu
)

add_library(mgrid_v2 SHARED ${GPU_SRCS_v2})
add_library(mgrid_v3 SHARED ${GPU_SRCS_v3})

if (USE_SYCL)
  file(GLOB CUH_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/multigrid_v2/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/multigrid_v3/*.cuh"
  )
  set_source_files_properties(${CUH_HEADERS} PROPERTIES LANGUAGE CXX)

  set_source_files_properties(${GPU_SRCS_v2} PROPERTIES LANGUAGE CXX)
  set_source_files_properties(${GPU_SRCS_v3} PROPERTIES LANGUAGE CXX)

  set_target_properties(mgrid_v2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
  target_compile_options(mgrid_v2 PRIVATE -x c++ -nocudainc -nocudalib)
  target_link_libraries(mgrid_v2 PRIVATE sycl_compat gsycl)

  set_target_properties(mgrid_v3 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
  target_compile_options(mgrid_v3 PRIVATE -x c++ -nocudainc -nocudalib)
  target_link_libraries(mgrid_v3 PRIVATE sycl_compat gsycl)
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

  set_target_properties(mgrid_v2 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    CUDA_SEPARABLE_COMPILATION ON)

  set_target_properties(mgrid_v3 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    CUDA_SEPARABLE_COMPILATION ON)
endif (USE_SYCL)

if(ENABLE_FP32_MULTIGRID)
  target_compile_definitions(mgrid_v2 PRIVATE GPU4PYSCF_ENABLE_FP32_MULTIGRID)
endif()
