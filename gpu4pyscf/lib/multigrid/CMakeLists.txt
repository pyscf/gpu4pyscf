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

# mgrid_v2 is not SYCL-ported (multigrid_v2.py tolerates a missing libmgrid_v2).
if (ENABLE_MULTIGRID_V2 AND NOT USE_SYCL)
add_library(mgrid_v2 SHARED
        multigrid_v2/drivers.cu
        multigrid_v2/screen.cu
        multigrid_v2/eval_xc.cu
        multigrid_v2/eval_xc_grad.cu
)

set_target_properties(mgrid_v2 PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
        CUDA_SEPARABLE_COMPILATION ON
)
if(ENABLE_FP32_MULTIGRID)
  target_compile_definitions(mgrid_v2 PRIVATE GPU4PYSCF_ENABLE_FP32_MULTIGRID)
endif()
endif()

add_library(mgrid_v3 SHARED
  multigrid_v3/screen.cu
  multigrid_v3/drivers.cu
  #multigrid_v3/eval_density.cu
  #multigrid_v3/eval_tau.cu
  #multigrid_v3/eval_lda_mat.cu
  #multigrid_v3/eval_mgga_mat.cu
  multigrid_v3/eval_density_v2.cu
  multigrid_v3/eval_tau_v2.cu
  multigrid_v3/eval_lda_mat_v2.cu
  multigrid_v3/eval_mgga_mat_v2.cu
  multigrid_v3/eval_lda_strain_grad.cu
  multigrid_v3/eval_mgga_strain_grad.cu
  multigrid_v3/aft_eval_density.cu
  multigrid_v3/aft_eval_tau.cu
  multigrid_v3/aft_eval_lda_mat.cu
  multigrid_v3/aft_eval_mgga_mat.cu
)

set_target_properties(mgrid_v3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON
)

if (USE_SYCL)
  file(GLOB CUH_HEADERS_v3
    "${CMAKE_CURRENT_SOURCE_DIR}/multigrid_v3/*.cuh"
  )
  get_target_property(GPU_SRCS_v3 mgrid_v3 SOURCES)
  set_source_files_properties(${CUH_HEADERS_v3} ${GPU_SRCS_v3} PROPERTIES LANGUAGE CXX)
  target_compile_options(mgrid_v3 PRIVATE -x c++ -nocudainc -nocudalib)
  # Header-defined device helpers must be inline: every TU of this library
  # includes them, and plain always_inline functions would be duplicate
  # symbols at link time. __device__ expands to this attribute (sycl_device.hpp).
  target_compile_definitions(mgrid_v3 PRIVATE
    "GPU4PYSCF_DEVICE_ATTR=inline __attribute__((always_inline))")
  target_link_libraries(mgrid_v3 PRIVATE sycl_compat gsycl)
endif()
