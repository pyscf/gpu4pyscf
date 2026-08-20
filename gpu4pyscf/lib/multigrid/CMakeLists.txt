set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

add_library(mgrid SHARED
  mg_driver.cu estimator.cu
  eval_rho.cu
  eval_mat_lda.cu eval_mat_gga.cu
  eval_tau.cu eval_mat_tau.cu
)

set_target_properties(mgrid PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

#target_link_libraries(ft_ao OpenMP::OpenMP_C)

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

add_library(mgrid_v3 SHARED
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

set_target_properties(mgrid_v3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON
)
