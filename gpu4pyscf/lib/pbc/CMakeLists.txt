set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

add_library(pbc SHARED
  ft_ao.cu ft_ao_ip1.cu contract_ft_ao.cu
  fill_int3c2e.cu contract_int3c2e.cu ejk_int3c2e_ip1.cu
  fill_int2c2e.cu fill_int2c2e_ip1.cu e_int2c2e_ip1.cu
  contract_int3c2e_pvp.cu
  overlap.cu ppnl.cu
  supmol_sr_estimator.cu
  rys_roots_dat.cu
  rys_contract_k.cu rys_contract_jk_ip1.cu
  rys_contract_j.cu
  unrolled_rys_k.cu
  nr_eval_gto.cu
  sorting.c write_cderi.c int3c2e_recontract.c
  screen.cu
  fill_triu.cu
)

set_target_properties(pbc PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)
gpu4pyscf_configure_gpu_target(pbc)

if (USE_SYCL)
  # rys_roots_dat.cu is included by rys_roots.cuh as constexpr tables; do not
  # also compile it as its own TU.
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/rys_roots_dat.cu
    PROPERTIES HEADER_FILE_ONLY ON)
  # gvhf-rys/vhf.cuh spells the gxyz device_global s_rys_gxyz_offset. libpbc.so
  # and libgvhf_rys.so are loaded into one process with no DT_NEEDED link and
  # default symbol visibility, so identical names would bind both libraries to
  # whichever loaded first and one library's host memcpy would land in the
  # other's device image (kernels then read uninitialised offsets). Give
  # libpbc its own name.
  target_compile_definitions(pbc PRIVATE s_rys_gxyz_offset=s_pbc_gxyz_offset)
endif()

target_link_libraries(pbc PRIVATE OpenMP::OpenMP_C)
