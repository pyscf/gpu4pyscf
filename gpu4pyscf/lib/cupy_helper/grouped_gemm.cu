/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/core_io.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

#define CUTLASS_CHECK(status)                                                                    \
{                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
        std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at line: " << __LINE__ \
                << std::endl;                                                                    \
        exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
}

static int get_device_compute_capability() {
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);

  return properties.major * 10 + properties.minor;
}

// A100
using cutlass_tensorop_d884gemm_grouped_64x128_16x3_tt_align1_base =
  typename cutlass::gemm::kernel::DefaultGemmGrouped<
    double, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,
    double, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,
    double, cutlass::layout::RowMajor,
    double,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    cutlass::arch::OpMultiplyAdd
>::GemmKernel;

// V100
using cutlass_simt_dgemm_grouped_64x128_8x2_tt_align1_base =
  typename cutlass::gemm::kernel::DefaultGemmGrouped<
    double, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,
    double, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,
    double, cutlass::layout::RowMajor,
    double,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    // cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
    2,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    cutlass::arch::OpMultiplyAdd
>::GemmKernel;

template<typename DeviceKernel>
cutlass::Status grouped_gemm_kernel_run(int problem_count, cutlass::gemm::GemmCoord* problem_sizes,
                        typename DeviceKernel::ElementA** A, typename DeviceKernel::ElementB** B, typename DeviceKernel::ElementC** C, typename DeviceKernel::ElementC** D,
                        int64_t* lda, int64_t* ldb, int64_t* ldc, int64_t* ldd,
                        typename DeviceKernel::EpilogueOutputOp::ElementCompute alpha, typename DeviceKernel::EpilogueOutputOp::ElementCompute beta) {

  int threadblock_count = DeviceKernel::sufficient();

  typename DeviceKernel::Arguments arguments {
    problem_sizes,
    problem_count,
    threadblock_count,
    {alpha, beta},
    A, B, C, D,
    lda, ldb, ldc, ldd
  };

  size_t workspace_size = DeviceKernel::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  DeviceKernel gemm_op;
  cutlass::Status status = gemm_op.initialize(arguments,
                                              workspace.get(),
                                              nullptr);     // CUDA stream

  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  status = gemm_op();
  return status;
}

template<typename DeviceKernel>
void grouped_gemm_kernel_launch(uint64_t *out, uint64_t *x, uint64_t *y, int64_t *Ms, int64_t *Ns, int64_t *Ks, int num)
{
  size_t total_size = sizeof(cutlass::gemm::GemmCoord) +
                      sizeof(typename DeviceKernel::ElementA*) +
                      sizeof(typename DeviceKernel::ElementB*) +
                      sizeof(typename DeviceKernel::ElementC*) +
                      sizeof(typename DeviceKernel::ElementC*) +
                      sizeof(int64_t) +
                      sizeof(int64_t) +
                      sizeof(int64_t);
  total_size *= num;

  int64_t padding = 8 - (total_size % 8);
  total_size += padding;

  uint8_t* host_data = new uint8_t[total_size];
  cutlass::DeviceAllocation<uint8_t> device_data(total_size);

  uint8_t* start = host_data;
  cutlass::gemm::GemmCoord* problem_sizes_host = reinterpret_cast<cutlass::gemm::GemmCoord*>(start);

  // Apply the padding after the list of GemmCoords
  start += num * sizeof(cutlass::gemm::GemmCoord) + padding;

  int64_t ptr_A_offset = start - host_data;
  typename DeviceKernel::ElementA** ptr_A_host = reinterpret_cast<typename DeviceKernel::ElementA**>(start);
  start += num * sizeof(typename DeviceKernel::ElementA*);

  int64_t ptr_B_offset = start - host_data;
  typename DeviceKernel::ElementB** ptr_B_host = reinterpret_cast<typename DeviceKernel::ElementB**>(start);
  start += num * sizeof(typename DeviceKernel::ElementB*);

  int64_t ptr_C_offset = start - host_data;
  typename DeviceKernel::ElementC** ptr_C_host = reinterpret_cast<typename DeviceKernel::ElementC**>(start);
  start += num * sizeof(typename DeviceKernel::ElementC*);

  int64_t ptr_D_offset = start - host_data;
  typename DeviceKernel::ElementC** ptr_D_host = reinterpret_cast<typename DeviceKernel::ElementC**>(start);
  start += num * sizeof(typename DeviceKernel::ElementC*);

  int64_t lda_offset = start - host_data;
  int64_t* lda_host = reinterpret_cast<int64_t*>(start);
  start += num * sizeof(int64_t);

  int64_t ldb_offset = start - host_data;
  int64_t* ldb_host = reinterpret_cast<int64_t*>(start);
  start += num * sizeof(int64_t);

  int64_t ldc_offset = start - host_data;
  int64_t* ldc_host = reinterpret_cast<int64_t*>(start);
  start += num * sizeof(int64_t);

  double alpha = 1.0;
  double beta = 0.0;

  for (size_t i = 0; i < num; ++i) {
      int M = Ms[i];
      int N = Ns[i];
      int K = Ks[i];
      *(problem_sizes_host + i) = {M, N, K};

      *(ptr_A_host + i) = reinterpret_cast<typename DeviceKernel::ElementA*>(x[i]);
      *(ptr_B_host + i) = reinterpret_cast<typename DeviceKernel::ElementB*>(y[i]);
      *(ptr_C_host + i) = nullptr;
      *(ptr_D_host + i) = reinterpret_cast<typename DeviceKernel::ElementC*>(out[i]);

      *(lda_host + i) = DeviceKernel::LayoutA::packed({M, K}).stride(0);
      *(ldb_host + i) = DeviceKernel::LayoutB::packed({K, N}).stride(0);
      *(ldc_host + i) = DeviceKernel::LayoutC::packed({M, N}).stride(0);
  }

  device_data.copy_from_host(host_data);

  cutlass::Status status = grouped_gemm_kernel_run<DeviceKernel>(
      num,
      reinterpret_cast<cutlass::gemm::GemmCoord*>(device_data.get()),
      reinterpret_cast<typename DeviceKernel::ElementA**>(device_data.get() + ptr_A_offset),
      reinterpret_cast<typename DeviceKernel::ElementB**>(device_data.get() + ptr_B_offset),
      reinterpret_cast<typename DeviceKernel::ElementC**>(device_data.get() + ptr_C_offset),
      reinterpret_cast<typename DeviceKernel::ElementC**>(device_data.get() + ptr_D_offset),
      reinterpret_cast<int64_t*>(device_data.get() + lda_offset),
      reinterpret_cast<int64_t*>(device_data.get() + ldb_offset),
      reinterpret_cast<int64_t*>(device_data.get() + ldc_offset),
      reinterpret_cast<int64_t*>(device_data.get() + ldc_offset),
      typename DeviceKernel::EpilogueOutputOp::ElementCompute(alpha), typename DeviceKernel::EpilogueOutputOp::ElementCompute(beta));

  delete[] host_data;

  CUTLASS_CHECK(status);
}

extern "C" {
// int dgemm(cudaStream_t stream, double **ptr_out, double **ptr_x, double **ptr_y, int64_t *Ms, int64_t *Ns, int64_t *Ks, int64_t *MNKs, int groups)
int grouped_gemm(cudaStream_t stream, uint64_t *out, uint64_t *x, uint64_t *y, int64_t *Ms, int64_t *Ns, int64_t *Ks, int num)
{
    int compute_capability = get_device_compute_capability();

    if(compute_capability < 80)
    {
      using DeviceKernel = cutlass::gemm::device::GemmGrouped<cutlass_simt_dgemm_grouped_64x128_8x2_tt_align1_base>;
      grouped_gemm_kernel_launch<DeviceKernel>(out, x, y, Ms, Ns, Ks, num);
    }
    else if(compute_capability >= 80)
    {
      using DeviceKernel = cutlass::gemm::device::GemmGrouped<cutlass_tensorop_d884gemm_grouped_64x128_16x3_tt_align1_base>;
      grouped_gemm_kernel_launch<DeviceKernel>(out, x, y, Ms, Ns, Ks, num);
    }
    else
    {
        printf("Unsupported GPU architecture: %d\n", compute_capability);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    // printf("%s\n", cudaGetErrorString(err));
    if (err != cudaSuccess)
        return 1;
    return 0;
}
}
