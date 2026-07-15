/*
 * Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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

#pragma once

// Each including .cu must define GVHF_FILE_TAG to a file-unique token before
// including this header, so auto-generated SYCL kernel names never collide.
#ifndef GVHF_FILE_TAG
#error "define GVHF_FILE_TAG before including launch.cuh"
#endif

#define GVHF_CAT_(a, b, c) a##_##b##_##c
#define GVHF_CAT(a, b, c)  GVHF_CAT_(a, b, c)
#define GVHF_TAG(KFN)      GVHF_CAT(GVHF_FILE_TAG, KFN, __LINE__)

// Launch macros expect `blocks`, `threads`, `stream` in scope. SYCL kernel type
// is auto-named GVHF_FILE_TAG_KFN_<line>; CUDA ignores the name.
#ifdef USE_SYCL

#define GVHF_LAUNCH(KFN) \
    stream.parallel_for<class GVHF_TAG(KFN)>( \
        sycl::nd_range<2>(blocks * threads, threads), \
        [=](auto item) { KFN(dev_envs, dev_jk, dev_offsets); })

#define GVHF_LAUNCH_T(KFN, ...) \
    stream.parallel_for<class GVHF_TAG(KFN)>( \
        sycl::nd_range<2>(blocks * threads, threads), \
        [=](auto item) { KFN<__VA_ARGS__>(dev_envs, dev_jk, dev_offsets); })

#define GVHF_LAUNCH_SHM(GSIZE, KFN) \
    stream.submit([&](sycl::handler &cgh) { \
        sycl::local_accessor<double, 1> local_acc(sycl::range<1>(GSIZE), cgh); \
        cgh.parallel_for<class GVHF_TAG(KFN)>( \
            sycl::nd_range<2>(blocks * threads, threads), \
            [=](auto item) { \
                KFN(dev_envs, dev_jk, dev_offsets, item, \
                    GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); })

#else

#define GVHF_LAUNCH(KFN) \
    KFN<<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets)

#define GVHF_LAUNCH_T(KFN, ...) \
    KFN<__VA_ARGS__><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets)

#define GVHF_LAUNCH_SHM(GSIZE, KFN) \
    KFN<<<blocks, threads, (GSIZE)*sizeof(double), stream>>>(*envs, *jk, *offsets)

#endif
