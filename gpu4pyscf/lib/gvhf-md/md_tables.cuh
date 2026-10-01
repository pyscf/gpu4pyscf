// Backend-split definition macros for the md_indices.cu lookup tables.
//
// md_indices.cu is dual-purpose: under CUDA it compiles as a translation
// unit defining __device__/__constant__ tables; under SYCL md_j.cuh
// #includes it directly so the tables become `inline constexpr` (SYCL has
// no cross-TU __device__ linkage, and compiling it as a TU as well would
// be redundant). The macros keep md_indices.cu itself free of backend
// branches.
#pragma once

#ifdef USE_SYCL
#define MD_TABLE(type, name) inline constexpr type name[] =
#define MD_CTABLE(type, name) inline constexpr type name[] =
#else
#define MD_TABLE(type, name) __device__ type name[] =
#define MD_CTABLE(type, name) __constant__ type name[] =
#endif
