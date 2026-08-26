/* CUB -> hipCUB compatibility stub. See gpu_runtime.h in this directory.
 * Lets existing `#include <cub/cub.cuh>` resolve on ROCm builds and maps the
 * cub:: namespace onto hipcub::. This directory is placed on the include path
 * only for HIP builds. Only block-level primitives (BlockReduce/BlockScan) and
 * their algorithm enums are needed by gpu4pyscf, all of which hipCUB provides. */
/* The C <complex.h> header (included by some sources) defines `I` as the
 * imaginary-unit macro, which collides with rocPRIM's `template<size_t I, ...>`
 * (pulled in via hipCUB). Undefine it before including hipCUB. */
#undef I
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
