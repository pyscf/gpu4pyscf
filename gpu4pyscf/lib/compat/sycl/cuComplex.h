#pragma once
// Drop-in replacement for <cuComplex.h> when building with SYCL.
// The real header also pulls in <vector_types.h>, so float2 lives here too.
#include "sycl_device.hpp"

// CUDA float2 stand-in. sycl::float2's element accessors are methods
// (v.x()), not members (v.x), so aliasing to it would break every
// `.x`/`.y` site that is shared with the CUDA build. This POD keeps both
// backends identical and supports brace-init assignment.
struct alignas(8) float2 { float x, y; };

// CUDA cuDoubleComplex stand-in. No SYCL analogue exists; this POD matches
// cuDoubleComplex's memory layout (two doubles) so device-pointer
// reinterpretation from Python callers is unaffected. A real struct with
// real members, so `.x`/`.y` access and brace-init need no further changes.
struct alignas(16) cuDoubleComplex { double x, y; };
