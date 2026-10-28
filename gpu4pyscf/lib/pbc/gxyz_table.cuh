/*
 * Backend-split gxyz offset table for libpbc.
 *
 * NOTE: this device_global and PBC_make_gxyz_offset() (defined in
 * rys_contract_k.cu) are deliberately named differently from their
 * libgvhf_rys counterparts. Both libraries are loaded into the SAME process
 * (verified via /proc/self/maps), neither lists the other in DT_NEEDED, and
 * the symbols are GLOBAL DEFAULT visibility. When both exported
 * `s_pbc_gxyz_offset` / `PBC_make_gxyz_offset`, the dynamic linker bound
 * every caller in the process to whichever library happened to load first --
 * so one library's device_global was written by the other library's host
 * memcpy, leaving its own copy uninitialised. Reading it yields garbage
 * int8_t offsets that flow into load_dm() as an arbitrary pointer
 * displacement. Keep these names library-unique.
 */

#pragma once

#ifdef USE_SYCL
// Declared extern in every TU, defined once in rys_contract_k.cu.
#define PBC_GXYZ_DECLARE() \
    extern SYCL_EXTERNAL sycl_device_global<GXYZOffset[625]> s_pbc_gxyz_offset
#define PBC_GXYZ_DEFINE() \
    SYCL_EXTERNAL sycl_device_global<GXYZOffset[625]> s_pbc_gxyz_offset
// Inside kernels: SYCL reads the pre-filled device_global (plus the
// template OFFSET); CUDA reads the chunk the launcher copied into the
// 256-entry __constant__ c_gxyz_offset, so no offset here.
#define PBC_GXYZ_SELECT(gxyz_offsets) \
    auto gxyz_offsets = s_pbc_gxyz_offset.get() + OFFSET; \
    (void)p_gxyz_offsets
// Tail of PBC_make_gxyz_offset(): publish the host table to the device.
#define PBC_GXYZ_FILL(goff, nf) \
    sycl_get_queue()->memcpy(s_pbc_gxyz_offset, goff, max(nf, 256) * sizeof(GXYZOffset)).wait(); \
    return nullptr
// Per-launch 256-tile chunk copy. SYCL needs nothing (full table already
// published); CUDA copies the chunk into c_gxyz_offset.
#define PBC_GXYZ_COPY_CHUNK(gxyz_offset, OFFSET, tile_chunk) \
    do { } while (0)
#else
#define PBC_GXYZ_DECLARE() /* c_gxyz_offset declared in gvhf-rys/vhf.cuh */
#define PBC_GXYZ_DEFINE() /* c_gxyz_offset defined in gvhf-rys/rys_constant.cu */
#define PBC_GXYZ_SELECT(gxyz_offsets) \
    const GXYZOffset *gxyz_offsets = p_gxyz_offsets
#define PBC_GXYZ_FILL(goff, nf) \
    GXYZOffset *p_gxyz_offset; \
    cudaGetSymbolAddress((void **)&p_gxyz_offset, c_gxyz_offset); \
    return p_gxyz_offset
#define PBC_GXYZ_COPY_CHUNK(gxyz_offset, OFFSET, tile_chunk) \
    checkCudaErrors( \
        cudaMemcpyToSymbol(c_gxyz_offset, gxyz_offset + OFFSET, \
                           tile_chunk * sizeof(GXYZOffset), \
                           0, cudaMemcpyHostToDevice))
#endif
