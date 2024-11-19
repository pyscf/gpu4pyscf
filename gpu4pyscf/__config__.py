import cupy

props = cupy.cuda.runtime.getDeviceProperties(0)
GB = 1024*1024*1024
min_ao_blksize = 128
min_grid_blksize = 128*128
ao_aligned = 32
grid_aligned = 256

# Use smaller blksize for old gaming GPUs
if props['totalGlobalMem'] < 16 * GB:
    min_ao_blksize = 64
    min_grid_blksize = 64*64

# Use 90% of the global memory for CuPy memory pool
mem_fraction = 0.9
cupy.get_default_memory_pool().set_limit(fraction=mem_fraction)
