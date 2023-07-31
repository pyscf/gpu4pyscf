import cupy

props = cupy.cuda.runtime.getDeviceProperties(0)

# for A100-80G
if props['name'][-4:] == b'80GB':
    min_ao_blksize = 256
    min_grid_blksize = 256*256
    ao_aligned = 64
    grid_aligned = 128
    mem_fraction = 0.9
    number_of_threads = 2048 * 108
# for V100-32G
elif props['name'][-4:] == b'32GB':
    min_ao_blksize = 128
    min_grid_blksize = 128*128
    ao_aligned = 32
    grid_aligned = 128
    mem_fraction = 0.9
    number_of_threads = 1024 * 80
else:
    min_ao_blksize = 64
    min_grid_blksize = 32*32
    ao_aligned = 16
    grid_aligned = 16*16
    mem_fraction = 0.6
    number_of_threads = 1024 * 80
    
cupy.get_default_memory_pool().set_limit(fraction=mem_fraction)