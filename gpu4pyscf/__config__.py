import cupy

_num_devices = cupy.cuda.runtime.getDeviceCount()

# TODO: switch to non_blocking stream (blocked by libxc)
_streams = [None] * _num_devices
for device_id in range(_num_devices):
    with cupy.cuda.Device(device_id):
        _streams[device_id] = cupy.cuda.stream.Stream(non_blocking=False)

props = cupy.cuda.runtime.getDeviceProperties(0)
GB = 1024*1024*1024
# such as A100-80G
if props['totalGlobalMem'] >= 64 * GB:
    min_ao_blksize = 128
    min_grid_blksize = 128*128
    ao_aligned = 32
    grid_aligned = 256
    mem_fraction = 0.9
    number_of_threads = 2048 * 108
# such as V100-32G
elif props['totalGlobalMem'] >= 32 * GB:
    min_ao_blksize = 128
    min_grid_blksize = 128*128
    ao_aligned = 32
    grid_aligned = 256
    mem_fraction = 0.9
    number_of_threads = 1024 * 80
# such as A30-24GB
elif props['totalGlobalMem'] >= 16 * GB:
    min_ao_blksize = 128
    min_grid_blksize = 128*128
    ao_aligned = 32
    grid_aligned = 256
    mem_fraction = 0.9
    number_of_threads = 1024 * 80
# other gaming cards
else:
    min_ao_blksize = 64
    min_grid_blksize = 64*64
    ao_aligned = 32
    grid_aligned = 128
    mem_fraction = 0.9
    number_of_threads = 1024 * 80

cupy.get_default_memory_pool().set_limit(fraction=mem_fraction)

