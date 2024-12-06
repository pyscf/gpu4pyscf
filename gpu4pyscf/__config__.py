import cupy

_num_devices = cupy.cuda.runtime.getDeviceCount()

# TODO: switch to non_blocking stream (currently blocked by libxc)
_streams = [None] * _num_devices
for device_id in range(_num_devices):
    with cupy.cuda.Device(device_id):
        _streams[device_id] = cupy.cuda.stream.Stream(non_blocking=False)

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

# Check P2P data transfer is available
_p2p_access = True
if _num_devices > 1:
    for src in range(_num_devices):
        for dst in range(_num_devices):
            if src != dst:
                can_access_peer = cupy.cuda.runtime.deviceCanAccessPeer(src, dst)
                _p2p_access &= can_access_peer
