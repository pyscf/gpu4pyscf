import dpnp as cp
import numpy as np

comp=4
nao_max=24
MIN_BLK_SIZE=4096

cu_array = cp.empty((comp, nao_max, MIN_BLK_SIZE), order='C')
np_array = np.empty((comp, nao_max, MIN_BLK_SIZE), order='C')

print("cu_array: ", hex(cu_array.data.ptr))
print("cu_array dtype/shape/strides:", cu_array.dtype, cu_array.shape, cu_array.strides)
print("np_array dtype/shape/strides:", np_array.dtype, np_array.shape, np_array.strides)
