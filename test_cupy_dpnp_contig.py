import numpy as np
import dpnp as dp

print("dpnp version:", getattr(dp, "__version__", "unknown"))

# Build the same data as in the larger code
ctr_offsets_slice = [
    dp.array([0, 0, 0], dtype=np.int32),
    dp.array([3, 3, 3], dtype=np.int32),
    dp.array([5, 5, 5], dtype=np.int32),
    dp.array([7, 7, 7], dtype=np.int32),
    dp.array([10, 10, 10], dtype=np.int32),
    dp.array([11, 11, 11], dtype=np.int32),
    dp.array([12, 12, 12], dtype=np.int32),
]

# 3a: stack -> dpnp_array (should be C-contiguous)
temp1 = dp.stack(ctr_offsets_slice)

# 3b: transpose -> dpnp_array view (not C-contiguous)
temp2 = dp.stack(ctr_offsets_slice).T

# 3c: device->host with "order='C'"
# EXPECTED (per docstring "works exactly like numpy.asarray"): C-contiguous NumPy array
# ACTUAL: order is ignored for dpnp_array, result keeps non-C layout
temp3 = dp.asnumpy(temp2, order='C')

print("3a. Testing : temp1: ", bool(temp1.flags['C_CONTIGUOUS']), type(temp1), len(temp1), temp1)
print("3b. Testing : temp2: ", bool(temp2.flags['C_CONTIGUOUS']), type(temp2), len(temp2), temp2)
print("3c. Testing : temp3: ", bool(temp3.flags['C_CONTIGUOUS']), type(temp3), len(temp3), temp3)

# Programmatic check to make the failure obvious:
if not temp3.flags['C_CONTIGUOUS']:
    print("\nBUG: dpnp.asnumpy(dpnp_array, order='C') returned a non-C-contiguous NumPy array.")
    print("     strides:", temp3.strides, "| shape:", temp3.shape)
