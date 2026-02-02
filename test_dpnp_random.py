import numpy as np
import dpnp

# Exact types/values you mentioned
M = np.int64(31)
N = np.int64(31)
K = 63  # plain Python int
dtype = dpnp.float64

sizes = [(M, K), (M, N), (K, N)]

tmp=[dpnp.random.random(size).astype(dtype) for size in sizes]
print(tmp)
