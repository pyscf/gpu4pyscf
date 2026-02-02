import dpnp as dp

nao, ngrids = 24, 4096
elems_plane = nao * ngrids

arena  = dp.empty(4 * elems_plane, dtype=dp.float64)
slice1 = arena[elems_plane:]  # non-zero offset view

# Construct shaped array over the *view*:
plane1 = dp.ndarray((nao, ngrids), dtype=arena.dtype, buffer=slice1)

def ptr(a):
    return int(a.data.ptr)

print("arena ptr :", hex(ptr(arena)))
print("slice1 ptr:", hex(ptr(slice1)))   # expected start for plane1
print("plane1 ptr:", hex(ptr(plane1)))   # BUG: equals arena ptr (offset lost)

# This should hold if buffer=view were respected:
print("EXPECT plane1.ptr == slice1.ptr:", ptr(plane1) == ptr(slice1))
