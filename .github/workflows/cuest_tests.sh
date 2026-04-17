# Given our CI machines are very old and CUDA 13 has already drop the support, we do not make a CI job for cuEST.

MY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

pytest ${MY_DIR}/../../gpu4pyscf/lib/tests/test_cuest_wrapper.py

# Notice there's a special test: test_K_spherical_mo_input_really_low_precision
# If it failes because of the last check (an error message will be printed),
# that indicates the low-precision emulation is not turned on. You can still use cuEST, it's just not
# as fast as it should be.
