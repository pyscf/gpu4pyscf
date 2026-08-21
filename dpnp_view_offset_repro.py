"""
dpnp: ndarray.view() ignores the array's USM element offset

dpnp_array._create_view() rebuilds the result with
    dpt.usm_ndarray(shape, dtype=dtype, buffer=self._array_obj, strides=...)
and never forwards self._array_obj._element_offset. dpctl interprets
`buffer=<usm_ndarray>` as the whole underlying USM allocation, so any array
that does not start at the base of its allocation gets a view onto the wrong
memory.

This is the same class of bug as the already-fixed #2641 / #2781
(".data.ptr on array views ignores USM offset"), but for _create_view()
rather than .data.ptr.

Impact is not limited to explicit .view() calls: dpnp.einsum() takes a
"returns_view" fast path for a single operand with no summed index -- any pure
permutation, including the identity 'abc->abc' -- and does
`operands = [a.view() for a in operands]`. So einsum over any sliced operand
silently returns values read from the base of the parent buffer. It is silent:
no exception, no warning, just wrong numbers.
"""
import sys

import numpy as np
import dpnp

if "gpu4pyscf" in sys.modules:
    raise SystemExit(
        "Run this with a clean interpreter: gpu4pyscf's compatibility shim "
        "patches dpnp_array._create_view and hides the bug.")

print("dpnp", dpnp.__version__)
fail = 0


def check(label, got, want):
    global fail
    got = dpnp.asnumpy(got) if isinstance(got, dpnp.ndarray) else np.asarray(got)
    ok = np.array_equal(got, want)
    fail += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        print(f"        got  {got.ravel()[:8]}")
        print(f"        want {want.ravel()[:8]}")


# ---------------------------------------------------------------- 1. view()
h = np.arange(10.0)
d = dpnp.asarray(h)
check("x[3:].view()                  ", d[3:].view(), h[3:])
check("x[3:].view(dpnp.float64)      ", d[3:].view(dpnp.float64), h[3:])

h2 = np.arange(12.0).reshape(3, 4)
d2 = dpnp.asarray(h2)
check("x[1:].view()  (2-D)           ", d2[1:].view(), h2[1:])

# Zero-offset views are fine -- shows the offset is precisely what is lost.
check("x[0:].view()  (offset 0, ok)  ", d[0:].view(), h[0:])

# --------------------------------------------------- 2. einsum on a view
# einsum's `returns_view` path calls .view() internally, so a pure
# permutation of a sliced operand silently reads the wrong memory.
h3 = np.arange(24.0).reshape(2, 3, 4)
d3 = dpnp.asarray(h3)
check("einsum('abc->abc', x[:,1:,:]) ",
      dpnp.einsum("abc->abc", d3[:, 1:, :]), np.einsum("abc->abc", h3[:, 1:, :]))
check("einsum('abc->cab', x[:,1:,:]) ",
      dpnp.einsum("abc->cab", d3[:, 1:, :]), np.einsum("abc->cab", h3[:, 1:, :]))
# Reducing and two-operand einsums do NOT take the view path and are correct,
# which is what makes the bug so easy to miss.
check("einsum('abc->ab',  x[:,1:,:]) ",
      dpnp.einsum("abc->ab", d3[:, 1:, :]), np.einsum("abc->ab", h3[:, 1:, :]))

print()
print(f"{fail} failure(s)")
raise SystemExit(1 if fail else 0)
