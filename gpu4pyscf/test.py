import inspect
import cupy as cp
import gpu4pyscf
import gpu4pyscf.pbc.gto.int1e as int1e

print("gpu4pyscf.__file__ =", gpu4pyscf.__file__)
#print("cupy.__file__      =", cp.__file__)
print("int1e.__file__     =", int1e.__file__)

src = inspect.getsource(int1e._Int1eOpt.generate_shl_pairs)
print("print-line-present =", "here from int1e.py" in src)
