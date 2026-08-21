"""
ExchCXX (the SYCL libxc shim) and libxc use DIFFERENT DENSITY CUTOFFS.

Root cause of every gpu4pyscf/dft/tests/test_libxc.py numerical failure on the
SYCL build. It is a convention difference at physically irrelevant densities,
NOT an accuracy bug in ExchCXX's functional forms.

At rho ~ 2e-15 on a real molecular grid:

    rho            exc_libxc(CPU)     exc_ExchCXX(GPU)    abs diff
    1.960751e-15   0.00000000e+00     -9.24398377e-06     9.24e-06

libxc returns exactly 0 because rho is under its density threshold; ExchCXX
evaluates the functional and returns the analytically CORRECT Slater value,
-Cx * rho^(1/3) = -0.7385587663820223 * (1.96e-15)^(1/3) = -9.2443e-06.

On smooth densities the two agree bit for bit -- LDA_X gpu/cpu ratio is
1.000000000000 across rho in [1e-3, 10], spread 3.3e-16.

The thresholds differ per functional and in both directions:

    functional     libxc zeroes below   ExchCXX zeroes below
    LDA_X          ~3e-15               never (tested to 1e-16)
    LDA_C_VWN      ~1e-15               ~3e-15   <-- GPU is stricter here
    GGA_C_LYP      ~1e-14               never

That asymmetry is why only some functionals trip the test's 1e-10 tolerance.
The metric is min(relative, absolute), so a functional only fails if its exc is
still LARGE at the disputed densities. exc ~ rho^(1/3) decays slowly, so LDA_X
is still 9e-6 at rho=2e-15 and fails; LDA_C_VWN's exc decays like rho, is tiny
there, and passes.

Component-by-component on a real grid (C2, ccpvtz, min(rel,abs) metric):

    LDA_X              exc=9.244e-06   vxc=1.233e-05   <-- cutoff
    LDA_C_VWN_RPA      exc=2.327e-05   vxc=3.089e-05   <-- cutoff
    LDA_C_VWN          exc=4.163e-17   vxc=5.551e-17       ok
    GGA_X_B88          exc=8.726e-16   vxc=8.986e-15       ok
    GGA_C_LYP          exc=2.928e-06   vxc=3.904e-06   <-- cutoff
    HYB_GGA_XC_B3LYP   exc=6.152e-06   vxc=8.178e-06   <-- inherits the above

B3LYP is simply the weighted mix of components that individually disagree; the
hybrid assembly itself is fine. (The comment at exchcxx.cpp:41 guessing at a
VWN5-vs-VWN_RPA mismatch is a red herring for this: GPU B3LYP is 6e-06 from
libxc B3LYP but 4e-03 from both B3LYP5 and B3LYP3, so the VWN variant is right.)

The fxc blow-ups have the same cause amplified: v2rho2 ~ rho^(-5/3) genuinely
diverges as rho -> 0, so at rho=2e-15 the correct value is ~1e20 while libxc
reports 0. Hence "fxc=1.0e+20" for B3LYP and "3.9e-02" for spin-polarized B88.

PRACTICAL IMPACT: essentially none for real calculations. These points carry
grid weight times a density of 1e-15; their contribution to Exc is far below any
convergence threshold. The consequence is that test_libxc.py's strict pointwise
comparison cannot pass until ExchCXX adopts libxc's per-functional
`dens_threshold` semantics (libxc exposes it as xc_func_type.dens_threshold).

Run:  python exchcxx_vs_libxc_repro.py
"""
import numpy as np
import pyscf
import cupy
from pyscf.dft import Grids
from pyscf.dft.numint import NumInt as numint_cpu
from gpu4pyscf.dft.numint import NumInt as numint_gpu

mol = pyscf.M(atom='''
C  -0.65830719,  0.61123287, -0.00800148
C   0.73685281,  0.61123287, -0.00800148
''', basis='ccpvtz', spin=None, output='/dev/null')

np.random.seed(1)
nao = mol.nao
dm0 = np.random.rand(nao, nao)
dm0 = dm0 + dm0.T


def _diff(dat, ref):
    """The metric test_libxc.py uses: min(relative, absolute) error."""
    d = dat - ref
    return np.min((abs(d / (ref + 1e-300)), abs(d)), axis=0)


def check(xc, spin, deriv=2):
    ni_cpu, ni_gpu = numint_cpu(), numint_gpu()
    xctype = ni_cpu._xc_type(xc)
    ao_deriv = 0 if xctype == 'LDA' else 1
    grids = Grids(mol).build()
    ao = ni_cpu.eval_ao(mol, grids.coords, ao_deriv)
    rho = ni_cpu.eval_rho(mol, ao, dm0, xctype=xctype)
    if spin != 0:
        rho = (rho, rho)

    c = ni_cpu.eval_xc_eff(xc, rho, deriv=deriv, xctype=xctype)
    g = ni_gpu.eval_xc_eff(xc, cupy.array(rho), deriv=deriv, xctype=xctype)

    out = []
    for name, gi, ci in zip(("exc", "vxc", "fxc"), g, c):
        if gi is None or ci is None:
            out.append(f"{name}=n/a")
            continue
        out.append(f"{name}={_diff(gi.get(), ci).max():.3e}")
    print(f"  {xc:22s} spin={spin}  " + "  ".join(out), flush=True)


print("ExchCXX (GPU) vs libxc (CPU); metric = min(relative, absolute) error")
print("tolerance used by test_libxc.py for exc and vxc is 1e-10\n")
for xc in ("LDA_C_VWN", "GGA_X_B88", "GGA_C_PBE", "HYB_GGA_XC_B3LYP"):
    for spin in (0, 1):
        try:
            check(xc, spin)
        except Exception as e:
            print(f"  {xc:22s} spin={spin}  ERROR {type(e).__name__}: {str(e)[:60]}", flush=True)
