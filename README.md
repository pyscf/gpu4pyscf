GPU plugin for PySCF
====================
Installation
--------

> [!NOTE]
> The compiled binary packages support compute capability 7.0 and later (Volta and later, such as Tesla V100, RTX 20 series and later). For older GPUs (GTX 10**, Tesla P100), please compile the package with the source code as follows.

Run ```nvidia-smi``` in your terminal to check the installed CUDA version.

Choose the proper package based on your CUDA environment.

| Platform      | Command                               |
----------------| --------------------------------------|
| **CUDA 11.x** |  ```pip3 install gpu4pyscf-cuda11x``` |
| **CUDA 12.x** |  ```pip3 install gpu4pyscf-cuda12x``` |

```cuTensor``` is **highly recommended** for accelerating tensor contractions.

For **CUDA 11.x**, ```pip3 install cutensor-cu11```

For **CUDA 12.x**, ```pip3 install cutensor-cu12```

Compilation
--------
The package provides ```dockerfiles/compile/Dockerfile``` for creating the CUDA environment. One can compile the package with
```sh
sh build.sh
```
This script will automatically download LibXC, and compile it with CUDA. The script will also build the wheel for installation. The compilation can take more than 5 mins. Then, one can either install the wheel with
```sh
cd output
pip3 install gpu4pyscf-*
```
or simply add it to ```PYTHONPATH```
```sh
export PYTHONPATH="${PYTHONPATH}:/your-local-path/gpu4pyscf"
```
Then install cutensor for acceleration
```sh
python -m cupyx.tools.install_library --cuda 11.x --library cutensor
```

Features
--------
- Density fitting scheme and direct SCF scheme;
- SCF, analytical Gradient, and analytical Hessian calculations for Hartree-Fock and DFT;
- LDA, GGA, mGGA, hybrid, and range-separated functionals via [libXC](https://gitlab.com/libxc/libxc/-/tree/master/);
- Geometry optimization and transition state search via [geomeTRIC](https://geometric.readthedocs.io/en/latest/);
- Dispersion corrections via [DFTD3](https://github.com/dftd3/simple-dftd3) and [DFTD4](https://github.com/dftd4/dftd4);
- Nonlocal functional correction (vv10) for SCF and gradient;
- ECP is supported and calculated on CPU;
- PCM models, SMD model, their analytical gradients, and semi-analytical Hessian matrix;

Limitations
--------
- Rys roots up to 8 for density fitting scheme;
- Rys roots up to 9 for direct scf scheme;
- Atomic basis up to g orbitals;
- Auxiliary basis up to h orbitals;
- Density fitting scheme up to ~168 atoms with def2-tzvpd basis, bounded CPU memory;
- Hessian is unavailable for Direct SCF yet;
- meta-GGA without density laplacian;

Examples
--------
```python
import pyscf
from gpu4pyscf.dft import rks

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf = rks.RKS(mol, xc='LDA').density_fit()

e_dft = mf.kernel()  # compute total energy
print(f"total energy = {e_dft}")

g = mf.nuc_grad_method()
g_dft = g.kernel()   # compute analytical gradient

h = mf.Hessian()
h_dft = h.kernel()   # compute analytical Hessian

```
Find more examples in gpu4pyscf/examples

Benchmarks
--------
Speedup with GPU4PySCF v0.6.0 on A100-80G over Q-Chem 6.1 on 32-cores CPU (Desity fitting, SCF, def2-tzvpp, def2-universal-jkfit, B3LYP, (99,590))

| mol               |   natm |    LDA |    PBE |   B3LYP |    M06 |   wB97m-v |
|:------------------|-------:|-------:|-------:|--------:|-------:|----------:|
| 020_Vitamin_C     |     20 |   2.86 |   6.09 |   13.11 |  11.58 |     17.46 |
| 031_Inosine       |     31 |  13.14 |  15.87 |   16.57 |  25.89 |     26.14 |
| 033_Bisphenol_A   |     33 |  12.31 |  16.88 |   16.54 |  28.45 |     28.82 |
| 037_Mg_Porphin    |     37 |  13.85 |  19.03 |   20.53 |  28.31 |     30.27 |
| 042_Penicillin_V  |     42 |  10.34 |  13.35 |   15.34 |  22.01 |     24.2  |
| 045_Ochratoxin_A  |     45 |  13.34 |  15.3  |   19.66 |  27.08 |     25.41 |
| 052_Cetirizine    |     52 |  17.79 |  17.44 |   19    |  24.41 |     25.87 |
| 057_Tamoxifen     |     57 |  14.7  |  16.57 |   18.4  |  24.86 |     25.47 |
| 066_Raffinose     |     66 |  13.77 |  14.2  |   20.47 |  22.94 |     25.35 |
| 084_Sphingomyelin |     84 |  14.24 |  12.82 |   15.96 |  22.11 |     24.46 |
| 095_Azadirachtin  |     95 |   5.58 |   7.72 |   24.18 |  26.84 |     25.21 |
| 113_Taxol         |    113 |   5.44 |   6.81 |   24.58 |  29.14 |    nan    |

Find more benchmarks in gpu4pyscf/benchmarks

