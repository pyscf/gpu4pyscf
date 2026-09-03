GPU plugin for PySCF
====================
![nightly](https://github.com/pyscf/gpu4pyscf/actions/workflows/nightly_build.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/gpu4pyscf-cuda11x.svg)](https://badge.fury.io/py/gpu4pyscf-cuda11x)

Installation
--------

> [!NOTE]
> The compiled binary packages support compute capability 7.0 and later (Volta and later, such as Tesla V100, RTX 20 series and later).

To check your installed CUDA Toolkit version, run
```sh
nvcc --version
```
Then, install the appropriate package based on your CUDA version:

| Platform      | Command                               | cutensor (**highly recommended**)|
----------------| --------------------------------------|----------------------------------|
| **CUDA 11.x** |  ```pip3 install gpu4pyscf-cuda11x``` | ```pip3 install cutensor-cu11``` |
| **CUDA 12.x** |  ```pip3 install gpu4pyscf-cuda12x``` | ```pip3 install cutensor-cu12``` |
| **CUDA 13.x** |  ```pip3 install gpu4pyscf-cuda13x``` | ```pip3 install cutensor-cu13``` |

The versions of CuPy and cuTENSOR are strongly interdependent and should not be combined arbitrarily.
The recommended combinations include:
1. CuPy 13.3.0 + cuTENSOR 2.0.2
2. CuPy 13.4.1 + cuTENSOR 2.2.0

Using other versions or combinations may lead to failures in functionality. 
We **recommend** creating a dedicated environment using:
```sh
pip3 install --no-cache-dir -r requirements.txt
```
This ensures compatibility and reproducibility, especially since this configuration is used in our nightly benchmarks.

Compilation
--------
To compile the package, run the following commands:
```sh
git clone https://github.com/pyscf/gpu4pyscf.git
cd gpu4pyscf
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf
cmake --build build/temp.gpu4pyscf -j 4
CURRENT_PATH=`pwd`
export PYTHONPATH="${PYTHONPATH}:${CURRENT_PATH}"
```
Then install cutensor and cupy for acceleration (please switch the versions according to your runtime CUDA environment!)
```sh
pip3 install cutensor-cu12 cupy-cuda12x
```
There shouldn't be cupy or cutensor compilation during pip install process. If you see the following warning at the beginning of a gpu4pyscf job, it implies problems with cupy and cutensor installation (likely a version mismatch, or multiple versions of same package installed).
```
<repo_path>/gpu4pyscf/lib/cutensor.py:<line_number>: UserWarning: using cupy as the tensor contraction engine.
```

Building for AMD GPUs (ROCm)
--------
gpu4pyscf also runs on AMD GPUs through ROCm/HIP. It is built from source with
the standard `pip`/`setup.py` path (`USE_HIP=1` selects the HIP backend). The HIP
kernels are compiled for each GPU's native wavefront automatically — 32-wide on
RDNA (Radeon) and 64-wide on CDNA (Instinct) — so the same source produces a
correct binary for one GPU, or a single "fat" binary for many.

Prerequisites: a working [ROCm](https://rocm.docs.amd.com) install (with
`hipconfig`/`amdclang++` on `PATH`, or `ROCM_PATH` set) and a ROCm build of
[CuPy](https://docs.cupy.dev/en/stable/install.html#using-cupy-on-amd-gpu-experimental).

Build and install:
```sh
git clone https://github.com/pyscf/gpu4pyscf.git
cd gpu4pyscf
USE_HIP=1 pip install .        # autodetect this machine's GPU
# add -e for an editable/development install
```

Use `GPU_TARGET` to choose what hardware to build for (an explicit
`GPU_ARCHITECTURES="gfx...;..."` list overrides it):

| `GPU_TARGET`             | Builds for                                                    |
|--------------------------|---------------------------------------------------------------|
| `native` (default)       | the GPU(s) installed in this machine (auto-detected)          |
| `gfx942`, `gfx1100;...`  | one specific GPU, or an explicit `;`-separated list           |
| `cdna`                   | AMD Instinct datacenter GPUs (MI100/MI200/MI300/MI350, wave64)|
| `rdna`                   | AMD Radeon RDNA2+ GPUs and APUs (RX 6000/7000/9000, wave32)   |
| `all`                    | all of the above, in a single binary (CDNA + RDNA)            |

```sh
USE_HIP=1 GPU_TARGET=all  pip install .          # one binary for all AMD GPUs
USE_HIP=1 GPU_TARGET=cdna pip wheel . -w dist/    # redistributable wheel
```

When working against an editable install, recompile the libraries in place with
`USE_HIP=1 python setup.py build_py` (it accepts the same `GPU_TARGET`).

> [!NOTE]
> AMD has no portable-IR fallback (unlike CUDA's PTX), so device code is emitted
> per architecture: each one adds ~40 MB. A single-GPU build is ~40 MB; an `all`
> binary is a few hundred MB. Build one family (`rdna`/`cdna`) or a specific
> `GPU_TARGET` to keep it small.

Features
--------

GPU4PySCF supports molecular and periodic Hartree-Fock and DFT, analytical
derivatives, excited states, solvent models, properties, geometry optimization,
and non-adiabatic dynamics.

See [FEATURES](FEATURES) for the detailed list of supported methods and
experimental capabilities.

Limitations
--------
- Atomic basis up to g orbitals;
- Auxiliary basis up to i orbitals;
- Density fitting scheme up to ~168 atoms with def2-tzvpd basis, bounded by CPU memory;
- meta-GGA with density laplacian;
- Double hybrid functionals are not supported;
- Hessian of TDDFT is not supported;

Examples
--------
```python
import pyscf

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf = rks.RKS(mol, xc='b3lyp').density_fit().to_gpu()  # move PySCF object to GPU4PySCF object

e_dft = mf.kernel()  # compute total energy
print(f"total energy = {e_dft}")

g = mf.Gradients()
g_dft = g.kernel()   # compute analytical gradient

h = mf.Hessian()
h_dft = h.kernel()   # compute analytical Hessian
```

Find more examples in [gpu4pyscf/examples](https://github.com/pyscf/gpu4pyscf/tree/master/examples)

Benchmarks
--------
Speedup with GPU4PySCF v0.6.0 on A100-80G over Q-Chem 6.1 on 32-cores CPU (density fitting, SCF, def2-tzvpp, def2-universal-jkfit, B3LYP, (99,590))

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

Find more benchmarks in [gpu4pyscf/benchmarks](https://github.com/pyscf/gpu4pyscf/tree/master/benchmarks)

References
---------
```
@misc{li2024introducting,
      title={Introducing GPU-acceleration into the Python-based Simulations of Chemistry Framework},
      author={Rui Li and Qiming Sun and Xing Zhang and Garnet Kin-Lic Chan},
      year={2024},
      eprint={2407.09700},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2407.09700},
}

@misc{wu2024enhancing,
      title={Enhancing GPU-acceleration in the Python-based Simulations of Chemistry Framework},
      author={Xiaojie Wu and Qiming Sun and Zhichen Pu and Tianze Zheng and Wenzhi Ma and Wen Yan and Xia Yu and Zhengxiao Wu and Mian Huo and Xiang Li and Weiluo Ren and Sheng Gong and Yumin Zhang and Weihao Gao},
      year={2024},
      eprint={2404.09452},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2404.09452},
}
```
