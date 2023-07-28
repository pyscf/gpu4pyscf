GPU plugin for PySCF
====================

Installation
--------
Build your own docker image (~5 mins)
```
docker pull nvidia/cuda:11.7.0-devel-ubuntu18.04
```
Compile and mannual install
```
sh build.sh
cd output
pip3 install gpu4pyscf-*
```
[optional] install cutensor for accerlation
```
python -m cupyx.tools.install_library --cuda 11.x --library cutensor
```

Features
--------
Density fitting scheme
- SCF, analytical Gradient, and analytical Hessian calculations for Hartree-Fock and DFT;
- LDA, GGA, mGGA without density laplacian, hybrid, and range-separated functionals;
- Geometry optimization and transition state search via [geomeTRIC](https://geometric.readthedocs.io/en/latest/);
- Dispersion corrections via [DFT3](https://github.com/dftd3/simple-dftd3) and [DFT4](https://github.com/dftd4/dftd4)
- Nonlocal functional correction (vv10) for SCF and gradient;
- ECP is supported and calculated on CPU

Direct SCF scheme (experimental)
- SCF for Hartree-Fock and DFT;
- LDA, GGA, mGGA without density laplacian, hybrid, and range-separated functionals;
- Dispersion corrections via [DFT3](https://github.com/dftd3/simple-dftd3) and [DFT4](https://github.com/dftd4/dftd4)
- Nonlocal functional correction (vv10) for SCF and gradient
- ECP is supported and calculated on CPU

Limitations
--------
- Rys roots up to 8 for density fitting scheme;
- Rys roots up to 9 for direct scf scheme;
- Atomic basis up to g orbitals;
- Auxiliary basis up to h orbitals;
- Up to ~70 atoms with def2-tzvpp basis on NVIDIA GPU with 80G memory;
- Up to ~50 atoms with def2-tzvpp basis on NVIDIA GPU with 32G memory;
- Gradient, Hessian and Geometry optimization are unavailable for Direct SCF yet;
- meta-GGA without density laplacian;


Examples
--------
Find examples in gpu4pyscf/examples

