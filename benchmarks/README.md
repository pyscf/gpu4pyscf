> [!NOTE]
> If you are using the following data as reference, please check out the settings carefully, such as threshold for integrals, convergence tolerance, cartesian or spherical basis, and grids. The default settings of quantum chemistry package can be significantly different. With different settings, the performances are not comparable.

# Benchmarks of algorithms

Machines and software versions
- GPU4PySCF on A100-SXM4-80G with Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz
    - CUDA driver version: 450.191.01
    - CUDA toolkit version: 11.7
- Q-Chem on 32-core vCPU, Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz
- Psi4 on 32-core vCPU, AMD EPYC 7Y83 64-Core Processor

Find more benchmarks in
- [DF SCF, Gradient, Hessian / Q-Chem v6.1](https://github.com/pyscf/gpu4pyscf/tree/master/benchmarks/df/df_pyscf_qchem.md)
- [DF SCF / Psi4 v1.8](https://github.com/pyscf/gpu4pyscf/tree/master/benchmarks/df/df_pyscf_psi4.md)
- [Direct SCF, Gradient / Q-Chem v6.1](https://github.com/pyscf/gpu4pyscf/tree/master/benchmarks/scf/scf_pyscf_qchem.md)
- [DF SCF, Gradient, Hessian with PCM / Q-Chem v6.1](https://github.com/pyscf/gpu4pyscf/tree/master/benchmarks/df/solvent_pyscf_qchem.md)

# Benchmark scripts of applications

- [Solvation free energy with SMD](https://github.com/pyscf/gpu4pyscf/tree/master/benchmarks/smd)
- [Transition state search for transition metals](https://github.com/pyscf/gpu4pyscf/tree/master/benchmarks/ts)
