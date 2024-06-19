# Benchmark details

Machines and software versions
- GPU4PySCF on A100-SXM4-80G with Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz
    - CUDA driver version: 450.191.01   
    - CUDA toolkit version: 11.7
- Q-Chem on 32-core vCPU, Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz
- Psi4 on 32-core vCPU, AMD EPYC 7Y83 64-Core Processor

# Benchmarks of algorithms 

Find more benchmarks in 
- [DF SCF, Gradient, Hessian / Q-Chem v6.1](https://github.com/pyscf/gpu4pyscf/tree/master/benchmarks/df_again_qchem.md)
- [DF SCF, Gradient / Psi4 v1.8]()(https://github.com/pyscf/gpu4pyscf/tree/master/benchmarks/df_again_psi4.md)
- [Direct SCF, Gradient/ Q-Chem v6.1](https://github.com/pyscf/gpu4pyscf/tree/master/benchmarks/scf_again_qchem.md)
- [DF Solvent, Gradient, Hessian / Q-Chem v6.1](https://github.com/pyscf/gpu4pyscf/tree/master/benchmarks/scf_again_qchem.md)

# Benchmarks of applications
- [Solvation free energy with SMD](https://github.com/pyscf/gpu4pyscf/tree/master/benchmarks/smd)
- [Transition state search for transition metals](https://github.com/pyscf/gpu4pyscf/tree/master/benchmarks/ts)
