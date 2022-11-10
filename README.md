GPU plugin for PySCF
====================


Build and Install
------------
- Install dependencies
Install cuda11.6, cudnn, Cupy and its denpendencies (Numpy, scipy, optuna, cuTENSOR v1.5, cuSPARSELt v0.2.0, cuDNN v8.3 / v8.4 / v8.5, NCCL v2.11 / v2.12 / v2.13 / v2.14)


```
apt install libnccl2=2.12.10-1+cuda11.6 libnccl-dev=2.12.10-1+cuda11.6  
python -m pip install -U setuptools pip  
pip install optuna scipy numpy cupy
nvcc --version

cd gpu4pyscf
python setup install
```


- Set enviromental variables
```
vi ~/.bashrc
export PATH=$PATH:/usr/local/cuda-11.6/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.6/lib64
export PYTHONPATH="/root/gpu4pyscf":$PYTHONPATH
source ~/.bashrc
```

- Troubles
```
ln -s ...../cint.h   ***/cint.h
```

Examples
--------
```
import pyscf
# Load the gpu4pyscf to enable GPU mode
from gpu4pyscf import patch_pyscf

mol = pyscf.M(atom='''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
basis='ccpvdz',
verbose=5)
mf = mol.RHF().run()

# Run SCF with CPU
mf.device = 'cpu'
mf.run()
```
