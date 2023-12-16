#!/bin/bash

#export OMP_NUM_THREADS=1 
#export PYTHONPATH=$(pwd):$PYTHONPATH 
ulimit -s 20000

pip install -r requirements.txt
pip install -e .

pytest gpu4pyscf/df 
pytest gpu4pyscf/dft 
pytest gpu4pyscf/rks
