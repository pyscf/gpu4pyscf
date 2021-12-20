#!/bin/bash

export OMP_NUM_THREADS=1 
export PYTHONPATH=$(pwd):$PYTHONPATH 
ulimit -s 20000

#pip install nose nose-exclude nose-timer nose-cov codecov
#pip install -e .

nosetests gpu4pyscf -v --with-timer
