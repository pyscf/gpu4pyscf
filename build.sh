#!/bin/bash

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
echo "PATH=${PATH}"
echo "CUDA_HOME=${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
python3 setup.py bdist_wheel
rm -rf output && mv dist output
CURRENT_PATH=`pwd`
echo "Current Path: ${CURRENT_PATH}"
export PYTHONPATH="${PYTHONPATH}:${CURRENT_PATH}"
export CUPY_ACCELERATORS=cub,cutensor
