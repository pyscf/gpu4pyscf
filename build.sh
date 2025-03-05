#!/bin/bash

export CUDA_HOME=${CUDA_HOME:-/usr/bin/}
echo "PATH=${PATH}"
echo "CUDA_HOME=${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

cmake -B build -S gpu4pyscf/lib -DCUDA_ARCHITECTURES=70 -DBUILD_LIBXC=OFF
cd build 
make -j8
cd ..

CURRENT_PATH=`pwd`
echo "Current Path: ${CURRENT_PATH}"
export PYTHONPATH="${PYTHONPATH}:${CURRENT_PATH}"
export CUPY_ACCELERATORS=cub,cutensor
