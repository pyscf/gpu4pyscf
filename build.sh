#!/bin/bash

export CUDA_HOME=${CUDA_HOME:-/usr/bin/}
echo "PATH=${PATH}"
echo "CUDA_HOME=${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

CURRENT_PATH=`pwd`

mkdir -p /tmp/build
cmake -B /tmp/build -S gpu4pyscf/lib -DCUDA_ARCHITECTURES="70-real;80" -DBUILD_LIBXC=OFF -DCMAKE_MESSAGE_LOG_LEVEL=WARNING
cd /tmp/build
make -j16
cd $CURRENT_PATH

echo "Current Path: ${CURRENT_PATH}"
export PYTHONPATH="${PYTHONPATH}:${CURRENT_PATH}"
export CUPY_ACCELERATORS=cub,cutensor
