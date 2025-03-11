#!/bin/bash

set -ex

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" -w /gpu4pyscf/wheelhouse/
    fi
}

export CUDA_HOME=/usr/local/cuda
export CUTENSOR_DIR=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

rm -rf /gpu4pyscf/build
rm -rf /gpu4pyscf/gpu4pyscf/lib/deps
rm -rf /gpu4pyscf/tmp/*
rm -rf /gpu4pyscf/put4pyscf/lib/*.so

setup_dir=$(dirname $0)

cmake -S /gpu4pyscf/gpu4pyscf/lib -B build/temp.gpu4pyscf-libxc -DBUILD_GINT=OFF -DBUILD_GVHF=OFF -DBUILD_GDFT=OFF -DBUILD_CUPY_HELPER=OFF -DBUILD_SOLVENT=OFF -DBUILD_GVHF_RYS=OFF -DBUILD_GVHF_MD=OFF -DBUILD_PBC=OFF -DCUDA_ARCHITECTURES="70"
cmake --build build/temp.gpu4pyscf-libxc -j 1

mkdir -p build/lib.gpu4pyscf-libxc/gpu4pyscf/lib/deps/lib
cp /gpu4pyscf/gpu4pyscf/lib/deps/lib/libxc.so build/lib.gpu4pyscf-libxc/gpu4pyscf/lib/deps/lib/
cd build/lib.gpu4pyscf-libxc

# Compile wheels
PYBIN=/opt/python/cp311-cp311/bin
"${PYBIN}/python3" $setup_dir/setup_libxc.py bdist_wheel
repair_wheel dist/*.whl
