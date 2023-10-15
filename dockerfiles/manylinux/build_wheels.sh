set -e -u -x

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

# Compile wheels
rm -rf /gpu4pyscf/wheelhouse
for PYBIN in /opt/python/cp311-cp311/bin; do
    rm -rf /gpu4pyscf/build
    rm -rf /gpu4pyscf/gpu4pyscf/lib/deps
    rm -rf /gpu4pyscf/tmp/*
    rm -rf /gpu4pyscf/put4pyscf/lib/*.so
    "${PYBIN}/python3" -m pip install --upgrade pip
    "${PYBIN}/pip" wheel /gpu4pyscf/ --no-deps -w /gpu4pyscf/tmp/
    repair_wheel /gpu4pyscf/tmp/*.whl
    rm -rf /gpu4pyscf/tmp/*.whl
done
rm -rf /gpu4pyscf/tmp

