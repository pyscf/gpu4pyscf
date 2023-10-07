set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" -w /gpu4pyscf/wheelhouse/
    fi
}

export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME="/usr/local/cuda" 
export CUTENSOR_DIR="/usr/local/cuda"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

echo "export CUDA_HOME=/usr/local/cuda" >> /etc/bash.bashrc
echo "export CUTENSOR_DIR=/usr/local/cuda" >> /etc/bash.bashrc
echo "export PATH=${CUDA_HOME}/bin:\$PATH" >> /etc/bash.bashrc
echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH" >> /etc/bash.bashrc

# Compile wheels
for PYBIN in /opt/python/{cp38-cp38,cp39-cp39,cp310-cp310,cp311-cp311}/bin; do
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

