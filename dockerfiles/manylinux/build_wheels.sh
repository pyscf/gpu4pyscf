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
export CUTENSOR_ROOT="/usr/local/cuda"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
echo "export PATH=${CUDA_HOME}/bin:\$PATH" >> /etc/bash.bashrc
echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/lib:\$LD_LIBRARY_PATH" >> /etc/bash.bashrc

# Compile wheels
for PYBIN in /opt/python/{cp38-cp38,cp39-cp39,cp310-cp310,cp311-cp311}/bin; do
    "${PYBIN}/python3" -m pip install --upgrade pip
    "${PYBIN}/pip" install -r /gpu4pyscf/dockerfiles/manylinux/requirements.txt
    "${PYBIN}/pip" wheel /gpu4pyscf/ --no-deps -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
for PYBIN in /opt/python/{cp38-cp38,cp39-cp39,cp310-cp310,cp311-cp311}/bin; do
    "${PYBIN}/pip" install gpu4pyscf --no-index -f /gpu4pyscf/wheelhouse
done
