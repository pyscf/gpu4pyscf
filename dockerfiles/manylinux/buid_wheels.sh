set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /gpu4pyscf/wheelhouse/
    fi
}

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [ ${PYBIN} != /opt/python/cp312-cp312/bin ]; then   
	"${PYBIN}/python3" -m pip install --upgrade pip
    	"${PYBIN}/pip" install -r /gpu4pyscf/dockerfiles/manylinux/requirements.txt
    	"${PYBIN}/pip" wheel /gpu4pyscf/ --no-deps -w wheelhouse/
    fi
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install gpu4pyscf --no-index -f /gpu4pyscf/wheelhouse
    (cd "$HOME"; "${PYBIN}/nosetests" gpu4pyscf)
done
