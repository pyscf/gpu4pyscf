: '
cd pyscf/lib
lib_path=$(pwd)

mkdir -p build
cd build 
cmake ..
make
echo ${lib_path}
# compile libxc, somehow local cmake is slow, and the compiled library is big
git clone https://gitlab.com/libxc/libxc.git
cd libxc
git checkout e48b9ee575867d6525f0f6fff05c3553d89628e1
cmake -H. -Bobjdir -DBUILD_SHARED_LIBS=ON -DENABLE_CUDA=ON -DBUILD_TESTING=OFF -DDISABLE_KXC=ON -DDISABLE_LXC=ON -DDISABLE_FHC=ON -DCMAKE_INSTALL_LIBDIR="${lib_path}/libxc"
cd objdir && make -j8
make install
'

#!/bin/bash

echo "PATH=${PATH}"
echo "CUDA_HOME=${CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
python3 setup.py bdist_wheel
rm -rf output && mv dist output
CURRENT_PATH=`pwd`
echo "Current Path: ${CURRENT_PATH}"
export PYTHONPATH="${PYTHONPATH}:${CURRENT_PATH}"
export CUPY_ACCELERATORS=cub,cutensor
