#!/bin/bash

set -ex

function install_118 {
    echo "Installing CUDA 11.8 and cutensor 1.6.1.5"
    rm -rf /usr/local/cuda-11.8 /usr/local/cuda
    # install CUDA 11.8.0 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    chmod +x cuda_11.8.0_520.61.05_linux.run
    ./cuda_11.8.0_520.61.05_linux.run --toolkit --silent
    rm -f cuda_11.8.0_520.61.05_linux.run
    
    # cutensor license: https://docs.nvidia.com/cuda/cutensor/license.html
    mkdir tmp_cutensor && cd tmp_cutensor
    mkdir /usr/local/cuda/lib
    wget -q https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-1.6.1.5-archive.tar.xz
    tar xf libcutensor-linux-x86_64-1.6.1.5-archive.tar.xz
    cp -a libcutensor-linux-x86_64-1.6.1.5-archive/include/* /usr/local/cuda-11.8/include/
    cp -a libcutensor-linux-x86_64-1.6.1.5-archive/lib/11/* /usr/local/cuda-11.8/lib64/
    cd ..
    rm -rf tmp_cutensor
    
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-11.8 /usr/local/cuda
    ldconfig
}

function install_121 {
    echo "Installing CUDA 12.1 and cutensor 1.7.0.1"
    rm -rf /usr/local/cuda-12.1 /usr/local/cuda
    # install CUDA 12.1.0 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
    chmod +x cuda_12.1.1_530.30.02_linux.run
    ./cuda_12.1.1_530.30.02_linux.run --toolkit --silent
    rm -f cuda_12.1.1_530.30.02_linux.run
    
    # cutensor license: https://docs.nvidia.com/cuda/cutensor/license.html
    mkdir tmp_cutensor && cd tmp_cutensor
    mkdir /usr/local/cuda/lib
    wget -q https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-1.7.0.1-archive.tar.xz
    tar xf libcutensor-linux-x86_64-1.7.0.1-archive.tar.xz
    cp -a libcutensor-linux-x86_64-1.7.0.1-archive/include/* /usr/local/cuda-12.1/include/
    cp -a libcutensor-linux-x86_64-1.7.0.1-archive/lib/11/* /usr/local/cuda-12.1/lib64/
    cd ..
    rm -rf tmp_cutensor
    
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.1 /usr/local/cuda
    ldconfig
}


function install_122 {
    echo "Installing CUDA 12.2 and cutensor 1.7.0.1"
    rm -rf /usr/local/cuda-12.2 /usr/local/cuda
    # install CUDA 12.2.0 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
    chmod +x cuda_12.2.2_535.104.05_linux.run
    ./cuda_12.2.2_535.104.05_linux.run --toolkit --silent
    rm -f cuda_12.2.2_535.104.05_linux.run
    
    # still need cuda 11.8 for cublas-11
    wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    chmod +x cuda_11.8.0_520.61.05_linux.run
    ./cuda_11.8.0_520.61.05_linux.run --toolkit --silent
    rm -f cuda_11.8.0_520.61.05_linux.run
    
    # cutensor license: https://docs.nvidia.com/cuda/cutensor/license.html
    mkdir tmp_cutensor && cd tmp_cutensor
    mkdir /usr/local/cuda/lib
    wget -q https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-1.7.0.1-archive.tar.xz
    tar xf libcutensor-linux-x86_64-1.7.0.1-archive.tar.xz
    cp -a libcutensor-linux-x86_64-1.7.0.1-archive/include/* /usr/local/cuda-12.2/include/
    cp -a libcutensor-linux-x86_64-1.7.0.1-archive/lib/11/* /usr/local/cuda-12.2/lib64/
    cd ..
    rm -rf tmp_cutensor
    
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.2 /usr/local/cuda
    ldconfig
}

function prune_122 {
    echo "Pruning CUDA 12.2 and cuDNN"
    #####################################################################################
    # CUDA 12.2 prune static libs
    #####################################################################################
    export NVPRUNE="/usr/local/cuda-12.2/bin/nvprune"
    export CUDA_LIB_DIR="/usr/local/cuda-12.2/lib64"
    
    export GENCODE="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
    export GENCODE_CUDNN="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

    if [[ -n "$OVERRIDE_GENCODE" ]]; then
        export GENCODE=$OVERRIDE_GENCODE
    fi

    # all CUDA libs except CuDNN and CuBLAS (cudnn and cublas need arch 3.7 included)
    ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
      | xargs -I {} bash -c \
                "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"
    
    # prune CuDNN and CuBLAS
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a
    
    #####################################################################################
    # CUDA 12.2 prune visual tools
    #####################################################################################
    export CUDA_BASE="/usr/local/cuda-12.2/"
    rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute* $CUDA_BASE/nsight-systems*
    rm -rf /opt/nvidia
}

function prune_121 {
  echo "Pruning CUDA 12.1"
  #####################################################################################
  # CUDA 12.1 prune static libs
  #####################################################################################
    export NVPRUNE="/usr/local/cuda-12.1/bin/nvprune"
    export CUDA_LIB_DIR="/usr/local/cuda-12.1/lib64"

    export GENCODE="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
    export GENCODE_CUDNN="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

    if [[ -n "$OVERRIDE_GENCODE" ]]; then
        export GENCODE=$OVERRIDE_GENCODE
    fi

    # all CUDA libs except CuDNN and CuBLAS
    ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
      | xargs -I {} bash -c \
                "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

    # prune CuDNN and CuBLAS
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

    #####################################################################################
    # CUDA 12.1 prune visual tools
    #####################################################################################
    export CUDA_BASE="/usr/local/cuda-12.1/"
    rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2023.1.0 $CUDA_BASE/nsight-systems-2023.1.2/
    rm -rf /opt/nvidia
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    11.8) install_118; prune_118
	  ;;
    12.1) install_121; prune_121
	  ;;
    12.2) install_122; prune_122
	  ;;
	*) echo "bad argument $1"; exit 1
	   ;;
    esac
    shift
done
