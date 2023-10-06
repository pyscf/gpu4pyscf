FROM debian:11
ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_REQUIRE_CUDA="cuda>=11.7 brand=tesla,driver>=450,driver<451 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=unknown,driver>=510,driver<511 brand=nvidia,driver>=510,driver<511 brand=nvidiartx,driver>=510,driver<511 brand=quadrortx,driver>=510,driver<511"

RUN apt-get update && \
    apt-get install -yq --no-install-recommends gnupg2 curl vim python3 python3-pip python3-setuptools cmake git \
    ca-certificates \
    && \
    curl -fsSL https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get install -yq --no-install-recommends --fix-missing \
    cuda-cudart-11-7=11.7.99-1 \
    cuda-compat-11-7 \
    cuda-libraries-11-7=11.7.1-1 \
    cuda-nvtx-11-7=11.7.91-1 \
    cuda-nvml-dev-11-7=11.7.50-1 \
    cuda-command-line-tools-11-7=11.7.1-1 \
    cuda-libraries-dev-11-7=11.7.1-1 \
    cuda-minimal-build-11-7=11.7.1-1 \
    libcublas-11-7=11.10.3.66-1 \
    libcublas-dev-11-7=11.10.3.66-1 \
    libcusparse-11-7=11.7.4.91-1 \
    libcusparse-dev-11-7=11.7.4.91-1 \
    libcudnn8=8.5.0.96-1+cuda11.7 \
    libcudnn8-dev=8.5.0.96-1+cuda11.7 \
    libnccl2=2.13.4-1+cuda11.7 \
    libnccl-dev=2.13.4-1+cuda11.7 \
    libcutensor1=1.6.0.3-1 \
    libcutensor-dev=1.6.0.3-1 \
    && \
    ln -s /usr/local/cuda-11.7 /usr/local/cuda && \
    find /usr/local/cuda-11.7/lib64/ -type f -name '*.a' -not -name 'libcudart_static.a' -not -name 'libcudadevrt.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_*static.a && \
    rm /usr/lib/x86_64-linux-gnu/libnccl_static.a && \
    rm /etc/apt/sources.list.d/cuda.list && \
    rm -rf /var/lib/apt/lists/*


ENV CUDA_HOME="/usr/local/cuda" LD_LIBRARY_PATH="${CUDA_HOME}/lib64::${LD_LIBRARY_PATH}"
RUN echo "export PATH=${CUDA_HOME}/bin:\$PATH" >> /etc/bash.bashrc
RUN echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH" >> /etc/bash.bashrc
RUN pip3 install -U --no-cache-dir pytest numpy cupy-cuda11x~=12.0 pyscf~=2.3
