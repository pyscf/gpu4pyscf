FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*


ENV CUDA_HOME="/usr/local/cuda" LD_LIBRARY_PATH="${CUDA_HOME}/lib64::${LD_LIBRARY_PATH}"
RUN echo "export PATH=${CUDA_HOME}/bin:\$PATH" >> /etc/bash.bashrc
RUN echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH" >> /etc/bash.bashrc

RUN pip3 install cutensor-cu11
RUN pip3 install gpu4pyscf-cuda11x