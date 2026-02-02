FROM quay.io/pypa/manylinux2014_x86_64:2025.11.11-1

ENV LC_ALL en_US.UTF-8

RUN yum install -y wget zlib-devel

# Install CUDA
ADD install_cuda.sh install_cuda.sh
ARG BASE_CUDA_VERSION=12.8
RUN bash ./install_cuda.sh ${BASE_CUDA_VERSION} && rm install_cuda.sh

ENV CUDA_HOME="/usr/local/cuda"
RUN echo "export PATH=${CUDA_HOME}/bin:\$PATH" >> /etc/bashrc
RUN echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH" >> /etc/bashrc
