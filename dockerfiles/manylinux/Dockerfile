FROM quay.io/pypa/manylinux2014_x86_64:2025.11.11-1

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

RUN yum install -y wget curl perl util-linux xz bzip2 git patch which perl zlib-devel
#RUN yum install -y yum-utils centos-release-scl
#RUN yum-config-manager --enable rhel-server-rhscl-7-rpms
#RUN yum install -y devtoolset-7-gcc devtoolset-7-gcc-c++ devtoolset-7-gcc-gfortran devtoolset-7-binutils
#ENV PATH=/opt/rh/devtoolset-7/root/usr/bin:$PATH
#ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:$LD_LIBRARY_PATH

# Install CUDA
ADD install_cuda.sh install_cuda.sh
ARG BASE_CUDA_VERSION=12.8
RUN bash ./install_cuda.sh ${BASE_CUDA_VERSION} && rm install_cuda.sh

ENV CUDA_HOME="/usr/local/cuda" LD_LIBRARY_PATH="${CUDA_HOME}/lib64::${LD_LIBRARY_PATH}"
RUN echo "export PATH=${CUDA_HOME}/bin:\$PATH" >> /etc/bashrc
RUN echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH" >> /etc/bashrc
