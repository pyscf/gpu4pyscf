ARG BASE_CUDA_VERSION=11.8
ARG GPU_IMAGE=nvidia/cuda:${BASE_CUDA_VERSION}.0-devel-centos7
FROM quay.io/pypa/manylinux2014_x86_64:2023-09-24-36b93e4 as base

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-9
ENV LANGUAGE en_US.UTF-8

RUN yum install -y wget curl perl util-linux xz bzip2 git patch which perl zlib-devel
RUN yum install -y yum-utils centos-release-scl
#RUN yum-config-manager --enable rhel-server-rhscl-7-rpms
#RUN yum install -y devtoolset-7-gcc devtoolset-7-gcc-c++ devtoolset-7-gcc-gfortran devtoolset-7-binutils
#ENV PATH=/opt/rh/devtoolset-7/root/usr/bin:$PATH
#ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:$LD_LIBRARY_PATH

# remove unncessary python versions
RUN rm -rf /opt/python/cp26-cp26m /opt/_internal/cpython-2.6.9-ucs2
RUN rm -rf /opt/python/cp26-cp26mu /opt/_internal/cpython-2.6.9-ucs4
RUN rm -rf /opt/python/cp33-cp33m /opt/_internal/cpython-3.3.6
RUN rm -rf /opt/python/cp34-cp34m /opt/_internal/cpython-3.4.6
RUN rm -rf /opt/python/cp36-cp36m /opt/_internal/cpython-3.6.15
RUN rm -rf /opt/python/cp37-cp37m /opt/_internal/cpython-cpython-3.7.17
RUN rm -rf /opt/python/pp37-pypy37_pp73 /opt/_internal/pp37-pypy37_pp73
RUN rm -rf /opt/python/pp38-pypy38_pp73 /opt/_internal/pp38-pypy38_pp73
RUN rm -rf /opt/python/pp39-pypy39_pp73 /opt/_internal/pp39-pypy39_pp73
RUN rm -rf /opt/python/pp310-pypy310_pp73 /opt/_internal/pp310-pypy310_pp73

# Install CUDA
ARG BASE_CUDA_VERSION=11.8
ADD install_cuda.sh install_cuda.sh
RUN bash ./install_cuda.sh ${BASE_CUDA_VERSION} && rm install_cuda.sh

ENV CUDA_HOME="/usr/local/cuda" LD_LIBRARY_PATH="${CUDA_HOME}/lib64::${LD_LIBRARY_PATH}"
RUN echo "export PATH=${CUDA_HOME}/bin:\$PATH" >> /etc/bash.bashrc
RUN echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH" >> /etc/bash.bashrc
