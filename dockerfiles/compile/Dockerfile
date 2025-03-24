# CUDA 11.8
#FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
#RUN apt-get update && \
#    apt-get install -yq --no-install-recommends gnupg2 curl vim python3 python3-pip python3-setuptools cmake git \
#    ca-certificates libopenblas-dev gfortran

#ENV CUDA_HOME="/usr/local/cuda" LD_LIBRARY_PATH="${CUDA_HOME}/lib64::${LD_LIBRARY_PATH}"
#RUN echo "export PATH=${CUDA_HOME}/bin:\$PATH" >> /etc/bash.bashrc
#RUN echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH" >> /etc/bash.bashrc

#RUN pip3 install -U --no-cache-dir pytest numpy cupy-cuda11x pyscf cutensor-cu11

# CUDA 12.4
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
RUN apt-get update && \
    apt-get install -yq --no-install-recommends gnupg2 curl vim python3 python3-pip python3-setuptools cmake git \
    ca-certificates libopenblas-dev gfortran

ENV CUDA_HOME="/usr/local/cuda" LD_LIBRARY_PATH="${CUDA_HOME}/lib64::${LD_LIBRARY_PATH}"
RUN echo "export PATH=${CUDA_HOME}/bin:\$PATH" >> /etc/bash.bashrc
RUN echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH" >> /etc/bash.bashrc

RUN pip3 install -U --no-cache-dir pytest numpy cupy-cuda12x pyscf cutensor-cu12 pytest-cov pytest-benchmark flake8 pyscf-dispersion
RUN pip3 install geometric
RUN pip3 install gpu4pyscf-libxc-cuda12x==0.5
RUN pip3 install git+https://github.com/pyscf/properties --upgrade
