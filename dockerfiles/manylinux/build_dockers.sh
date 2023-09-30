#!/usr/bin/env bash

#:set -eou pipefail

docker build -t gpu4pyscf-manylinux2014-builder:cuda11.8 --build-arg BASE_CUDA_VERSION=11.8 .

docker build -t gpu4pyscf-manylinux2014-builder:cuda12.1 --build-arg BASE_CUDA_VERSION=12.1 .
