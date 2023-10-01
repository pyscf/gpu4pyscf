#!/usr/bin/env bash

docker build -t wxj6000/manylinux2014:cuda118 --build-arg BASE_CUDA_VERSION=11.8 .

docker build -t wxj6000/manylinux2014:cuda121 --build-arg BASE_CUDA_VERSION=12.1 .
