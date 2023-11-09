#!/bin/bash

DIR="./organic/xc"
[ ! -d "$DIR" ] && mkdir -p "$DIR"

run --cpu 64 --memory 128 --gpu 1 -- python3 dft_driver.py --input_path ../molecules/organic/ --output_path ./organic/xc/LDA/ --xc LDA
run --cpu 64 --memory 128 --gpu 1 -- python3 dft_driver.py --input_path ../molecules/organic/ --output_path ./organic/xc/PBE/ --xc PBE
run --cpu 64 --memory 128 --gpu 1 -- python3 dft_driver.py --input_path ../molecules/organic/ --output_path ./organic/xc/B3LYP/ --xc B3LYP
run --cpu 64 --memory 128 --gpu 1 -- python3 dft_driver.py --input_path ../molecules/organic/ --output_path ./organic/xc/M06/ --xc M06
run --cpu 64 --memory 128 --gpu 1 -- python3 dft_driver.py --input_path ../molecules/organic/ --output_path ./organic/xc/wB97m-v/ --xc wB97m-v

DIR="./organic/basis"
[ ! -d "$DIR" ] && mkdir -p "$DIR"

run --cpu 64 --memory 128 --gpu 1 -- python3 dft_driver.py --input_path ../molecules/organic/ --output_path ./organic/basis/def2-svp/ --basis def2-svp
run --cpu 64 --memory 128 --gpu 1 -- python3 dft_driver.py --input_path ../molecules/organic/ --output_path ./organic/basis/def2-tzvpp/ --basis def2-tzvpp
run --cpu 64 --memory 128 --gpu 1 -- python3 dft_driver.py --input_path ../molecules/organic/ --output_path ./organic/basis/def2-tzvpd/ --basis def2-tzvpd
run --cpu 64 --memory 128 --gpu 1 -- python3 dft_driver.py --input_path ../molecules/organic/ --output_path ./organic/basis/sto-3g/ --basis sto-3g
run --cpu 64 --memory 128 --gpu 1 -- python3 dft_driver.py --input_path ../molecules/organic/ --output_path ./organic/basis/6-31g/ --basis 6-31g

DIR="./organic/solvent"
[ ! -d "$DIR" ] && mkdir -p "$DIR"

run --cpu 64 --memory 128 --gpu 1 -- python3 dft_driver.py --input_path ../molecules/organic/ --output_path ./organic/solvent/def2-tzvpp/ --basis def2-tzvpp --with_hessian True --solvent C-PCM
