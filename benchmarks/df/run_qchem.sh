#!/bin/bash

export QCSCRATCH=/tmp/

DIR="./organic/xc"
[ ! -d "$DIR" ] && mkdir -p "$DIR"

DIR="./organic/basis"
[ ! -d "$DIR" ] && mkdir -p "$DIR"

#for xc in LDA PBE B3LYP M06 wB97m-v
run --cpu 64 --memory 128 --gpu 0 -- python3 qchem.py --input_path ../molecules/organic/ --output_path ./organic/xc/LDA/ --xc LDA &&
run --cpu 64 --memory 128 --gpu 0 -- python3 qchem.py --input_path ../molecules/organic/ --output_path ./organic/xc/PBE/ --xc PBE &&
run --cpu 64 --memory 128 --gpu 0 -- python3 qchem.py --input_path ../molecules/organic/ --output_path ./organic/xc/B3LYP/ --xc B3LYP &&
run --cpu 64 --memory 128 --gpu 0 -- python3 qchem.py --input_path ../molecules/organic/ --output_path ./organic/xc/M06/ --xc M06 &&
run --cpu 64 --memory 128 --gpu 0 -- python3 qchem.py --input_path ../molecules/organic/ --output_path ./organic/xc/wB97m-v/ --xc wB97m-v &&

#for basis in def2-svp def2-tzvpp def2-tzvpd sto-3g 6-31g 6-31g*
run --cpu 64 --memory 128 --gpu 0 -- python3 qchem.py --input_path ../molecules/organic/ --output_path ./organic/basis/def2-svp/ --basis def2-svp &&
run --cpu 64 --memory 128 --gpu 0 -- python3 qchem.py --input_path ../molecules/organic/ --output_path ./organic/basis/def2-tzvpp/ --basis def2-tzvpp &&
run --cpu 64 --memory 128 --gpu 0 -- python3 qchem.py --input_path ../molecules/organic/ --output_path ./organic/basis/def2-tzvpd/ --basis def2-tzvpd &&
run --cpu 64 --memory 128 --gpu 0 -- python3 qchem.py --input_path ../molecules/organic/ --output_path ./organic/basis/sto-3g/ --basis sto-3g &&
run --cpu 64 --memory 128 --gpu 0 -- python3 qchem.py --input_path ../molecules/organic/ --output_path ./organic/basis/6-31g/ --basis 6-31g &&
run --cpu 64 --memory 128 --gpu 0 -- python3 qchem.py --input_path ../molecules/organic/ --output_path ./organic/basis/6-31g*/ --basis 6-31g*
