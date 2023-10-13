#!/bin/bash

DIR="./water_clusters/xc"
[ ! -d "$DIR" ] && mkdir -p "$DIR"

DIR="./water_clusters/basis"
[ ! -d "$DIR" ] && mkdir -p "$DIR"

#for xc in LDA PBE B3LYP M06 wB97m-v
run --cpu 64 --memory 256 --gpu 0 -- python3 qchem.py --input_path ../molecules/water_clusters/ --output_path ./water_clusters/xc/LDA/ --xc LDA &&
run --cpu 64 --memory 256 --gpu 0 -- python3 qchem.py --input_path ../molecules/water_clusters/ --output_path ./water_clusters/xc/PBE/ --xc PBE && 
run --cpu 64 --memory 256 --gpu 0 -- python3 qchem.py --input_path ../molecules/water_clusters/ --output_path ./water_clusters/xc/B3LYP/ --xc B3LYP &&
run --cpu 64 --memory 256 --gpu 0 -- python3 qchem.py --input_path ../molecules/water_clusters/ --output_path ./water_clusters/xc/M06/ --xc M06 && 
run --cpu 64 --memory 256 --gpu 0 -- python3 qchem.py --input_path ../molecules/water_clusters/ --output_path ./water_clusters/xc/wB97m-v/ --xc wB97m-v &&

#for basis in def2-svp def2-tzvpp def2-tzvpd sto-3g 6-31g 6-31g*
run --cpu 64 --memory 256 --gpu 0 -- python3 qchem.py --input_path ../molecules/water_clusters/ --output_path ./water_clusters/basis/def2-svp/ --basis def2-svp &&
run --cpu 64 --memory 256 --gpu 0 -- python3 qchem.py --input_path ../molecules/water_clusters/ --output_path ./water_clusters/basis/def2-tzvpp/ --basis def2-tzvpp &&
run --cpu 64 --memory 256 --gpu 0 -- python3 qchem.py --input_path ../molecules/water_clusters/ --output_path ./water_clusters/basis/def2-tzvpd/ --basis def2-tzvpd &&
run --cpu 64 --memory 256 --gpu 0 -- python3 qchem.py --input_path ../molecules/water_clusters/ --output_path ./water_clusters/basis/sto-3g/ --basis sto-3g &&
run --cpu 64 --memory 256 --gpu 0 -- python3 qchem.py --input_path ../molecules/water_clusters/ --output_path ./water_clusters/basis/6-31g/ --basis 6-31g
