#!/bin/bash

DIR="./water_clusters/xc"
[ ! -d "$DIR" ] && mkdir -p "$DIR"
for xc in B3LYP M06 wB97m-v
do 
    python3 dft_driver.py --input_path ../molecules/water_clusters/ --output_path ./water_clusters/xc/$xc/ --xc $xc
done

DIR="./water_clusters/basis"
[ ! -d "$DIR" ] && mkdir -p "$DIR"

for basis in def2-svp def2-tzvpp def2-tzvpd sto-3g 6-31g
do
    python3 dft_driver.py --input_path ../molecules/water_clusters/ --output_path ./water_clusters/basis/$basis/ --basis $basis
done
