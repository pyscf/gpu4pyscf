#!/bin/bash

DIR="./organic/xc"
[ ! -d "$DIR" ] && mkdir -p "$DIR"
for xc in LDA PBE B3LYP M06 wB97m-v
do 
    python3 dft_driver.py --input_path ../molecules/organic/ --output_path ./organic/xc/$xc/ --xc $xc --device CPU
done

DIR="./organic/basis"
[ ! -d "$DIR" ] && mkdir -p "$DIR"

for basis in def2-svp def2-tzvpp def2-tzvpd sto-3g 6-31g
do
    python3 dft_driver.py --input_path ../molecules/organic/ --output_path ./organic/basis/$basis/ --basis $basis --deivce CPU
done
