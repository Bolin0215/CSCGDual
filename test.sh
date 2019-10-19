#!/bin/bash

src=$1
tgt=$2
mdl=$3
txt=$4
gpu=$5

python nmt/nmt.py --mode test --test_src $src --test_tgt $tgt --load_model $mdl --save_to_file $txt --cuda --gpu $gpu