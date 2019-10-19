#!/bin/bash

nmtdir=pretrain_models/
srcdir=data/py/

nmtA=$nmtdir/c2nl.bin
nmtB=$nmtdir/nl2c.bin
srcA=$srcdir/train.token.code
srcB=$srcdir/train.token.nl
valA=$srcdir/valid.token.code
valB=$srcdir/valid.token.nl

saveA="dual_models/c2nl/"
saveB="dual_models/nl2c/"
python dsl_dual.py \
    --nmt $nmtA $nmtB \
    --src $srcA $srcB \
    --val $valA $valB \
    --log_n_iter 10 \
    --val_n_iter 20 \
    --lr 0.2 \
    --beta1 1e-3 \
    --beta2 1e-2 \
    --beta3 1e-2 \
    --beta4 1e-1 \
    --model $saveA $saveB \
    --cuda \
    --gpu 0

##