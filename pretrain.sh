#!/bin/sh

L1='code'
L2='nl'
JOB='pretrain'

data_dir="data/py"
vocab_bin="$data_dir/vocab.${L1}${L2}.bin"
train_src="$data_dir/train.token.${L1}"
train_tgt="$data_dir/train.token.${L2}"
test_src="$data_dir/valid.token.${L1}"
test_tgt="$data_dir/valid.token.${L2}"

job_name="$JOB"
model_name="pretrain_models/c2nl.${job_name}"

python2 nmt/nmt.py \
    --cuda \
    --gpu 0 \
    --mode train \
    --vocab ${vocab_bin} \
    --save_to ${model_name} \
    --log_every 500 \
    --valid_niter 500 \
    --valid_metric bleu \
    --save_model_after 2 \
    --beam_size 10 \
    --batch_size 32 \
    --hidden_size 512 \
    --embed_size 512 \
    --uniform_init 0.1 \
    --dropout 0.2 \
    --clip_grad 5.0 \
    --decode_max_time_step 50 \
    --lr_decay 0.8 \
    --lr 0.002 \
    --train_src ${train_src} \
    --train_tgt ${train_tgt} \
    --dev_src ${test_src} \
    --dev_tgt ${test_tgt}
    

