#!/usr/bin/env sh

WORK_PATH=$(dirname $0)

CUDA_VISIBLE_DEVICES=9 python -u -W ignore main.py \
--weights_root pretrained \
--load_weights 128 \
--setting pggan \
--seed 2 \
--exp_path $WORK_PATH \
--root_dir data/others \
--list_file data/others/list.txt \
--G_lrs 5e-5 5e-5 2e-5 2e-5 1e-5 \
--z_lrs 2e-3 1e-3 2e-5 2e-5 1e-5 \
--no_tb