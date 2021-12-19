#!/usr/bin/env sh

WORK_PATH=$(dirname $0)

CUDA_VISIBLE_DEVICES=5 python -u -W ignore main.py \
--exp_path $WORK_PATH/EC_200_psnr \
--root_dir data/others \
--list_file data/others/list_200.txt \
--seed 2 \
--dgp_mode inpainting \
--update_G \
--update_embed \
--ftr_num 8 8 8 8 8 \
--ft_num 7 7 7 7 7 \
--lr_ratio 1.0 1.0 1.0 1.0 1.0 \
--w_D_loss 1 1 1 1 0.5 \
--w_nll 0.02 \
--w_mse 1 1 1 1 10 \
--w_perceptual_edge 0 0 0.01 0.001 0.0001 \
--select_num 1000 \
--sample_std 0.3 \
--iterations 2000 2000 2000 2000 2000 \
--G_lrs 5e-5 5e-5 2e-5 2e-5 1e-5 \
--z_lrs 2e-3 1e-3 2e-5 2e-5 1e-5 \
--use_in False False False False False \
--resolution 256 \
--weights_root pretrained \
--load_weights 256 \
--G_ch 96 --D_ch 96 \
--G_shared \
--hier --dim_z 120 --shared_dim 128 \
--skip_init --use_ema --no_tb
