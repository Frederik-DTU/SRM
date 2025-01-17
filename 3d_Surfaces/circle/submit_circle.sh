#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J circle
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u s164222@student.dtu.dk
#BSUB -o output/output_%J.out
#BSUB -e error/error_%J.err
#BSUB -B
#BSUB -N

#Load the following in case
#module load python/3.8
module swap cuda/8.0
module swap cudnn/v7.0-prod-cuda8

python3 train_circle.py \
    --data_path Data/circle.csv \
    --save_model_path trained_models/latent_dim_2/circle \
    --save_step 5000 \
    --device cuda \
    --epochs 100000 \
    --batch_size 100 \
    --workers 4 \
    --latent_dim 2 \
    --lr 0.0001  \
    --con_training 0 \
    --load_model_path trained_models/main/circle_epoch_70000.pt
