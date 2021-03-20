#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J svhn
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

python3 train_svhn.py \
    --svhn_path ../../Data/svhn \
    --save_model_path trained_models/svhn \
    --save_step 1000 \
    --num_img 1 \
    --train_type train \
    --device cuda \
    --workers 2 \
    --epochs 50000 \
    --batch_size 100 \
    --lr 0.0002 \
    --con_training 1 \
    --load_model_path trained_models/svhn_epoch_5000.pt
