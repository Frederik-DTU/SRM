#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J parabolic
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

python3 train_surface3d.py \
    --data_path ../Data/parabolic.csv \
    --save_model_path trained_models/parabolic/parabolic \
    --save_step 5000 \
    --device cuda \
    --epochs 100000 \
    --batch_size 100 \
    --lr 0.0001 \
    --workers 4 \
    --con_training 0 \
    --load_model_path trained_models/parabolic/parabolic_epoch_80000.pt
