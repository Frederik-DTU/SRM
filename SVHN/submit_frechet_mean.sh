#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J group1
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

python3 rm_frechet_mean.py \
    --data_path Data_groups/group1.pt \
    --save_path rm_computations/frechet_group1.pt \
    --device cpu \
    --epochs 100000 \
    --T 10 \
    --batch_size 10 \
    --lr 0.0002 \
    --size 32 \
    --load_model_path trained_models/main/svhn_epoch_50000.pt
