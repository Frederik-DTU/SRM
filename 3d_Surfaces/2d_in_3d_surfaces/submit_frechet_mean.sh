#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J frechet_mean
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
    --data_name hyperbolic_paraboloid \
    --device cpu \
    --epochs 100000 \
    --T 100 \
    --batch_size 100 \
    --lr 0.0001 \
    --save_step 100 \
    --load_epoch 100000.pt
