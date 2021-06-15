#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J distance_matrix
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

python3 rm_distance_matrix.py \
    --data_path ../../Data/CelebA/celeba \
    --save_path rm_computations/dmat.pt \
    --group1 Data_groups/group_blond_closed/ \
    --group2 Data_groups/group_blond_open/ \
    --group3 Data_groups/group_black_closed/ \
    --group4 Data_groups/group_black_open/ \
    --device cpu \
    --epochs 10000 \
    --T 10 \
    --batch_size 10 \
    --lr 0.0001 \
    --size 64 \
    --load_model_path trained_models/main/celeba_epoch_6300.pt
