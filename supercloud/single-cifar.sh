#!/bin/bash

#SBATCH -o supercloud_logs/%j.log
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20

source /etc/profile
module load anaconda/2023a

export OMP_NUM_THREADS=20
export IMAGENET_PATH=/home/gridsan/groups/datasets/ImageNet

python main.py --train_bs 400 --test_bs 400 --arch resnet50 --dataset cifar10 --epochs 300 --loss xent
