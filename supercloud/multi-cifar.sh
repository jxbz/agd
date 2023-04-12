#!/bin/bash
#SBATCH --job-name pytorch
#SBATCH -o supercloud_logs/%j.log
#SBATCH -N 2
#SBATCH --tasks-per-node=2
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=20

# Initialize the module command
source /etc/profile

# Load modules
module load anaconda/2021a
module load cuda/10.1
module load mpi/openmpi-4.0
module load nccl/2.5.6-cuda10.1

export OMP_NUM_THREADS=20
export MPI_FLAGS="--tag-output --bind-to socket -map-by core -mca btl ^openib -mca pml ob1 -x PSM2_GPUDIRECT=1 -x NCCL_NET_GDR_LEVEL=5 -x NCCL_P2P_LEVEL=5 -x NCCL_NET_GDR_READ=1"
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "MASTER_ADDR : ${MASTER_ADDR}"
echo "MASTER_PORT : ${MASTER_PORT}"

export IMAGENET_PATH=/home/gridsan/groups/datasets/ImageNet
mpirun ${MPI_FLAGS} python main.py --distribute --train_bs 400 --test_bs 400 --arch resnet50 --dataset cifar10 --epochs 300 --loss xent
