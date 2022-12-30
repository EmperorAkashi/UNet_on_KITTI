#! /bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH --constraint=a100
#SBATCH -c 8
#SBATCH --mem=80gb
#SBATCH --time=4:00:00

# Training script for the KITTI semantic dataset

OUTPUT_DIR=/mnt/ceph/users/$USER/projects/unet/outputs/$SLURM_JOB_ID
mkdir -p $OUTPUT_DIR

python -um UNet_on_Kitti.trainer data.file_path="/mnt/home/clin/ceph/dataset/kitti_semantic/training" \
hydra.run.dir="$OUTPUT_DIR" num_gpus=$SLURM_GPUS_ON_NODE num_epochs=10 