#!/bin/bash

#SBATCH -J cop-tsm-ucf2hmdb_from_vanilla
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=10G
#SBATCH -t 4-0
#SBATCH --array 1-2%2
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')

lrs=(4e-2 8e-3 4e-3 8e-4 4e-4 4e-5)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"

N=$SLURM_GPUS_ON_NODE
config=configs/recognition/hello/cop/cop_tsm_ucf2hmdb.py
ckpt=work_dirs/train_output/ucf2hmdb/tsm/vanilla/source-only/3509__tsm-ucf-vanilla-source-only/2/20220722-020357/best_mean_class_accuracy_epoch_10.pth  # k400 pretrained
workdir=work_dirs/train_output/ucf2hmdb/tsm/cop-from-vanilla/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        optimizer.lr=$lr \
        load_from=$ckpt

exit
