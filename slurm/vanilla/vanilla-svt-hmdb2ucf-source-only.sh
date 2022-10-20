#!/bin/bash

#SBATCH -J vanilla-svt-hmdb2ucf-source-only
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=15G
#SBATCH -t 8:00:00
#SBATCH -x agi1
#SBATCH --array 0-5%2
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')

lrs=(.3 .1 8e-2 5e-2 1e-2 5e-3)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"

N=$SLURM_GPUS_ON_NODE
task='source-only'

config=configs/recognition/hello/vanilla/vanilla_svt_hmdb51_closed.py
workdir=work_dirs/train_output/hmdb2ucf/svt/vanilla/${task}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        optimizer.lr=$lr \
    --validate --test-last --test-best

echo 'done'
exit 0
