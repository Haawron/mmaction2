#!/bin/bash

#SBATCH -J dann-tsm_hmdb2ucf
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=15G
#SBATCH -t 4-0
#SBATCH --array 0-5%2
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=gunsbrother@khu.ac.kr

current_time=$(date +'%Y%m%d-%H%M%S')

lrs=(4e-2 8e-3 4e-3 8e-4 4e-4 4e-5)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"

N=$SLURM_GPUS_ON_NODE

config=configs/recognition/hello/dann/dann_tsm_hmdb2ucf.py
workdir=work_dirs/train_output/hmdb2ucf/tsm/dann/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        optimizer.lr=$lr \
    --validate \
    --test-last --test-best

exit
