#!/bin/bash

#SBATCH -J cdar-phase0-tsm_ucf2hmdb-from-vanilla-sanity
#SBATCH --gres=gpu:4
#SBATCH -t 4-0
#SBATCH -p batch
#SBATCH --array 1-2%2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=15G
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

echo 'extracting the dataset from NAS ...'
. slurm/utils/copy_ucf_hmdb_to_node_and_untar.sh
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')

lrs=(4e-2 8e-3 4e-3 8e-4 4e-4 4e-5)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"

N=$SLURM_GPUS_ON_NODE

config=configs/recognition/hello/gcd4da/plain/cdar_phase0_tsm_ucf2hmdb_sanity.py
workdir=work_dirs/train_output/ucf2hmdb/tsm/cdar/vanilla/phase0/sanity/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        optimizer.lr=$lr

echo 'done'
exit
