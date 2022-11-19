#!/bin/bash

#SBATCH -J gcd4da-phase0-tsm-hmdb2ucf-plain
#SBATCH --gres=gpu:4
#SBATCH -t 8:00:00
#SBATCH -p batch
#SBATCH --array 0-0%1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=50G
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

echo 'extracting the dataset from NAS ...'
. slurm/utils/copy_ucf_hmdb_to_node_and_untar.sh
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')

# lrs=(4e-2 8e-3 4e-3 8e-4 4e-4 4e-5)
# lr="${lrs[SLURM_ARRAY_TASK_ID]}"

N=$SLURM_GPUS_ON_NODE

config=configs/recognition/hello/gcd4da/plain/gcd4da_phase0_tsm_hmdb2ucf.py
workdir=work_dirs/train_output/hmdb2ucf/tsm/gcd4da/plain/phase0/default/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --validate --test-best --test-last

echo 'done'
exit 0
