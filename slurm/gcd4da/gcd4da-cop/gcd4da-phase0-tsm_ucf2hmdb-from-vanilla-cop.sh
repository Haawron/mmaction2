#!/bin/bash

#SBATCH -J gcd4da-cop-phase0-tsm_ucf2hmdb
#SBATCH --gres=gpu:4
#SBATCH -t 4-0
#SBATCH -p batch
#SBATCH --array 0-5%2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=15G
#SBATCH -x agi1
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

echo 'extracting the dataset from NAS ...'
. slurm/utils/copy_ucf_hmdb_to_node_and_untar.sh
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')

lrs=(4e-2 8e-3 4e-3 8e-4 4e-4 4e-5)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"

N=$SLURM_GPUS_ON_NODE

config=configs/recognition/hello/gcd4da/plain/gcd4da_phase0_tsm_ucf2hmdb.py
# vanilla -> COP weight
ckpt=work_dirs/train_output/ucf2hmdb/tsm/cop-from-vanilla/7970__cop-tsm-ucf2hmdb_from_vanilla/1/20220902-074331/best_mean_class_accuracy_epoch_30.pth
workdir=work_dirs/train_output/ucf2hmdb/tsm/gcd4da/cop/phase0/default/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        optimizer.lr=$lr \
        load_from=$ckpt \
    --validate \
    --test-best --test-last

echo 'done'
exit 0
