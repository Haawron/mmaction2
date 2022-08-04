#!/bin/bash

#SBATCH -J gcd4da-phase0-svt_hmdb2ucf-from-vanilla
#SBATCH --gres=gpu:4
#SBATCH -t 6:00:00
#SBATCH -p batch
#SBATCH --array 0-5%2
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

config=configs/recognition/hello/gcd4da/plain/gcd4da_phase0_svt_hmdb2ucf.py
workdir=work_dirs/train_output/hmdb2ucf/svt/gcd4da/vanilla/phase0/one_way/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}
ckpt=work_dirs/train_output/hmdb2ucf/svt/vanilla/source-only/4372__vanilla-svt-hmdb2ucf-source-only/1/20220728-192146/best_mean_class_accuracy_epoch_20.pth

OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        optimizer.lr=$lr \
        load_from=$ckpt \
    --validate \
    --test-best --test-last

echo done
exit
