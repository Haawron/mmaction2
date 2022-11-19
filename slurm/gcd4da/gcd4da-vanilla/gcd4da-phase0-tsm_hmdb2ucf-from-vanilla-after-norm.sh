#!/bin/bash

#SBATCH -J gcd4da-phase0-tsm_hmdb2ucf-from-vanilla-after-norm
#SBATCH --gres=gpu:4
#SBATCH -t 4-0
#SBATCH -p batch
#SBATCH --array 0-35%2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=10G
#SBATCH -x agi1
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

echo 'extracting the dataset from NAS ...'
. slurm/utils/copy_ucf_hmdb_to_node_and_untar.sh
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')

lrs=(4e-2 8e-3 4e-3 8e-4 4e-4 4e-5)
layer_nums=(1 2 3)  # 1: linear
feature_nums=(512 768)
idx2=$(( SLURM_ARRAY_TASK_ID % ${#feature_nums[@]} ))
idx1=$(( SLURM_ARRAY_TASK_ID / ${#feature_nums[@]} % ${#layer_nums[@]} ))
idx0=$(( SLURM_ARRAY_TASK_ID / ${#feature_nums[@]} / ${#layer_nums[@]} ))

lr="${lrs[idx0]}"
num_layers="${layer_nums[idx1]}"
num_features="${feature_nums[idx2]}"

N=$SLURM_GPUS_ON_NODE

config=configs/recognition/hello/gcd4da/plain/gcd4da_phase0_tsm_hmdb2ucf.py
workdir=work_dirs/train_output/hmdb2ucf/tsm/gcd4da/vanilla/phase0/default/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        optimizer.lr=$lr \
        cls_head.num_layers=$num_layers \
        cls_head.num_features=$num_features \
        evaluation.metric_options.logits.p_out_dir=$workdir \
    --validate --test-best --test-last

echo 'done'
exit 0