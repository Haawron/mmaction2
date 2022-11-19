#!/bin/bash

#SBATCH -J gcd4da-median-warmup-phase0-svt-hmdb2ucf
#SBATCH --gres=gpu:4
#SBATCH -t 8:00:00
#SBATCH -p batch
#SBATCH --array 0-29%2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=15G
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out
#SBATCH -x agi1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gunsbrother@khu.ac.kr

echo 'extracting the dataset from NAS ...'
. slurm/utils/copy_ucf_hmdb_to_node_and_untar.sh
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')

lrs=(.3 .1 8e-2 5e-2 1e-2 5e-3)
loss_ratios=(.1 .3 .35 .5 .7)

idx0=$(( SLURM_ARRAY_TASK_ID / ${#loss_ratios[@]} ))
idx1=$(( SLURM_ARRAY_TASK_ID % ${#loss_ratios[@]} ))
lr="${lrs[idx0]}"
loss_ratio="${loss_ratios[idx1]}"

N=$SLURM_GPUS_ON_NODE

config=configs/recognition/hello/gcd4da/median/gcd4da_median_phase0_svt_hmdb2ucf.py
workdir=work_dirs/train_output/hmdb2ucf/svt/gcd4da/median/phase0/warmup/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        optimizer.lr=$lr \
        model.cls_head.loss_cls.loss_ratio=$loss_ratio \
        load_from=work_dirs/train_output/hmdb2ucf/svt/vanilla/source-only/13763__vanilla-svt-hmdb2ucf-source-only/0/20221009-145733/best_mean_class_accuracy_epoch_30.pth \
    --validate --test-best --test-last

echo 'done'
exit 0
