#!/bin/bash

#SBATCH -J vanilla-svt-ek100-{{ domain }}-{{ task }}
#SBATCH -p batch_grad
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=20G
#SBATCH -t 8:00:00
#SBATCH --array 0-3%4
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')

# lrs=(.3 .1)
# fzss=(10 9 8 7 6 -1)
# bszs=(64 44 36 28 24 10)
# idx1=$(( SLURM_ARRAY_TASK_ID % ${#fzss[@]} ))
# idx0=$(( SLURM_ARRAY_TASK_ID / ${#fzss[@]} ))

# lr="${lrs[idx0]}"
# fzs="${fzss[idx1]}"
# bsz="${bszs[idx1]}"

# lrs=(.3 .1 5e-2 1e-2)
lrs=(5e-3 1e-3 5e-4 1e-4)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"
fzs=-1
bsz=10

domain='{{ domain }}'
task='{{ task }}'
openness='{{ openness }}'


N=$SLURM_GPUS_ON_NODE

config=configs/recognition/hello/vanilla/vanilla_svt_ek100_${openness}_aug.py
workdir=work_dirs/train_output/ek100/svt/vanilla/${domain}/${task}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

# 11-96 / 10-64 / 9-44 / 8-36 / 7-28 / 6-24
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py "$config" \
    --launcher pytorch \
    --work-dir "$workdir" \
    --cfg-options \
        data.train.ann_file=data/_filelists/ek100/filelist_"${domain}"_train_"${openness}".txt \
        data.val.ann_file=data/_filelists/ek100/filelist_"${domain}"_valid_"${openness}".txt \
        data.test.ann_file=data/_filelists/ek100/filelist_"${domain}"_test_"${openness}".txt \
        optimizer.lr="$lr" \
        lr_config.min_lr="$(echo "${lr}" | awk '{print $1 *1e-3}') "\
        model.backbone.frozen_stages=$fzs \
        data.videos_per_gpu=$bsz \
    --validate --test-best --test-last

exit $?
