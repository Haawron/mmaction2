#!/bin/bash

#SBATCH -J dann-svt-ek100-{{ source }}_{{ target }}
#SBATCH -p batch_grad
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=20G
#SBATCH -t 8:00:00
#SBATCH --array 0-95%4
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')
lrs=(.3 .1 8e-2 5e-2 1e-2 5e-3)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"


source='{{ source }}'
target='{{ target }}'


task=${source}_${target}

lrs=(.3 .1 8e-2 5e-2 1e-2 5e-3)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"

N=$SLURM_GPUS_ON_NODE

config=configs/recognition/hello/dann/dann_svt_ek100.py
workdir=work_dirs/train_output/ek100/svt/dann/${task}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        data.train.0.ann_file=data/_filelists/ek100/filelist_${source}_train_closed.txt \
        data.train.1.ann_file=data/_filelists/ek100/filelist_${target}_train_open.txt \
        data.val.ann_file=data/_filelists/ek100/filelist_${target}_valid_closed.txt \
        data.test.ann_file=data/_filelists/ek100/filelist_${target}_test_closed.txt \
        optimizer.lr=$lr \
        load_from=★★★★★★★★★★★★★★★★★source-only★★★★★★★★★★★★★★★★★
    --validate --test-best --test-last

echo 'done'
exit 0
