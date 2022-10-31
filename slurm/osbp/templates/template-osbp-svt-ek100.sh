#!/bin/bash

#SBATCH -J osbp-svt-ek100-{{ source }}_{{ target }}
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

lrs=(.3 .1 5e-2 1e-2)
weights=(.5 .1 5e-2 1e-2 1e-3 5e-3)
ts=(.3 .4 .5 .7)
idx2=$(( SLURM_ARRAY_TASK_ID % ${#ts[@]} ))
idx1=$(( SLURM_ARRAY_TASK_ID / ${#ts[@]} % ${#weights[@]} ))
idx0=$(( SLURM_ARRAY_TASK_ID / ${#ts[@]} / ${#weights[@]} ))

lr="${lrs[idx0]}"
weight="${weights[idx1]}"
t="${ts[idx2]}"

N=$SLURM_GPUS_ON_NODE
workers=$SLURM_CPUS_PER_GPU

config=configs/recognition/hello/osbp/osbp_svt_ek100.py
workdir=work_dirs/train_output/ek100/svt/osbp/default/${task}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        data.workers_per_gpu=$workers \
        data.train.0.ann_file=data/_filelists/ek100/filelist_${source}_train_closed.txt \
        data.train.1.ann_file=data/_filelists/ek100/filelist_${target}_train_open.txt \
        data.val.ann_file=data/_filelists/ek100/filelist_${target}_valid_open.txt \
        data.test.ann_file=data/_filelists/ek100/filelist_${target}_test_open.txt \
        model.cls_head.loss_cls.weight_loss=$weight \
        model.cls_head.loss_cls.target_domain_label=$t \
        optimizer.lr=$lr \
    --validate --test-best --test-last

echo 'done'
exit 0
