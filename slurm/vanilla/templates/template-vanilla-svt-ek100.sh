#!/bin/bash

#SBATCH -J vanilla-svt-ek100-{{ domain }}-{{ task }}
#SBATCH -p batch_grad
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=20G
#SBATCH -t 8:00:00
#SBATCH --array 0-5%4
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')
lrs=(.3 .1 8e-2 5e-2 1e-2 5e-3)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"


domain='{{ domain }}'
task='{{ task }}'
openness='{{ openness }}'
num_classes='{{ num_classes }}'


N=$SLURM_GPUS_ON_NODE
workers=$SLURM_CPUS_PER_GPU

config=configs/recognition/hello/vanilla/vanilla_svt_ek100.py
workdir=work_dirs/train_output/ek100/svt/vanilla/${domain}/${task}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        data.workers_per_gpu=$workers \
        model.cls_head.num_classes=$num_classes \
        data.train.ann_file=data/_filelists/ek100/filelist_${domain}_train_${openness}.txt \
        data.val.ann_file=data/_filelists/ek100/filelist_${domain}_valid_${openness}.txt \
        data.test.ann_file=data/_filelists/ek100/filelist_${domain}_test_${openness}.txt \
        optimizer.lr=$lr \
    --validate --test-best --test-last

echo 'done'
exit 0
