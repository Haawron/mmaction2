#!/bin/bash

#SBATCH -J vanilla-timesformer-babel-target-only
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=15G
#SBATCH -t 3-0
#SBATCH -x agi[1-2],augi2,vll[2-4]
#SBATCH --array 0-3%4
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')

lrs=(1e-2 5e-3 1e-3 5e-4)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"

ckpt='data/weights/vit/vit_base_patch16_224.pth'

N=$SLURM_GPUS_ON_NODE
task='target-only'

config=configs/recognition/hello/vanilla/vanilla_timesformer_babel_open.py
workdir=work_dirs/train_output/babel2kinetics/timesformer/vanilla/${task}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir "$workdir" \
    --cfg-options optimizer.lr="$lr" model.backbone.pretrained=$ckpt total_epochs=50 \
    --validate --test-last --test-best

echo 'done'
exit 0