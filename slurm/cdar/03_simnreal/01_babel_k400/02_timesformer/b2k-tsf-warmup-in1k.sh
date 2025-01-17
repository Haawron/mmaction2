#!/bin/bash

#SBATCH -J b2k-tsf-warmup-in1k
#SBATCH -p batch_vll
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=15G
#SBATCH -t 1-0
#SBATCH -x vll1
#SBATCH --array 0-3%4
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')


# overleaf table 기준
project='cdar'
task='03_simnreal'  # table name
subtask='01_babel_k400'  # column
model='02_timesformer'  # row
add_on='warmup'
extra_setting='in1k'  # default if none
path_experiment="${project}/${task}/${subtask}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}"


# lrs=(1e-2 8e-3 5e-3 1e-3)  # 1e-3에서 best 뜸
lrs=(5e-4 1e-4)  # --array 0-1%2, 5e-4에서 best
lr="${lrs[SLURM_ARRAY_TASK_ID]}"

ckpt='data/weights/vit/vit_base_patch16_224.pth'
config='configs/recognition/cdar/03_simnreal/01_babel_k400/02_timesformer/b2k_tsf_warmup.py'

N=$SLURM_GPUS_ON_NODE
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir "$workdir" \
    --cfg-options optimizer.lr="$lr" model.backbone.pretrained="$ckpt" \
    --validate --test-last --test-best

exit 0
