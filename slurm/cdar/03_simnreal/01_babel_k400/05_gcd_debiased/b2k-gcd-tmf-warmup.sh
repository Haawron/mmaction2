#!/bin/bash

#SBATCH -J b2k-gcd-tmf-warmup
#SBATCH -p batch_vll
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=25G
#SBATCH -t 1-0
#SBATCH -x vll1
#SBATCH --array 0-3%4
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')


# overleaf table 기준
project='cdar'
task='03_simnreal'  # table name
subtask='01_babel_k400'  # column
model='05_gcd_debiased'  # row
add_on='tmf'
extra_setting='warmup'  # 'default' if none
path_experiment="${project}/${task}/${subtask}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}"


lrs=(1e-2 5e-3 1e-3 5e-4)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"

ckpt='work_dirs/train_output/cdar/03_simnreal/01_babel_k400/02_timesformer/warmup/in1k/27898__b2k-tsf-warmup-in1k/0/20230129-105526/best_kmeans_epoch_40.pth'
config='configs/recognition/cdar/03_simnreal/01_babel_k400/05_gcd_debaised/b2k_gcd_tmf.py'

N=$SLURM_GPUS_ON_NODE
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir "$workdir" \
    --cfg-options optimizer.lr="$lr" load_from="$ckpt" \
    --validate --test-last --test-best

exit 0
