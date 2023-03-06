#!/bin/bash

#SBATCH -J k2b-gcd
#SBATCH -p batch_vll
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=25G
#SBATCH -t 3-0
#SBATCH -x vll1
#SBATCH --array 0-3%4
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')


# overleaf table 기준
project='cdar'
task='03_simnreal'  # table name
subtask='02_k400_babel'  # column
model='03_gcd'  # row
add_on='default'
extra_setting='default'  # 'default' if none
path_experiment="${project}/${task}/${subtask}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}"


lrs=(1e-2 5e-3 1e-3 5e-4)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"

ckpt='data/weights/vit/vit_base_patch16_224.pth'
config='configs/recognition/cdar/03_simnreal/02_k400_babel/03_gcd/k2b_gcd.py'

N=$SLURM_GPUS_ON_NODE
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir "$workdir" \
    --cfg-options optimizer.lr="$lr" model.backbone.pretrained="$ckpt" log_config.interval=100 \
    --validate --test-last --test-best

source /data/hyogun/send_slack_message_mmaction.sh
exit 0
