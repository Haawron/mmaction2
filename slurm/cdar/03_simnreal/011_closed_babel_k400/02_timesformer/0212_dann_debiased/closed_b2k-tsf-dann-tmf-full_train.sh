#!/bin/bash

#SBATCH -J closed_b2k-tsf-dann-tmf-full_train
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=25G
#SBATCH -t 4-0
#SBATCH -x vll1,augi1,agi[1-2]
#SBATCH --array 0-3%2
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')


# overleaf table 기준
project='cdar'
task='03_simnreal'  # table name
subtask='011_closed_babel_k400'  # column
model='0212_dann_debiased'  # row
add_on='tmf_h100m'
extra_setting='full_train'  # 'default' if none
path_experiment="${project}/${task}/${subtask}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}"


lrs=(8e-3 5e-3 1e-3 5e-4)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"
weight=10
batch=4

config='configs/recognition/cdar/03_simnreal/011_closed_babel_k400/0212_dann_debiased/b2k_dann_tmf.py'
ckpt='work_dirs/train_output/cdar/03_simnreal/02_k400_babel/02_timesformer/warmup/h100m/32235__k2b-tsf-warmup-h100m/1/20230317-184200/latest.pth'

# ckpt loaded from config
N=$SLURM_GPUS_ON_NODE
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py "$config" \
    --launcher pytorch \
    --work-dir "$workdir" \
    --cfg-options \
        optimizer.lr="$lr" \
        model.backbone.frozen_stages=-1 data.videos_per_gpu="$batch" find_unused_parameters=False \
        load_from="$ckpt" model.neck.loss_weight="$weight" \
    --validate --test-last --test-best

source /data/hyogun/send_slack_message_mmaction.sh
exit 0