#!/bin/bash

#SBATCH -J k2b-tsf-dann-from_best
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=25G
#SBATCH -t 4-0
#SBATCH -x vll1,augi1,agi[1-2]
#SBATCH --array 0-3%4
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')


# overleaf table 기준
project='cdar'
task='03_simnreal'  # table name
subtask='02_k400_babel'  # column
model='021_dann'  # row
add_on='h100m'
extra_setting='from_best'  # 'default' if none
path_experiment="${project}/${task}/${subtask}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}"


lrs=(8e-3 5e-3 1e-3 5e-4)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"

config='configs/recognition/cdar/03_simnreal/02_k400_babel/021_dann/k2b_tsf_dann.py'
ckpt='work_dirs/train_output/cdar/03_simnreal/02_k400_babel/021_dann/default/default/32414__k2b-tsf-dann/0/20230318-145433/best_kmeans_balanced_epoch_10.pth'

# ckpt loaded from config
N=$SLURM_GPUS_ON_NODE
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py "$config" \
    --launcher pytorch \
    --work-dir "$workdir" \
    --cfg-options optimizer.lr="$lr" log_config.interval=90 load_from="$ckpt" \
    --validate --test-last --test-best

source /data/hyogun/send_slack_message_mmaction.sh
exit 0
