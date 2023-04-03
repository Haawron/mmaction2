#!/bin/bash

#SBATCH -J closed_k2b-tsf-dann-full_train
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=25G
#SBATCH -t 4-0
#SBATCH -x vll1,augi[1-2],agi[1-2]
#SBATCH --array 0-3%2
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')


# shellcheck source=./_vars.sh
source slurm/cdar/03_simnreal/021_closed_k400_babel/02_timesformer/020_dann/_vars.sh
add_on='h100m'
extra_setting='full_train'  # 'default' if none
path_experiment="${project}/${task}/${subtask}/${backbone}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}"


lrs=(8e-3 5e-3 1e-3 5e-4)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"
weight=1
batch=4

N=$SLURM_GPUS_ON_NODE
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py "$config" \
    --launcher pytorch \
    --work-dir "$workdir" \
    --cfg-options \
        optimizer.lr="$lr" log_config.interval=30 \
        model.backbone.frozen_stages=-1 data.videos_per_gpu="$batch" find_unused_parameters=False \
        load_from="$ckpt" model.neck.loss_weight="$weight" \
        ckpt_revise_keys='' model.neck.dropout_ratio=.5 \
        optimizer.paramwise_cfg.custom_keys.fc_domain.lr_mult=.1 \
    --validate --test-last --test-best

source /data/hyogun/send_slack_message_mmaction.sh
exit 0
