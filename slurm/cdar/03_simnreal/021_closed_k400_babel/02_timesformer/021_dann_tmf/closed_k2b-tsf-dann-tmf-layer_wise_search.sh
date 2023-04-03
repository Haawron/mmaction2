#!/bin/bash

#SBATCH -J closed_k2b-tsf-dann-tmf-layer_wise_search
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=25G
#SBATCH -t 4-0
#SBATCH -x vll1,augi1,agi[1-2]
#SBATCH --array 0-39:2%4
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')


# shellcheck source=./_vars.sh
source slurm/cdar/03_simnreal/021_closed_k400_babel/02_timesformer/021_dann_tmf/_vars.sh
add_on='h100m'
extra_setting='layer_wise_search'  # 'default' if none
path_experiment="${project}/${task}/${subtask}/${backbone}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}"


# frozen layers: corresponding batches
# 1: 6, 2: 6, 3: 7, 4: 8, 5: 8, 6: 10, 7: 12, 8: 14, 9: 20, 10: 26, 11: 40
weight=10
lrs=(8e-3 5e-3 1e-3 5e-4)
frozens=(1 2 3 4 5 6  7  8  9  10)
batches=(6 6 7 8 8 10 12 14 20 26)
idx0=$(( SLURM_ARRAY_TASK_ID / ${#frozens[@]} ))
idx1=$(( SLURM_ARRAY_TASK_ID % ${#frozens[@]} ))
lr="${lrs[idx0]}"
frozen="${frozens[idx1]}"
batch="${batches[idx1]}"
echo -e "hostname: $(hostname), lr: $lr, frozen_layers: $frozen, batch_size: $batch\n\n\n\n\n\n\n"

# ckpt loaded from config
N=$SLURM_GPUS_ON_NODE
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py "$config" \
    --launcher pytorch \
    --work-dir "$workdir" \
    --cfg-options \
        optimizer.lr="$lr" log_config.interval=90 \
        model.backbone.frozen_stages="$frozen" data.videos_per_gpu="$batch" \
        load_from="$ckpt" model.neck.loss_weight="$weight" \
    --validate --test-last --test-best

source /data/hyogun/send_slack_message_mmaction.sh
exit 0
