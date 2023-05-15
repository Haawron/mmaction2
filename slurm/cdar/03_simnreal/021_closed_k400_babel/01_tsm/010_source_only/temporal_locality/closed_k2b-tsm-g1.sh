#!/bin/bash

#SBATCH -J closed_k2b-tsm-g1
#SBATCH -p batch_vll
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=29G
#SBATCH -t 4-0
#SBATCH -x vll1
#SBATCH --array 0-7
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')

global_indices=(3 4)
idx0=$(( SLURM_ARRAY_TASK_ID % ${#global_indices[@]} ))
seed=$(( SLURM_ARRAY_TASK_ID / ${#global_indices[@]} ))

global_index="${global_indices[idx0]}"

# overleaf table 기준
project='cdar'
task='03_simnreal'  # table name
subtask='021_closed_k400_babel'  # column
backbone='01_tsm'
model='010_source_only'  # row
add_on='locality'
extra_setting="g1_${global_index}"  # 'default' if none
path_experiment="${project}/${task}/${subtask}/${backbone}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${seed}/${current_time}"


lr=1e-3

config='configs/recognition/cdar/03_simnreal/021_closed_k400_babel/01_tsm/010_source_only/temporal_locality/k2b_tsm_lngn.py'

N=$SLURM_GPUS_ON_NODE
calibed_lr="$(perl -le "print $lr * $N / 4")"
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py "$config" \
    --launcher pytorch --seed "$seed" \
    --work-dir "$workdir" \
    --validate --test-last --test-best \
    --cfg-options \
        optimizer.lr="$calibed_lr" \
        log_config.interval=10 evaluation.interval=1 \
        model.forward_kwargs.index._delete_='True' model.forward_kwargs.index.g="$global_index" \
        data.videos_per_gpu=18 data.workers_per_gpu=10

# shellcheck source=../../../../_extract_backbone_tsm.sh
source slurm/cdar/03_simnreal/_extract_backbone_tsm.sh
# shellcheck source=../../../../_tsne.sh
source slurm/cdar/03_simnreal/_tsne.sh
source /data/hyogun/send_slack_message_mmaction.sh
exit 0
