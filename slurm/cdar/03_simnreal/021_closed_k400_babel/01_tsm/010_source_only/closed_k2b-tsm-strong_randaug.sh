#!/bin/bash

#SBATCH -J closed_k2b-tsm-strong_randaug
#SBATCH -p batch
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=25G
#SBATCH -t 4-0
#SBATCH -x vll1,augi[1-2],agi[1-2]
#SBATCH --array 0-3%4
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')


# overleaf table 기준
project='cdar'
task='03_simnreal'  # table name
subtask='021_closed_k400_babel'  # column
backbone='01_tsm'
model='010_source_only'  # row
add_on='default'
extra_setting='randaug2'  # 'default' if none
path_experiment="${project}/${task}/${subtask}/${backbone}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}"


lrs=(8e-3 5e-3 1e-3 5e-4)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"

config='configs/recognition/cdar/03_simnreal/021_closed_k400_babel/01_tsm/010_source_only/k2b_tsm_strong_randaug.py'

N=$SLURM_GPUS_ON_NODE
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py "$config" \
    --launcher pytorch \
    --work-dir "$workdir" \
    --cfg-options \
        optimizer.lr="$(echo "$(awk '{ print +$1 }' <<< "$lr") * $N / 4" | bc -l)" \
        log_config.interval=10 \
    --validate --test-last --test-best

# shellcheck source=../../../_extract_backbone_tsm.sh
source slurm/cdar/03_simnreal/_extract_backbone_tsm.sh
source /data/hyogun/send_slack_message_mmaction.sh
exit 0