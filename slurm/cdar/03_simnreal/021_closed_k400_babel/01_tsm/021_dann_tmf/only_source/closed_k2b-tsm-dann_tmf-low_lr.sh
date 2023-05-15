#!/bin/bash

#SBATCH -J closed_k2b-tsm-dann_tmf-low_lr
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=25G
#SBATCH -t 4-0
#SBATCH -x vll1,augi[1-2],agi[1-2]
#SBATCH --array 0
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')


# overleaf table 기준
project='cdar'
task='03_simnreal'  # table name
subtask='021_closed_k400_babel'  # column
backbone='01_tsm'
model='021_dann_tmf'  # row
add_on='only_source'
extra_setting='low_lr'  # 'default' if none
path_experiment="${project}/${task}/${subtask}/${backbone}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}"


lr=1e-6

config='configs/recognition/cdar/03_simnreal/021_closed_k400_babel/01_tsm/021_dann_tmf/k2b_tsm_dann_tmf_only_source.py'
ckpt='work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/010_source_only/default/tmf/35428__closed_k2b-tsm-tmf/2/20230404-182711/best_mean_class_accuracy_epoch_35.pth'

N=$SLURM_GPUS_ON_NODE
calibed_lr="$(perl -le "print $lr * $N / 4")"
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py "$config" \
    --launcher pytorch \
    --work-dir "$workdir" \
    --cfg-options \
        load_from="$ckpt" optimizer.lr="$calibed_lr" log_config.interval=20 \
        ckpt_revise_keys='' \
        total_epochs=20 evaluation.interval=1 checkpoint_config.interval=5 \
        lr_config.step='[2,10]' \
    --validate --test-last --test-best

# shellcheck source=../../../../_extract_backbone_tsm.sh
source slurm/cdar/03_simnreal/_extract_backbone_tsm.sh
# shellcheck source=../../../../_tsne.sh
source slurm/cdar/03_simnreal/_tsne.sh
source /data/hyogun/send_slack_message_mmaction.sh
exit 0
