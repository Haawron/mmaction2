#!/bin/bash

#SBATCH -J closed_k2b-tsm-dann_tmf-only_source_accumx2_lr_search
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=25G
#SBATCH -t 4-0
#SBATCH -x vll1,augi[1-2],agi[1-2]
#SBATCH --array 0-7
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')


# overleaf table 기준
project='cdar'
task='03_simnreal'  # table name
subtask='021_closed_k400_babel'  # column
backbone='01_tsm'
model='021_dann_tmf'  # row
add_on='only_source'
extra_setting='accumx2'  # 'default' if none
path_experiment="${project}/${task}/${subtask}/${backbone}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}"


lrs=(1e-3 5e-4 3e-4 1e-4)
lr="${lrs[SLURM_ARRAY_TASK_ID/2]}"

config='configs/recognition/cdar/03_simnreal/021_closed_k400_babel/01_tsm/021_dann_tmf/k2b_tsm_dann_tmf_only_source.py'
# ckpt='work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/010_source_only/default/tmf/35479__closed_k2b-tsm-tmf/0/20230405-001320/best_mean_class_accuracy_epoch_30.pth'
ckpt='work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/010_source_only/tmf/all_bg/35802__closed_k2b-tsm-tmf-all_bg/0/20230407-210118/best_mean_class_accuracy_epoch_20.pth'

N=$SLURM_GPUS_ON_NODE
calibed_lr="$(perl -le "print $lr * $N / 4")"
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py "$config" \
    --launcher pytorch \
    --work-dir "$workdir" \
    --cfg-options \
        load_from="$ckpt" optimizer.lr="$calibed_lr" log_config.interval=20 \
        ckpt_revise_keys='' \
        total_epochs=50 evaluation.interval=1 checkpoint_config.interval=25 \
        optimizer_config.type='GradientCumulativeOptimizerHook' optimizer_config.cumulative_iters=2 \
    --validate --test-last --test-best

# shellcheck source=../../../../_extract_backbone_tsm.sh
source slurm/cdar/03_simnreal/_extract_backbone_tsm.sh
# shellcheck source=../../../../_tsne.sh
source slurm/cdar/03_simnreal/_tsne.sh
source /data/hyogun/send_slack_message_mmaction.sh
exit 0
