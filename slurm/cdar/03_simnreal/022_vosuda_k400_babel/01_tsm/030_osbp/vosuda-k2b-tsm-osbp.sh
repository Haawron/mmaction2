#!/bin/bash

#SBATCH -J vosuda-k2b-tsm-osbp
#SBATCH -p batch_vll
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=29G
#SBATCH -t 4-0
#SBATCH -x vll1
#SBATCH --array 0-3
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')


# overleaf table 기준
project='cdar'
task='03_simnreal'  # table name
subtask='022_vosuda_k400_babel'  # column
backbone='01_tsm'
model='030_osbp'  # row
add_on='default'
extra_setting='default'  # 'default' if none
path_experiment="${project}/${task}/${subtask}/${backbone}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}"


lr=1e-3

config='configs/recognition/cdar/03_simnreal/022_vosuda_k400_babel/01_tsm/030_osbp/vosuda_k2b_tsm_osbp.py'
ckpts=(
    work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/010_source_only/default/default/39176__closed_k2b-tsm/0/20230504-162831/best_mean_class_accuracy_epoch_32.pth
    work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/010_source_only/default/default/39176__closed_k2b-tsm/1/20230504-162831/best_mean_class_accuracy_epoch_16.pth
    work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/010_source_only/default/default/39176__closed_k2b-tsm/2/20230504-170916/best_mean_class_accuracy_epoch_23.pth
    work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/010_source_only/default/default/39176__closed_k2b-tsm/3/20230504-170916/best_mean_class_accuracy_epoch_10.pth
)
ckpt="${ckpts[$SLURM_ARRAY_TASK_ID]}"
config='configs/recognition/cdar/03_simnreal/022_vosuda_k400_babel/01_tsm/030_osbp/vosuda_k2b_tsm_osbp.py'


N=$SLURM_GPUS_ON_NODE
calibed_lr="$(perl -le "print $lr * $N / 4")"
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py "$config" \
    --launcher pytorch --seed "$SLURM_ARRAY_TASK_ID" \
    --work-dir "$workdir" \
    --cfg-options \
        optimizer.lr="$calibed_lr" \
        load_from="$ckpt" \
        log_config.interval=10 evaluation.interval=1 \
        ckpt_revise_keys='' \
    --validate --test-last --test-best

# shellcheck source=../../../_extract_backbone_tsm.sh
source slurm/cdar/03_simnreal/_extract_backbone_tsm.sh
# shellcheck source=../../../_tsne.sh
source slurm/cdar/03_simnreal/_tsne.sh
source /data/hyogun/send_slack_message_mmaction.sh
exit 0