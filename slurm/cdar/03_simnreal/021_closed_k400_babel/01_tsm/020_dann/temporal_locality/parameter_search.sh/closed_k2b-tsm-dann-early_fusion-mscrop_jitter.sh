#!/bin/bash

#SBATCH -J closed_k2b-tsm-dann-early_fusion-mscrop_jitter
#SBATCH -p batch_vll
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=29G
#SBATCH -t 4-0
#SBATCH -x vll[1-2]
#SBATCH --array 0-3
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')


# overleaf table 기준
project='cdar'
task='03_simnreal'  # table name
subtask='021_closed_k400_babel'  # column
backbone='01_tsm'
model='020_dann'  # row
add_on='locality'
extra_setting='early_fusion-mscrop_jitter'  # 'default' if none
path_experiment="${project}/${task}/${subtask}/${backbone}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}"


lr=1e-3

config='configs/recognition/cdar/03_simnreal/021_closed_k400_babel/01_tsm/020_dann/temporal_locality/parameter_search/k2b_tsm_dann_lngn_early_fusion_mscrop.py'
ckpts=(
    work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/010_source_only/locality/l2g1_34/39030__closed_k2b-tsm-l2g1/0/20230502-112831/best_mean_class_accuracy_epoch_33.pth
    work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/010_source_only/locality/l2g1_34/39030__closed_k2b-tsm-l2g1/1/20230502-122315/best_mean_class_accuracy_epoch_5.pth
    work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/010_source_only/locality/l2g1_34/39030__closed_k2b-tsm-l2g1/2/20230502-141738/best_mean_class_accuracy_epoch_10.pth
    work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/010_source_only/locality/l2g1_34/39030__closed_k2b-tsm-l2g1/3/20230502-142328/best_mean_class_accuracy_epoch_25.pth
)
ckpt="${ckpts[$SLURM_ARRAY_TASK_ID]}"

N=$SLURM_GPUS_ON_NODE
calibed_lr="$(perl -le "print $lr * $N / 4")"
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py "$config" \
    --launcher pytorch --seed "$SLURM_ARRAY_TASK_ID" \
    --work-dir "$workdir" \
    --cfg-options \
        optimizer.lr="$calibed_lr" \
        ckpt_revise_keys='' \
        load_from="$ckpt" \
        log_config.interval=10 evaluation.interval=1 \
    --validate --test-last --test-best

# shellcheck source=../../../../../_extract_backbone_tsm.sh
source slurm/cdar/03_simnreal/_extract_backbone_tsm.sh
# shellcheck source=../../../../../_tsne.sh
source slurm/cdar/03_simnreal/_tsne.sh
source /data/hyogun/send_slack_message_mmaction.sh
exit 0
