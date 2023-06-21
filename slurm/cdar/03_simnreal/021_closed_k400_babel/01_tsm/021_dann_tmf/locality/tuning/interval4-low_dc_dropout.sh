#!/bin/bash

#SBATCH -J closed_k2b-tsm-dann-tmf-dc_glx+cop-interval4-low_dc_dropout
#SBATCH -p batch_grad
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -t 4-0
#SBATCH -x ariel-v[1-13]
#SBATCH --array 0-3%2
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')


# overleaf table 기준
project='cdar'
task='03_simnreal'  # table name
subtask='021_closed_k400_babel'  # column
backbone='01_tsm'
model='021_dann_tmf'  # row
add_on='multiscale-tuning'
extra_setting='dc_glx+cop-interval4-low_dc_dropout'  # 'default' if none
path_experiment="${project}/${task}/${subtask}/${backbone}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}"


lr=1e-3

config='configs/recognition/cdar/03_simnreal/021_closed_k400_babel/01_tsm/021_dann_tmf/temporal_locality/k2b_tsm_dann_tmf_cross_dcx2.py'
ckpts=(
    'work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/011_cop_pretrain/phase0/no_aug/11432__closed_k2b-tsm-cop_pretrain-phase0-no_sched_no_aug/0/20230520-161635/epoch_500.pth'
    'work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/011_cop_pretrain/phase0/no_aug/11432__closed_k2b-tsm-cop_pretrain-phase0-no_sched_no_aug/0/20230520-161635/epoch_500.pth'
    'work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/011_cop_pretrain/phase0/no_aug/11432__closed_k2b-tsm-cop_pretrain-phase0-no_sched_no_aug/0/20230520-161635/epoch_500.pth'
    'work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/011_cop_pretrain/phase0/no_aug/11432__closed_k2b-tsm-cop_pretrain-phase0-no_sched_no_aug/0/20230520-161635/epoch_500.pth'
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
        data.train.0.pipeline.0.frame_interval=4 \
        model.neck.0.dropout_ratio=.1 model.neck.1.dropout_ratio=.1 model.neck.2.dropout_ratio=.1 \
    --validate --test-last --test-best

# # shellcheck source=../../../../_extract_backbone_tsm.sh
# source slurm/cdar/03_simnreal/_extract_backbone_tsm.sh
# # shellcheck source=../../../../_tsne.sh
# source slurm/cdar/03_simnreal/_tsne.sh
# source /data/hyogun/send_slack_message_mmaction.sh
exit 0
