#!/bin/bash

#SBATCH -J k2b-tsf-osbp
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=25G
#SBATCH -t 4-0
#SBATCH -x vll1,augi1,agi[1-2]
#SBATCH --array 0-95%2
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')


# overleaf table 기준
project='cdar'
task='03_simnreal'  # table name
subtask='02_k400_babel'  # column
model='022_osbp'  # row
add_on='default'
extra_setting='default'  # 'default' if none
path_experiment="${project}/${task}/${subtask}/${model}/${add_on}/${extra_setting}"

workdir="work_dirs/train_output"
workdir="${workdir}/${path_experiment}"
workdir="${workdir}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}"


nlayerses=(1 2)
lrs=(8e-3 5e-3 1e-3 5e-4)
weights=(1e-2 1e-3 5e-3)
ts=(.3 .4 .5 .7)
idx0=$(( SLURM_ARRAY_TASK_ID / ${#ts[@]} / ${#weights[@]} / ${#lrs[@]} ))
idx1=$(( SLURM_ARRAY_TASK_ID / ${#ts[@]} / ${#weights[@]} % ${#lrs[@]} ))
idx2=$(( SLURM_ARRAY_TASK_ID / ${#ts[@]} % ${#weights[@]} ))
idx3=$(( SLURM_ARRAY_TASK_ID % ${#ts[@]} ))
nlayers="${nlayerses[idx0]}"
lr="${lrs[idx1]}"
weight="${weights[idx2]}"
t="${ts[idx3]}"

config='configs/recognition/cdar/03_simnreal/02_k400_babel/022_osbp/k2b_tsf_osbp.py'
ckpt='work_dirs/train_output/cdar/03_simnreal/02_k400_babel/02_timesformer/warmup/h100m/32235__k2b-tsf-warmup-h100m/1/20230317-184200/latest.pth'

# ckpt loaded from config
N=$SLURM_GPUS_ON_NODE
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py "$config" \
    --launcher pytorch \
    --work-dir "$workdir" \
    --cfg-options \
        model.neck.num_hidden_layers="$nlayers" \
        model.neck.target_domain_label="$t" \
        model.neck.weight_loss_target="$weight" \
        optimizer.lr="$lr" \
        log_config.interval=90 load_from="$ckpt" \
    --validate --test-last --test-best

source /data/hyogun/send_slack_message_mmaction.sh
exit 0
