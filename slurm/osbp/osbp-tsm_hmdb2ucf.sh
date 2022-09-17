#!/bin/bash

#SBATCH -J osbp-tsm_hmdb2ucf
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=15G
#SBATCH -t 4-0
#SBATCH --array 0-35%2
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=gunsbrother@khu.ac.kr

echo 'extracting the dataset from NAS ...'
python /data/hyogun/lab/copy_ek_to_node_and_untar.py
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')

lrs=(4e-2 8e-3 4e-3 8e-4 4e-4 4e-5)
weights=(1e-3 5e-3)
ts=(.3 .5 .7)
idx2=$(( SLURM_ARRAY_TASK_ID % ${#ts[@]} ))
idx1=$(( SLURM_ARRAY_TASK_ID / ${#ts[@]} % ${#weights[@]} ))
idx0=$(( SLURM_ARRAY_TASK_ID / ${#ts[@]} / ${#weights[@]} ))

lr="${lrs[idx0]}"
weight="${weights[idx1]}"
t="${ts[idx2]}"

N=$SLURM_GPUS_ON_NODE
config=configs/recognition/hello/osbp/osbp_tsm_hmdb2ucf.py
workdir=work_dirs/train_output/hmdb2ucf/tsm/osbp/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        model.cls_head.loss_cls.weight_loss=$weight \
        model.cls_head.loss_cls.target_domain_label=$t \
        optimizer.lr=$lr \
    --validate \
    --test-best --test-last

echo 'done'
exit
