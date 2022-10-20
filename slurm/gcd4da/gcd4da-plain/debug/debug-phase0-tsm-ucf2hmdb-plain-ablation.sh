#!/bin/bash

#SBATCH -J debug-phase0-tsm-ucf2hmdb-plain-ablation
#SBATCH --gres=gpu:4
#SBATCH -t 12:00:00
#SBATCH -p batch
#SBATCH --array 0-23%2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out
#SBATCH -x agi1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gunsbrother@khu.ac.kr

echo 'extracting the dataset from NAS ...'
. slurm/utils/copy_ucf_hmdb_to_node_and_untar.sh
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')

p_vanilla_weight='work_dirs/train_output/ucf2hmdb/tsm/vanilla/source-only/4380__vanilla-tsm-ucf2hmdb-source-only/4/20220728-204923/best_mean_class_accuracy_epoch_20.pth'

lrs=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3)
frozens=(3 '[0,2,3,4]')
batches=(24 6)
epoches=(200 70)
ckpts=($p_vanilla_weight '')

idx0=$(( SLURM_ARRAY_TASK_ID / ${#ckpts[@]} / ${#frozens[@]} ))
idx1=$(( SLURM_ARRAY_TASK_ID / ${#ckpts[@]} % ${#frozens[@]} ))
idx2=$(( SLURM_ARRAY_TASK_ID % ${#ckpts[@]} ))

lr="${lrs[idx0]}"
frozen="${frozens[idx1]}"
batch="${batches[idx1]}"
epoch="${epoches[idx1]}"
ckpt="${ckpts[idx2]}"

N=$SLURM_GPUS_ON_NODE

workdir=work_dirs/train_output/ucf2hmdb/tsm/gcd4da/debug__plain/phase0/default/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

config=configs/recognition/hello/gcd4da/plain/debug/tsm_04_ucfclosed2hmdbopen.py
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
        --cfg-options \
            total_epochs=$epoch \
            model.backbone.frozen_stages=$frozen \
            data.videos_per_gpu=$batch \
            optimizer.lr=$lr \
            lr_config.min_lr="${lr}*1e-3" \
            load_from=$ckpt \
    --validate --test-best --test-last

echo 'done'
exit 0
