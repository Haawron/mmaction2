#!/bin/bash

#SBATCH -J cop-svt-hmdb2ucf_from_vanilla
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=15G
#SBATCH -t 4-0
#SBATCH --array 0-5%2
#SBATCH -x agi1
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gunsbrother@khu.ac.kr

current_time=$(date +'%Y%m%d-%H%M%S')

# lrs=(4e-2 8e-3 4e-3 8e-4 4e-4 4e-5)
# lr="${lrs[SLURM_ARRAY_TASK_ID]}"

lr=8e-3

frozen_nums=(11 10 9 8 7 6)
batch_nums=(16 9 7 5 4 3)
frozen_num="${frozen_nums[SLURM_ARRAY_TASK_ID]}"
batch_num="${batch_nums[SLURM_ARRAY_TASK_ID]}"

N=$SLURM_GPUS_ON_NODE
config=configs/recognition/hello/cop/cop_svt_hmdb2ucf.py
ckpt=work_dirs/train_output/hmdb2ucf/svt/vanilla/source-only/4372__vanilla-svt-hmdb2ucf-source-only/1/20220728-192146/best_mean_class_accuracy_epoch_20.pth

workdir=work_dirs/train_output/hmdb2ucf/svt/cop/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        optimizer.lr=$lr \
        model.backbone.norm_eval=True \
        model.backbone.frozen_stages=$frozen_num \
        data.videos_per_gpu=$batch_num \
    --validate \
    --test-last --test-best

exit 0
