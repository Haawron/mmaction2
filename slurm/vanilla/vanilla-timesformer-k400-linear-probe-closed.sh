#!/bin/bash

#SBATCH -J vanilla-timesformer-k400-linear-probe-closed
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=40G
#SBATCH -t 3-0
#SBATCH -x agi[1-2],augi1,vll1
#SBATCH --array 0-5%3
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

current_time=$(date +'%Y%m%d-%H%M%S')

lrs=(8e-2 1e-2 1e-3)
ckpts=(
    'data/weights/timesformer/timesformer_8x32_224_howto100M_mmaction.pth'
    'data/weights/timesformer/timesformer_8_224_ssv2_mmaction.pyth'
    'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth'
)
names=('h100m' 'ssv2' 'in1k')
idx1=$(( SLURM_ARRAY_TASK_ID % ${#lrs[@]} ))
idx0=$(( SLURM_ARRAY_TASK_ID / ${#lrs[@]} ))
ckpt="${ckpts[idx0]}"
name="${names[idx0]}"
lr="${lrs[idx1]}"

interface=$([ "$name" = 'in1k' ] && echo 'model.backbone.pretrained' || echo 'load_from')

N=$SLURM_GPUS_ON_NODE

config=configs/recognition/hello/vanilla/vanilla_timesformer_k400_closed.py
workdir=work_dirs/train_output/kinetics2babel/timesformer/linear_probe/${name}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        optimizer.lr=$lr "${interface}"="${ckpt}" \
        log_config.interval=10 model.backbone.frozen_stages=12 data.videos_per_gpu=96 \
        find_unused_parameters=True total_epochs=50 \
    --validate --test-last --test-best

# TODO: 이 파일 closed, open template 화 시키기
# closed는 tested on target도 있어야 함
# B2K K2B 양방향으로 해야 함
# N=$SLURM_GPUS_ON_NODE
# config=configs/recognition/hello/vanilla/vanilla_timesformer_k400_closed.py  # 위랑 똑같음
# ckpt=
# OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/test.py $config \
#     --launcher pytorch $ckpt \
#     --eval top_k_accuracy H_mean_class_accuracy mean_class_accuracy confusion_matrix \
#     --cfg-options data.test.ann_file='data/_filelists/babel/processed/filelist_babel_test_closed.txt'

echo 'done'
exit 0
