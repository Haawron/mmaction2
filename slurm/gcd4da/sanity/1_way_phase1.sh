#!/bin/bash

#SBATCH -J GCD4DA_tsm_ek100_sanity_one_way_phase1
#SBATCH --gres=gpu:4
#SBATCH -t 4-0
#SBATCH --array 0-1%2
#SBATCH -p vll
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

echo 'extracting the dataset from NAS ...'
python /data/hyogun/lab/copy_ek_to_node_and_untar.py
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')
lr=4e-3
source=${source:-'P02'}
targets=(P04 P22)
target="${targets[SLURM_ARRAY_TASK_ID]}"
task=${source}_${target}
echo $lr $source $target

# get source-only best ckpt to init the model
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 torchrun --nproc_per_node=4 --master_port=$((10000+$RANDOM%20000)) tools/train.py configs/recognition/hello/ours/gcd4da_phase1_tsm_ek100.py --launcher pytorch \
    --work-dir work_dirs/sanity/ek100/tsm/GCD4DA/${task}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time} \
    --cfg-options \
        data.train.0.ann_file=data/epic-kitchens-100/filelist_${source}_train_closed.txt \
        data.train.1.ann_file=work_dirs/sanity/ek100/tsm/GCD4DA/P02_P04/19135__GCD4DA_tsm_ek100_sanity_one_way/0/20220506-045539/filelist_pseudo_P04_k009_one_way.txt \
        data.val.ann_file=data/epic-kitchens-100/filelist_${target}_valid_open.txt \
        data.test.ann_file=data/epic-kitchens-100/filelist_${target}_test_open.txt \
        optimizer.lr=$lr \
        model.cls_head.num_layers=1 \
        model.cls_head.num_features=512 \
        model.cls_head.hsic_factor=.05 \
        model.cls_head.dropout_ratio=0. \
        model.cls_head.loss_cls.loss_ratio=1. \
        model.cls_head.loss_cls.tau=.1 \
        model.cls_head.debias=True \
        model.cls_head.bias_input=False \
        model.cls_head.bias_network=False \
        evaluation.interval=5 \
        optimizer.paramwise_cfg.fc_lr5=False \
        load_from=work_dirs/sanity/ek100/tsm/GCD4DA/P02_P04/19135__GCD4DA_tsm_ek100_sanity_one_way/0/20220506-045539/best_mean_class_accuracy_epoch_30.pth \
    --validate \
    --test-best --test-last

echo done
exit
