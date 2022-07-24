#!/bin/bash

#SBATCH -J GCD4DA_tsm_ek100
#SBATCH --gres=gpu:4
#SBATCH -t 4-0
#SBATCH --array 0-1%1
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

echo 'extracting the dataset from NAS ...'
python tools/data/ek100/copy_ek_to_node_and_untar.py
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')
loss_ratios=(.1 .3)
loss_ratio="${loss_ratios[SLURM_ARRAY_TASK_ID]}"
source=${source:-'P02'}
target=${target:-'P22'}
task=${source}_${target}
echo $lr $source $target

# get source-only best ckpt to init the model
output=$(python slurm/print_best_scores.py -d ek100 -b tsm -m vanilla -dom ${source} -t source-only -o)
read _dataset _backbone _model _domain _task _acc _mca _unk _jid ckpt _config <<< $output

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 torchrun --nproc_per_node=4 --master_port=$((10000+$RANDOM%20000)) tools/train.py configs/recognition/hello/ours/gcd4da_phase0_tsm_ek100.py --launcher pytorch \
    --work-dir work_dirs/train_output/ek100/tsm/gcd4da_phase0/${task}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time} \
    --cfg-options \
        data.train.0.ann_file=data/_filelists/ek100/filelist_${source}_train_closed.txt \
        data.train.1.ann_file=data/_filelists/ek100/filelist_${target}_train_open.txt \
        data.val.ann_file=data/_filelists/ek100/filelist_${target}_valid_closed.txt \
        data.test.ann_file=data/_filelists/ek100/filelist_${target}_test_closed.txt \
        optimizer.lr=4e-3 \
        model.cls_head.num_layers=2 \
        model.cls_head.num_features=512 \
        model.cls_head.hsic_factor=.01 \
        model.cls_head.dropout_ratio=.05 \
        model.cls_head.loss_cls.loss_ratio=$loss_ratio \
        model.cls_head.loss_cls.tau=.5 \
        model.cls_head.debias=True \
        model.cls_head.bias_input=True \
        model.cls_head.bias_network=True \
        evaluation.interval=5 \
        optimizer.paramwise_cfg.fc_lr5=False \
        lr_config.warmup=exp \
        load_from=$ckpt \
    --validate \
    --test-best --test-last

echo done
exit
