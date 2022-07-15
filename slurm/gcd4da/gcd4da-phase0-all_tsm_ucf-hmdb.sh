#!/bin/bash

#SBATCH -J GCD4DA-phase0-all_tsm_ucf-hmdb
#SBATCH --gres=gpu:4
#SBATCH -t 4-0
#SBATCH -p batch
#SBATCH --array 0-3%4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=50G
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

echo 'extracting the dataset from NAS ...'
. slurm/utils/copy_ucf_hmdb_to_node_and_untar.sh
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')

bias_inputs=(False False True True)
bias_networks=(False True False True)
ablations=(one-way lower upper full)

bias_input="${bias_inputs[SLURM_ARRAY_TASK_ID]}"
bias_network="${bias_networks[SLURM_ARRAY_TASK_ID]}"
ablation="${ablations[SLURM_ARRAY_TASK_ID]}"
task='ucf-hmdb'

# get source-only best ckpt to init the model
ckpt=work_dirs/hello/ucf101/vanilla/best_mean_class_accuracy_epoch_40.pth

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 torchrun --nproc_per_node=4 --master_port=$((10000+$RANDOM%20000)) tools/train.py configs/recognition/hello/ours/gcd4da_phase0_tsm_ucf-hmdb.py --launcher pytorch \
    --work-dir work_dirs/train_output/ucf-hmdb/tsm/gcd4da/phase0/${ablation}/${task}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time} \
    --cfg-options \
        optimizer.lr=4e-3 \
        model.cls_head.num_layers=1 \
        model.cls_head.num_features=512 \
        model.cls_head.hsic_factor=.01 \
        model.cls_head.dropout_ratio=.05 \
        model.cls_head.loss_cls.loss_ratio=.3 \
        model.cls_head.loss_cls.tau=1. \
        model.cls_head.debias=True \
        model.cls_head.bias_input=$bias_input \
        model.cls_head.bias_network=$bias_network \
        evaluation.interval=5 \
        optimizer.paramwise_cfg.fc_lr5=False \
        lr_config.warmup=exp \
        load_from=$ckpt \
    --validate \
    --test-best --test-last

echo done
exit
