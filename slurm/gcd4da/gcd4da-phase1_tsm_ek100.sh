#!/bin/bash

#SBATCH -J GCD4DA_tsm_ek100_phase1
#SBATCH --gres=gpu:4
#SBATCH -t 4-0
#SBATCH --array 0-1%1
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

echo 'extracting the dataset from NAS ...'
python /data/hyogun/lab/copy_ek_to_node_and_untar.py
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')
lrs=(4e-3 4e-4)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"
source=${source:-'P02'}
target=${target:-'P22'}
task=${source}_${target}
echo $lr $source $target

# get source-only best ckpt to init the model
output=$(python slurm/print_best_scores.py -m gcd4da_phase0 -o --task ${source}_${target})
read _dataset _backbone _model _task _acc _mca _unk _jid ckpt config <<< $output

model_dir=${ckpt%/*}
pseudo=$model_dir/$(ls $model_dir | grep pseudo)
centroid=$model_dir/$(ls $model_dir | grep centroid)
k=${pseudo##*_k}
k=${k%.*}

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 torchrun --nproc_per_node=4 --master_port=$((10000+$RANDOM%20000)) tools/train.py configs/recognition/hello/ours/gcd4da_phase1_tsm_ek100.py --launcher pytorch \
    --work-dir work_dirs/train_output/ek100/tsm/gcd4da_phase0/${task}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time} \
    --cfg-options \
        model.cls_head.num_classes=$k \
        model.cls_head.loss_cls.num_classes=$k \
        model.cls_head.centroids.p_centroid=$centroid \
        data.train.0.ann_file=data/epic-kitchens-100/filelist_${source}_train_closed.txt \
        data.train.1.ann_file=$pseudo \
        data.val.ann_file=data/epic-kitchens-100/filelist_${target}_valid_open.txt \
        data.test.ann_file=data/epic-kitchens-100/filelist_${target}_test_open.txt \
        optimizer.lr=$lr \
        optimizer.paramwise_cfg.fc_lr5=False \
        load_from=$ckpt \
    --validate \
    --test-best --test-last

echo done
exit
