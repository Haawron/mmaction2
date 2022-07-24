#!/bin/bash

#SBATCH -J GCD4DA_tsm_ek100
#SBATCH --gres=gpu:4
#SBATCH -t 4-0
#SBATCH --array 0-1%1
#SBATCH -p vll
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

echo 'extracting the dataset from NAS ...'
python /data/hyogun/lab/copy_ek_to_node_and_untar.py
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')
lr=4e-3
# loss_ratios=(.3 .5 .7)
# loss_ratio="${loss_ratios[SLURM_ARRAY_TASK_ID]}"
loss_ratio=.35
source=${source:-'P02'}
target=${target:-'P04'}
task=${source}_${target}
echo $lr $source $target

# get source-only best ckpt to init the model
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
output=$(python slurm/print_best_scores.py -d ek100 -b tsm -m vanilla -dom ${source} -t source-only -o)
read _dataset _backbone _model _domain _task _acc _mca _unk _jid ckpt _config <<< $output
python -m torch.distributed.launch --nproc_per_node=4 --master_port=$((10000+$RANDOM%20000)) tools/train.py configs/recognition/hello/ours/gcd4da_phase0_tsm_ek100.py --launcher pytorch \
    --work-dir work_dirs/sanity/ek100/tsm/GCD4DA/${task}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time} \
    --cfg-options \
        data.videos_per_gpu=4 \
        data.train.0.ann_file=data/epic-kitchens-100/filelist_${source}_train_closed.txt \
        data.train.1.ann_file=data/epic-kitchens-100/filelist_${target}_train_open.txt \
        data.val.ann_file=data/epic-kitchens-100/filelist_${target}_valid_closed.txt \
        data.test.ann_file=data/epic-kitchens-100/filelist_${target}_test_closed.txt \
        optimizer.lr=$lr \
        model.cls_head.num_layers=1 \
        model.cls_head.num_features=512 \
        model.cls_head.hsic_factor=.05 \
        model.cls_head.dropout_ratio=0. \
        model.cls_head.loss_cls.loss_ratio=$loss_ratio \
        model.cls_head.loss_cls.tau=.1 \
        model.cls_head.debias=False \
        evaluation.interval=5 \
        load_from=$ckpt \
    --validate \
    --test-best --test-last

echo done
exit
