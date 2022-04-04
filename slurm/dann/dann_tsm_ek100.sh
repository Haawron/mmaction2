#!/bin/bash

#SBATCH -J dann_tsm_02_to_22
#SBATCH --gres=gpu:4
#SBATCH -t 4-0
#SBATCH --array 0-1%1
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

echo 'extracting the dataset from NAS ...'
python /data/hyogun/lab/copy_ek_to_node_and_untar.py
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')
# lrs=(4e-3 4e-4 4e-5 4e-6)
lrs=(8e-4 4e-4)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"
source=${source:-'P02'}
target=${target:-'P22'}
task=${source}_${target}
echo $lr $source

# get source-only best ckpt to init the model
output=$(python slurm/print_best_scores.py -d ek100 -b tsm -m vanilla -dom ${source} -t source-only -o)
read _dataset _backbone _model _domain _task _acc _mca _jid ckpt _config <<< $output

python -m torch.distributed.launch --nproc_per_node=4 --master_port=$((10000+$RANDOM%20000)) tools/train.py configs/recognition/hello/dann/dann_tsm_ek100.py --launcher pytorch \
    --work-dir work_dirs/train_output/ek100/tsm/dann/${task}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time} \
    --cfg-options \
        data.train.0.ann_file=data/epic-kitchens-100/filelist_${source}_train_closed.txt \
        data.train.1.ann_file=data/epic-kitchens-100/filelist_${target}_train_open.txt \
        data.val.ann_file=data/epic-kitchens-100/filelist_${target}_valid_closed.txt \
        data.test.ann_file=data/epic-kitchens-100/filelist_${target}_test_closed.txt \
        optimizer.lr=$lr \
        model.cls_head.num_cls_layers=1 \
        model.cls_head.num_domain_layers=4 \
        model.cls_head.loss_domain.loss_weight=1.5 \
        load_from=$ckpt \
    --validate \
    --test-best --test-last

echo done
exit
