#!/bin/bash

#SBATCH -J vanilla_tsm_ek100
#SBATCH --gres=gpu:4
#SBATCH -t 4-0
#SBATCH --array 0-1%1
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

echo 'extracting the dataset from NAS ...'
python /data/hyogun/lab/copy_ek_to_node_and_untar.py
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')
# lrs=(4e-3 4e-4 4e-5 4e-6)
# lrs=(4e-3 1e-3 8e-4 4e-4)
lrs=(8e-3 4e-3)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"
domain=${domain:-'P02'}
echo $lr $domain

# source-only: train only for shared labels
# target-only: train for all labels but unshared ones as a single label
echo "task: $task"
echo "openness: $openness"
echo "num_classes: $num_classes"
if [ -z $task ] || [ -z $openness ] || [ -z $num_classes ]; then
    echo "At least one of doamin, openness, num_classes is not set"
    exit 1
fi

workdir=work_dirs/train_output/ek100/tsm/vanilla/${domain}/${task}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
python -m torch.distributed.launch --nproc_per_node=4 --master_port=$((10000+$RANDOM%20000)) tools/train.py configs/recognition/hello/vanilla/vanilla_tsm_ek100.py \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        data.train.ann_file=data/epic-kitchens-100/filelist_${domain}_train_${openness}.txt \
        data.val.ann_file=data/epic-kitchens-100/filelist_${domain}_valid_${openness}.txt \
        data.test.ann_file=data/epic-kitchens-100/filelist_${domain}_test_${openness}.txt \
        optimizer.lr=$lr \
        model.cls_head.num_classes=$num_classes \
    --validate \
    --test-best --test-last

echo 'done'
exit
