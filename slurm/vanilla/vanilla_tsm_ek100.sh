#!/bin/bash

#SBATCH -J source_only_tsm_02_to_22
#SBATCH --gres=gpu:4
#SBATCH -t 4-0
#SBATCH --array 0-3%2
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

# outfile: slurm-{main_job_id}_{array_idx}-{job_name}.out
python /data/hyogun/lab/copy_ek_to_node_and_untar.py

current_time=$(date +'%Y%m%d-%H%M%S')
lrs=(4e-3 4e-4 4e-5 4e-6)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"
source=${source:-'P02'}
echo $lr $source $target_ann_file

python -m torch.distributed.launch --nproc_per_node=4 --master_port=$((10000+$RANDOM%20000)) tools/train.py configs/recognition/hello/vanilla/vanilla_tsm_ek100.py --launcher pytorch \
    --work-dir work_dirs/ek100/vanilla/${source}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time} \
    --cfg-options \
        data.train.ann_file=data/epic-kitchens-100/filelist_${source}_train_closed.txt \
        data.val.ann_file=data/epic-kitchens-100/filelist_${source}_valid_closed.txt \
        data.test.ann_file=data/epic-kitchens-100/filelist_${source}_test_open.txt \
        optimizer.lr=$lr \
        cls_head.num_layers=2 \
    --validate \
    --test-best

echo done
exit
