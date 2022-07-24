#!/bin/bash

#SBATCH -J osbp_tsm_sanity
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
target=${target:-'P04'}
echo $lr $source $target

python -m torch.distributed.launch --nproc_per_node=4 --master_port=$((10000+$RANDOM%20000)) tools/train.py configs/recognition/hello/tsm_ek_02_to_22.py --launcher pytorch \
    --work-dir work_dirs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}__${SLURM_JOB_NAME}/${current_time} \
    --cfg-options \
        data.train.source_ann_file=data/epic-kitchens-100/filelist_${source}_train_closed.txt \
        data.train.target_ann_file=data/epic-kitchens-100/empty.txt \
        data.val.source_ann_file=data/epic-kitchens-100/filelist_${source}_valid_closed.txt \
        data.val.target_ann_file=data/epic-kitchens-100/empty.txt \
        data.test.source_ann_file=data/epic-kitchens-100/empty.txt \
        data.test.target_ann_file=data/epic-kitchens-100/filelist_${source}_test_closed.txt \
        optimizer.lr=$lr \
    --validate \
    --test-best

echo done
exit
