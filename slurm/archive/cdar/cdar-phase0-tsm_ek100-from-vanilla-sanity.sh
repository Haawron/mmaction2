#!/bin/bash

#SBATCH -J cdar-phase0-tsm_ek100-from-vanilla-sanity
#SBATCH -p batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=15G
#SBATCH -t 4-0
#SBATCH --array 1-2%2
#SBATCH -x agi1
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

echo 'extracting the dataset from NAS ...'
python /data/hyogun/lab/copy_ek_to_node_and_untar.py
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')

lrs=(4e-2 8e-3 4e-3 8e-4 4e-4 4e-5)
lr="${lrs[SLURM_ARRAY_TASK_ID]}"
domain=${domain:-'P02'}

N=$SLURM_GPUS_ON_NODE
workers=$SLURM_CPUS_PER_GPU

config=configs/recognition/hello/gcd4da/plain/cdar_phase0_tsm_ek100_sanity.py
workdir=work_dirs/train_output/ek100/tsm/cdar/vanilla/phase0/sanity/${domain}/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

output=$(python slurm/utils/print_best_scores.py -d ek100 -m vanilla -dom ${domain} -t 'source-only' -o)
read _dataset _backbone _model _domain _task _acc _mca _unk _jid ckpt _config <<< $output

OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        data.workers_per_gpu=$workers \
        data.train.0.ann_file=data/_filelists/ek100/filelist_${domain}_train_closed.txt \
        data.train.1.ann_file=data/_filelists/ek100/filelist_${domain}_valid_open.txt \
        optimizer.lr=$lr \
        load_from=$ckpt \
    --validate \
    --test-best --test-last

echo 'done'
exit
