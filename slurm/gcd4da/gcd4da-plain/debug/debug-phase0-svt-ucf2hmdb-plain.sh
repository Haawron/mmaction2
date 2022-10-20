#!/bin/bash

#SBATCH -J debug-phase0-svt-ucf2hmdb-plain
#SBATCH --gres=gpu:4
#SBATCH -t 4-0
#SBATCH -p batch
#SBATCH --array 0-3%2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out
#SBATCH -x agi1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gunsbrother@khu.ac.kr

echo 'extracting the dataset from NAS ...'
. slurm/utils/copy_ucf_hmdb_to_node_and_untar.sh
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')

N=$SLURM_GPUS_ON_NODE

workdir=work_dirs/train_output/ucf2hmdb/svt/gcd4da/debug__plain/phase0/default/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

configs=(
    configs/recognition/hello/gcd4da/plain/debug/svt_01_ucfclosed2ucfclosed.py
    configs/recognition/hello/gcd4da/plain/debug/svt_02_ucfclosed2ucfopen.py
    configs/recognition/hello/gcd4da/plain/debug/svt_03_ucfclosed2hmdbclosed.py
    configs/recognition/hello/gcd4da/plain/debug/svt_04_ucfclosed2hmdbopen.py
)

config="${configs[SLURM_ARRAY_TASK_ID]}"
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --validate --test-best --test-last

echo 'done'
exit 0
