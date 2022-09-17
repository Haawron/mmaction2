#!/bin/bash

#SBATCH -J debug-cop_multi-phase0-tsm-hmdb2ucf
#SBATCH --gres=gpu:4
#SBATCH -t 4-0
#SBATCH -p batch
#SBATCH --array 0-17%2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=15G
#SBATCH -x agi1
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out

echo 'extracting the dataset from NAS ...'
. slurm/utils/copy_ucf_hmdb_to_node_and_untar.sh
echo 'done'

current_time=$(date +'%Y%m%d-%H%M%S')

lrs=(4e-2 8e-3 4e-3 8e-4 4e-4 4e-5)
weights=(1 3 5)  # bash doesn't do floating point operation

idx0=$(( SLURM_ARRAY_TASK_ID / ${#weights[@]} ))
idx1=$(( SLURM_ARRAY_TASK_ID % ${#weights[@]} ))

lr="${lrs[idx0]}"
weight="${weights[idx1]}"

N=$SLURM_GPUS_ON_NODE

config=configs/recognition/hello/gcd4da/cop/phase0_tsm_hmdb2ucf.py
workdir=work_dirs/train_output/hmdb2ucf/tsm/gcd4da/debug__cop_multi/phase0/default/${SLURM_ARRAY_JOB_ID}__${SLURM_JOB_NAME}/${SLURM_ARRAY_TASK_ID}/${current_time}

# closed -> closed로 open-set setting이 없는 상태에서는 잘 동작하는지 확인
# => TSM이 문제인 건지 확인

OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
    --launcher pytorch \
    --work-dir $workdir \
    --cfg-options \
        optimizer.lr=$lr \
        model.neck.loss_cls.loss_weight=".$weight" \
        model.cls_head.loss_cls.loss_weight=".$(( 10 - $weight ))" \
        data.train.1.ann_file=data/_filelists/ucf101/filelist_ucf_train_closed.txt \
        data.val.ann_file=data/_filelists/ucf101/filelist_ucf_val_closed.txt \
        data.test.ann_file=data/_filelists/ucf101/filelist_ucf_test_closed.txt \
        evaluation.metric_options.logits.p_out_dir=$workdir \
    --validate --test-best --test-last

echo 'done'
exit 0
