#!/bin/bash

#SBATCH -J GCD4DA_tsm_ucf-hmdb_bias-check
#SBATCH --gres=gpu:4
#SBATCH -t 4-0
#SBATCH -p batch
#SBATCH --array 0-7%4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=50G
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out


echo 'extracting the dataset from NAS ...'
. slurm/utils/copy_ucf_hmdb_to_node_and_untar.sh
echo 'done'

N=4
config="configs/recognition/hello/ours/gcd4da_phase0_tsm_ucf-hmdb_bias-check.py"

workdirs=(
  "work_dirs/train_output/ucf-hmdb/tsm/gcd4da/phase0/one-way/ucf-hmdb/570__GCD4DA-phase0-all_tsm_ucf-hmdb/0/20220615-040530"
  "work_dirs/train_output/ucf-hmdb/tsm/gcd4da/phase0/full/ucf-hmdb/570__GCD4DA-phase0-all_tsm_ucf-hmdb/3/20220615-040530"
  "work_dirs/train_output/ucf-hmdb/tsm/gcd4da/phase0/upper/ucf-hmdb/570__GCD4DA-phase0-all_tsm_ucf-hmdb/2/20220615-040530"
  "work_dirs/train_output/ucf-hmdb/tsm/gcd4da/phase0/lower/ucf-hmdb/20146__GCD4DA-phase0-all_tsm_ucf-hmdb/1/20220518-142721"
)
ckpts=(
  "best_mean_class_accuracy_epoch_30.pth"
  "best_mean_class_accuracy_epoch_45.pth"
  "best_mean_class_accuracy_epoch_40.pth"
  "best_mean_class_accuracy_epoch_40.pth"
)
datasets=(ucf101 hmdb51)
dataset_names=(ucf hmdb)

i=`expr $SLURM_ARRAY_TASK_ID % 4`
workdir="${workdirs[i]}"
ckpt="${workdir}/${ckpts[i]}"

i=`expr $SLURM_ARRAY_TASK_ID / 4`
dataset="${datasets[i]}"
dataset_name="${dataset_names[i]}"

data_prefix="/local_datasets/${dataset}/rawframes"
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/train.py $config \
  --launcher pytorch \
  --work-dir ${workdir}/bias-check/${dataset_name}/${SLURM_ARRAY_JOB_ID} \
  --cfg-options \
    data.train.ann_file="data/_filelists/${dataset}/scene/filelist_scene_${dataset_name}_train_open.txt" \
    data.val.ann_file="data/_filelists/${dataset}/scene/filelist_scene_${dataset_name}_val_open.txt" \
    data.test.ann_file="data/_filelists/${dataset}/scene/filelist_scene_${dataset_name}_test_open.txt" \
    data.train.data_prefix=${data_prefix} \
    data.val.data_prefix=${data_prefix} \
    data.test.data_prefix=${data_prefix} \
    optimizer.lr=4e-2 \
    evaluation.interval=10 \
    load_from=$ckpt \
  --validate \
  --test-best --test-last
