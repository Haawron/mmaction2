#!/bin/bash


if [ -z "$config" ]; then
    echo 'config not defined'
fi
if [ -z "$workdir" ]; then
    echo 'workdir not defined'
fi

echo -e '\n\nEvaluating the test results ...'
echo "$config"
echo "$workdir"

eval_ckpt="$(find "$workdir" -name 'best*.pth' -type f)"
echo "$eval_ckpt"

N=$SLURM_GPUS_ON_NODE
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/test.py \
	"$config" --launcher pytorch "$eval_ckpt" \
	--eval 'top_k_accuracy' 'mean_class_accuracy' 'confusion_matrix' \
	--cfg-options \
		data.test_dataloader.videos_per_gpu=40 \
		data.test_dataloader.workers_per_gpu=$(( SLURM_CPUS_ON_NODE / SLURM_GPUS_ON_NODE ))

echo -e 'Done\n\n'
