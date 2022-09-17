N=$SLURM_GPUS_ON_NODE

echo "ucf(source-only target-test)"
config=configs/recognition/hello/vanilla/vanilla_svt_hmdb51_closed.py
ckpt=work_dirs/train_output/ucf2hmdb/svt/vanilla/source-only/4370__vanilla-svt-ucf2hmdb-source-only/2/20220728-182744/best_mean_class_accuracy_epoch_5.pth
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/test.py $config --launcher pytorch \
    $ckpt \
    --cfg-options \
        data.videos_per_gpu=8 \
    --eval top_k_accuracy mean_class_accuracy confusion_matrix

echo "hmdb(source-only target-test)"
config=configs/recognition/hello/vanilla/vanilla_svt_ucf101_closed.py
ckpt=work_dirs/train_output/hmdb2ucf/svt/vanilla/source-only/4372__vanilla-svt-hmdb2ucf-source-only/1/20220728-192146/best_mean_class_accuracy_epoch_20.pth
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/test.py $config --launcher pytorch \
    $ckpt \
    --cfg-options \
        data.videos_per_gpu=8 \
    --eval top_k_accuracy mean_class_accuracy confusion_matrix
