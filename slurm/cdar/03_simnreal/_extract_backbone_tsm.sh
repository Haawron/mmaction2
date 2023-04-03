#!/bin/bash

if [ -z "$workdir" ]; then
    echo 'workdir not defined'
fi

echo -e '\n\nExtracting Backbone Features ...'
echo "$workdir"

backbone_extractor_config='configs/recognition/cdar/03_simnreal/021_closed_k400_babel/01_tsm/_extact_backbone.py'
backbone_outfile="$workdir/backbone.pkl"
backbone_ckpt="$(find "$workdir" -name 'best*.pth' -type f)"

echo -e "$backbone_ckpt" '\n-->' "$backbone_outfile"

N=$SLURM_GPUS_ON_NODE
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/test.py \
    "$backbone_extractor_config" \
    --launcher pytorch "$backbone_ckpt" \
    --out "$backbone_outfile" \
    --cfg-options \
        model.test_cfg.feature_extraction=True

echo -e 'Done\n\n'
