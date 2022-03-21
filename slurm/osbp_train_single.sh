#!/bin/bash

conda activate open-mmlab
python tools/train.py configs/recognition/hello/tsm_ek_02_to_22.py --gpu-ids 0 \
    --cfg-options \
        evaluation.interval=5 \
        optimizer.lr=1e-5 \
    --validate \
    --test-best

echo
echo done
