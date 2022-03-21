#!/bin/bash

conda activate open-mmlab

jid=14967
read checkpoint src tgt <<< $(python slurm/find_best.py -j ${jid})
printf "\n\n%s\n\n" ${checkpoint}
python tools/test.py configs/recognition/hello/tsm_ek_02_to_22.py \
    $checkpoint \
    --out work_dirs/test_output/${jid}_${src}_${tgt}.json \
    --eval top_k_accuracy mean_class_accuracy \
    --cfg-options \
        data.test.source_ann_file=data/epic-kitchens-100/empty.txt \
        data.test.target_ann_file=data/epic-kitchens-100/filelist_${tgt}_test_closed.txt
echo $src $tgt
