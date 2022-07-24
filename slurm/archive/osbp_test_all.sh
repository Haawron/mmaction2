#!/bin/bash


# valid best vs. just last

jid=${jid:-'14967'}

while IFS= read -r line; do
    read i checkpoint src tgt <<< $line
    
    printf "\n\nCheckpoint: %s\nSource: %s\nTarget: %s\n\n" $checkpoint $src $tgt
    python tools/test.py configs/recognition/hello/tsm_ek_02_to_22.py \
        $checkpoint \
        --out work_dirs/test_output/model_selection/${jid}_${i}_${src}_${tgt}_source_valid.json \
        --eval top_k_accuracy \
        --cfg-options \
            data.test.source_ann_file=data/epic-kitchens-100/empty.txt \
            data.test.target_ann_file=data/epic-kitchens-100/filelist_${src}_valid_closed.txt

    python tools/test.py configs/recognition/hello/tsm_ek_02_to_22.py \
        $checkpoint \
        --out work_dirs/test_output/model_selection/${jid}_${i}_${src}_${tgt}_target_test.json \
        --eval top_k_accuracy \
        --cfg-options \
            data.test.source_ann_file=data/epic-kitchens-100/empty.txt \
            data.test.target_ann_file=data/epic-kitchens-100/filelist_${tgt}_test_open.txt

done <<< $(python slurm/find_all.py -j ${jid})
