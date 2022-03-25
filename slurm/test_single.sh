#!/bin/bash

# 예시) . slurm/osbp_test_single.sh 15588

conda activate open-mmlab

jid=$1
openness=${2:-'closed'}

if [ -z $jid ]; then
    echo 'No jid is passed'
else
    echo "Testing the best model of jid:$jid on $openness set"
    output=$(python slurm/find_best.py -j ${jid})

    if [ $? -eq 0 ]; then
        read checkpoint src tgt <<< $output

        echo "Checkpoint: $checkpoint"
        echo "source: $src, target: $tgt"
        outdir="work_dirs/test_output/${jid}_${src}_${tgt}_${openness}.json"
        echo "Outdir: $outdir"

        python tools/test.py configs/recognition/hello/osbp_tsm_same_batch.py \
            $checkpoint \
            --out $outdir \
            --eval top_k_accuracy mean_class_accuracy \
            --cfg-options \
                data.test.ann_file=data/epic-kitchens-100/filelist_${tgt}_test_${openness}.txt
    else
        echo "invalid jid $jid"
    fi
fi
