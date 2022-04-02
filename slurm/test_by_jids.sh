#!/bin/bash

# 예시) . slurm/test_single.sh 15588

conda activate open-mmlab

jid=$1
config=${2:-'configs/recognition/hello/osbp/osbp_tsm_same_batch.py'}
openness=${3:-'closed'}
dataset=${dataset:-'ek100'}

model=${config##*/}
model=${model%%_*}

if [ -z $jid ]; then
    echo 'No jid is passed'
else
    echo "Testing the best model of jid:$jid on [$openness] set"
    output=$(python slurm/find_best.py -j ${jid})

    if [ $? -eq 0 ]; then
        read checkpoint src tgt <<< $output

        echo "Checkpoint: $checkpoint"
        echo "Model: $model"
        echo "source: $src, target: $tgt"
        outdir="work_dirs/test_output/${dataset}/${model}/${openness}/${src}_${tgt}/${jid}"
        outfile="${outdir}/${jid}.json"
        annfile="data.test.ann_file=data/epic-kitchens-100/filelist_${tgt}_test_${openness}.txt"
        echo "Outfile: $outfile"

        python tools/test.py $config \
            $checkpoint \
            --out $outfile \
            --eval top_k_accuracy mean_class_accuracy \
            --cfg-options $annfile \

    else
        echo "Invalid jid $jid or something's wrong with the training result."
    fi
fi

echo -e "\n\n\n"
