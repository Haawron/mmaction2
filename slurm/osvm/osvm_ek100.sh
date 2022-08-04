#!/bin/bash


num_classes=5


for target in P02 P04 P22; do
    for source in P02 P04 P22; do
        if [[ $source == $target ]]; then
            continue
        fi

        echo -e "$source -> $target \n\n"

        # output=$(python slurm/utils/print_best_scores.py -d ek100 -b tsm -m vanilla -dom $source -t 'source-only' -o)
        output=$(python slurm/utils/print_best_scores.py -d ek100 -b tsm -m dann -t ${source}_${target} -o)
        read _dataset _backbone _model _domain _task _acc _mca _unk _jid ckpt config <<< $output
        feature_dir=${ckpt%/*}/features

        python slurm/osvm/osvm.py \
            -n $num_classes \
            -pf $feature_dir \
            -s $source -t $target
        
        echo -e "============================================\n\n"
    done
done
