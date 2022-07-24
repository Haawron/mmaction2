#!/bin/bash

for source in P02 P04 P22; do
    for target in P02 P04 P22; do

        if [[ $source = $target ]]; then
            continue
        fi

        echo Task $source_$target

        output=$(python slurm/print_best_scores.py -m gcd4da_phase0 -o --task ${source}_${target})
        read _dataset _backbone _model _task _acc _mca _unk _jid ckpt _config <<< $output

        python slurm/gcd4da/kmeans.py -r ${ckpt%/*} -c

        echo -e "\n\n"
    done
done
