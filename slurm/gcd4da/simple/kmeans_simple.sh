#!/bin/bash

model=${1:-'gcd4da_phase0'}
sources=(P02 P02)
targets=(P04 P22)

for i in {0..1}; do
    source="${sources[i]}"
    target="${targets[i]}"

    echo "Task ${source}_${target}"

    output=$(python slurm/print_best_scores.py -m ${model} -o --task ${source}_${target} -ig)
    read _dataset _backbone _model _task _acc _mca _unk _jid ckpt _config <<< $output

    python slurm/gcd4da/kmeans.py -r ${ckpt%/*} -c

    echo -e "\n\n"
done


