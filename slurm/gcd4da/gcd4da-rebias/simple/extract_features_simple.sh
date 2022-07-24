#!/bin/bash

# Extract features for each best model for each task

N=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

model=${1:-'gcd4da_phase0'}
sources=(P02 P02)
targets=(P04 P22)

for i in 0 1; do
    source="${sources[i]}"
    target="${targets[i]}"

    output=$(python slurm/print_best_scores.py -m ${model} -o --task ${source}_${target} -ig)
    read _dataset _backbone _model _task _acc _mca _unk _jid ckpt config <<< $output

    domains=($source $target $target $target)
    splits=(train train valid test)
    for i in {0..3}; do
        outfile=${ckpt%/*}/features/${domains[i]}_${splits[i]}_open.pkl
        annfile=data/epic-kitchens-100/filelist_${domains[i]}_${splits[i]}_open.txt

        echo $outfile
        echo $annfile

        OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/test.py $config --launcher pytorch \
            $ckpt \
            --out $outfile \
            --cfg-options \
                data.test.ann_file=$annfile \
                model.test_cfg.average_clips='feature'

        echo -e "\n\n"
    done
done
