#!/bin/bash

# Extract features for each best model for each task

N=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

for source in P02 P04 P22; do
    for target in P02 P04 P22; do

        if [[ $source = $target ]]; then
            continue
        fi
        echo -e "$source -> $target\n"
        output=$(python slurm/utils/print_best_scores.py -o -m gcd4da --phase phase0 --ablation one_way --task ${source}_${target})
        read _dataset _backbone _model _phase _ablation _task _acc _mca _unk _jid ckpt config <<< $output

        echo
        echo "ckpt:   ${ckpt}"
        echo "config: ${config}"
        echo

        domains=($source $target $target $target)
        splits=(train train valid test)
        for i in {0..3}; do
            outfile=${ckpt%/*}/features/${domains[i]}_${splits[i]}_open.pkl
            annfile=data/_filelists/ek100/filelist_${domains[i]}_${splits[i]}_open.txt

            echo "Outfile:" $outfile
            echo "Annfile:" $annfile
            echo 

            OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/test.py $config --launcher pytorch \
                $ckpt \
                --out $outfile \
                --cfg-options \
                    data.test.ann_file=$annfile \
                    data.videos_per_gpu=8 \
                    model.test_cfg.average_clips='feature'

            echo -e "\n=====================================================\n\n"
        done
    done
done
