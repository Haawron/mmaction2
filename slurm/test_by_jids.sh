#!/bin/bash

conda activate open-mmlab

if [ -z $1 ]; then
    echo 'No jid is passed'
else
    OMP_NUM_THREADS=2
    MKL_NUM_THREADS=2
    for jid in "$@"; do
        output=$(python slurm/print_best_scores.py -j ${jid} -o)
        if [ $? -eq 0 ]; then
            read dataset backbone model task acc mca unk jid ckpt config <<< $output
            echo -e "\n====================================================================\n"
            echo "Testing the best model of [jid $jid]"
            echo 
            echo -e "Dataset:\t$dataset"
            echo -e "Backbone:\t$backbone"
            echo -e "Model:\t\t$model"
            echo -e "Task:\t\t$task"
            echo -e "Test ACC:\t$acc"
            echo -e "Test MCA:\t$mca"
            echo -e "Test UNK:\t$unk"
            echo -e "Checkpoint:\t$ckpt"
            echo -e "Config:\t\t$config\n"

            target=${task#*_}
            echo -e "Target: ${target}\n"

            for openness in open closed; do
                outfile="work_dirs/test_output/${dataset}/${backbone}/${model}/${task}/${openness}/${jid}.json"
                annfile="data/epic-kitchens-100/filelist_${target}_test_${openness}.txt"

                python tools/test.py $config \
                    $ckpt \
                    --out $outfile \
                    --eval top_k_accuracy mean_class_accuracy confusion_matrix $([[ $openness = 'open' ]] && echo 'recall_unknown' || echo '') \
                    --average-clips score \
                    --cfg-options \
                        data.test.ann_file=$annfile
                if [ $? -ne 0 ]; then
                    break
                fi
            done
        else
            echo "Invalid jid $jid or something's wrong with the training result."
        fi
    echo -e "\n====================================================================\n"
    done
fi
