#!/bin/bash

# . slurm/osvm/extract_features.sh {16383..16388}  # vanilla
# . slurm/osvm/extract_features.sh {16748..16753}  # dann

if [ -z $1 ]; then
    echo 'No jid is passed'
else
    OMP_NUM_THREADS=2
    MKL_NUM_THREADS=2
    for jid in "$@"; do
        output=$(python slurm/print_best_scores.py -j ${jid} -o)
        if [ $? -eq 0 ]; then
            let "num_elements = $(echo $output | tr -cd ' \t' | wc -c) + 1"
            if [[ $num_elements == 10 ]]; then  # vanilla
                read dataset backbone model domain task acc mca jid ckpt config <<< $output
                if [[ $task == source-only ]]; then
                    echo -e "\n====================================================================\n"
                    echo "Info of loaded model: [jid $jid]"
                    echo 
                    echo -e "Dataset:\t$dataset"
                    echo -e "Backbone:\t$backbone"
                    echo -e "Model:\t\t$model"
                    echo -e "Domain:\t\t$domain"
                    echo -e "Task:\t\t$task"
                    echo -e "Test ACC:\t$acc"
                    echo -e "Test Mean-Class ACC:\t$mca"
                    echo -e "Checkpoint:\t$ckpt"
                    echo -e "Config:\t\t$config\n"
                    for target in P02 P04 P22; do
                        for split in 'train' 'valid' 'test'; do
                            openness=$([ "$split" == 'train' ] && echo "closed" || echo "open")
                            outfile="work_dirs/train_output/${dataset}/${backbone}/osvm/${model}/${domain}/${target}/${jid}_${split}.pkl"
                            annfile="data/epic-kitchens-100/filelist_${target}_${split}_${openness}.txt"
                            python tools/test.py $config \
                                $ckpt \
                                --out $outfile \
                                --eval top_k_accuracy mean_class_accuracy confusion_matrix recall_unknown \
                                --average-clips score \
                                --cfg-options \
                                    data.test.ann_file=$annfile
                            if [ $? -ne 0 ]; then
                                break
                            fi
                        done
                    done
                else
                    echo "Given model is vanilla but the task is not source-only"
                fi
            else  # not vanilla
                read dataset backbone model task acc mca unk jid ckpt config <<< $output
                echo -e "\n====================================================================\n"
                echo "Testing the best model of [jid $jid]"
                echo 
                echo -e "Dataset:\t$dataset"
                echo -e "Backbone:\t$backbone"
                echo -e "Model:\t\t$model"
                echo -e "Task:\t\t$task"
                echo -e "Test ACC:\t$acc"
                echo -e "Test Mean-Class ACC:\t$mca"
                echo -e "Test UNK:\t$unk"
                echo -e "Checkpoint:\t$ckpt"
                echo -e "Config:\t\t$config\n"

                source=${task%_*}
                target=${task#*_}
                echo -e "Task: ${source} --> ${target}\n"
                
                for split in "train" "valid" "test"; do
                    domain=$([ "$split" == 'train' ] && echo $source || echo $target)
                    openness=$([ "$split" == 'train' ] && echo "closed" || echo "open")
                    outfile="work_dirs/train_output/${dataset}/${backbone}/osvm/${model}/${task}/${jid}_${split}.pkl"
                    annfile="data/epic-kitchens-100/filelist_${domain}_${split}_${openness}.txt"

                    python tools/test.py $config \
                        $ckpt \
                        --out $outfile \
                        --eval top_k_accuracy mean_class_accuracy confusion_matrix recall_unknown \
                        --average-clips score \
                        --cfg-options \
                            data.test.ann_file=$annfile \
                            data.videos_per_gpu=20
                    if [ $? -ne 0 ]; then
                        break
                    fi
                done
            fi
        fi
    done
fi
