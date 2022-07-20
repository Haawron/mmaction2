#!/bin/bash

# . slurm/osvm/extract_features.sh {16383..16388}  # vanilla
# . slurm/osvm/extract_features.sh {16748..16753}  # dann

if [ -z $1 ]; then
    echo 'No jid is passed'
else
    N=$SLURM_GPUS_ON_NODE
    for jid in "$@"; do
        output=$(python slurm/utils/print_best_scores.py -j ${jid} -o)
        if [ $? -eq 0 ]; then
            let "num_elements = $(echo $output | tr -cd ' \t' | wc -c)"
            if [[ $num_elements == 10 ]]; then  # vanilla
                read dataset backbone model domain task acc mca _unk jid ckpt config <<< $output
                if [[ $task == 'source-only' ]]; then
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
                            _domain=$([ "$split" = 'train' ] && echo "$domain" || echo "$target")
                            outfile="${ckpt%/*}/features/${_domain}_${split}_open.pkl"
                            annfile="data/epic-kitchens-100/filelist_${_domain}_${split}_open.txt"
                            OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/test.py $config --launcher pytorch \
                                $ckpt \
                                --out $outfile \
                                --eval top_k_accuracy mean_class_accuracy confusion_matrix \
                                --average-clips score \
                                --cfg-options \
                                    data.test.ann_file=$annfile \
                                    data.videos_per_gpu=20
                            if [ $? -ne 0 ]; then
                                break
                            fi
                        done
                    done
                else
                    echo "Given model is vanilla but the task is not source-only"
                fi
            else  # not vanilla
                read dataset backbone model task acc mca _unk jid ckpt config <<< $output
                echo -e "\n====================================================================\n"
                echo "Testing the best model of [jid $jid]"
                echo 
                echo -e "Dataset:\t$dataset"
                echo -e "Backbone:\t$backbone"
                echo -e "Model:\t\t$model"
                echo -e "Task:\t\t$task"
                echo -e "Test ACC:\t$acc"
                echo -e "Test Mean-Class ACC:\t$mca"
                echo -e "Checkpoint:\t$ckpt"
                echo -e "Config:\t\t$config\n"

                source=${task%_*}
                target=${task#*_}
                echo -e "Task: ${source} --> ${target}\n"
                
                for split in 'train' 'valid' 'test'; do
                    domain=$([ "$split" = 'train' ] && echo "$source" || echo "$target")
                    outfile="${ckpt%/*}/features/${domain}_${split}_open.pkl"
                    annfile="data/epic-kitchens-100/filelist_${domain}_${split}_open.txt"

                    echo "Outfile:" $outfile
                    echo "Annfile:" $annfile
                    echo 

                    OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/test.py $config --launcher pytorch \
                        $ckpt \
                        --out $outfile \
                        --eval top_k_accuracy mean_class_accuracy confusion_matrix \
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
