#!/bin/bash

# bash slurm/vanilla/vanilla_test_single.sh {16279..16284}

N=$SLURM_GPUS_ON_NODE

if [ -z $1 ]; then
    echo 'No jid is passed'
else
    export OMP_NUM_THREADS=${N}
    export MKL_NUM_THREADS=${N}
    for jid in "$@"; do
        output=$(python slurm/utils/print_best_scores.py -j ${jid} -o)
        if [ $? -eq 0 ]; then
            read dataset backbone model domain task top1 top5 mca jid ckpt config <<< $output
            if [ $model == "vanilla" ]; then
                echo -e "\n====================================================================\n"
                echo "Testing the best model of [jid $jid]"
                echo
                echo -e "Dataset:\t$dataset"
                echo -e "Backbone:\t$backbone"
                echo -e "Model:\t\t$model"
                echo -e "Domain:\t\t$domain"
                echo -e "Task:\t\t$task"
                echo -e "Test MCA:\t$mca"
                echo -e "Test ACC:\t$top1"
                echo -e "Checkpoint:\t$ckpt"
                echo -e "Config:\t\t$config\n"

                for domain_tested in P02 P04 P22; do
                    for openness in open closed; do
                        outfile="work_dirs/test_output/${dataset}/${backbone}/${model}/${domain}/${task}/tested_on_${domain_tested}/${openness}/${jid}.json"
                        annfile="data/_filelists/ek100/filelist_${domain_tested}_test_${openness}.txt"

                        python tools/test.py $config \
                            $ckpt \
                            --out $outfile \
                            --eval top_k_accuracy mean_class_accuracy confusion_matrix \
                            --average-clips score \
                            --cfg-options \
                                data.test.ann_file=$annfile
                        if [ $? -ne 0 ]; then
                            break
                        fi
                    done
                done
            else
                echo "This model is not a vanilla"
            fi
        else
            echo "Invalid jid $jid or something's wrong with the training result."
        fi
    echo -e "\n====================================================================\n"
    done
fi
