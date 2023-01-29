#!/bin/bash

# bash slurm/osvm/extract_features.sh {16383..16388}  # vanilla
# bash slurm/osvm/extract_features.sh {16748..16753}  # dann

if [ -z $1 ]; then
    echo 'No jid is passed'
else
    N=$SLURM_GPUS_ON_NODE
    for jid in "$@"; do
        output=$(python slurm/utils/print_best_scores.py -j ${jid} -o)
        read dataset backbone model output <<< $output
        if [ $? -eq 0 ]; then
            let "num_elements = $(echo $output | tr -cd ' \t' | wc -c)"
            if [[ $model == 'vanilla' ]]; then  # vanilla
                echo -e "\n====================================================================\n"
                echo "Info of loaded model: [jid $jid]"
                echo
                echo -e "Dataset:\t$dataset"
                echo -e "Backbone:\t$backbone"
                echo -e "Model:\t\t$model"
                if [[ $dataset == 'ek100' ]]; then
                    read domain task top1 top5 mca jid ckpt config <<< $output
                    echo -e "Domain:\t\t$domain"
                    echo -e "Task:\t\t$task"
                    echo -e "Test ACC:\t$top1"
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
                else  # not ek100
                    read task top1 top5 mca jid ckpt config <<< $output
                    domain=${dataset%2*}
                    target=${dataset#*2}
                    echo -e "Domain:\t\t$domain"
                    echo -e "Target:\t\t$target"
                    echo -e "Task:\t\t$task"
                    echo -e "Test ACC:\t$top1"
                    echo -e "Test MCA:\t$mca"
                    echo -e "Checkpoint:\t$ckpt"
                    splits=('train' 'train' 'val' 'test')
                    configs=(
                        'configs/recognition/hello/vanilla/vanilla_timesformer_k400_open.py'
                        'configs/recognition/hello/vanilla/vanilla_timesformer_babel_open.py'
                    )
                    echo
                    for i in $(seq 0 3) ; do
                        split="${splits[i]}"
                        _domain=$([ $i -eq 0 ] && echo "$domain" || echo "$target")
                        config="${configs[(( i < 1 ? i : 1 ))]}"
                        outfile="${ckpt%/*}/features/${_domain}_${split}_open.pkl"
                        echo -e "Config:\t" $config
                        echo -e "Outfile:\t" $outfile
                        echo
                        OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/test.py $config --launcher pytorch \
                            $ckpt \
                            --out $outfile \
                            --cfg-options \
                                model.test_cfg.feature_extraction=True \
                                data.videos_per_gpu=16
                        if [ $? -ne 0 ]; then
                            break
                        fi
                    done
                fi
            else  # not vanilla
                read task top1 mca _unk jid ckpt config <<< $output
                echo -e "\n====================================================================\n"
                echo "Testing the best model of [jid $jid]"
                echo
                echo -e "Dataset:\t$dataset"
                echo -e "Backbone:\t$backbone"
                echo -e "Model:\t\t$model"
                echo -e "Task:\t\t$task"
                echo -e "Test ACC:\t$top1"
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
