#!/bin/bash

# Extract features for each best model for each task

# bash slurm/gcd4da/commons/extract_features_ucf-hmdb.sh work_dirs/train_output/ucf2hmdb/svt/gcd4da/vanilla/phase0/one_way/4770__gcd4da-phase0-svt_ucf2hmdb-from-vanilla/4/20220731-142832 ucf hmdb

N=$SLURM_GPUS_ON_NODE

workdir=$1
source=$2
target=$3

sources=(ucf hmdb)
targets=(hmdb ucf)

for i in 0 1; do

    if [[ -z $workdir ]]; then

        source="${sources[i]}"
        target="${targets[i]}"

        output=$(python slurm/utils/print_best_scores.py -d ${source}2${target} -m gcd4da --debias vanilla-cop --phase phase0 --ablation one_way -o)
        read _dataset _backbone _debias model _phase _ablation _task _acc _mca _unk _jid ckpt config <<< $output

    else

        if [[ $i -eq 1 ]]; then
            break
        fi

        if [[ -z $source || -z $target ]]; then
            echo 'workdir is specified but source or target are null'
            exit 1
        fi

        echo "workdir is given: ${workdir}"
        best=$(find "${workdir}" -maxdepth 1 -type f -iname "best*.pth")
        latest="${workdir}/latest.pth"
        ckpt="${best:-$latest}"
        config=$(find "${workdir}" -maxdepth 1 -type f -iname "*.py")

        model=$(python slurm/utils/commons/patterns.py ${workdir} model)

    fi


    if [[ ! -f $ckpt ]]; then
        echo "$ckpt does not exist"
        exit 2
    fi
    if [[ ! -f $config ]]; then
        echo "$config does not exist"
        exit 2
    fi

    echo "$source -> $target"
    echo
    echo "ckpt:   ${ckpt}"
    echo "config: ${config}"
    echo

    domains=($source $target $target $target)
    splits=(train train "valid" test)
    for i in {0..3}; do
        dataset=$([ "${domains[i]}" == 'ucf' ] && echo 'ucf101' || echo 'hmdb51')
        # split=$([ "${splits[i]}" == 'vaild' ] && echo 'val' || echo "${splits[i]}")  # 이거 왜 안 됨?
        if [[ "${splits[i]}" == 'valid' ]]; then
            split='val'
        else
            split="${splits[i]}"
        fi

        outfile=${ckpt%/*}/features/${domains[i]}_${splits[i]}_open.pkl
        annfile=data/_filelists/${dataset}/filelist_${domains[i]}_${split}_open.txt
        outtype=$([ "$model" == 'gcd4da' -o "$model" == 'cdar' ] && echo "feature" || echo "score")

        if [[ ! -f $annfile ]]; then
            echo "$annfile does not exist"
            exit 2
        fi

        echo "Outfile:" $outfile
        echo "Annfile:" $annfile
        echo "OutType:" $outtype
        echo 

        OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/test.py $config --launcher pytorch \
            $ckpt \
            --out $outfile \
            --cfg-options \
                data.test.ann_file=$annfile \
                data.test.data_prefix=/local_datasets/${dataset}/rawframes \
                data.videos_per_gpu=8 \
                model.test_cfg.average_clips=$outtype

        echo -e "\n=====================================================\n\n"
    done
done
