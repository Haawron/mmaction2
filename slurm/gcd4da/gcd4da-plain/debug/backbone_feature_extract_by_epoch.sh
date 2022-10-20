#!/bin/bash

set -e

jid=${1:-'10608_0'}
echo $jid

read backbone source target config <<< $(python slurm/utils/commons/patterns.py 'backbone' 'source' 'target' 'config' -j $jid)
workdir=${config%/*}

echo "$source -> $target"
echo "backbone: ${backbone}"
echo "workdir: ${workdir}"
echo

N=$SLURM_GPUS_ON_NODE
dataset_abbrevs=($source $target)
domain_types=('source' 'target')
for ckpt in $(ls $workdir | egrep '^epoch' | sort -n -t _ -k 2); do
    for i in 0 1; do
        dataset=$([ "${dataset_abbrevs[i]}" == 'ucf' ] && echo 'ucf101' || echo 'hmdb51')

        outfile=${workdir}/features/${ckpt%.*}_${domain_types[i]}.pkl  # features/epoch_##_source.pkl
        annfile=data/_filelists/${dataset}/filelist_${dataset_abbrevs[i]}_train_open.txt

        if [[ ! -f $annfile ]]; then
            echo "$annfile does not exist"
            exit 2
        fi

        echo "Outfile:" $outfile
        echo "Annfile:" $annfile
        echo

        OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node=${N} --master_port=$((10000+$RANDOM%20000)) tools/test.py $config --launcher pytorch \
            ${workdir}/${ckpt} \
            --out $outfile \
            --cfg-options \
                data.test.ann_file=$annfile \
                data.test.data_prefix=/local_datasets/${dataset}/rawframes \
                data.videos_per_gpu=32 \
                model.test_cfg.feature_extraction=True

        echo
    done
    echo -e "\n=====================================================\n\n"
done
