#!/bin/bash

set -e


jid=${1:-'4770_4'}
echo $jid

read model source target config ckpt <<< $(python slurm/utils/commons/patterns.py 'model' 'source' 'target' 'config' 'ckpt' -j $jid)


echo "$source -> $target"
echo
echo "ckpt:   ${ckpt}"
echo "config: ${config}"
echo

N=$SLURM_GPUS_ON_NODE
domains=($source $target $target $target)
splits=(train train "valid" test)
for i in {0..3}; do
    dataset=$([ "${domains[i]}" == 'ucf' ] && echo 'ucf101' || echo 'hmdb51')
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
