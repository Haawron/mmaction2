#!/bin/bash

set -e

ckpts=(
    'data/weights/svt/releases/download/v1.0/SVT_mmaction.pth'
    'data/weights/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth.1'
    'data/weights/timesformer/timesformer_8x32_224_howto100m_mmaction.pyth'
    'data/weights/timesformer/timesformer_8_224_ssv2_mmaction.pyth'
    'data/weights/vit/vit_base_patch16_224.pth'
)

names=(
    'k400-SVT'
    'k400'
    'h100m'
    'ssv2'
    'in1k'
)

p_target_dir='/data/hyogun/repos/haawron_mmaction2/data/features'

declare -A tasks=(
  ['hello']="ucf hmdb"
  ['ek100']="P02 P04 P22"
  ['simnreal']="k400 babel"
)

declare -A configs=(
    ['ucf']='configs/recognition/hello/vanilla/vanilla_svt_ucf101_open.py'
    ['hmdb']='configs/recognition/hello/vanilla/vanilla_svt_hmdb51_open.py'
    ['P02']='configs/recognition/hello/vanilla/vanilla_svt_ek100_open.py'
    ['P04']='configs/recognition/hello/vanilla/vanilla_svt_ek100_open.py'
    ['P22']='configs/recognition/hello/vanilla/vanilla_svt_ek100_open.py'
    ['k400']='configs/recognition/hello/vanilla/vanilla_timesformer_k400_open.py'
    ['babel']='configs/recognition/hello/vanilla/vanilla_timesformer_babel_open.py'
)

declare -A annfiles=(
    ['ucf']='data/_filelists/ucf101/filelist_ucf_train_open.txt'
    ['hmdb']='data/_filelists/hmdb51/filelist_hmdb_train_open.txt'
    ['P02']='data/_filelists/ek100/processed/filelist_P02_train_open_all.txt'
    ['P04']='data/_filelists/ek100/processed/filelist_P04_train_open_all.txt'
    ['P22']='data/_filelists/ek100/processed/filelist_P22_train_open_all.txt'
    ['k400']='data/_filelists/k400/processed/filelist_k400_train_open.txt'
    ['babel']='data/_filelists/babel/processed/filelist_babel_train_open.txt'
)

declare -A dataset_names=(
    ['ucf']='ucf101'
    ['hmdb']='hmdb51'
    ['P02']='epic-kitchens-100'
    ['P04']='epic-kitchens-100'
    ['P22']='epic-kitchens-100'
    ['k400']='kinetics400'
    ['babel']='babel'
)

declare -A dataset_prefixes=(
    ['ucf']='/local_datasets/ucf101/rawframes'
    ['hmdb']='/local_datasets/hmdb51/rawframes'
    ['P02']='/local_datasets/epic-kitchens-100/EPIC-KITCHENS'
    ['P04']='/local_datasets/epic-kitchens-100/EPIC-KITCHENS'
    ['P22']='/local_datasets/epic-kitchens-100/EPIC-KITCHENS'
    ['k400']='/local_datasets/kinetics400/videos'  # /train or /val
    ['babel']='/local_datasets/babel'
)

splits=( 'train' 'test_merged' )

N=$SLURM_GPUS_ON_NODE

for i in $(seq 0 $(( ${#ckpts[@]} - 1 ))); do  # for each ckpt
    ckpt="${ckpts[i]}"
    name="${names[i]}"
    echo -e "$name\t$ckpt"
    for task in "${!tasks[@]}"; do  # for each task
        echo -e "\t$task"
        datasets="${tasks[$task]}"
        for dataset in $datasets; do  # for each subtask=dataset (probed on)
            dataset_name="${dataset_names[$dataset]}"
            [ ! -d "/local_datasets/${dataset_name}" ] && { continue; }
            echo -e "\t\t$dataset"
            config="${configs[$dataset]}"
            [ ! -f "$config" ] && { echo "$config does not exist"; exit; }
            for split in "${splits[@]}"; do  # for each split
                annfile="${annfiles[$dataset]/train/$split}"
                echo -e "\t\t\t$config\n\t\t\t$annfile"
                [ ! -f "$annfile" ] && { echo "$annfile does not exist"; exit; }
                outfile="${p_target_dir}/${name}/${task}/${dataset}/${split}.pkl"
                if [ "$dataset" == 'k400' ]; then
                    split_=$([ \( "$split" == 'test_merged' \) -o \( "$split" == 'test' \) ] && echo 'val' || echo 'train' )
                    dataset_prefix="${dataset_prefixes[$dataset]}/${split_}"
                else
                    dataset_prefix="${dataset_prefixes[$dataset]}"
                fi
                [ ! -d "$dataset_prefix" ] && { echo "wrong data_prefix: $dataset_prefix"; continue; }
                echo -e "\t\t\t\t$outfile"
                [ -f "$outfile" ] && { echo -e "\t\t\t\t\tpassed, already exists"; continue; }
                OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/test.py \
                    "$config" --launcher pytorch "$ckpt" \
                    --out "$outfile" \
                    --cfg-options \
                        model.test_cfg.feature_extraction=True \
                        data.test.ann_file="$annfile" \
                        data.test.data_prefix="$dataset_prefix" \
                        data.videos_per_gpu=16 model.backbone.pretrained="$ckpt"
            done
        done
        echo
    done
done
