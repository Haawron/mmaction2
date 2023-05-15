# CDAR Implementation Based on mmaction2
> This project is supported by [NCSoft](https://kr.ncsoft.com/kr/index.do)

# Setup
## Environment
```bash
conda env create -f open-mmlab.conda.env.yaml  # Comman AI packages and mmcv-full an AI-training engine mmaction will be running on
conda activate open-mmlab
pip install .  # installs mmaction
```

## Dataset
- Download the datasets
    - [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
    - [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
    - [Epic-kitchens-100](https://epic-kitchens.github.io/2023)
- [Optional] Extract Rawframes. Refer to [this](https://mmaction2.readthedocs.io/en/latest/datasetzoo.html#step-4-extract-rgb-frames)
- Put the datasets under any preferred location
- Link the datasets
    ```bash
    ln -s /PATH/TO/UCF101 data/ucf101
    ln -s /PATH/TO/HMDB51 data/hmdb51
    ln -s /PATH/TO/EK100 data/epic-kitchens-100
    ```
- Edit dataset paths in config files
    ```bash
    # from
    data_prefix_source = '/local_datasets/hmdb51/rawframes'
    data_prefix_target = '/local_datasets/ucf101/rawframes'

    # to
    data_prefix_source = 'data/hmdb51/rawframes'
    data_prefix_target = 'data/ucf101/rawframes'
    ```

## [Optional] Extract Backgrounds for Temporal Median Filter
```bash
# NOTE: please edit some important path in this python file. didn't make use of argparse

# for ucf, hmdb, ek100, babel
python slurm/utils/extract_median_by_rawframes.py

# for kinetics
python slurm/utils/extract_median_by_videos.py  
```

---

# Train
```bash
workdir=/PATH/TO/WORKDIR  # ex: work_dirs/train_output/cdar/tsm/dann/randaug/k400_pretrained/0/$(date +'%Y%m%d-%H%M%S')
config=/PATH/TO/CONFIG  # ex: configs/recognition/vosuda/gcd4da/median/gcd4da_median_phase0_svt_hmdb2ucf.py

lr=1e-3

N=$CUDA_VISIBLE_DEVICES
calibed_lr="$(perl -le "print $lr * $N / 4")"  # default lr in the config is 4-GPU based
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/train.py "$config" \
    --launcher pytorch \
    --work-dir "$workdir" \
    --cfg-options \
        optimizer.lr="$calibed_lr" \
    --validate --test-last --test-best
``` 
- This will generate
    - The rendered config file used for training this model, `CONFIGNAME.py`
    - Log files `.log`, `.json`
    - `pth` files per specified epochs
        - You can change the checkpointing period by adding argument `checkpoint_config.interval=EPOCHS` right after `--cfg-options`
    - Feature files, `best_pred.pkl`
        - The default will be their logits. How to extract backbone feature is described in Test section.
    
    under specified `work_dir`

# Test

## Just Evaluate
```bash
config=/PATH/TO/CONFIG
ckpt=/PATH/TO/CHECKPOINT

N=$CUDA_VISIBLE_DEVICES
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/test.py \
	"$config" --launcher pytorch "$ckpt" \
	--cfg-options \
		data.test_dataloader.videos_per_gpu=40 \
		data.test_dataloader.workers_per_gpu=$(( SLURM_CPUS_ON_NODE / SLURM_GPUS_ON_NODE )) \
	--eval 'top_k_accuracy' 'mean_class_accuracy' 'confusion_matrix' 'kmeans' 'gcd_v2'
``` 


## Extract Backbone Features
```bash
config=/PATH/TO/CONFIG
ckpt=/PATH/TO/CHECKPOINT
outfile=/PATH/TO/OUTFILE  # ex: outfile=$(dirname $ckpt)/backbone.pkl
annfile=/PATH/TO/ANNFILE # ex: 'data/_filelists/babel/processed/filelist_babel_test_open.txt', data you want to extract backbone features of
dataset_prefix=/PATH/TO/DATASET  # ex: data/babel

N=$SLURM_GPUS_ON_NODE
OMP_NUM_THREADS=${N} MKL_NUM_THREADS=${N} torchrun --nproc_per_node="${N}" --master_port=$((10000+RANDOM%20000)) tools/test.py \
	"$config" --launcher pytorch "$ckpt" \
	--out "$outfile" \
	--cfg-options \
		model.test_cfg.feature_extraction=True \
		data.test.ann_file="$annfile" \
		data.test.data_prefix="$dataset_prefix" \
		data.test_dataloader.videos_per_gpu=40 model.backbone.pretrained="$ckpt"
```
- You should add arguments `'data.test.filename_tmpl='img_{:05d}.jpg' data.test.with_offset=True data.test.start_index=1'` to `--cfg-options` when it comes to BABEL.
- If you meet OOM, this would be caused by the excessive number of workers. You can change this by adding `data.workers_per_gpu=4` to `--cfg-options`.
