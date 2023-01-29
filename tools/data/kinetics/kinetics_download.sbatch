#!/bin/bash

#SBATCH -J kinetics-download
#SBATCH -t 14-0
#SBATCH -p batch
#SBATCH --cpus-per-task=48
#SBATCH --mem=150G
#SBATCH -o tools/data/kinetics/logs/slurm-%j-%x.out
#SBATCH -w vll4

cd /data/hyogun/repos/haawron_mmaction2/tools/data/kinetics
bash download_videos.sh kinetics400

exit $?
