#!/bin/bash

#SBATCH -J sskmeans
#SBATCH --gres=gpu:2
#SBATCH -t 8:00:00
#SBATCH -p batch
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=10G
#SBATCH -x agi[1-2],augi[1-2],vll1
#SBATCH --array 0-11
#SBATCH -o slurm/logs/slurm-%A_%a-%x.out


jids=(
    # ucf2hmdb
    11850_0 12604_9 15201_2 13781_7  17442_16 17444_9
    # hmdb2ucf
    12811_21 13771_1 15103_1 13780_4  17440_1 17443_0
)
jid="${jids[SLURM_ARRAY_TASK_ID]}"

python slurm/gcd4da/commons/kmeans_copy.py -n 100 -k 22 -j $jid

exit $?
