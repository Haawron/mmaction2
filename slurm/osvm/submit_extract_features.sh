#!/bin/bash

#SBATCH -J extract_features_for_osvm
#SBATCH --gres=gpu:1
#SBATCH -t 4-0
#SBATCH -p vll
#SBATCH -o slurm/logs/slurm-%j-%x.out


echo -e "Extracting features with vanilla models...\n"
. slurm/osvm/extract_features.sh {16904..16909}  # vanilla

echo -e "\n\n\n\n"

echo -e "Extracting features with DANNs...\n"
. slurm/osvm/extract_features.sh {16910..16915}  # dann


echo "done"
exit
