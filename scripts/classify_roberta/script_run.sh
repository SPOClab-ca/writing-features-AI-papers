#!/bin/bash

#SBATCH --mem=10G
#SBATCH -c 1
#SBATCH -p rtx6000
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --output=slurm_roberta_%j.log


. /etc/profile.d/lmod.sh
module use $HOME/env_scripts
module load transformers4

python -u main.py --seed $1 --venue $2
