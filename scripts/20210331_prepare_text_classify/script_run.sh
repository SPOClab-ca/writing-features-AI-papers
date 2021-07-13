#!/bin/bash

#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH -p cpu
#SBATCH --qos=nopreemption
#SBATCH --output=slurm_%j.log

. /etc/profile.d/lmod.sh
module use $HOME/env_scripts
module load features

python -u filter_articles.py --include_arxiv
