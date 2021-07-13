#!/bin/bash

#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -p cpu
#SBATCH --output=slurm_%j.log

#python -u count_categories.py
#python -u categorize.py
python -u count_journals.py
