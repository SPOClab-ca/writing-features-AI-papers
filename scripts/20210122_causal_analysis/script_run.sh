#!/bin/bash

#SBATCH --mem=4G
#SBATCH -c 2
#SBATCH -p cpu
#SBATCH --qos=nopreemption
#SBATCH --output=slurm_classification_%j.log
#SBATCH --open-mode=append


# Regression:
#python -u full_linear_model.py --by_venue --venue_name_id $1 --target annual_citations
#python -u full_linear_model.py --by_category --cat_id $1 --target annual_citations

# Classification:
python -u full_linear_model.py --by_venue --venue_name_id $1 --target venue_is_top --remove_redundant_features
#python -u full_linear_model.py --by_category --cat_id $1 --target venue_is_top --remove_redundant_features
