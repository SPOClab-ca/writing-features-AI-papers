#!/bin/bash

#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.log
#SBATCH --open-mode=append


. /etc/profile.d/lmod.sh 
module use $HOME/env_scripts 
module load features  # An environment I created

#python -u main.py --chunk $1 --export feature_v2/lex_chunk_${1}.csv --checkpoint ${ckpt_dir}/checkpoint.pkl --slurm_jid $SLURM_JOB_ID --include_lex

# Use the 1109_gector extractor. Don't include grammar here.
#python -u main.py --chunk $1 --export feature_v2/grammar_chunk_${1}.csv --checkpoint ${ckpt_dir}/checkpoint.pkl --slurm_jid $SLURM_JOB_ID  --include_grammar

# Use the 1017_rst extractor. Don't include RST here
#python -u main.py --chunk $1 --export feature_v2/rst_chunk_${1}.csv --checkpoint ${ckpt_dir}/checkpoint.pkl --slurm_jid $SLURM_JOB_ID  --include_rst  

python -u main.py --chunk $1 --export feature_v2/surprisal_chunk_${1}.csv --checkpoint ${ckpt_dir}/checkpoint.pkl --slurm_jid $SLURM_JOB_ID --include_surprisal

#python -u main.py --chunk $1 --export feature_v2/syntax_chunk_${1}.csv --checkpoint ${ckpt_dir}/checkpoint.pkl --slurm_jid $SLURM_JOB_ID --include_syntax

# Comprehensive: see if everything runs well together
#python -u main.py --chunk $1 --export feature_v2/all_chunk_${1}.csv --checkpoint ${ckpt_dir}/checkpoint.pkl --slurm_jid $SLURM_JOB_ID --include_lex --include_surprisal --include_syntax
