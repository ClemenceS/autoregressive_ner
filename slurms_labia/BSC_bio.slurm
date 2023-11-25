#!/bin/bash

#SBATCH --job-name=BSC_bio
#SBATCH --output=BSC_bio.out
#SBATCH --error=BSC_bio.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "PlanTL-GOB-ES/bsc-bio-es" --dataset_name "mnaguib/WikiNER/es" 
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "PlanTL-GOB-ES/bsc-bio-es" --dataset_name "conll2002/es" 
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "PlanTL-GOB-ES/bsc-bio-es" --dataset_name "/mnt/beegfs/home/naguib/e3c_es" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "PlanTL-GOB-ES/bsc-bio-es" --dataset_name "/mnt/beegfs/home/naguib/cwlc" -d
