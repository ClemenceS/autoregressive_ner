#!/bin/bash

#SBATCH --job-name=cmbio
#SBATCH --output=cmbio.out
#SBATCH --error=cmbio.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "almanach/camembert-bio-base" --dataset_name "mnaguib/WikiNER/fr" 
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "almanach/camembert-bio-base" --dataset_name "/mnt/beegfs/home/naguib/medline" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "almanach/camembert-bio-base" --dataset_name "/mnt/beegfs/home/naguib/emea" -d

