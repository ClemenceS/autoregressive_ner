#!/bin/bash

#SBATCH --job-name=BSC_bio_ehr
#SBATCH --output=BSC_bio_ehr.out
#SBATCH --error=BSC_bio_ehr.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "PlanTL-GOB-ES/bsc-bio-ehr-es" --dataset_name "mnaguib/WikiNER/es" 
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "PlanTL-GOB-ES/bsc-bio-ehr-es" --dataset_name "conll2002/es" 
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "PlanTL-GOB-ES/bsc-bio-ehr-es" --dataset_name "/mnt/beegfs/home/naguib/e3c_es" -d
