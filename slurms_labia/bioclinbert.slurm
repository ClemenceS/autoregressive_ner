#!/bin/bash

#SBATCH --job-name=bioclinbert
#SBATCH --output=bioclinbert.out
#SBATCH --error=bioclinbert.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "emilyalsentzer/Bio_ClinicalBERT" --dataset_name "conll2003" 
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "emilyalsentzer/Bio_ClinicalBERT" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "emilyalsentzer/Bio_ClinicalBERT" --dataset_name "mnaguib/WikiNER/en" 
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "emilyalsentzer/Bio_ClinicalBERT" --dataset_name "/mnt/beegfs/home/naguib/e3c_en" -d
