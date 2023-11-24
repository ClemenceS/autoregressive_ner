#!/bin/bash

#SBATCH --job-name=ncbi
#SBATCH --output=ncbi.out
#SBATCH --error=ncbi.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "bert-large-cased" --dataset_name "/mnt/beegfs/home/naguib/NCBI" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "roberta-large" --dataset_name "/mnt/beegfs/home/naguib/NCBI" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "medicalai/ClinicalBERT" --dataset_name "/mnt/beegfs/home/naguib/NCBI" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "Charangan/MedBERT" --dataset_name "/mnt/beegfs/home/naguib/NCBI" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "xlm-roberta-large" --dataset_name "/mnt/beegfs/home/naguib/NCBI" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "emilyalsentzer/Bio_ClinicalBERT" --dataset_name "/mnt/beegfs/home/naguib/NCBI" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "bert-base-multilingual-cased" --dataset_name "/mnt/beegfs/home/naguib/NCBI" -d
