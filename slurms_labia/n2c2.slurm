#!/bin/bash

#SBATCH --job-name=n2c2
#SBATCH --output=n2c2.out
#SBATCH --error=n2c2.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "bert-large-cased" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "roberta-large" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "medicalai/ClinicalBERT" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "Charangan/MedBERT" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "xlm-roberta-large" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -d
