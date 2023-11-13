#!/bin/bash

#SBATCH --job-name=wnfr
#SBATCH --output=wnfr.out
#SBATCH --error=wnfr.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "camembert/camembert-large" --dataset_name "WikiNER/fr" 
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "Dr-BERT/DrBERT-4GB" --dataset_name "WikiNER/fr" 
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "Dr-BERT/DrBERT-7GB" --dataset_name "WikiNER/fr" 
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "xlm-roberta-large" --dataset_name "WikiNER/fr" 
