#!/bin/bash

#SBATCH --job-name=camembert_full
#SBATCH --output=camembert_full.out
#SBATCH --error=camembert_full.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "camembert/camembert-large" --dataset_name "/mnt/beegfs/home/naguib/emea" -d -s "-1"
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "camembert/camembert-large" --dataset_name "/mnt/beegfs/home/naguib/medline" -d -s "-1"
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "camembert/camembert-large" --dataset_name "mnaguib/WikiNER/fr"  -s "-1"
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "camembert/camembert-large" --dataset_name "/mnt/beegfs/home/naguib/e3c_fr" -d -s "-1"
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "camembert/camembert-large" --dataset_name "/mnt/beegfs/home/naguib/QFP" -d -s "-1"
