#!/bin/bash

#SBATCH --job-name=roberta_full
#SBATCH --output=roberta_full.out
#SBATCH --error=roberta_full.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "roberta-large" --dataset_name "conll2003"  -s "-1"
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "roberta-large" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -d -s "-1"
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "roberta-large" --dataset_name "mnaguib/WikiNER/en"  -s "-1"
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "roberta-large" --dataset_name "/mnt/beegfs/home/naguib/e3c_en" -d -s "-1"
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "roberta-large" --dataset_name "/mnt/beegfs/home/naguib/NCBI" -d -s "-1"
