#!/bin/bash

#SBATCH --job-name=emea
#SBATCH --output=emea.out
#SBATCH --error=emea.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "camembert/camembert-large" --dataset_name "/mnt/beegfs/home/naguib/emea" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "Dr-BERT/DrBERT-4GB" --dataset_name "/mnt/beegfs/home/naguib/emea" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "Dr-BERT/DrBERT-7GB" --dataset_name "/mnt/beegfs/home/naguib/emea" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "almanach/camembert-bio-base" --dataset_name "/mnt/beegfs/home/naguib/emea" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "xlm-roberta-large" --dataset_name "/mnt/beegfs/home/naguib/emea" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "bert-base-multilingual-cased" --dataset_name "/mnt/beegfs/home/naguib/emea" -d
