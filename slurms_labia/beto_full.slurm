#!/bin/bash

#SBATCH --job-name=beto_full
#SBATCH --output=beto_full.out
#SBATCH --error=beto_full.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "dccuchile/bert-base-spanish-wwm-uncased" --dataset_name "mnaguib/WikiNER/es"  -s "-1"
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "dccuchile/bert-base-spanish-wwm-uncased" --dataset_name "conll2002/es"  -s "-1"
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "dccuchile/bert-base-spanish-wwm-uncased" --dataset_name "/mnt/beegfs/home/naguib/e3c_es" -d -s "-1"
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "dccuchile/bert-base-spanish-wwm-uncased" --dataset_name "/mnt/beegfs/home/naguib/cwlc" -d -s "-1"
