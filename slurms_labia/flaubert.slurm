#!/bin/bash

#SBATCH --job-name=flaubert
#SBATCH --output=flaubert.out
#SBATCH --error=flaubert.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "flaubert/flaubert_large_cased" --dataset_name "/mnt/beegfs/home/naguib/emea" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "flaubert/flaubert_large_cased" --dataset_name "/mnt/beegfs/home/naguib/medline" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "flaubert/flaubert_large_cased" --dataset_name "mnaguib/WikiNER/fr" 
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "flaubert/flaubert_large_cased" --dataset_name "/mnt/beegfs/home/naguib/e3c_fr" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "flaubert/flaubert_large_cased" --dataset_name "/mnt/beegfs/home/naguib/QFP" -d
