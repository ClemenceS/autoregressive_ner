#!/bin/bash

#SBATCH --job-name=conll2002_clm
#SBATCH --output=conll2002_clm.out
#SBATCH --error=conll2002_clm.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:2

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "mistralai/Mistral-7B-v0.1" --dataset_name "conll2002/es" 
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "lmsys/vicuna-7b-v1.5" --dataset_name "conll2002/es" 
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "lmsys/vicuna-13b-v1.5" --dataset_name "conll2002/es" 
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "tiiuae/falcon-40b" --dataset_name "conll2002/es"
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "bofenghuang/vigogne-2-7b-instruct" --dataset_name "conll2002/es"

