#!/bin/bash

#SBATCH --job-name=cwlc
#SBATCH --output=cwlc.out
#SBATCH --error=cwlc.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "dccuchile/bert-base-spanish-wwm-uncased" --dataset_name "/mnt/beegfs/home/naguib/cwlc" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "dccuchile/tulio-chilean-spanish-bert" --dataset_name "/mnt/beegfs/home/naguib/cwlc" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "dccuchile/patana-chilean-spanish-bert" --dataset_name "/mnt/beegfs/home/naguib/cwlc" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "xlm-roberta-large" --dataset_name "/mnt/beegfs/home/naguib/cwlc" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "IIC/BETO_Galen" --dataset_name "/mnt/beegfs/home/naguib/cwlc" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "bert-base-multilingual-cased" --dataset_name "/mnt/beegfs/home/naguib/cwlc" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "PlanTL-GOB-ES/bsc-bio-ehr-es" --dataset_name "/mnt/beegfs/home/naguib/cwlc" -d
