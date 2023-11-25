#!/bin/bash

#SBATCH --job-name=xlmr_sampling_n2c2
#SBATCH --output=xlmr_sampling_n2c2.out
#SBATCH --error=xlmr_sampling_n2c2.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1

# python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "xlm-roberta-large" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -s 100 -p 1
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "xlm-roberta-large" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -s 100 -p 2
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "xlm-roberta-large" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -s 100 -p 3

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "xlm-roberta-large" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -s 50 -p 1
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "xlm-roberta-large" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -s 50 -p 2
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "xlm-roberta-large" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -s 50 -p 3

python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "xlm-roberta-large" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -s 25 -p 1
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "xlm-roberta-large" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -s 25 -p 2
python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "xlm-roberta-large" --dataset_name "/mnt/beegfs/home/naguib/n2c2" -s 25 -p 3
