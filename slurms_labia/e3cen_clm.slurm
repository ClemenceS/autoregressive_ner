#!/bin/bash

#SBATCH --job-name=e3cen_clm
#SBATCH --output=e3cen_clm.out
#SBATCH --error=e3cen_clm.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2

python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py -n 2 --model_name "tiiuae/falcon-7b" --dataset_name "/mnt/beegfs/home/naguib/e3c_en" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py -n 2 --model_name "lmsys/vicuna-7b-v1.5" --dataset_name "/mnt/beegfs/home/naguib/e3c_en" -d
python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py -n 2 --model_name "mistralai/Mistral-7B-v0.1" --dataset_name "/mnt/beegfs/home/naguib/e3c_en" -d
