#!/bin/bash

#SBATCH --job-name=mistral
#SBATCH --output=mistral.out
#SBATCH --error=mistral.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=45:00:00
#SBATCH --gres=gpu:2

model=mistralai/Mistral-7B-v0.1

python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name conll2003  --n_gpus 2
python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /mnt/beegfs/home/naguib/emea  --n_gpus 2
python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /mnt/beegfs/home/naguib/medline  --n_gpus 2
python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /mnt/beegfs/home/naguib/n2c2  --n_gpus 2
python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name mnaguib/WikiNER/en  --n_gpus 2
python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name mnaguib/WikiNER/fr  --n_gpus 2
python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name mnaguib/WikiNER/es  --n_gpus 2
python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /mnt/beegfs/home/naguib/e3c_en  --n_gpus 2
python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /mnt/beegfs/home/naguib/e3c_fr  --n_gpus 2
python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /mnt/beegfs/home/naguib/e3c_es  --n_gpus 2
