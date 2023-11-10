#!/bin/bash

#SBATCH --job-name=mistral_wnen
#SBATCH --output=mistral_wnen.out
#SBATCH --error=mistral_wnen.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=lak@a100
#SBATCH -C a100


module purge
module load llm

my_datasets_pref="/gpfswork/rech/lak/utb11pp/data"
dataset="$my_datasets_pref/mnaguib/WikiNER/en"
my_models_pref="/gpfswork/rech/lak/utb11pp/models"
model="$my_models_pref/mistralai/Mistral-7B-v0.1"

python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name $dataset  --n_gpus 2 -d 