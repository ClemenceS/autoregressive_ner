#!/bin/bash

#SBATCH --job-name=mistral_wnes
#SBATCH --output=mistral_wnes.out
#SBATCH --error=mistral_wnes.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=lak@a100
#SBATCH -C a100


module purge
module load llm

python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/mistralai/Mistral-7B-v0.1 --dataset_name /gpfswork/rech/lak/utb11pp/data/mnaguib/WikiNER/es  --n_gpus 2 -d