#!/bin/bash

#SBATCH --job-name=vic13_wnen
#SBATCH --output=vic13_wnen.out
#SBATCH --error=vic13_wnen.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=lak@a100
#SBATCH -C a100


module purge
module load llm

python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/lmsys/vicuna-13b-v1.5 --dataset_name /gpfswork/rech/lak/utb11pp/data/mnaguib/WikiNER/en  --n_gpus 2 -d