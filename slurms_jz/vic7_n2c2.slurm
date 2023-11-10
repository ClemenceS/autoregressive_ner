#!/bin/bash

#SBATCH --job-name=vic7_n2c2
#SBATCH --output=vic7_n2c2.out
#SBATCH --error=vic7_n2c2.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=lak@a100
#SBATCH -C a100


module purge
module load llm

python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/lmsys/vicuna-7b-v1.5 --dataset_name /gpfswork/rech/lak/utb11pp/data/n2c2  --n_gpus 2 -d