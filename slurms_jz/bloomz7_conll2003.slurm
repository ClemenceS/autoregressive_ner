#!/bin/bash

#SBATCH --job-name=bloomz7_conll2003
#SBATCH --output=bloomz7_conll2003.out
#SBATCH --error=bloomz7_conll2003.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=lak@a100
#SBATCH -C a100


module purge
module load llm

python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfsdswork/dataset/HuggingFace_Models/bigscience/bloomz-7b1 --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2003  --n_gpus 2 -d