#!/bin/bash

#SBATCH --job-name=bloombig_medline
#SBATCH --output=bloombig_medline.out
#SBATCH --error=bloombig_medline.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=lak@a100
#SBATCH -C a100


module purge
module load llm

python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfsdswork/dataset/HuggingFace_Models/bigscience/bloom --dataset_name /gpfswork/rech/lak/utb11pp/data/medline  --n_gpus 2 -d