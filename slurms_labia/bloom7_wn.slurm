#!/bin/bash

#SBATCH --job-name=bloom7
#SBATCH --output=bloom7.out
#SBATCH --error=bloom7.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=45:00:00
#SBATCH --gres=gpu:2

model=bigscience/bloom-7b1

# python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name conll2003  --transformers
# python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /mnt/beegfs/home/naguib/emea  --transformers
# python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /mnt/beegfs/home/naguib/medline  --transformers
# python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /mnt/beegfs/home/naguib/n2c2  --transformers
# python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name mnaguib/WikiNER/en  --transformers
python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name mnaguib/WikiNER/fr  --transformers
python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name mnaguib/WikiNER/es  --transformers
# python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /mnt/beegfs/home/naguib/e3c_en  --transformers
# python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /mnt/beegfs/home/naguib/e3c_fr  --transformers
# python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /mnt/beegfs/home/naguib/e3c_es  --transformers
# python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name conll2002/es  --transformers
