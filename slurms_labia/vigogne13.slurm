#!/bin/bash

#SBATCH --job-name=vigogne13
#SBATCH --output=vigogne13.out
#SBATCH --error=vigogne13.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=45:00:00
#SBATCH --gres=gpu:2

model=bofenghuang/vigogne-2-13b-instruct

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
