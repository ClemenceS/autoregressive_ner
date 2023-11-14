#!/bin/bash

#SBATCH --job-name=biomedlm
#SBATCH --output=biomedlm.out
#SBATCH --error=biomedlm.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=24
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=lak@a100
#SBATCH -C a100


module purge
module load llm

#python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/lmsys/vicuna-7b-v1.5 --dataset_name /gpfswork/rech/lak/utb11pp/data/n2c2  --n_gpus 2 --transformers
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/stanford-crfm/BioMedLM --dataset_name /gpfswork/rech/lak/utb11pp/data/n2c2  --n_gpus 2 --transformers

#python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/stanford-crfm/BioMedLM --dataset_name /mnt/beegfs/home/naguib/emea  --n_gpus 2
#python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/stanford-crfm/BioMedLM --dataset_name /mnt/beegfs/home/naguib/medline  --n_gpus 2
#python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/stanford-crfm/BioMedLM --dataset_name /mnt/beegfs/home/naguib/n2c2  --n_gpus 2
#python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/stanford-crfm/BioMedLM --dataset_name mnaguib/WikiNER/en  --n_gpus 2
#python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/stanford-crfm/BioMedLM --dataset_name mnaguib/WikiNER/fr  --n_gpus 2
#python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/stanford-crfm/BioMedLM --dataset_name mnaguib/WikiNER/es  --n_gpus 2
