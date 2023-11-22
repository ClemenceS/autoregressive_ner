#!/bin/bash

#SBATCH --job-name=bloom7_wn
#SBATCH --output=bloom7_wn.out
#SBATCH --error=bloom7_wn.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=lak@a100
#SBATCH -C a100


module purge
module load llm

model=/gpfsdswork/dataset/HuggingFace_Models/bigscience/bloom-7b1

# python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2003  --n_gpus 2 -d
# python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/emea  --n_gpus 2 -d
# python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/medline  --n_gpus 2 -d
# python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/n2c2  --n_gpus 2 -d
# python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/mnaguib/WikiNER/en  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/mnaguib/WikiNER/fr  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/mnaguib/WikiNER/es  --n_gpus 2 -d
# python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
# python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/e3c_en  --n_gpus 2 -d
# python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/e3c_fr  --n_gpus 2 -d
# python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/e3c_es  --n_gpus 2 -d
