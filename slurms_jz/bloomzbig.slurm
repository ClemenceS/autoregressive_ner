#!/bin/bash

#SBATCH --job-name=bloomzbig
#SBATCH --output=bloomzbig.out
#SBATCH --error=bloomzbig.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:8
#SBATCH --account=lak@a100
#SBATCH -C a100


module purge
module load llm

model=/gpfsdswork/dataset/HuggingFace_Models/bigscience/bloomz

python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2003  --transformers -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/emea  --transformers -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/medline  --transformers -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/n2c2  --transformers -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/mnaguib/WikiNER/en  --transformers -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/mnaguib/WikiNER/fr  --transformers -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/mnaguib/WikiNER/es  --transformers -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --transformers -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/e3c_en  --transformers -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/e3c_fr  --transformers -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name /gpfswork/rech/lak/utb11pp/data/e3c_es  --transformers -d
