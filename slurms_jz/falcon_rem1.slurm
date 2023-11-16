#!/bin/bash

#SBATCH --job-name=falcon_rem1
#SBATCH --output=falcon_rem1.out
#SBATCH --error=falcon_rem1.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=lak@a100
#SBATCH -C a100


module purge
module load llm

e3c_en=/gpfswork/rech/lak/utb11pp/data/e3c_en
e3c_fr=/gpfswork/rech/lak/utb11pp/data/e3c_fr
e3c_es=/gpfswork/rech/lak/utb11pp/data/e3c_es
emea=/gpfswork/rech/lak/utb11pp/data/emea
falcon=/gpfsdswork/dataset/HuggingFace_Models/tiiuae/falcon-40b

python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $falcon --dataset_name $e3c_en --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $falcon --dataset_name $e3c_fr --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $falcon --dataset_name $e3c_es --n_gpus 2 -d
# python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $falcon --dataset_name $emea --n_gpus 2 -d
