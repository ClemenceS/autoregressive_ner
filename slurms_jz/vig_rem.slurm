#!/bin/bash

#SBATCH --job-name=vig_rem
#SBATCH --output=vig_rem.out
#SBATCH --error=vig_rem.out
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
wn_en=/gpfswork/rech/lak/utb11pp/data/mnaguib/WikiNER/en
wn_es=/gpfswork/rech/lak/utb11pp/data/mnaguib/WikiNER/es
vigogne=/gpfsdswork/dataset/HuggingFace_Models/bofenghuang/vigogne-2-13b-instruct/
# falcon=/gpfsdswork/dataset/HuggingFace_Models/tiiuae/falcon-40b

python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $vigogne --dataset_name $e3c_en --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $vigogne --dataset_name $e3c_fr --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $vigogne --dataset_name $e3c_es --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $vigogne --dataset_name $wn_en --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name $vigogne --dataset_name $wn_es --n_gpus 2 -d