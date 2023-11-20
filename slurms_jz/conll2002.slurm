#!/bin/bash

#SBATCH --job-name=conll2002
#SBATCH --output=conll2002.out
#SBATCH --error=conll2002.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=lak@a100
#SBATCH -C a100


module purge
module load llm


python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/mistralai/Mistral-7B-v0.1 --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/lmsys/vicuna-7b-v1.5 --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/lmsys/vicuna-13b-v1.5 --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/stanford-crfm/BioMedLM --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfswork/rech/lak/utb11pp/models/nomic-ai/gpt4all-j --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfsdswork/dataset/HuggingFace_Models/bigscience/bloom-7b1 --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfsdswork/dataset/HuggingFace_Models/bigscience/bloomz-7b1 --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfsdswork/dataset/HuggingFace_Models/bigscience/bloom --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfsdswork/dataset/HuggingFace_Models/bigscience/bloomz --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfsdswork/dataset/HuggingFace_Models/tiiuae/falcon-7b --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfsdswork/dataset/HuggingFace_Models/tiiuae/falcon-40b --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfsdswork/dataset/HuggingFace_Models/bofenghuang/vigogne-2-13b-instruct --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfsdswork/dataset/HuggingFace_Models/EleutherAI/gpt-j-6B --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfsdswork/dataset/HuggingFace_Models/EleutherAI/gpt-neo-2.7B/ --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfsdswork/dataset/HuggingFace_Models/facebook/opt-66b --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
python3 $WORK/autoregressive_ner/clm_experiment.py --model_name /gpfsdswork/dataset/HuggingFace_Models/meta-llama/Llama-2-70b-hf --dataset_name /gpfswork/rech/lak/utb11pp/data/conll2002  --n_gpus 2 -d
