#!/bin/bash

#SBATCH --mail-user=marco.naguib@universite-paris-saclay.fr
#SBATCH --mail-type=ALL
#SBATCH --job-name=ner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1

export http_proxy=http://webproxy.lab-ia.fr:8080
export https_proxy=http://webproxy.lab-ia.fr:8080
export HTTP_PROXY=http://webproxy.lab-ia.fr:8080
export HTTPS_PROXY=http://webproxy.lab-ia.fr:8080

#python3 ~/autoregressive_ner/bloom_ner.py --model_name "lmsys/vicuna-13b-v1.5" -g --prompt_dict fr --language fr --n_few_shot 5 10 --random_seed 1 2 3 --ner_tag PER
python3 ~/autoregressive_ner/bloom_ner.py --model_name "lmsys/vicuna-13b-v1.5" -g --prompt_dict fr --language fr --n_few_shot 5 10 --random_seed 1 2 3 --ner_tag LOC
#python3 ~/autoregressive_ner/bloom_ner.py --model_name "lmsys/vicuna-13b-v1.5" -g --prompt_dict en --language en --n_few_shot 5 10 --random_seed 1 2 3 --ner_tag PER
python3 ~/autoregressive_ner/bloom_ner.py --model_name "lmsys/vicuna-13b-v1.5" -g --prompt_dict en --language en --n_few_shot 5 10 --random_seed 1 2 3 --ner_tag LOC
python3 ~/autoregressive_ner/bloom_ner.py --model_name "lmsys/vicuna-13b-v1.5" -g --prompt_dict fr --language fr --n_few_shot 5 10 --random_seed 1 2 3  --domain clinical
