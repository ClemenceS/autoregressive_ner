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

python3 bloom_ner.py --domain clinical --top_p 0.4 0.5 0.6 --top_k 7 8 9 --temperature 0.7 0.8 0.9