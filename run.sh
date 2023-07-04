#!/bin/bash

#SBATCH --mail-user=marco.naguib@universite-paris-saclay.fr
#SBATCH --mail-type=ALL
#SBATCH --job-name=ner
#SBATCH --nodes=1

python3 bloom_ner.py
