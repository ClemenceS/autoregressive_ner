models = {
    "fr":{
        "camembert/camembert-large": "camembert",
        "Dr-BERT/DrBERT-4GB": "drbert4",
        "Dr-BERT/DrBERT-7GB": "drbert7",
        "xlm-roberta-large": "xlmr",
    },
    "en":{
        "bert-large-cased": "bert",
        "roberta-large" : "roberta",
        "medicalai/ClinicalBERT": "clinbert",
        "Charangan/MedBERT": "medbert",
        "xlm-roberta-large": "xlmr",
        "almanach/camembert-bio-base": "camembertbio",
    },
    "es":{
        "dccuchile/bert-base-spanish-wwm-uncased" : "beto",
        "dccuchile/tulio-chilean-spanish-bert": "tulio",
        "dccuchile/patana-chilean-spanish-bert": "patana",
        "xlm-roberta-large": "xlmr",
        "IIC/BETO_Galen" : "beto_galen",
    },
}
datasets = {
    "fr":{
        "/mnt/beegfs/home/naguib/emea": "emea",
        "/mnt/beegfs/home/naguib/medline": "medline",
        "WikiNER/fr": "wnfr",
    },
    "en":{
        "/mnt/beegfs/home/naguib/conll2003": "conll2003",
        "/mnt/beegfs/home/naguib/n2c2": "n2c2",
        "WikiNER/en": "wnen",
        },
    "es":{
        "WikiNER/es": "wnes",
    },
}
fixed_header="""#!/bin/bash

#SBATCH --job-name={dataset}
#SBATCH --output={dataset}.out
#SBATCH --error={dataset}.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
"""
line = 'python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "{model}" --dataset_name "{dataset}" -d'

def generate_slurm(dataset, language):
    dataset_short_name = datasets[language][dataset]
    slurm_name = f"slurms_labia/{dataset_short_name}.slurm"
    slurm_header = fixed_header.format(dataset=dataset_short_name)
    with open(slurm_name, "w") as f:
        f.write(slurm_header + "\n")
        for model in models[language]:
            f.write(line.format(model=model, dataset=dataset) + "\n")

for language in datasets:
    for dataset in datasets[language]:
        generate_slurm(dataset, language)