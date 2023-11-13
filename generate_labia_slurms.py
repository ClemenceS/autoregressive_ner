models = {
    "fr":{
        "camembert/camembert-large": "camembert",
        "Dr-BERT/DrBERT-4GB": "drbert4",
        "Dr-BERT/DrBERT-7GB": "drbert7",
        "almanach/camembert-bio-base": "camembertbio",
        "xlm-roberta-large": "xlmr",
    },
    "en":{
        "bert-large-cased": "bert",
        "roberta-large" : "roberta",
        "medicalai/ClinicalBERT": "clinbert",
        "Charangan/MedBERT": "medbert",
        "xlm-roberta-large": "xlmr",
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
        "mnaguib/WikiNER/fr": "wnfr",
    },
    "en":{
        "conll2003": "conll2003",
        "/mnt/beegfs/home/naguib/n2c2": "n2c2",
        "mnaguib/WikiNER/en": "wnen",
        },
    "es":{
        "mnaguib/WikiNER/es": "wnes",
        "conll2002/es": "conll2002",
    },
}
disk = ['emea', 'medline', 'n2c2']
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
line = 'python3 /mnt/beegfs/home/naguib/autoregressive_ner/mlm_experiment.py --model_name "{model}" --dataset_name "{dataset}" {disk}'

def generate_slurm(dataset, language):
    dataset_short_name = datasets[language][dataset]
    slurm_name = f"slurms_labia/{dataset_short_name}.slurm"
    slurm_header = fixed_header.format(dataset=dataset_short_name)
    with open(slurm_name, "w") as f:
        f.write(slurm_header + "\n")
        for model in models[language]:
            f.write(line.format(model=model, dataset=dataset, disk='-d' if dataset_short_name in disk else '') + "\n")

for language in datasets:
    for dataset in datasets[language]:
        generate_slurm(dataset, language)