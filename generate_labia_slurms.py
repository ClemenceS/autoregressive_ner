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
        "/mnt/beegfs/home/naguib/e3c_fr": "e3cfr",
    },
    "en":{
        "conll2003": "conll2003",
        "/mnt/beegfs/home/naguib/n2c2": "n2c2",
        "mnaguib/WikiNER/en": "wnen",
        "/mnt/beegfs/home/naguib/e3c_en": "e3cen",
        },
    "es":{
        "mnaguib/WikiNER/es": "wnes",
        "conll2002/es": "conll2002",
        "/mnt/beegfs/home/naguib/e3c_es": "e3ces",
    },
}
disk = ['emea', 'medline', 'n2c2', 'e3cfr', 'e3cen', 'e3ces']
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

for dataset in datasets['fr']:
    generate_slurm(dataset, 'fr')
for dataset in datasets['en']:
    generate_slurm(dataset, 'en')
for dataset in datasets['es']:
    generate_slurm(dataset, 'es')