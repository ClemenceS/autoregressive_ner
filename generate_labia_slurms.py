################################### MLMs ###################################
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


################################### CLMs ###################################
# models = {
#     #local models
#     "/gpfswork/rech/lak/utb11pp/models/mistralai/Mistral-7B-v0.1" : "mistral",
#     "/gpfswork/rech/lak/utb11pp/models/lmsys/vicuna-7b-v1.5" : "vic7",
#     "/gpfswork/rech/lak/utb11pp/models/lmsys/vicuna-13b-v1.5" : "vic13",
#     "/gpfswork/rech/lak/utb11pp/models/stanford-crfm/BioMedLM": "biomedlm",
#     #common models
#     "/gpfsdswork/dataset/HuggingFace_Models/bigscience/bloom-7b1": "bloom7",
#     "/gpfsdswork/dataset/HuggingFace_Models/bigscience/bloomz-7b1": "bloomz7",
#     "/gpfsdswork/dataset/HuggingFace_Models/bigscience/bloom" : "bloombig",
#     "/gpfsdswork/dataset/HuggingFace_Models/bigscience/bloomz" : "bloomzbig",
#     "/gpfsdswork/dataset/HuggingFace_Models/tiiuae/falcon-7b": "falcon7",
#     "/gpfsdswork/dataset/HuggingFace_Models/tiiuae/falcon-40b": "falcon40",
#     "/gpfsdswork/dataset/HuggingFace_Models/bofenghuang/vigogne-2-13b-instruct/": "vigogne13",
#     "/gpfsdswork/dataset/HuggingFace_Models/EleutherAI/gpt-j-6B": "gptj6",
# }
models = {
    "mistralai/Mistral-7B-v0.1" : "mistral",
    "lmsys/vicuna-7b-v1.5" : "vic7",
    "lmsys/vicuna-13b-v1.5" : "vic13",
    "stanford-crfm/BioMedLM": "biomedlm",
    "bigscience/bloom-7b1": "bloom7",
    "bigscience/bloomz-7b1": "bloomz7",
    "bigscience/bloom" : "bloombig",
    "bigscience/bloomz" : "bloomzbig",
    "tiiuae/falcon-7b": "falcon7",
    "tiiuae/falcon-40b": "falcon40",
    "bofenghuang/vigogne-2-13b-instruct/": "vigogne13",
    "EleutherAI/gpt-j-6B": "gptj6",
}
datasets = {
    #local datasets
    "conll2003": "conll2003",
    "/mnt/beegfs/home/naguib/emea": "emea",
    "/mnt/beegfs/home/naguib/medline": "medline",
    "/mnt/beegfs/home/naguib/n2c2": "n2c2",
    "mnaguib/WikiNER/en": "wnen",
    "mnaguib/WikiNER/fr": "wnfr",
    "mnaguib/WikiNER/es": "wnes",
    "/mnt/beegfs/home/naguib/e3c_en": "e3cen",
    "/mnt/beegfs/home/naguib/e3c_fr": "e3cfr",
    "/mnt/beegfs/home/naguib/e3c_es": "e3ces",
}
fixed_header="""#!/bin/bash

#SBATCH --job-name={dataset}
#SBATCH --output={dataset}.out
#SBATCH --error={dataset}.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=45:00:00
#SBATCH --gres=gpu:2

model={model}
"""

line = "python3 /mnt/beegfs/home/naguib/autoregressive_ner/clm_experiment.py --model_name $model --dataset_name {dataset}  --n_gpus 2"

def generate_slurm(model):
    model_short_name = models[model]
    with open(f"slurms_labia/{model_short_name}.slurm", "w") as f:
        f.write(fixed_header.format(dataset=model_short_name, model=model))
        f.write("\n")
        for dataset in datasets:
            f.write(line.format(dataset=dataset))
            f.write("\n")

for model in models:
    generate_slurm(model)
