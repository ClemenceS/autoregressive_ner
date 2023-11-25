################################### MLMs ###################################
models = {
    "fr":{
        "camembert/camembert-large": "camembert",
        "flaubert/flaubert_large_cased": "flaubert",
        "Dr-BERT/DrBERT-4GB": "drbert4",
        "Dr-BERT/DrBERT-7GB": "drbert7",
        "almanach/camembert-bio-base": "camembertbio",
        "xlm-roberta-large": "xlmr",
        "bert-base-multilingual-cased" : "mbert",
    },
    "en":{
        "bert-large-cased": "bert",
        "roberta-large" : "roberta",
        "medicalai/ClinicalBERT": "clinbert",
        "Charangan/MedBERT": "medbert",
        "xlm-roberta-large": "xlmr",
        "emilyalsentzer/Bio_ClinicalBERT" : "bioclinbert",
        "bert-base-multilingual-cased" : "mbert",
    },
    "es":{
        "dccuchile/bert-base-spanish-wwm-uncased" : "beto",
        "dccuchile/tulio-chilean-spanish-bert": "tulio",
        "dccuchile/patana-chilean-spanish-bert": "patana",
        "xlm-roberta-large": "xlmr",
        "IIC/BETO_Galen" : "beto_galen",
        "bert-base-multilingual-cased" : "mbert",
        "PlanTL-GOB-ES/bsc-bio-ehr-es" : "BSC_bio_ehr",
        "PlanTL-GOB-ES/bsc-bio-es" : "BSC_bio",
    },
}
datasets = {
    "fr":{
        "/mnt/beegfs/home/naguib/emea": "emea",
        "/mnt/beegfs/home/naguib/medline": "medline",
        "mnaguib/WikiNER/fr": "wnfr",
        "/mnt/beegfs/home/naguib/e3c_fr": "e3cfr",
        "/mnt/beegfs/home/naguib/QFP": "qfp",
    },
    "en":{
        "conll2003": "conll2003",
        "/mnt/beegfs/home/naguib/n2c2": "n2c2",
        "mnaguib/WikiNER/en": "wnen",
        "/mnt/beegfs/home/naguib/e3c_en": "e3cen",
        "/mnt/beegfs/home/naguib/NCBI": "ncbi",
        },
    "es":{
        "mnaguib/WikiNER/es": "wnes",
        "conll2002/es": "conll2002",
        "/mnt/beegfs/home/naguib/e3c_es": "e3ces",
        "/mnt/beegfs/home/naguib/cwlc": "cwlc",
    },
}
disk = ['emea', 'medline', 'n2c2', 'e3cfr', 'e3cen', 'e3ces', 'cwlc', 'qfp', 'ncbi']
fixed_header="""#!/bin/bash

#SBATCH --job-name={dataset}
#SBATCH --output={dataset}.out
#SBATCH --error={dataset}.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
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

def generate_slurms_for_model(model, full=False):
    for lang_models in models.values():
        if model in lang_models:
            model_short_name = lang_models[model]
            break
    if full:
        model_short_name += "_full"
    with open(f"slurms_labia/{model_short_name}.slurm", "w") as f:
        f.write(fixed_header.format(dataset=model_short_name))
        f.write("\n")    
    for language, lang_models in models.items():
        if model in lang_models:        
            for dataset, dataset_short_name in datasets[language].items():
                with open(f"slurms_labia/{model_short_name}.slurm", "a") as f:
                    f.write(line.format(dataset=dataset, model=model, disk='-d' if dataset_short_name in disk else '') + " -s \"-1\"" * full)
                    f.write("\n")

for dataset in datasets['fr']:
    generate_slurm(dataset, 'fr')
for dataset in datasets['en']:
    generate_slurm(dataset, 'en')
for dataset in datasets['es']:
    generate_slurm(dataset, 'es')
rem_models = [
    "bert-base-multilingual-cased",
    "flaubert/flaubert_large_cased",
    "PlanTL-GOB-ES/bsc-bio-ehr-es",
    "PlanTL-GOB-ES/bsc-bio-es",
    "emilyalsentzer/Bio_ClinicalBERT",
]
for model in rem_models:
    generate_slurms_for_model(model)

models_for_fully_supervised = [
    'roberta-large',
    "camembert/camembert-large",
    "dccuchile/bert-base-spanish-wwm-uncased",
]
for model in models_for_fully_supervised:
    generate_slurms_for_model(model, full=True)


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
    "bofenghuang/vigogne-2-13b-instruct": "vigogne13",
    "EleutherAI/gpt-j-6B": "gptj6",
    "nomic-ai/gpt4all-j": "gpt4allj",
    "gpt2-xl": "gpt2xl",
    "EleutherAI/gpt-neox-20b" : "gptneox20",
    "medalpaca/medalpaca-7b": "medalpaca",
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
    "conll2002/es": "conll2002",
    "/mnt/beegfs/home/naguib/NCBI": "ncbi",
    "/mnt/beegfs/home/naguib/cwlc": "cwlc",
    "/mnt/beegfs/home/naguib/QFP": "qfp",
}
fixed_header="""#!/bin/bash

#SBATCH --job-name={dataset}
#SBATCH --output={dataset}.out
#SBATCH --error={dataset}.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
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
