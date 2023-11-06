from glob import glob
import json
import os
import pandas as pd

B=10**9
M=10**6

dataset_hierarchy = {
    "en":{
        "General": [
            {"conll2003": "CoNLL2003"},
            {"WikiNER-en": "WikiNER"}
        ],
        "Clinical": [
            {"naguib-n2c2": "n2c2"}
        ],
        },
    "fr": {
        "General": [
            {"WikiNER-fr": "WikiNER"}
        ],
        "Clinical": [
            {"naguib-emea": "EMEA"},
            {"naguib-medline": "MEDLINE"}
        ],
    },
    "es": {
        "General": [
            {"conll2002-es": "CoNLL2002"},
            {"WikiNER-es": "WikiNER"}
        ],
        "Clinical": []
    }
}
model_hierarchy = {
    "Masked": {
        "General":[
            {
                "name": "roberta-large", 
                "clean_name": "RoBERTa-large",
                "size": 355*M,
                "languages" : ["en",],
            },
            {
                "name": "camembert-base",
                "clean_name": "CamemBERT-base",
                "size": 110*M,
                "languages" : ["fr",],
            },
            {
                "name": "camembert-large",
                "clean_name": "CamemBERT-large",
                "size": 355*M,
                "languages" : ["fr",],
            },
            {
                "name": "bert-large-cased",
                "clean_name": "BERT-large",
                "size": 340*M,
                "languages" : ["en",],
            },
            {
                "name": "tulio-chilean-spanish-bert",
                "clean_name": "TulioBERT",
                "size": 110*M,
                "languages" : ["es",],
            },
            {
                "name": "patana-chilean-spanish-bert",
                "clean_name": "PatanaBERT",
                "size": 110*M,
                "languages" : ["es",],
            },
            {
                "name": "bert-base-spanish-wwm-cased",
                "clean_name": "BERT-base-Spanish",
                "size": 110*M,
                "languages" : ["es",],
            },
            {
                "name": "xlm-roberta-large",
                "clean_name": "XLM-RoBERTa-large",
                "size": 355*M,
                "languages" : ["en", "fr", "es"],
            }
        ],
        "Clinical":[
            {
                "name": "ClinicalBERT",
                "clean_name": "ClinicalBERT",
                "size": 110*M,
                "languages" : ["en",],
            },
            {
                "name": "MedBERT",
                "clean_name": "MedBERT",
                "size": 110*M,
                "languages" : ["en",],
            },
            {
                "name": "camembert-bio-base",
                "clean_name": "CamemBERT-bio",
                "size": 110*M,
                "languages" : ["en",],
            },
            {
                "name": "DrBERT-4GB-CP-PubMedBERT",
                "clean_name": "DrBERT",
                "size": 110*M,
                "languages" : ["en",],
            },

        ],
    },
    "Causal": {
        "General":[
            {
                "name": "bloom-560m",
                "clean_name": "BLOOM-560M",
                "size": 560*M,
                "languages" : ["en", "fr", "es"],
            },
            {
                "name": "Mistral-7B-Instruct-v0.1",
                "clean_name": "Mistral-7B-Instruct",
                "size": 7*B,
                "languages" : ["en", "fr", "es"],
            },
            {
                "name": "bloom-7b1",
                "clean_name": "BLOOM-7B1",
                "size": 7*B,
                "languages" : ["en", "fr", "es"],
            },
            {
                "name": "falcon-40b-instruct",
                "clean_name": "Falcon-40B-Instruct",
                "size": 40*B,
                "languages" : ["en", "fr", "es"],
            },
            {
                "name": "Mistral-7B-v0.1",
                "clean_name": "Mistral-7B",
                "size": 7*B,
                "languages" : ["en", "fr", "es"],
            },
            {
                "name": "vicuna-13b-v1.5",
                "clean_name": "Vicuna-13B",
                "size": 13*B,
                "languages" : ["en", "fr", "es"],
            },
            {
                "name": "falcon-40b",
                "clean_name": "Falcon-40B",
                "size": 40*B,
                "languages" : ["en", "fr", "es"],
            },
        ],
    }
}

dataset_domains = {}
dataset_langs = {}
model_domains = {}
model_types = {}
model_sizes = {}
model_clean_names = {}
dataset_names = {}
for model_type in model_hierarchy:
    for model_domain in model_hierarchy[model_type]:
        for model in model_hierarchy[model_type][model_domain]:
            model_name = model['name']
            model_domains[model_name] = model_domain
            model_types[model_name] = model_type
            model_sizes[model_name] = model['size']
            model_clean_names[model_name] = model['clean_name']
for lang in dataset_hierarchy:
    for domain in dataset_hierarchy[lang]:
        for dataset in dataset_hierarchy[lang][domain]:
            for dataset_name in dataset:
                dataset_domains[dataset_name] = domain
                dataset_langs[dataset_name] = lang
                dataset_names[dataset_name] = dataset[dataset_name]

def read_jsons(path):
    jsons = glob(os.path.join(path, '*.json'))
    data = []
    for json_file in jsons:
        with open(json_file, 'r') as f:
            data.append(json.load(f))
    
    df = pd.DataFrame(data)


    df['lang'] = df['dataset_name'].apply(lambda name: dataset_langs[name])
    df['dataset_domain'] = df['dataset_name'].apply(lambda name: dataset_domains[name])
    df['model_name'] = df['model_name'].apply(lambda name: name.split('/')[-1])
    df['model_domain'] = df['model_name'].apply(lambda name: model_domains[name])
    df['model_type'] = df['model_name'].apply(lambda name: model_types[name])
    df['model_size'] = df['model_name'].apply(lambda name: model_sizes[name])
    df['model_clean_name'] = df['model_name'].apply(lambda name: model_clean_names[name])
    df['f1'] = df['exact'].apply(lambda x: round(x['f1'],3))
    df['lang'] = df['dataset_name'].apply(lambda name: dataset_langs[name])

    df = df.drop_duplicates(subset=['model_name', 'dataset_name'], keep='last')
    return df