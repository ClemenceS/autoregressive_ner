from glob import glob
import json
import os
import pandas as pd

B=10**9
M=10**6

dataset_hierarchy = {
    "english":{
        "General": {
            "WikiNER-en": "WikiNER-en",
            "conll2003": "CoNLL2003",
        },
        "Clinical": {
            "e3c_en": "E3C-en",
            "n2c2": "n2c2",
            "NCBI": "NCBI",
        },
    },
    "french": {
        "General": {
            "WikiNER-fr": "WikiNER-fr",
            "QFP": "QFP",
        },
        "Clinical": {
            "e3c_fr": "E3C-fr",
            "emea": "EMEA",
            "medline": "MEDLINE",
        },
    },
    "spanish": {
        "General": {
            "WikiNER-es": "WikiNER-es",
            "conll2002": "CoNLL2002",
        },
        "Clinical": {
            "e3c_es": "E3C-es",
            "cwlc": "CWL",
        },
    },
}
model_hierarchy = {
    "Masked": {
        "General":[
            {
                "name": "roberta-large", 
                "clean_name": "RoBERTa-large",
                "description": "160GB of text from CommonCrawl, a dataset of web crawl data.",
                "size": 355*M,
                "languages" : "english",
            },
            {
                "name": "camembert-large",
                "clean_name": "CamemBERT-large",
                "size": 335*M,
                "languages" : "french",
            },
            {
                "name": "flaubert_large_cased",
                "clean_name": "FlauBERT-large",
                "size": 335*M,
                "languages" : "french",
            },
            {
                "name": "bert-large-cased",
                "clean_name": "BERT-large",
                "size": 345*M,
                "languages" : "english",
            },
            {
                "name": "tulio-chilean-spanish-bert",
                "clean_name": "TulioBERT",
                "size": 110*M,
                "languages" : "spanish",
            },
            {
                "name": "patana-chilean-spanish-bert",
                "clean_name": "PatanaBERT",
                "size": 110*M,
                "languages" : "spanish",
            },
            {
                "name": "bert-base-spanish-wwm-uncased",
                "clean_name": "BETO",
                "size": 110*M,
                "languages" : "spanish",
            },
            {
                "name": "xlm-roberta-large",
                "clean_name": "XLM-RoBERTa-large",
                "size": 355*M,
                "languages" : "all",
            },
            {
                "name": "bert-base-multilingual-cased",
                "clean_name": "mBERT",
                "size": 110*M,
                "languages" : "all",
            },
        ],
        "Clinical":[
            {
                "name": "ClinicalBERT",
                "clean_name": "ClinicalBERT",
                "size": 110*M,
                "languages" : "english",
            },
            {
                "name": "MedBERT",
                "clean_name": "MedBERT",
                "size": 110*M,
                "languages" : "english",
            },
            {
                "name": "Bio_ClinicalBERT",
                "clean_name": "Bio_ClinicalBERT",
                "size": 110*M,
                "languages" : "english",
            },
            {
                "name": "camembert-bio-base",
                "clean_name": "CamemBERT-bio",
                "size": 110*M,
                "languages" : "french",
            },
            {
                "name": "DrBERT-4GB",
                "clean_name": "DrBERT-4GB",
                "size": 4*B,
                "languages" : "french",
            },
            # {
                
            #     "name": "BETO_Galen",
            #     "clean_name": "BETO-Galen",
            #     "size": 110*M,
            #     "languages" : "spanish",
            # },
            {
                "name": "bsc-bio-ehr-es",
                "clean_name": "BSC-BioEHR",
                "size": 110*M,
                "languages" : "spanish",
            },
            {
                "name": "bsc-bio-es",
                "clean_name": "BSC-Bio",
                "size": 110*M,
                "languages" : "spanish",
            }
        ],
    },
    "Causal": {
        "General":[
            {
                "name": "bloom-7b1",
                "clean_name": "BLOOM-7B1",
                "size": 7*B,
                "languages" : "all",
            },
            {
                "name": "Mistral-7B-v0.1",
                "clean_name": "Mistral-7B",
                "size": 7*B,
                "description": "Not shared yet.",
                "languages" : "all",
            },
            {
                "name": "vicuna-7b-v1.5",
                "clean_name": "Vicuna-7B",
                "size": 7*B,
                "languages" : "all",
            },
            {
                "name": "vicuna-13b-v1.5",
                "clean_name": "Vicuna-13B",
                "size": 13*B,
                "languages" : "all",
            },
            {
                "name": "falcon-40b",
                "clean_name": "Falcon-40B",
                "size": 40*B,
                "languages" : "all",
            },
            {
                "name": "vigogne-2-13b-instruct",
                "clean_name": "Vigogne-13B",
                "size": 13*B,
                "languages" : "french",
            },
            {
                "name": "gpt-j-6B",
                "clean_name": "GPT-J-6B",
                "size": 6*B,
                "languages" : "all",
            },
           {
                "name" : "opt-66b",
                "clean_name": "OPT-66B",
                "size": 66*B,
                "languages" : "all",
            },
            {
                "name": "Llama-2-70b-hf",
                "clean_name": "LLAMA-70B",
                "size": 70*B,
                "languages" : "all",
            },
        ],
        "Clinical":[
            {
                "name": "medalpaca-7b",
                "clean_name": "Medalpaca-7B",
                "size": 7*B,
                "languages" : "all",
            },
            # {
            #     "name": "BioMedLM",
            #     "clean_name": "BioMedLM",
            #     "size": 7*B,
            #     "languages" : "all",
            # }
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
model_langs = {}
model_descriptions = {}
for model_type in model_hierarchy:
    for model_domain in model_hierarchy[model_type]:
        for model in model_hierarchy[model_type][model_domain]:
            model_name = model['name']
            model_domains[model_name] = model_domain
            model_types[model_name] = model_type
            model_langs[model_name] = model['languages']
            model_sizes[model_name] = model['size']
            model_clean_names[model_name] = model['clean_name']
            model_descriptions[model_name] = model['description'] if 'description' in model else '-'
for lang in dataset_hierarchy:
    for domain in dataset_hierarchy[lang]:
        for dataset_name in dataset_hierarchy[lang][domain]:
            dataset_domains[dataset_name] = domain
            dataset_langs[dataset_name] = lang
            dataset_names[dataset_name] = dataset_hierarchy[lang][domain][dataset_name]

def read_jsons(path):
    jsons = glob(os.path.join(path, '*.json'))
    data = []
    for json_file in jsons:
        with open(json_file, 'r') as f:
            data.append(json.load(f))
    
    df = pd.DataFrame(data)
    
    df['dataset_name'] = df['dataset_name'].apply(lambda name: name.replace('data-', ''))
    df['dataset_name'] = df['dataset_name'].apply(lambda name: name.replace('naguib-', ''))
    df['dataset_name'] = df['dataset_name'].apply(lambda name: name.replace('conll2002-es', 'conll2002'))
    df['lang'] = df['dataset_name'].apply(lambda name: dataset_langs[name])
    df['dataset_domain'] = df['dataset_name'].apply(lambda name: dataset_domains[name])
    df['model_name'] = df['model_name'].apply(lambda name: name.split('/')[-1] if not name.endswith('/') else name.split('/')[-2])

    df['model_domain'] = df['model_name'].apply(lambda name: model_domains[name])
    df['model_type'] = df['model_name'].apply(lambda name: model_types[name])
    df['model_size'] = df['model_name'].apply(lambda name: model_sizes[name])
    df['model_clean_name'] = df['model_name'].apply(lambda name: model_clean_names[name])
    df['f1'] = df['exact'].apply(lambda x: round(x['f1'],3))
    df['lang'] = df['dataset_name'].apply(lambda name: dataset_langs[name])
    df['fully_supervised'] = df['training_size'].apply(lambda x: x != 100)
    df['listing'] = df['listing'].fillna(False)
    #get only experiments where test_on_test_set is True
    df = df[df['test_on_test_set'] == True]

    initial_len = len(df)
    #sort df by time_str
    df = df.sort_values(by=["model_name", "dataset_name", "time_str"])
    df = df.drop_duplicates(subset=['model_name', 'dataset_name', 'fully_supervised', 'listing'], keep='last')
    final_len = len(df)
    print(f'Dropped {initial_len - final_len} duplicates.')
    df_few_shot = df[df['fully_supervised'] == False]
    df_fully_supervised = df[df['fully_supervised'] == True]
    return df_few_shot, df_fully_supervised