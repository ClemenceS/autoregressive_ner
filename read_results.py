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
                "size": 355*M,
                "languages" : "english",
                "training_data_size": "160 GiB",
                "training_data_languages": "English",
                "reference": "liu2019roberta",
                "language_markers": ["en"],
            },
            {
                "name": "camembert-large",
                "clean_name": "CamemBERT-large",
                "size": 335*M,
                "languages" : "french",
                "training_data_languages": "French",
                "reference": "martin2019camembert",
                "training_data_size": "64 billion tokens",
                "language_markers": ["fr"],
            },
            {
                "name": "flaubert_large_cased",
                "clean_name": "FlauBERT-large",
                "size": 335*M,
                "languages" : "french",
                "training_data_languages": "French",
                "reference": "le2019flaubert",
                "training_data_size": "13 billion tokens",
                "language_markers": ["fr"],
            },
            {
                "name": "bert-large-cased",
                "clean_name": "BERT-large",
                "size": 345*M,
                "languages" : "english",
                "training_data_languages": "English",
                "reference": "devlin2019bert",
                "training_data_size": "3,3 billion words",
                "language_markers": ["en"],
            },
            {
                "name": "tulio-chilean-spanish-bert",
                "clean_name": "TulioBERT",
                "size": 110*M,
                "languages" : "spanish",
                "training_data_languages": "Spanish",
                "training_data_size": "Undisclosed",
                "language_markers": ["es"],
            },
            {
                "name": "patana-chilean-spanish-bert",
                "clean_name": "PatanaBERT",
                "size": 110*M,
                "languages" : "spanish",
                "training_data_languages": "Spanish",
                "training_data_size": "Undisclosed",
                "language_markers": ["es"],
            },
            {
                "name": "bert-base-spanish-wwm-uncased",
                "clean_name": "BETO",
                "size": 110*M,
                "languages" : "spanish",
                "training_data_languages": "Spanish",
                "training_data_size": "3 billion words",
                "reference": "canete2020beto",
                "language_markers": ["es"],
            },
            {
                "name": "xlm-roberta-large",
                "clean_name": "XLM-RoBERTa-large",
                "size": 355*M,
                "languages" : "all",
                "training_data_languages": "100 languages including English, French and Spanish",
                "training_data_size": "2.5 TB",
                "reference": "conneau2020unsupervised",
                "language_markers": ["en", "fr", "es"],
            },
            {
                "name": "bert-base-multilingual-cased",
                "clean_name": "mBERT",
                "size": 110*M,
                "languages" : "all",
                "training_data_languages": "104 languages including English, French and Spanish",
                "training_data_size": "Undisclosed",
                "reference": "devlin2019bert",
                "language_markers": ["en", "fr", "es"],
            },
        ],
        "Clinical":[
            {
                "name": "ClinicalBERT",
                "clean_name": "ClinicalBERT",
                "size": 110*M,
                "languages" : "english",
                "training_data_languages": "English",
                "training_data_size": "1.2 billion words",
                "reference": "wang2023optimized",
                "language_markers": ["en"],
            },
            {
                "name": "MedBERT",
                "clean_name": "MedBERT",
                "size": 110*M,
                "languages" : "english",
                "training_data_languages": "English",
                "training_data_size": "57 million words",
                "reference": "charangan2022medbert",
                "language_markers": ["en"],
            },
            {
                "name": "Bio_ClinicalBERT",
                "clean_name": "Bio_ClinicalBERT",
                "size": 110*M,
                "languages" : "english",
                "training_data_languages": "English",
                "training_data_size": "2 million clinical notes",
                "reference": "alsentzer2019publicly",
                "language_markers": ["en"],
            },
            {
                "name": "camembert-bio-base",
                "clean_name": "CamemBERT-bio",
                "size": 110*M,
                "languages" : "french",
                "training_data_languages": "French",
                "reference": "touchent2023camembertbio",
                "training_data_size": "413 million words",
                "language_markers": ["fr"],
            },
            {
                "name": "DrBERT-4GB",
                "clean_name": "DrBERT-4GB",
                "size": 110*M,
                "languages" : "french",
                "training_data_languages": "French",
                "reference": "labrak2023drbert",
                "training_data_size": "1 billion words",
                "language_markers": ["fr"],
            },
            {
                "name": "bsc-bio-ehr-es",
                "clean_name": "BSC-BioEHR",
                "size": 110*M,
                "languages" : "spanish",
                "training_data_languages": "Spanish",
                "reference": "carrino2022pretrained",
                "training_data_size": "1.1 billion tokens",
                "language_markers": ["es"],
            },
            {
                "name": "bsc-bio-es",
                "clean_name": "BSC-Bio",
                "size": 110*M,
                "languages" : "spanish",
                "training_data_languages": "Spanish",
                "reference": "carrino2022pretrained",
                "training_data_size": "963 million tokens",
                "language_markers": ["es"],
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
                "training_data_size": "1600 GiB",
                "training_data_languages": "46 languages including English, French and Spanish",
                "reference": "workshop2022bloom",
                "language_markers": ["en", "fr", "es"],
            },
            {
                "name": "Mistral-7B-v0.1",
                "clean_name": "Mistral-7B",
                "size": 7*B,
                "training_data_size": "Undisclosed",
                "training_data_languages": "Undisclosed",
                "languages" : "all",
                "reference": "jiang2023mistral",
                "language_markers": ["?"],
            },
            {
                "name": "vicuna-7b-v1.5",
                "clean_name": "Vicuna-7B",
                "size": 7*B,
                "languages" : "all",
                "training_data_languages": "LLAMA 2, fine-tuned on conversations collected from ShareGPT.com, mainly in English",
                # "training_data_languages": "Mainly English",
                "training_data_size": "125K conversations",
                "reference": "zheng2023judging",
                "language_markers": ["en", "*"],
            },
            {
                "name": "vicuna-13b-v1.5",
                "clean_name": "Vicuna-13B",
                "size": 13*B,
                "languages" : "all",
                "training_data_languages": "LLAMA 2, fine-tuned on conversations collected from ShareGPT.com, mainly in English",
                # "training_data_languages": "Mainly English",
                "training_data_size": "125K conversations",
                "reference": "zheng2023judging",
                "language_markers": ["en", "*"],
            },
            {
                "name": "falcon-40b",
                "clean_name": "Falcon-40B",
                "size": 40*B,
                "languages" : "all",
                "training_data_size": "1 trillion tokens",
                "training_data_languages": "Mainly English, French, Spanish and German",
                "language_markers": ["en", "fr", "es"],
            },
            {
                "name": "vigogne-2-13b-instruct",
                "clean_name": "Vigogne-13B",
                "size": 13*B,
                "languages" : "french",
                "training_data_size": "52K instructions",
                "training_data_languages": "LLAMA 2, fine-tuned on English instructions automatically translated to French",
                "language_markers": ["fr", "en", "*"],
            },
            {
                "name": "gpt-j-6B",
                "clean_name": "GPT-J-6B",
                "size": 6*B,
                "languages" : "all",
                "training_data_size": "825 GiB",
                "training_data_languages": "English",
                "reference": "wang2021gptj",
                "language_markers": ["en"],
            },
           {
                "name" : "opt-66b",
                "clean_name": "OPT-66B",
                "size": 66*B,
                "languages" : "all",
                "training_data_size": "180 billion tokens",
                "training_data_languages": "Mainly English",
                "reference": "zhang2022opt",
                "language_markers": ["en"],
            },
            {
                "name": "Llama-2-70b-hf",
                "clean_name": "LLAMA-2-70B",
                "size": 70*B,
                "languages" : "all",
                "training_data_size": "2 trillion tokens",
                "training_data_languages": "Mainly English",
                "reference": "touvron2023llama",
                "language_markers": ["en"],
            },
        ],
        "Clinical":[
            {
                "name": "medalpaca-7b",
                "clean_name": "Medalpaca-7B",
                "size": 7*B,
                "languages" : "all",
                "training_data_languages": "LLAMA 2, fine-tuned on semi-generated medical question-answer pairs in English",
                # "training_data_languages": "Mainly English",
                "training_data_size": "400K Q.A. pairs",
                "reference": "han2023medalpaca",
                "language_markers": ["en", "*"],
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
model_training_data_sizes = {}
model_training_data_languages = {}
model_reference = {}
model_language_markers = {}
for model_type in model_hierarchy:
    for model_domain in model_hierarchy[model_type]:
        for model in model_hierarchy[model_type][model_domain]:
            model_name = model['name']
            model_domains[model_name] = model_domain
            model_types[model_name] = model_type
            model_langs[model_name] = model['languages']
            model_sizes[model_name] = model['size']
            model_clean_names[model_name] = model['clean_name']
            model_training_data_sizes[model_name] = model['training_data_size'] if 'training_data_size' in model else '-'
            model_training_data_languages[model_name] = model['training_data_languages'] if 'training_data_languages' in model else '-'
            model_reference[model_name] = model['reference'] if 'reference' in model else '-'
            model_language_markers[model_name] = model['language_markers'] if 'language_markers' in model else '-'
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
    df['fully_supervised'] = df['training_size'].apply(lambda x: x == -1)
    df['listing'] = df['listing'].fillna(False)
    #get only experiments where test_on_test_set is True
    df = df[df['test_on_test_set'] == True]

    initial_len = len(df)
    #sort df by time_str
    df = df.sort_values(by=["model_name", "dataset_name", "time_str"])
    df = df.drop_duplicates(subset=['model_name', 'dataset_name', 'fully_supervised', 'listing', 'training_size', 'partition_seed'], keep='last') #keep last to keep the best result
    final_len = len(df)
    print(f'Dropped {initial_len - final_len} duplicates.')
    df_few_shot = df[df['fully_supervised'] == False]
    df_fully_supervised = df[df['fully_supervised'] == True]
    return df_few_shot, df_fully_supervised