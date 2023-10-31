import json
import os
import pandas as pd
from glob import glob

def read_jsons(path):
    jsons = glob(os.path.join(path, '*.json'))
    data = []
    for json_file in jsons:
        with open(json_file, 'r') as f:
            data.append(json.load(f))
    return data

data = read_jsons('results')
df = pd.DataFrame(data)

dataset_names = {
    "conll2002-es":{
        "name": "CoNLL2002",
        "lang": "es",
        "domain": "general"
    },
    "conll2003": {
        "name": "CoNLL2003",
        "lang": "en",
        "domain": "general"
    },
    "naguib-emea": {
        "name": "EMEA",
        "lang": "fr",
        "domain": "clinical"
    },
    "naguib-medline": {
        "name": "MEDLINE",
        "lang": "fr",
        "domain": "clinical"
    },
    "WikiNER-en": {
        "name": "WikiNER",
        "lang": "en",
        "domain": "general"
    },
    "WikiNER-fr": {
        "name": "WikiNER",
        "lang": "fr",
        "domain": "general"
    },
    "WikiNER-es": {
        "name": "WikiNER",
        "lang": "es",
        "domain": "general"
    },
    "naguib-n2c2": {
        "name": "n2c2",
        "lang": "en",
        "domain": "clinical"
    },
    # "Mistral-7B-Instruct-v0.1" :{
    #     "name": "Mistral-Instruct-7B",
    #     "lang": "en",
    #     "domain": "general",
    # },
    # "Mistral-7B-v0.1" :{
    #     "name": "Mistral-7B",
    #     "lang": "en",
    #     "domain": "general",
    # },
    # "bloom-7b1": {
    #     "name": "Bloom-7B",
    #     "lang": "en",
    #     "domain": "general"
    # },
    # "vicuna-13b-v1.5": {
    #     "name": "Vicuna-13B",
    #     "lang": "en",
    #     "domain": "general"
    # },
}

os.makedirs('tables', exist_ok=True)


df['lang'] = df['dataset_name'].apply(lambda name: dataset_names[name]['lang'])
df['dataset_domain'] = df['dataset_name'].apply(lambda name: dataset_names[name]['domain'])
df['model_name'] = df['model_name'].apply(lambda name: name.split('/')[-1])
gdf = df[df['dataset_domain'] == 'general']
print(gdf)
gscores = {}
n_discarded_general = 0
for dataset_name in gdf['dataset_name'].unique():
    for model_name in gdf['model_name'].unique():
        if len(gdf[(gdf['dataset_name'] == dataset_name) & (gdf['model_name'] == model_name)]) > 0:
            line = gdf[(gdf['dataset_name'] == dataset_name) & (gdf['model_name'] == model_name)].iloc[-1]
            n_discarded_general += len(gdf[(gdf['dataset_name'] == dataset_name) & (gdf['model_name'] == model_name)]) - 1
            gscores[(dataset_name, model_name)] = round(line['exact']['f1'],3)
print('n_discarded: ', n_discarded_general)

with open("tables/general_results.tex", 'w') as f:
    f.write('\\begin{table}')
    f.write('\\resizebox{\\textwidth}{!}{')
    f.write('\\begin{tabular}{l|l|' + 'c|' * len(gdf['model_name'].unique()) + '}')
    f.write('\\toprule')
    f.write(' & & ' + ' & '.join([model_name for model_name in gdf['model_name'].unique()]) + '\\\\')
    f.write('\\midrule')
    for lang in ['en', 'es', 'fr']:
        nb_lines = len(gdf.query('lang == @lang and dataset_domain == "general"')['dataset_name'].unique())
        if nb_lines > 0:
            f.write('\\multirow{' + str(nb_lines) + '}{*}{' + lang + '} & ')
            for dataset_name in gdf[gdf['lang'] == lang]['dataset_name'].unique():
                if dataset_names[dataset_name]['domain'] == 'general':
                    f.write(dataset_names[dataset_name]['name'] + ' & ' + ' & '.join([str(gscores[(dataset_name, model_name)]) if (dataset_name, model_name) in gscores else '-' for model_name in gdf['model_name'].unique()]) + '\\\\')
                    #if last line, don't write '&'
                    if dataset_name != gdf[gdf['lang'] == lang]['dataset_name'].unique()[-1]:
                        f.write(' & ')
        if lang != 'fr':
            f.write('\\midrule')
    f.write('\\bottomrule')
    f.write('\\end{tabular}')
    f.write('}')
    f.write('\\end{table}')


cdf = df[df['dataset_domain'] == 'clinical']
cscores = {}
n_discarded_clinical = 0
for dataset_name in cdf['dataset_name'].unique():
    for model_name in cdf['model_name'].unique():
        if len(cdf[(cdf['dataset_name'] == dataset_name) & (cdf['model_name'] == model_name)]) > 0:
            line = cdf[(cdf['dataset_name'] == dataset_name) & (cdf['model_name'] == model_name)].iloc[-1]
            n_discarded_clinical += len(cdf[(cdf['dataset_name'] == dataset_name) & (cdf['model_name'] == model_name)]) - 1
            cscores[(dataset_name, model_name)] = round(line['exact']['f1'],3)
print('n_discarded: ', n_discarded_clinical)

with open("tables/clinical_results.tex", 'w') as f:
    f.write('\\begin{table}')
    f.write('\\resizebox{\\textwidth}{!}{')
    f.write('\\begin{tabular}{l|l|' + 'c|' * len(cdf['model_name'].unique()) + '}')
    f.write('\\toprule')
    f.write(' & & ' + ' & '.join([model_name for model_name in cdf['model_name'].unique()]) + '\\\\')
    f.write('\\midrule')
    for lang in ['en', 'fr']:
        nb_lines = len(cdf.query('lang == @lang and dataset_domain == "clinical"')['dataset_name'].unique())
        if nb_lines > 0:
            f.write('\\multirow{' + str(nb_lines) + '}{*}{' + lang + '} & ')
            for dataset_name in cdf[cdf['lang'] == lang]['dataset_name'].unique():
                if dataset_names[dataset_name]['domain'] == 'clinical':
                    f.write(dataset_names[dataset_name]['name'] + ' & ' + ' & '.join([str(cscores[(dataset_name, model_name)]) if (dataset_name, model_name) in cscores else '-' for model_name in cdf['model_name'].unique()]) + '\\\\')
                    #if last line, don't write '&'
                    if dataset_name != cdf[cdf['lang'] == lang]['dataset_name'].unique()[-1]:
                        f.write(' & ')
        if lang != 'fr':
            f.write('\\midrule')
    f.write('\\bottomrule')
    f.write('\\end{tabular}')
    f.write('}')
    f.write('\\end{table}')