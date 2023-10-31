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

scores = {}

#replace each model_name with a shorter version, which is the name.split('/')[-1]
df['model_name'] = df['model_name'].apply(lambda name: name.split('/')[-1])
#for each pair of dataset_name and model_name, print all the lines containing the pair, if any
for dataset_name in df['dataset_name'].unique():
    for model_name in df['model_name'].unique():
        if len(df[(df['dataset_name'] == dataset_name) & (df['model_name'] == model_name)]) > 0:
            # print('dataset_name: ', dataset_name, ' model_name: ', model_name)
            #if more than one line, get the most recent one
            line = df[(df['dataset_name'] == dataset_name) & (df['model_name'] == model_name)].iloc[-1]
            scores[(dataset_name, model_name)] = line['metrics']['f1']

#print a latex table where each row is a dataset and each column is a model
print('\\begin{tabular}{l|' + 'c|' * len(df['model_name'].unique()) + '}')
print(' & ' + ' & '.join(df['model_name'].unique()) + '\\\\')
print('\\hline')
for dataset_name in df['dataset_name'].unique():
    print(dataset_name + ' & ' + ' & '.join([str(scores[(dataset_name, model_name)]) if (dataset_name, model_name) in scores else '-' for model_name in df['model_name'].unique()]) + '\\\\')
print('\\end{tabular}')
