import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from read_results import read_jsons, model_domains, model_types, model_sizes, model_clean_names, dataset_names, model_langs, model_training_data_sizes, model_training_data_languages, model_reference, model_language_markers
from read_results import dataset_hierarchy, model_hierarchy
from prompt_strings import strings
from plot_tools import plot_data
from latex_tools import latex_results, latex_models, latex_listing, latex_sampling, latex_ner_descriptions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--local', action='store_true')
parser.add_argument('-s', '--segur', action='store_true')
args = parser.parse_args()

script_dir = os.path.dirname(os.path.realpath(__file__))

if not args.local:
    print('Downloading results...')
    if args.segur:
        cmd1='rsync -r segur.limsi.fr:~/autoregressive_ner/results/ {}/results/'.format(script_dir)    
    else:
        cmd1='rsync -r jean-zay.idris.fr:/gpfswork/rech/lak/utb11pp/autoregressive_ner/results/ {}/results/'.format(script_dir)
    cmd2='rsync -r slurm.lab-ia.fr:/mnt/beegfs/home/naguib/autoregressive_ner/results/ {}/results/'.format(script_dir)
    os.system(cmd1)
    os.system(cmd2)
    print('Done.')
df_few_shot, df_fully_supervised = read_jsons(os.path.join(script_dir, 'results'))

output_folder = os.path.join(script_dir, 'tabs_and_plots')
os.makedirs(output_folder, exist_ok=True)

model_order = latex_results(df_few_shot, df_fully_supervised, output_folder, model_domains, model_types, dataset_names, model_langs, model_clean_names, dataset_hierarchy, model_hierarchy)
latex_listing(df_few_shot, output_folder, model_domains, model_types, dataset_names, model_langs, model_clean_names, dataset_hierarchy, model_hierarchy)
latex_sampling(df_few_shot, dataset_names=dataset_names, model_clean_names=model_clean_names, output_folder=output_folder)
latex_models(df_few_shot, output_folder, model_domains, model_types, model_sizes, model_clean_names, model_training_data_sizes, model_training_data_languages, model_reference, model_order, model_language_markers)
latex_ner_descriptions(strings)
model_numbers = {model_order[i]:i+1 for i in range(len(model_order))}
plot_data(df_few_shot, output_folder, model_domains, model_types, model_sizes, model_clean_names, model_numbers)
