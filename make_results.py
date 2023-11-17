import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from read_results import read_jsons, model_domains, model_types, model_sizes, model_clean_names, dataset_names, model_langs
from read_results import dataset_hierarchy, model_hierarchy
from plot_tools import plot_data
from latex_tools import latex_data
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
df = read_jsons(os.path.join(script_dir, 'results'))
# df = df[~df['model_name'].str.contains('DrBERT-7GB')]
# df = df[~df['model_name'].str.contains('neo')]
# df = df[~df['model_name'].str.contains('4all')]

output_folder = os.path.join(script_dir, 'result_tabs_and_plots')
os.makedirs(output_folder, exist_ok=True)

plot_data(df, output_folder, model_domains, model_types, model_sizes, model_clean_names)
latex_data(df, output_folder, model_domains, model_types, dataset_names, model_langs, model_clean_names, dataset_hierarchy, model_hierarchy)