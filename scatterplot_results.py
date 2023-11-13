import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from read_jsons import read_jsons, model_domains, model_types, model_sizes, model_clean_names, dataset_names, model_langs
from read_jsons import dataset_hierarchy, model_hierarchy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--local', action='store_true')
args = parser.parse_args()

script_dir = os.path.dirname(os.path.realpath(__file__))

if not args.local:
    print('Downloading results...')
    cmd1='rsync -r jean-zay.idris.fr:/gpfswork/rech/lak/utb11pp/autoregressive_ner/results/ {}/results/'.format(script_dir)
    cmd2='rsync -r slurm-int:/mnt/beegfs/home/naguib/autoregressive_ner/results/ {}/results/'.format(script_dir)
    os.system(cmd1)
    os.system(cmd2)
    print('Done.')
df = read_jsons(os.path.join(script_dir, 'results'))
#clear DrBERT
df = df[~df['model_name'].str.contains('DrBERT-7GB')]

scatter_data = []
for language, df_lang in df.groupby('lang'):
    print(f'================{language}================')
    for model_name, model_performance in df_lang.groupby('model_name'):
        model_name = model_name.split('/')[-1]
        model_domain = model_domains[model_name]
        model_type = model_types[model_name]
        model_size = model_sizes[model_name]
        print(f'------------{model_clean_names[model_name]}------------')
        print(model_performance[['dataset_name','f1']])
        general_performance = model_performance[model_performance['dataset_domain'] == 'General']['f1'].mean()
        clinical_performance = model_performance[model_performance['dataset_domain'] == 'Clinical']['f1'].mean()
        print(f'=== General: {general_performance} - Clinical: {clinical_performance} ===')
        print()
        scatter_data.append({
            'model_name': model_clean_names[model_name],
            'model_domain': model_domain,
            'model_type': model_type,
            'model_size': model_size,
            'general_performance': general_performance,
            'clinical_performance': clinical_performance,
            'model_language': language,
        })

scatter_df = pd.DataFrame(scatter_data)

output_folder = os.path.join(script_dir, 'result_tabs_and_plots')
os.makedirs(output_folder, exist_ok=True)

for lang in ['en', 'fr', 'es']:
    plt.clf()
    sns.scatterplot(
        x="general_performance",
        y="clinical_performance",
        hue="model_domain",
        hue_order=["General","", "Clinical"],
        size="model_size",
        style="model_type",
        style_order=["Causal","", "Masked"],
        sizes=(500,10000),
        data=scatter_df.query('model_language == @lang'),
    )
    for i in range(len(scatter_df)):
        if scatter_df.model_language[i] == lang:
            plt.text(
                x=scatter_df.general_performance[i]+0.01,
                y=scatter_df.clinical_performance[i]+0.01,
                s=scatter_df.model_name[i],
                fontdict=dict(color='black',size=10),
                bbox=dict(facecolor='white',alpha=0.5,edgecolor='black',boxstyle='round,pad=0.5')
            )

    plt.title("General vs Clinical NER Performance of Language Models pretrained on " + lang.upper())
    plt.xlabel("General Performance")
    plt.ylabel("Clinical Performance")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., scatterpoints=1, labelspacing=3)
    plt.gcf().set_size_inches(15, 10)
    plt.savefig(os.path.join(output_folder, f'{lang}_scatterplot.png'), dpi=300)

df_table = df.pivot(index='model_name', columns='dataset_name', values='f1')
#unsafe way to get the order of the datasets.. TODO: find a better way
ordered_datasets = {k:list(v['General'].values())+list(v['Clinical'].values()) for k,v in dataset_hierarchy.items()}
df_table = df_table.rename(columns=dataset_names)
for lang in ordered_datasets:
    for dataset in ordered_datasets[lang]:
        if dataset not in df_table.columns:
            df_table[dataset] = '-'

#sort lines by model type, language and domain
df_table['model_type'] = df_table.index.map(lambda x: model_types[x])
df_table['model_language'] = df_table.index.map(lambda x: model_langs[x])
df_table['model_domain'] = df_table.index.map(lambda x: model_domains[x])
df_table.index = df_table.index.map(lambda x: model_clean_names[x])
df_table = df_table.sort_values(by=['model_type', 'model_language', 'model_domain'], ascending=[True, True, False])
df_table.drop(columns=['model_domain'], inplace=True)
df_table = df_table.fillna('-')

df_table = df_table.rename_axis(None, axis=1)
df_table = df_table.rename_axis(None, axis=0)

df_table = df_table[ordered_datasets['en'] + ordered_datasets['es'] + ordered_datasets['fr'] + ['model_type']]
#print this table as a latex table and add a top line with dataset languages and domains
latex = "\\scalebox{0.7}{\\begin{tabular}"
latex += "{ll|" + "c"*len(ordered_datasets['en']) + "|" + "c"*len(ordered_datasets['es']) + "|" + "c"*len(ordered_datasets['fr']) + "}\n"
latex += "\\toprule\n"
latex += " & & \\multicolumn{" + str(len(ordered_datasets['en'])) + "}{c|}{English} & \\multicolumn{" + str(len(ordered_datasets['es'])) + "}{c|}{Spanish} & \\multicolumn{" + str(len(ordered_datasets['fr'])) + "}{c}{French} \\\\\n"
latex += "\\cmidrule{3-" + str(len(ordered_datasets['en'])+2) + "} \\cmidrule{" + str(len(ordered_datasets['en'])+3) + "-" + str(len(ordered_datasets['en'])+len(ordered_datasets['es'])+2) + "} \\cmidrule{" + str(len(ordered_datasets['en'])+len(ordered_datasets['es'])+3) + "-" + str(len(ordered_datasets['en'])+len(ordered_datasets['es'])+len(ordered_datasets['fr'])+2) + "}\n"
latex += "Type & Model & " + " & ".join(ordered_datasets['en']) + " & " + " & ".join(ordered_datasets['es']) + " & " + " & ".join(ordered_datasets['fr']) + " \\\\\n"
latex += "\\midrule\n"
n_causal = len(df_table[df_table.model_type == 'Causal'])
latex += "\\multirow{" + str(n_causal) + "}{*}{Causal} & " + df_table.index[0] + " & " + " & ".join([str(x) for x in df_table.iloc[0][:-1]]) + " \\\\\n"
for model_name, row in df_table.iloc[1:n_causal].iterrows():
    latex += " & " + model_name + " & " + " & ".join([str(x) for x in row[:-1]]) + " \\\\\n"
latex += "\\midrule\n"
n_masked = len(df_table[df_table.model_type == 'Masked'])
latex += "\\multirow{" + str(n_masked) + "}{*}{Masked} & " + df_table.index[n_causal] + " & " + " & ".join([str(x) for x in df_table.iloc[n_causal][:-1]]) + " \\\\\n"
for model_name, row in df_table.iloc[n_causal+1:].iterrows():
    latex += " & " + model_name + " & " + " & ".join([str(x) for x in row[:-1]]) + " \\\\\n"
latex += "\\bottomrule\n"
latex += "\\end{tabular}}"
with open(os.path.join(output_folder, 'results_table.tex'), 'w') as f:
    f.write(latex)