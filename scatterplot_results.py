import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from read_jsons import read_jsons, model_domains, model_types, model_sizes, model_clean_names, dataset_names


df = read_jsons('results')

#clean camembert-base
df = df[df['model_name'] != 'camembert-base']

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

os.makedirs('tables', exist_ok=True)
os.makedirs('figures', exist_ok=True)

for lang in ['en', 'fr', 'es']:
    sns.scatterplot(
        x="general_performance",
        y="clinical_performance",
        hue="model_domain",
        hue_order=["General","", "Clinical"],
        size="model_size",
        style="model_type",
        style_order=["Causal","", "Masked"],
        sizes=(500,10000),
        data=scatter_df.query('model_language == "fr"'),
    )
    for i in range(len(scatter_df)):
        if scatter_df.model_language[i] == 'fr':
            plt.text(
                x=scatter_df.general_performance[i]+0.01,
                y=scatter_df.clinical_performance[i]+0.01,
                s=scatter_df.model_name[i],
                fontdict=dict(color='black',size=10),
                bbox=dict(facecolor='white',alpha=0.5,edgecolor='black',boxstyle='round,pad=0.5')
            )

    plt.title("General vs Clinical NER Performance of Language Models")
    plt.xlabel("General Performance")
    plt.ylabel("Clinical Performance")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., scatterpoints=1, labelspacing=3)
    plt.gcf().set_size_inches(15, 10)
    plt.savefig(f'figures/{lang}_scatterplot.png', dpi=300)

    df_lang = df[df['lang'] == lang]
    df_lang = df_lang.sort_values(by=['dataset_domain', 'dataset_name'])
    df_lang = df_lang[['model_name', 'model_clean_name', 'dataset_name', 'f1']]
    df_lang = df_lang.pivot(index='model_name', columns='dataset_name', values='f1')
    df_lang['model_type'] = df_lang.index.map(lambda x: model_types[x])
    df_lang['model_domain'] = df_lang.index.map(lambda x: model_domains[x])
    df_lang.index = df_lang.index.map(lambda x: model_clean_names[x])

    df_lang = df_lang.sort_values(by=['model_type', 'model_domain'], ascending=[True, False])
    df_lang.drop(columns=['model_type', 'model_domain'], inplace=True)
    df_lang = df_lang.rename_axis(None, axis=1)

    df_lang = df_lang.rename(columns=dataset_names)
    df_lang = df_lang.rename_axis(None, axis=1)
    df_lang = df_lang.rename_axis(None, axis=0)
    df_lang = df_lang.fillna('-')
    with open(f'tables/{lang}_results.tex', 'w') as f:
        f.write(df_lang.to_latex(escape=False))
