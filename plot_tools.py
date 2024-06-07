import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

MARKER_SIZES = (200,4000)
def add_text(ax, i, df):
    ax.text(
        x=df.general_performance[i]-0.02 if df.model_type[i] == 'Masked' else df.general_performance[i]-0.01,
        # y=scatter_df.clinical_performance[i]-0.02,
        y=df.clinical_performance[i]-0.015,
        s=df.model_number[i],
        fontdict=dict(color='black',size=10),
        # bbox=dict(facecolor='white',alpha=0.5,edgecolor='black',boxstyle='round,pad=0.5')
    )

def plot_data(df, output_folder, model_domains, model_types, model_sizes, model_clean_names, model_numbers, print_results=False):
    df = df[df['listing'] == False]
    df = df[df['partition_seed'] == 1]
    df = df[df['training_size'] == 100]
    scatter_data = []
    for language, df_lang in df.groupby('lang'):
        if print_results:
            print(f'================{language}================')
        for model_name, model_performance in df_lang.groupby('model_name'):
            model_name = model_name.split('/')[-1]
            model_domain = model_domains[model_name]
            model_type = model_types[model_name]
            model_size = model_sizes[model_name]
            general_performance = model_performance[model_performance['dataset_domain'] == 'General']['f1'].mean()
            clinical_performance = model_performance[model_performance['dataset_domain'] == 'Clinical']['f1'].mean()
            if print_results:
                print(f'------------{model_clean_names[model_name]}------------')
                print(model_performance[['dataset_name','f1']])
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
    scatter_df['model_number'] = scatter_df['model_name'].map(lambda x: model_numbers[x])
    #sort by model number
    scatter_df = scatter_df.sort_values('model_number')

    for lang in ['english', 'french', 'spanish']:
        # scatter_df_lang = scatter_df[scatter_df.model_language == lang]
        # resetting the index to avoid the warning
        plt.figure()
        scatter_df_lang = scatter_df[scatter_df.model_language == lang].reset_index(drop=True)
        sns.scatterplot(
            data=scatter_df_lang,
            x="general_performance",
            y="clinical_performance",
            size="model_size",
            style="model_type",
            hue="model_domain", 
            hue_order=["General","", "Clinical"], 
            palette=['#1f77b4', '#ff7f0e', '#2ca02c'], 
            
            markers={"Causal": 'o', "Masked": 's'},
            sizes=MARKER_SIZES,
            alpha=0.8
            )

        #set limits
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('General Performance')
        plt.ylabel('Clinical Performance')
        #plot model names
        for i in range(len(scatter_df_lang)):
            if scatter_df_lang.general_performance[i]>0 and scatter_df_lang.clinical_performance[i]>0:
                add_text(plt.gca(), i, scatter_df_lang)
        #make the figure square
        plt.gcf().set_figheight(6)
        plt.gcf().set_figwidth(6)

        h, l = plt.gca().get_legend_handles_labels()
        #remove the size legend
        h = h[:3] + h[-2:]
        l = l[:3] + l[-2:]
        
        ax = plt.gca()
        ax.legend(h, l)

        #show and save
        plt.savefig(os.path.join(output_folder, f'scatter_{lang}.png'))