import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

MARKER_SIZES = (100,2000)
def add_text(ax, i, scatter_df):
    ax.text(
        x=scatter_df.general_performance[i]-0.02 if scatter_df.model_type[i] == 'Masked' else scatter_df.general_performance[i]-0.01,
        y=scatter_df.clinical_performance[i]-0.02,
        s=scatter_df.model_number[i],
        fontdict=dict(color='black',size=10),
        # bbox=dict(facecolor='white',alpha=0.5,edgecolor='black',boxstyle='round,pad=0.5')
    )
    # e = 0.1
    # below = True
    # #if 3 or more models are too close, don't plot any
    # if len(scatter_df[(scatter_df.general_performance>scatter_df.general_performance[i]-e) & (scatter_df.general_performance<scatter_df.general_performance[i]+e) & (scatter_df.clinical_performance>scatter_df.clinical_performance[i]-e) & (scatter_df.clinical_performance<scatter_df.clinical_performance[i]+e)]) > 3:
    #     below = False
    # if below:
    #     ax.text(
    #         x=scatter_df.general_performance[i]-0.07,
    #         y=scatter_df.clinical_performance[i]-0.1,
    #         s=scatter_df.model_name[i],
    #         fontdict=dict(color='black',size=10),
    #         bbox=dict(facecolor='white',alpha=0.5,edgecolor='black',boxstyle='round,pad=0.5')
    #     )
    # else:
    #     ax.text(
    #         x=scatter_df.general_performance[i]-0.07,
    #         y=scatter_df.clinical_performance[i]+0.05,
    #         s=scatter_df.model_name[i],
    #         fontdict=dict(color='black',size=10),
    #         bbox=dict(facecolor='white',alpha=0.5,edgecolor='black',boxstyle='round,pad=0.5')
    #     )
    pass

def plot_data(df, output_folder, model_domains, model_types, model_sizes, model_clean_names, model_numbers):
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
    # model_numbers = {}
    # for i, model_name in enumerate(scatter_df.model_name.unique()):
    #     model_numbers[model_name] = i+1

    scatter_df['model_number'] = scatter_df['model_name'].map(lambda x: model_numbers[x])


    grid = sns.FacetGrid(
        scatter_df,
        col="model_language", 
        hue="model_domain", 
        hue_order=["General","", "Clinical"], 
        palette=['#1f77b4', '#ff7f0e', '#2ca02c'], 
        col_wrap=3, 
        height=4, 
        aspect=1.5
        )
    grid.map_dataframe(
        sns.scatterplot,
        "general_performance",
        "clinical_performance",
        size="model_size",
        style="model_type",
        markers={"Causal": 'o', "Masked": 's'},
        sizes=MARKER_SIZES,
        alpha=0.8
        )
    grid.set(xlim=(0, 1), ylim=(0, 1))
    grid.set_titles("Language: {col_name}")
    grid.set_axis_labels("General Performance", "Clinical Performance")
    # grid.add_legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., scatterpoints=1, labelspacing=3)
    grid.fig.tight_layout(w_pad=1)
    grid.fig.subplots_adjust(top=0.9)
    grid.fig.suptitle("General vs Clinical NER Performance of Language Models")
    #plot model names
    for ax in grid.axes.flatten():
        for i in range(len(scatter_df)):
            if scatter_df.model_language[i] == ax.get_title().split(': ')[-1]:
                if scatter_df.general_performance[i]>0 and scatter_df.clinical_performance[i]>0:
                    add_text(ax, i, scatter_df)
    #make the figure square
    grid.fig.set_figheight(5)
    grid.fig.set_figwidth(15)
    grid.savefig(os.path.join(output_folder, f'scatterplot.png'), dpi=300)