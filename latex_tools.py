import os

def latex_data(df, output_folder, model_domains, model_types, dataset_names, model_langs, model_clean_names, dataset_hierarchy, model_hierarchy):
    df_table = df.pivot(index='model_name', columns='dataset_name', values='f1')
    lang_shortname = {"english": "en", "french": "fr", "spanish": "es", "all": "all"}
    #unsafe way to get the order of the datasets.. TODO: find a better way
    ordered_datasets = {lang_shortname[k]:list(v['General'].values())+list(v['Clinical'].values()) for k,v in dataset_hierarchy.items()}
    df_table = df_table.rename(columns=dataset_names)
    for lang in ordered_datasets:
        for dataset in ordered_datasets[lang]:
            if dataset not in df_table.columns:
                df_table[dataset] = '-'
    for model in model_hierarchy['Causal']['General']+model_hierarchy['Causal']['Clinical']+model_hierarchy['Masked']['General']+model_hierarchy['Masked']['Clinical']:
        if model['name'] not in df_table.index:
            df_table.loc[model['name']] = '-'

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

    df_table = df_table[ordered_datasets['en'] + ordered_datasets['fr'] + ordered_datasets['es'] + ['model_type']]
    latex = "\\scalebox{0.7}{\\begin{tabular}"
    latex += "{ll|" + "c"*len(ordered_datasets['en']) + "|" + "c"*len(ordered_datasets['fr']) + "|" + "c"*len(ordered_datasets['es']) + "}\n"
    # latex += "\\toprule\n"
    latex += " & & \\multicolumn{" + str(len(ordered_datasets['en'])) + "}{c|}{English} & \\multicolumn{" + str(len(ordered_datasets['fr'])) + "}{c|}{French} & \\multicolumn{" + str(len(ordered_datasets['es'])) + "}{c}{Spanish} \\\\\n"
    latex += "\\cmidrule{3-" + str(len(ordered_datasets['en'])+2) + "} \\cmidrule{" + str(len(ordered_datasets['en'])+3) + "-" + str(len(ordered_datasets['en'])+len(ordered_datasets['fr'])+2) + "} \\cmidrule{" + str(len(ordered_datasets['en'])+len(ordered_datasets['fr'])+3) + "-" + str(len(ordered_datasets['en'])+len(ordered_datasets['fr'])+len(ordered_datasets['es'])+2) + "}\n"
    latex += ("& Model & " + " & ".join(ordered_datasets['en']) + " & " + " & ".join(ordered_datasets['fr']) + " & " + " & ".join(ordered_datasets['es']) + " \\\\\n").replace('-en', '').replace('-fr', '').replace('-es', '')
    latex += "\\midrule\n"
    n_causal = len(df_table[df_table.model_type == 'Causal'])
    latex += "\\multirow{" + str(n_causal) + "}{*}{\\rotatebox[origin=c]{90}{Causal}} & 1- " + df_table.index[0] + " & " + " & ".join([str(x) for x in df_table.iloc[0][:-1]]) + " \\\\\n"
    for i, (model_name, row) in enumerate(df_table.iloc[1:n_causal].iterrows()):
        latex += " & " + str(i+2) + "- " + model_name.replace('_','\\_') + " & " + " & ".join([str(x) for x in row[:-1]]) + " \\\\\n"
    latex += "\\midrule\n"
    n_masked = len(df_table[df_table.model_type == 'Masked'])
    latex += "\\multirow{" + str(n_masked) + "}{*}{\\rotatebox[origin=c]{90}{Masked}} & "+ str(n_causal+1) + "- " + df_table.index[n_causal] + " & " + " & ".join([str(x) for x in df_table.iloc[n_causal][:-1]]) + " \\\\\n"
    for i, (model_name, row) in enumerate(df_table.iloc[n_causal+1:].iterrows()):
        latex += " & " + str(i+n_causal+2) + "- " + model_name.replace('_','\\_') + " & " + " & ".join([str(x) for x in row[:-1]]) + " \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}}"
    with open(os.path.join(output_folder, 'results_table.tex'), 'w') as f:
        f.write(latex)
    
    return df_table.index.tolist()