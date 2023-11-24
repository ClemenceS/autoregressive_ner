import os

def latex_results(df, df_fully_sup, output_folder, model_domains, model_types, dataset_names, model_langs, model_clean_names, dataset_hierarchy, model_hierarchy, output_name='results_table.tex'):
    df = df[df.listing == False]
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
    
    df_fully_sup_table = df_fully_sup.pivot(index='model_name', columns='dataset_name', values='f1')
    df_fully_sup_table = df_fully_sup_table.rename(columns=dataset_names)
    for lang in ordered_datasets:
        for dataset in ordered_datasets[lang]:
            if dataset not in df_fully_sup_table.columns:
                df_fully_sup_table[dataset] = '-'

    df_fully_sup_table['model_language'] = df_fully_sup_table.index.map(lambda x: model_langs[x])
    df_fully_sup_table.index = df_fully_sup_table.index.map(lambda x: model_clean_names[x])
    df_fully_sup_table = df_fully_sup_table.sort_values(by='model_language', ascending=True)
    df_fully_sup_table.drop(columns='model_language', inplace=True)
    df_fully_sup_table = df_fully_sup_table.fillna('-')
    df_fully_sup_table = df_fully_sup_table[ordered_datasets['en'] + ordered_datasets['fr'] + ordered_datasets['es']]
    
    latex = "\\scalebox{0.7}{\\begin{tabular}"
    latex += "{lll|" + "c"*len(ordered_datasets['en']) + "|" + "c"*len(ordered_datasets['fr']) + "|" + "c"*len(ordered_datasets['es']) + "}\n"
    # latex += "\\toprule\n"
    latex += " & & & \\multicolumn{" + str(len(ordered_datasets['en'])) + "}{c|}{English} & \\multicolumn{" + str(len(ordered_datasets['fr'])) + "}{c|}{French} & \\multicolumn{" + str(len(ordered_datasets['es'])) + "}{c}{Spanish} \\\\\n"
    latex += "\\cmidrule{3-" + str(len(ordered_datasets['en'])+2) + "} \\cmidrule{" + str(len(ordered_datasets['en'])+3) + "-" + str(len(ordered_datasets['en'])+len(ordered_datasets['fr'])+2) + "} \\cmidrule{" + str(len(ordered_datasets['en'])+len(ordered_datasets['fr'])+3) + "-" + str(len(ordered_datasets['en'])+len(ordered_datasets['fr'])+len(ordered_datasets['es'])+2) + "}\n"
    latex += ("& \# & Model & " + " & ".join(ordered_datasets['en']) + " & " + " & ".join(ordered_datasets['fr']) + " & " + " & ".join(ordered_datasets['es']) + " \\\\\n").replace('-en', '').replace('-fr', '').replace('-es', '')
    latex += "\\midrule\n"
    latex += "\\midrule\n"
    n_datasets = len(ordered_datasets['en']) + len(ordered_datasets['fr']) + len(ordered_datasets['es'])
    latex += " \\multicolumn{" + str(n_datasets+3) + "}{l}{\\textit{Few-shot approaches}} \\\\\n"
    latex += "\\midrule\n"
    n_causal = len(df_table[df_table.model_type == 'Causal'])
    latex += "\\multirow{" + str(n_causal) + "}{*}{\\rotatebox[origin=c]{90}{Causal}} & 1 & " + df_table.index[0] + " & " + " & ".join([str(x) for x in df_table.iloc[0][:-1]]) + " \\\\\n"
    for i, (model_name, row) in enumerate(df_table.iloc[1:n_causal].iterrows()):
        latex += " & " + str(i+2) + " & " + model_name.replace('_','\\_') + " & " + " & ".join([str(x) for x in row[:-1]]) + " \\\\\n"
    latex += "\\midrule\n"
    n_masked = len(df_table[df_table.model_type == 'Masked'])
    latex += "\\multirow{" + str(n_masked) + "}{*}{\\rotatebox[origin=c]{90}{Masked}} & "+ str(n_causal+1) + " & " + df_table.index[n_causal] + " & " + " & ".join([str(x) for x in df_table.iloc[n_causal][:-1]]) + " \\\\\n"
    for i, (model_name, row) in enumerate(df_table.iloc[n_causal+1:].iterrows()):
        latex += " & " + str(i+n_causal+2) + " & " + model_name.replace('_','\\_') + " & " + " & ".join([str(x) for x in row[:-1]]) + " \\\\\n"
    latex += "\\midrule\n"
    latex += "\\midrule\n"
    latex += "\\multicolumn{" + str(n_datasets+3) + "}{l}{\\textit{Masked fully-supervised (skyline)}} \\\\\n"
    latex += "\\midrule\n"
    for i, (model_name, row) in enumerate(df_fully_sup_table.iterrows()):
        latex += " & & " + model_name.replace('_','\\_') + " & " + " & ".join([str(x) for x in row]) + " \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}}"
    with open(os.path.join(output_folder, output_name), 'w') as f:
        f.write(latex)
    
    return df_table.index.tolist()

def scientific_notation(x):
    #X is a float, usually in millions or billions
    #return a string with the number in scientific notation
    if x == '-':
        return '-'
    if x < 1:
        return str(round(x, 2))
    if x < 1000:
        return str(round(x))
    if x < 1000000:
        return str(round(x/1000, 2)) + 'K'
    if x < 1000000000:
        return str(round(x/1000000, 2)) + 'M'
    return str(round(x/1000000000, 2)) + 'B'


def latex_models(df, output_folder, model_domains, model_types, model_sizes, model_clean_names, model_descriptions):
    #get a row from each model type
    df_table = df['model_name'].drop_duplicates().to_frame()
    df_table = df_table.set_index('model_name')
    df_table['model_type'] = df_table.index.map(lambda x: model_types[x])
    df_table['model_domain'] = df_table.index.map(lambda x: model_domains[x])
    df_table['model_size'] = df_table.index.map(lambda x: model_sizes[x])
    df_table['model_size'] = df_table['model_size'].map(scientific_notation)
    df_table['model_description'] = df_table.index.map(lambda x: model_descriptions[x])
    df_table.index = df_table.index.map(lambda x: model_clean_names[x])
    #sort by type and by then by index
    df_table.index.name = 'model_name'
    df_table = df_table.sort_values(by=['model_type', 'model_name'], ascending=[True, True])
    n_causal = len(df_table[df_table.model_type == 'Causal'])
    n_masked = len(df_table[df_table.model_type == 'Masked'])
    print(df_table)
    #print a table with the model names
    latex = "\\scalebox{1}{\\begin{tabular}"
    latex += "{clll}\n"
    latex += "\\toprule\n"
    latex += "& Model & \# of parameters & Brief description of the training data \\\\\n"
    latex += "\\midrule\n"
    latex += "\\multirow{" + str(n_causal) + "}{*}{\\rotatebox[origin=c]{90}{Causal}} & " + df_table.index[0] + " & " + df_table.iloc[0]['model_size'] + " & " + df_table.iloc[0]['model_description'] + " \\\\\n"
    for i, (model_name, row) in enumerate(df_table.iloc[1:n_causal].iterrows()):
        latex += " & " + model_name.replace('_','\\_') + " & " + row['model_size'] + " & " + row['model_description'] + " \\\\\n"
    latex += "\\midrule\n"
    latex += "\\multirow{" + str(n_masked) + "}{*}{\\rotatebox[origin=c]{90}{Masked}} & "+ df_table.index[n_causal] + " & " + df_table.iloc[n_causal]['model_size'] + " & " + df_table.iloc[n_causal]['model_description'] + " \\\\\n"
    for i, (model_name, row) in enumerate(df_table.iloc[n_causal+1:].iterrows()):
        latex += " & " + model_name.replace('_','\\_') + " & " + row['model_size'] + " & " + row['model_description'] + " \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}}"
    with open(os.path.join(output_folder, 'model_names_table.tex'), 'w') as f:
        f.write(latex)

def latex_listing(df, output_folder, model_domains, model_types, dataset_names, model_langs, model_clean_names, dataset_hierarchy, model_hierarchy):
    df_base = df[df.listing == False]
    df = df[df.listing == True]
    df_table = df.pivot(index='model_name', columns='dataset_name', values='f1')
    lang_shortname = {"english": "en", "french": "fr", "spanish": "es", "all": "all"}
    #unsafe way to get the order of the datasets.. TODO: find a better way
    ordered_datasets = {lang_shortname[k]:list(v['General'].values())+list(v['Clinical'].values()) for k,v in dataset_hierarchy.items()}
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
    df_table = df_table[ordered_datasets['en'] + ordered_datasets['fr'] + ordered_datasets['es'] + ['model_type']]
    
    df_base_table = df_base.pivot(index='model_name', columns='dataset_name', values='f1')
    df_base_table = df_base_table.rename(columns=dataset_names)
    for lang in ordered_datasets:
        for dataset in ordered_datasets[lang]:
            if dataset not in df_base_table.columns:
                df_base_table[dataset] = '-'
    df_base_table['model_type'] = df_base_table.index.map(lambda x: model_types[x])
    df_base_table['model_language'] = df_base_table.index.map(lambda x: model_langs[x])
    df_base_table['model_domain'] = df_base_table.index.map(lambda x: model_domains[x])
    df_base_table.index = df_base_table.index.map(lambda x: model_clean_names[x])
    df_base_table = df_base_table.sort_values(by=['model_type', 'model_language', 'model_domain'], ascending=[True, True, False])
    df_base_table.drop(columns=['model_domain'], inplace=True)
    df_base_table = df_base_table.fillna('-')


    df_base_table = df_base_table.rename_axis(None, axis=1)
    df_base_table = df_base_table.rename_axis(None, axis=0)
    df_base_table = df_base_table[ordered_datasets['en'] + ordered_datasets['fr'] + ordered_datasets['es'] + ['model_type']]

    #select only the models that are in the listing
    df_base_table = df_base_table.loc[df_table.index]
    
    latex = "\\scalebox{0.7}{\\begin{tabular}"
    latex += "{l|" + "c"*len(ordered_datasets['en']) + "|" + "c"*len(ordered_datasets['fr']) + "|" + "c"*len(ordered_datasets['es']) + "}\n"
    latex += " & \\multicolumn{" + str(len(ordered_datasets['en'])) + "}{c|}{English} & \\multicolumn{" + str(len(ordered_datasets['fr'])) + "}{c|}{French} & \\multicolumn{" + str(len(ordered_datasets['es'])) + "}{c}{Spanish} \\\\\n"
    latex += "\\cmidrule{2-" + str(len(ordered_datasets['en'])+1) + "} \\cmidrule{" + str(len(ordered_datasets['en'])+2) + "-" + str(len(ordered_datasets['en'])+len(ordered_datasets['fr'])+1) + "} \\cmidrule{" + str(len(ordered_datasets['en'])+len(ordered_datasets['fr'])+2) + "-" + str(len(ordered_datasets['en'])+len(ordered_datasets['fr'])+len(ordered_datasets['es'])+1) + "}\n"
    latex += (" Model & " + " & ".join(ordered_datasets['en']) + " & " + " & ".join(ordered_datasets['fr']) + " & " + " & ".join(ordered_datasets['es']) + " \\\\\n").replace('-en', '').replace('-fr', '').replace('-es', '')
    latex += "\\midrule\n"
    latex += "\\midrule\n"
    n_datasets = len(ordered_datasets['en']) + len(ordered_datasets['fr']) + len(ordered_datasets['es'])
    latex += " \\multicolumn{" + str(n_datasets+3) + "}{l}{\\textit{Listing prompts}} \\\\\n"
    latex += "\\midrule\n"
    latex += df_table.index[0] + " & " + " & ".join([str(x) for x in df_table.iloc[0][:-1]]) + " \\\\\n"
    for model_name, row in df_table.iloc[1:].iterrows():
        latex += model_name.replace('_','\\_') + " & " + " & ".join([str(x) for x in row[:-1]]) + " \\\\\n"
    latex += "\\midrule\n"
    latex += "\\midrule\n"
    latex += "\\multicolumn{" + str(n_datasets+3) + "}{l}{\\textit{Tagging prompts}} \\\\\n"
    latex += "\\midrule\n"
    latex += df_base_table.index[0] + " & " + " & ".join([str(x) for x in df_base_table.iloc[0][:-1]]) + " \\\\\n"
    for model_name, row in df_base_table.iloc[1:].iterrows():
        latex += model_name.replace('_','\\_') + " & " + " & ".join([str(x) for x in row[:-1]]) + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}}"
    with open(os.path.join(output_folder, "listing.tex"), 'w') as f:
        f.write(latex)