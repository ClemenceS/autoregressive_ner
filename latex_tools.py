import os
from itertools import product

def latex_results(df, df_fully_sup, output_folder, model_domains, model_types, dataset_names, model_langs, model_clean_names, dataset_hierarchy, model_hierarchy, output_name='results_table.tex'):
    df = df[df.listing == False]
    df = df[df.partition_seed == 1]
    df = df[df.training_size == 100]
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
    latex = "\\begin{table}[ht]\n"
    latex += "\\centering\n"
    latex += "\\scalebox{0.7}{\\begin{tabular}"
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
    #to print floats up to 3 decimals, replace str(x) by "{:.3f}".format(x)
    latex += "\\multirow{" + str(n_causal) + "}{*}{\\rotatebox[origin=c]{90}{Causal}} & 1 & " + df_table.index[0] + " & " + " & ".join(["{:.3f}".format(x) if x!='-' else x for x in df_table.iloc[0][:-1]]) + " \\\\\n"
    for i, (model_name, row) in enumerate(df_table.iloc[1:n_causal].iterrows()):
        latex += " & " + str(i+2) + " & " + model_name.replace('_','\\_') + " & " + " & ".join(["{:.3f}".format(x) if x!='-' else x for x in row[:-1]]) + " \\\\\n"
    latex += "\\midrule\n"
    n_masked = len(df_table[df_table.model_type == 'Masked'])
    latex += "\\multirow{" + str(n_masked) + "}{*}{\\rotatebox[origin=c]{90}{Masked}} & "+ str(n_causal+1) + " & " + df_table.index[n_causal] + " & " + " & ".join(["{:.3f}".format(x) if x!='-' else x for x in df_table.iloc[n_causal][:-1]]) + " \\\\\n"
    for i, (model_name, row) in enumerate(df_table.iloc[n_causal+1:].iterrows()):
        latex += " & " + str(i+n_causal+2) + " & " + model_name.replace('_','\\_') + " & " + " & ".join(["{:.3f}".format(x) if x!='-' else x for x in row[:-1]]) + " \\\\\n"
    latex += "\\midrule\n"
    latex += "\\midrule\n"
    latex += "\\multicolumn{" + str(n_datasets+3) + "}{l}{\\textit{Masked fully-supervised (skyline)}} \\\\\n"
    latex += "\\midrule\n"
    for i, (model_name, row) in enumerate(df_fully_sup_table.iterrows()):
        latex += " & & " + model_name.replace('_','\\_') + " & " + " & ".join(["{:.3f}".format(x) if x!='-' else x for x in row]) + " \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}}"
    #\caption{This table presents the (macro?)-F1 obtained from few-shot experiments. skyline results are obtained using all training data available instead of the few-shot setting.}
    latex += "\\caption{This table presents the micro-F1 obtained from few-shot experiments. Skyline results are obtained using all training data available instead of the few-shot setting.}\n"
    latex += "\\label{tab:results}\n"
    latex += "\\end{table}\n"
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
        return str(round(x/1000, 2)) + ' thousand'
    if x < 1000000000:
        return str(round(x/1000000, 2)) + ' million'
    return str(round(x/1000000000, 2)) + ' billion'

def million_notation(x):
    #X is a float, usually in millions or billions
    #return a string with the number divided by 1 million
    if x == '-':
        return '-'
    return str(round(x/1000000))

def latex_models(df, output_folder, model_domains, model_types, model_sizes, model_clean_names, model_training_data_sizes, model_training_data_languages, model_reference, model_order, model_language_markers):
    #get a row from each model type
    df_table = df['model_name'].drop_duplicates().to_frame()
    df_table = df_table.set_index('model_name')
    df_table['model_type'] = df_table.index.map(lambda x: model_types[x])
    df_table['model_domain'] = df_table.index.map(lambda x: model_domains[x])
    df_table['model_size'] = df_table.index.map(lambda x: model_sizes[x])
    df_table['model_size'] = df_table['model_size'].map(million_notation)
    df_table['model_training_data_size'] = df_table.index.map(lambda x: model_training_data_sizes[x])
    df_table['model_training_data_languages'] = df_table.index.map(lambda x: model_training_data_languages[x])
    df_table['model_reference'] = df_table.index.map(lambda x: model_reference[x])
    df_table['model_language_markers'] = df_table.index.map(lambda x: ''.join(['\\textsuperscript{\\texttt{['+lang+']}}' if lang!="*" else lang for lang in model_language_markers[x]]))
    df_table.index = df_table.index.map(lambda x: model_clean_names[x])
    df_table['model_latex_name'] = df_table.index.map(lambda x: x.replace('_','\\_'))
    #add reference to the model name if available
    df_table['model_latex_name'] = df_table['model_latex_name'] + df_table['model_language_markers'] + df_table['model_reference'].map(lambda x: ' \\cite{' + x + '}' if x != '-' else '')
    #sort by type and by then by index
    df_table.index.name = 'model_name'
    df_table = df_table.sort_values(by=['model_type', 'model_name'], ascending=[True, True])
    n_causal = len(df_table[df_table.model_type == 'Causal'])
    n_masked = len(df_table[df_table.model_type == 'Masked'])
    #order the model by model_order
    df_table['model_order'] = df_table.index.map(lambda x: model_order.index(x))
    df_table = df_table.sort_values(by=['model_order'])    
    #print a table with the model names
    latex = "\\begin{table}[ht]\n"
    latex += "\\centering\n"
    latex += "\\scalebox{0.7}{\\begin{tabular}"
    # latex += "{clllll}\n"
    latex += "{cllrrl}\n"
    latex += "\\toprule\n"
    latex += "& \# & Model & \makecell{Number of\\\\ parameters\\\\(in millions)} & \makecell{Training data\\\\ size} & \makecell{Training corpus\\\\ language(s) and details} \\\\\n"
    latex += "\\midrule\n"
    # latex += "\\multirow{" + str(n_causal) + "}{*}{\\rotatebox[origin=c]{90}{Causal}} & 1 & " + df_table.index[0] + " & " + df_table.iloc[0]['model_size'] + " & " + df_table.iloc[0]['model_training_data_size'] + " & " + df_table.iloc[0]['model_training_data_languages'] + " \\\\\n"
    latex += "\\multirow{" + str(n_causal) + "}{*}{\\rotatebox[origin=c]{90}{Causal}} & 1 & " + df_table['model_latex_name'][0] + " & " + df_table.iloc[0]['model_size'] + " & " + df_table.iloc[0]['model_training_data_size'] + " & " + df_table.iloc[0]['model_training_data_languages'] + " \\\\\n"
    for i, (model_name, row) in enumerate(df_table.iloc[1:n_causal].iterrows()):
        latex += " & " + str(i+2) + " & " + row['model_latex_name'] + " & " + row['model_size'] + " & " + row['model_training_data_size'] + " & " + row['model_training_data_languages'] + " \\\\\n"
    latex += "\\midrule\n"
    # latex += "\\multirow{" + str(n_masked) + "}{*}{\\rotatebox[origin=c]{90}{Masked}} & " + str(n_causal+1) + " & " + df_table.index[n_causal] + " & " + df_table.iloc[n_causal]['model_size'] + " & " + df_table.iloc[n_causal]['model_training_data_size'] + " & " + df_table.iloc[n_causal]['model_training_data_languages'] + " \\\\\n"
    latex += "\\multirow{" + str(n_masked) + "}{*}{\\rotatebox[origin=c]{90}{Masked}} & " + str(n_causal+1) + " & " + df_table['model_latex_name'][n_causal] + " & " + df_table.iloc[n_causal]['model_size'] + " & " + df_table.iloc[n_causal]['model_training_data_size'] + " & " + df_table.iloc[n_causal]['model_training_data_languages'] + " \\\\\n"
    for i, (model_name, row) in enumerate(df_table.iloc[n_causal+1:].iterrows()):
        latex += " & " + str(i+n_causal+2) + " & " + row['model_latex_name'] + " & " + row['model_size'] + " & " + row['model_training_data_size'] + " & " + row['model_training_data_languages'] + " \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}}\n"
    latex += "\\caption{Characterization of the language models used in our experiments in terms of parameters and training corpus. CLMs marked with * are fine-tuned versions of other CLMs. MLMs marked with \\textsuperscript{\\texttt{[en]}} (respectively \\textsuperscript{\\texttt{[fr]}}, \\textsuperscript{\\texttt{[es]}}) are mainly trained on English (respectively French, Spanish).}\n"
    latex += "\\label{tab:LM_features}\n"
    latex += "\\end{table}\n"
    with open(os.path.join(output_folder, 'model_names_table.tex'), 'w') as f:
        f.write(latex)

def latex_listing(df, output_folder, model_domains, model_types, dataset_names, model_langs, model_clean_names, dataset_hierarchy, model_hierarchy):
    df_base = df[df.listing == False]
    df_base = df_base[df_base.training_size == 100]
    df_base = df_base[df_base.partition_seed == 1]
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
    
    latex = "\\begin{table}[ht]\n"
    latex += "\\centering\n"
    latex += "\\scalebox{0.7}{%\n"
    latex += "\\begin{tabular}"
    latex += "{l|" + "c"*len(ordered_datasets['en']) + "|" + "c"*len(ordered_datasets['fr']) + "|" + "c"*len(ordered_datasets['es']) + "}\n"
    latex += " & \\multicolumn{" + str(len(ordered_datasets['en'])) + "}{c|}{English} & \\multicolumn{" + str(len(ordered_datasets['fr'])) + "}{c|}{French} & \\multicolumn{" + str(len(ordered_datasets['es'])) + "}{c}{Spanish} \\\\\n"
    latex += "\\cmidrule{2-" + str(len(ordered_datasets['en'])+1) + "} \\cmidrule{" + str(len(ordered_datasets['en'])+2) + "-" + str(len(ordered_datasets['en'])+len(ordered_datasets['fr'])+1) + "} \\cmidrule{" + str(len(ordered_datasets['en'])+len(ordered_datasets['fr'])+2) + "-" + str(len(ordered_datasets['en'])+len(ordered_datasets['fr'])+len(ordered_datasets['es'])+1) + "}\n"
    latex += (" Model & " + " & ".join(ordered_datasets['en']) + " & " + " & ".join(ordered_datasets['fr']) + " & " + " & ".join(ordered_datasets['es']) + " \\\\\n").replace('-en', '').replace('-fr', '').replace('-es', '')
    latex += "\\midrule\n"
    latex += "\\midrule\n"
    n_datasets = len(ordered_datasets['en']) + len(ordered_datasets['fr']) + len(ordered_datasets['es'])
    latex += " \\multicolumn{" + str(n_datasets+1) + "}{l}{\\textit{Listing prompts}} \\\\\n"
    latex += "\\midrule\n"
    latex += df_table.index[0] + " & " + " & ".join(["{:.3f}".format(x) if x!='-' else x for x in df_table.iloc[0][:-1]]) + " \\\\\n"
    for model_name, row in df_table.iloc[1:].iterrows():
        latex += model_name.replace('_','\\_') + " & " + " & ".join(["{:.3f}".format(x) if x!='-' else x for x in row[:-1]]) + " \\\\\n"
    latex += "\\midrule\n"
    latex += "\\midrule\n"
    latex += "\\multicolumn{" + str(n_datasets+1) + "}{l}{\\textit{Tagging prompts}} \\\\\n"
    latex += "\\midrule\n"
    latex += df_base_table.index[0] + " & " + " & ".join(["{:.3f}".format(x) if x!='-' else x for x in df_base_table.iloc[0][:-1]]) + " \\\\\n"
    for model_name, row in df_base_table.iloc[1:].iterrows():
        latex += model_name.replace('_','\\_') + " & " + " & ".join(["{:.3f}".format(x) if x!='-' else x for x in row[:-1]]) + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}}"
    # latex += "\\caption{This table presents the F1 obtained from the listing and tagging prompts.}\n"
    latex += "\\caption{F1 scores obtained with the listing and tagging prompts.}\n"
    latex += "\\label{tab:listing}\n"
    latex += "\\end{table}\n"
    with open(os.path.join(output_folder, "listing.tex"), 'w') as f:
        f.write(latex)

def latex_sampling(df, dataset_names, model_clean_names, output_folder):
    n_values = [25, 50, 100]
    p_values = [1, 2, 3]
    studied_models = ['Mistral-7B-v0.1', "xlm-roberta-large"]
    studied_datasets = ['conll2003', 'n2c2']
    df = df[df.listing == False]
    df = df[df.training_size.isin(n_values)]
    df = df[df.partition_seed.isin(p_values)]
    df = df[df.model_name.isin(studied_models)]
    df = df[df.dataset_name.isin(studied_datasets)]
    df['dataset_partition'] = df['dataset_name'] + ' ' + df['partition_seed'].astype(str)
    df['model_training_size'] = df['model_name'] + ' ' + df['training_size'].astype(str)
    df_table = df.pivot(index='model_training_size', columns='dataset_partition', values='f1')
    df_table = df_table.reindex(sorted(df_table.columns), axis=1)
    
    #if a model/training size is missing, add it with a NaN value
    for model in studied_models:
        for training_size in n_values:
            if model + ' ' + str(training_size) not in df_table.index:
                df_table.loc[model + ' ' + str(training_size)] = '-'
    #if a dataset/partition is missing, add it with a NaN value
    for dataset in studied_datasets:
        for partition in p_values:
            if dataset + ' ' + str(partition) not in df_table.columns:
                df_table[dataset + ' ' + str(partition)] = '-'
    
    #sort lines by model (alphabetical order) and then by decreasing training size
    df_table['model_name'] = df_table.index.map(lambda x: x.split()[0])
    df_table['training_size'] = df_table.index.map(lambda x: int(x.split()[1]))
    
    df_table = df_table.sort_values(by=['training_size', 'model_name'], ascending=[False, True])
    df_table.drop(columns=['training_size', 'model_name'], inplace=True)
    df_table = df_table.rename_axis(None, axis=1)
    df_table = df_table.rename_axis(None, axis=0)
    column_order = [" ".join(x) for x in product(studied_datasets, [str(x) for x in p_values])]
    df_table = df_table[column_order]
    df_table = df_table.fillna('-')
    
    latex = "\\begin{table}[ht]\n"
    latex += "\\centering\n"
    latex += "\\scalebox{1}{\\begin{tabular}"
    latex += "{l|" + "c"*len(p_values) + "|" + "c"*len(p_values) + "}\n"
    latex += "\\toprule\n"
    # latex += "\\multicolumn{" + str(len(studied_datasets)) + "}{c}{" + dataset_names[studied_datasets[0]] + "} & \\multicolumn{" + str(len(studied_datasets)) + "}{c}{" + dataset_names[studied_datasets[1]] + "} \\\\\n"
    latex += "& \\multicolumn{" + str(len(p_values)) + "}{c|}{" + dataset_names[studied_datasets[0]] + "} & \\multicolumn{" + str(len(p_values)) + "}{c}{" + dataset_names[studied_datasets[1]] + "} \\\\\n"
    latex += "\\midrule\n"
    latex += "\\multicolumn{" + str(len(studied_datasets)*2+1) + "}{c}{\\textit{100 annotated instances}} \\\\\n"
    latex += "\\midrule\n"
    latex += " & " + " & ".join(["\\textit{p="+str(p)+'}' for p in p_values]) + " & " + " & ".join(["\\textit{p="+str(p)+'}' for p in p_values]) + " \\\\\n"
    latex += "\\midrule\n"
    latex += "\\midrule\n"
    for i, (model_name, row) in enumerate(df_table.iloc[:len(studied_models)].iterrows()):
        latex += model_clean_names[model_name.split()[0]].replace('_','\\_') + " & " + " & ".join([str(x) for x in row[:len(p_values)]]) + " & " + " & ".join([str(x) for x in row[len(p_values):]]) + " \\\\\n"
    latex += "\\midrule\n"
    latex += "\\midrule\n"
    latex += "\\multicolumn{" + str(len(studied_datasets)*2+1) + "}{c}{\\textit{50 annotated instances}} \\\\\n"
    latex += "\\midrule\n"
    latex += " & " + " & ".join(["\\textit{p="+str(p)+'}' for p in p_values]) + " & " + " & ".join(["\\textit{p="+str(p)+'}' for p in p_values]) + " \\\\\n"
    latex += "\\midrule\n"
    latex += "\\midrule\n"
    for i, (model_name, row) in enumerate(df_table.iloc[len(studied_models):-len(studied_models)].iterrows()):
        latex += model_clean_names[model_name.split()[0]].replace('_','\\_') + " & " + " & ".join([str(x) for x in row[:len(p_values)]]) + " & " + " & ".join([str(x) for x in row[len(p_values):]]) + " \\\\\n"
    latex += "\\midrule\n"
    latex += "\\midrule\n"
    latex += "\\multicolumn{" + str(len(studied_datasets)*2+1) + "}{c}{\\textit{25 annotated instances}} \\\\\n"
    latex += "\\midrule\n"
    latex += " & " + " & ".join(["\\textit{p="+str(p)+'}' for p in p_values]) + " & " + " & ".join(["\\textit{p="+str(p)+'}' for p in p_values]) + " \\\\\n"
    latex += "\\midrule\n"
    latex += "\\midrule\n"
    for i, (model_name, row) in enumerate(df_table.iloc[-len(studied_models):].iterrows()):
        latex += model_clean_names[model_name.split()[0]].replace('_','\\_') + " & " + " & ".join([str(x) for x in row[:len(p_values)]]) + " & " + " & ".join([str(x) for x in row[len(p_values):]]) + " \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}}"
    latex += "\\caption{F1 scores obtained over experiments with different training samples and different training sample sizes.}\n"
    latex += "\\label{tab:sampling}\n"
    latex += "\\end{table}\n"

    with open(os.path.join(output_folder, "sampling.tex"), 'w') as f:
        f.write(latex)
