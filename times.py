import os
from glob import glob

times_strings = {}
cmd = "cat {fn} | sed -n -e 's/^.*Processed prompts: 100//p' | cut -d'[' -f2 | cut -d'<' -f1 "
folder = "times"
for fn in glob(os.path.join(folder, "*.out")):
    model_dataset = fn.split("/")[-1].split(".")[0]
    model, dataset = model_dataset.split("_")
    if model not in times_strings:
        times_strings[model] = {}
    with open(fn, "r") as f:
        file_content = f.read()
        if len(file_content.split("\n")) < 50:
            string = file_content
        else:
            string = os.popen(cmd.format(fn=fn)).read()
    #make sure output has < 18 lines
    if len(string.split("\n")) > 20:
        print("ERROR: too many lines in output")
        print(string)
        print(fn)
        exit()
    times_strings[model][dataset] = string

def second_string_to_int(s):
    if s == "":
        return 0
    s = s.strip()
    if s.count(":") == 1:
        return int(s.split(":")[0]) * 60 + int(s.split(":")[1])
    elif s.count(":") == 2:
        return int(s.split(":")[0]) * 3600 + int(s.split(":")[1]) * 60 + int(s.split(":")[2])

validation_times = {}
test_times = {}
for model in times_strings:
    validation_times[model] = {}
    test_times[model] = {}
    for dataset in times_strings[model]:
        #convert to list of strings
        string_list = [s.strip() for s in times_strings[model][dataset].split("\n") if 0 < len(s.strip()) < 8]
        int_list = [second_string_to_int(s) for s in string_list]
        if len(int_list) < 2:
            print("ERROR: not enough times for model {} and dataset {}".format(model, dataset))
            print(times_strings[model][dataset])
            continue
        

        if len(int_list) > 14:
            split = -2
        else:
            split = -1
        validation_t = sum(int_list[:split])
        test_t = sum(int_list[split:])
        if "falcon" in model and "emea" in dataset:
            print(string_list)
            print(len(int_list))
            print(validation_t)
            print(test_t)


        validation_times[model][dataset] = validation_t
        test_times[model][dataset] = test_t
column_order = ["wnen", "conll2003", "e3cen", "n2c2", "ncbi", "wnfr", "qfp", "e3cfr", "emea", "medline", "wnes", "conll2002", "e3ces", "cwlc"]
row_order = ['gptj','vic7', 'medalpaca7', 'mistral', 'bloom7', 'vigogne', 'vic13', 'falcon40', 'opt66', 'llama70']
import pandas as pd
df_v = pd.DataFrame(validation_times).transpose()
df_t = pd.DataFrame(test_times).transpose() 
#insert columns for empty datasets
for dataset in column_order:
    if dataset not in df_v.columns:
        df_v[dataset] = None
    if dataset not in df_t.columns:
        df_t[dataset] = None
#insert rows for empty models
for model in row_order:
    if model not in df_v.index:
        df_v.loc[model] = None
    if model not in df_t.index:
        df_t.loc[model] = None
df_v = df_v.loc[row_order]
df_t = df_t.loc[row_order]

# print(df_v[column_order])
print(df_v[column_order])
print()
print(df_t[column_order])