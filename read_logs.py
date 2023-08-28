import argparse
import glob
import datetime
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--start-date', help="Start date for the log file")
parser.add_argument('-e', '--end-date', help="End date for the log file")
args = parser.parse_args()

#parse start-date and end-date
start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d_%H-%M-%S" if '_' in args.start_date else "%Y-%m-%d")
end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d_%H-%M-%S" if '_' in args.end_date else "%Y-%m-%d")

#compare the datetime objects
assert start_date < end_date, "Start date should be before end date"

data = []
for folder in glob.glob("hyp_search_*"):
    folder_date = folder.split("_", 2)[2]
    folder_date = datetime.datetime.strptime(folder_date, "%Y-%m-%d_%H-%M-%S")
    if start_date <= folder_date <= end_date:
        for file in glob.glob(folder + "/*"):
            file_dict = {}
            with open(file, 'r') as f:
                s=f.read()
            header_separator = "=================================================="
            footer_separator = "--------------------------------------------------"
            header = s.split(header_separator)[0]
            header_keywords = [
                "language",
                "domain",
                "ner_tag",
                "begin_tag",
                "end_tag",
                "n_few_shot",
                "model_name",
                "criterion",
                "prompt_dict",
                "training_size",
                "random_seed",
                "self verification",
                "example prompt",
                "self_verif_template",
                "greedy"
            ]
            for i, keyword in enumerate(header_keywords):
                if i < len(header_keywords) - 1:
                    split_kw = header.split(keyword + ":")
                    after_keyword = split_kw[1]
                    value = after_keyword.split('\n'+header_keywords[i+1])[0].strip()
                    if keyword == 'example prompt':
                        value = ' '.join(value.split()[:30])
                    file_dict[keyword] = value
                else:
                    file_dict[keyword] = header.split(keyword + ":")[1].split('\n'+header_separator)[0].strip()
            
            #get the footer
            footer = s.split(footer_separator)[-1].split(header_separator)[0].strip()
            footer_keywords = [
                "precision",
                "recall",
                "f1"
            ]
            values = [line.split(" ")[-1] for line in footer.split("\n")]
            for i, keyword in enumerate(footer_keywords):
                file_dict[keyword] = values[i]
            data.append(file_dict)

df = pd.DataFrame(data)

#convert to numeric
df["precision"] = pd.to_numeric(df["precision"])
df["recall"] = pd.to_numeric(df["recall"])
df["f1"] = pd.to_numeric(df["f1"])

#add a mean_f1 column that is the mean over every group with same ['language', 'ner_tag', 'n_few_shot']
df["mean_f1"] = df.groupby(['language', 'ner_tag', 'n_few_shot', 'example prompt'])['f1'].transform('mean')

print(df.sort_values(by=['mean_f1'], ascending=False)[['language', 'ner_tag', 'n_few_shot', 'random_seed',  'f1', 'mean_f1']].to_string(index=False))