import datetime
import os
import numpy as np
import argparse
import logging
import random
import json

from predict import predict_for_dataset, MODEL_INSTRUCTION_TEMPLATES
from nlstruct import BRATDataset, HuggingfaceNERDataset
from nlstruct.metrics import MetricsCollection
from nlstruct.registry import get_instance
from nlstruct.data_utils import sentencize
from prompt_templates import prompt_templates


args = argparse.ArgumentParser()
args.add_argument("--dataset_name", type=str, help="dataset name")
args.add_argument('-d', "--load_dataset_from_disk", action="store_true")
args.add_argument("--begin_tag", type=str, default="@@")
args.add_argument("--end_tag", type=str, default="##")
args.add_argument("--n_few_shot", type=int, default=5)
args.add_argument("--model_name", type=str, default="bigscience/bloom")
args.add_argument("--criterion", type=str, default="most_occurences")
args.add_argument("--prompt_dict", type=str, default="en")
args.add_argument('--top_p', type=float, default=1.0)
args.add_argument('--top_k', type=int, default=50)
args.add_argument('--temperature', type=float, default=1.0)
args.add_argument('--num_beams', type=int, default=1)
args.add_argument('--partition_seed', type=int, default=1)
args.add_argument('--random_seed', type=int, default=42)
args.add_argument('-n', '--n_gpus', type=int, default=1)
args.add_argument('-s', '--training_size', type=int, default=100)
args.add_argument('-t', '--test_on_test_set', action="store_true")
args.add_argument('--do_sample', action="store_true")
args.add_argument('--no_write_log', dest='write_log', action='store_false')
args.add_argument('--no_control', dest='control', action='store_false')
args.add_argument('--no_self_verification', dest='self_verification', action='store_false')
args = args.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("experiment")

#random deals with choosing the few-shot examples, so we want that fixed
random.seed(args.random_seed)

assert args.dataset_name is not None

ner_tags_by_dataset = {
    "WikiNER" : ["PER", "LOC", "ORG"],
    "conll2003" : ["PER", "LOC", "ORG"],
    "conll2002" : ["PER", "LOC", "ORG"],
    "medline" : ["ANAT", "CHEM", "DEVI", "DISO", "GEOG", "LIVB", "OBJC", "PHEN", "PHYS", "PROC"],
    "emea" : ["ANAT", "CHEM", "DEVI", "DISO", "GEOG", "LIVB", "OBJC", "PHEN", "PHYS", "PROC"],
    "n2c2" : ["ACTI", "ANAT", "CHEM", "CONC", "DEVI", "DISO", "LIVB", "OBJC", "PHEN", "PHYS", "PROC"],
}
colnames_by_hf_dataset = {
    "WikiNER" : ("id", "words", "ner_tags"),
    "conll2003" : ("id", "tokens", "ner_tags"),
    "conll2002" : ("id", "tokens", "ner_tags"),
}
tag_map_by_hf_dataset = {
    "WikiNER" : {
        0: "O",
        1: "LOC",
        2: "PER",
        3: "FAC",
        4: "ORG",
    },
    "conll2003" : {
        0: "O",
        1: "PER",
        2: "PER",
        3: "ORG",
        4: "ORG",
        5: "LOC",
        6: "LOC",
        7: "O",
        8: "O",
    },
    "conll2002" : {
        0: "O",
        1: "PER",
        2: "PER",
        3: "ORG",
        4: "ORG",
        5: "LOC",
        6: "LOC",
        7: "O",
        8: "O",
    },
}

def get_if_key_in_x(dict, x):
    return next((dict[key] for key in dict if key in x), None)

try:
    doc_id_colname, words_colname, ner_tags_colname = get_if_key_in_x(colnames_by_hf_dataset, args.dataset_name)
    dataset = HuggingfaceNERDataset(
        dataset_name=args.dataset_name,
        tag_map=get_if_key_in_x(tag_map_by_hf_dataset, args.dataset_name),
        doc_id_colname=doc_id_colname,
        words_colname=words_colname,
        ner_tags_colname=ner_tags_colname,
        load_from_disk=args.load_dataset_from_disk,
    )
    #This is not supposed to be here, but WikiNER is a mess for now and I have no time to fix it
    if args.dataset_name.endswith("WikiNER/en"):
        dataset.train_data = [e for e in dataset.train_data if e['doc_id'].startswith('en')]
    elif args.dataset_name.endswith("WikiNER/fr"):
        dataset.train_data = [e for e in dataset.train_data if e['doc_id'].startswith('fr')]
    elif args.dataset_name.endswith("WikiNER/es"):
        dataset.train_data = [e for e in dataset.train_data if e['doc_id'].startswith('es')]
except:
    dataset = BRATDataset(
        train= f"{args.dataset_name}/train",
        val= 0, 
        test= f"{args.dataset_name}/test",
    )
ner_tags = get_if_key_in_x(ner_tags_by_dataset, args.dataset_name)

traindev_dataset = []
for e in dataset.train_data:
    sentences = sentencize(e, reg_split=r"(?<=[.|\s])(?:\s+)(?=[A-Z])", entity_overlap="split")
    traindev_dataset.extend([s for s in sentences if len(s['text']) < 512])
test_dataset = []
for e in dataset.test_data:
    sentences = sentencize(e, reg_split=r"(?<=[.|\s])(?:\s+)(?=[A-Z])", entity_overlap="split")
    test_dataset.extend([s for s in sentences if len(s['text']) < 512])

folder_name = 'results'
script_dir = os.path.dirname(__file__)
os.makedirs(os.path.join(script_dir, folder_name), exist_ok=True)

np.random.seed(args.partition_seed)
traindev_dataset_this_seed = [traindev_dataset[i] for i in np.random.choice(len(traindev_dataset), size=args.training_size, replace=False)]

res_dict = {}
time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
last_two_dirs = '-'.join(args.dataset_name.split('/')[-2:])
model_base_name = os.path.basename(args.model_name)

res_dict['dataset_name'] = last_two_dirs
res_dict['begin_tag'] = args.begin_tag
res_dict['end_tag'] = args.end_tag
res_dict['n_few_shot'] = args.n_few_shot
res_dict['model_name'] = args.model_name
res_dict['criterion'] = args.criterion
res_dict['prompt_dict'] = args.prompt_dict
res_dict['training_size'] = args.training_size
res_dict['partition_seed'] = args.partition_seed
res_dict['random_seed'] = args.random_seed
res_dict['control'] = args.control
res_dict['num_beams'] = args.num_beams
res_dict['self_verification'] = args.self_verification
res_dict['do_sample'] = args.do_sample
res_dict['top_p'] = args.top_p
res_dict['top_k'] = args.top_k
res_dict['temperature'] = args.temperature

model_kwargs = {
    "num_beams": args.num_beams,
}
if args.do_sample:
    model_kwargs.update({
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "temperature": args.temperature,
    })
else:
    model_kwargs.update({
        "do_sample": False,
    })

res_dict['prompt'] = prompt_templates[args.prompt_dict]
res_dict['chat_template'] = MODEL_INSTRUCTION_TEMPLATES[args.model_name] if args.model_name in MODEL_INSTRUCTION_TEMPLATES else ""
res_dict['ner_tags'] = ner_tags
res_dict['first_sentence'] = traindev_dataset_this_seed[0]['text']
res_dict['last_sentence'] = traindev_dataset_this_seed[-1]['text']
res_dict['test_on_test_set'] = args.test_on_test_set

textual_outputs, predicted_dataset = predict_for_dataset(
    training_data=traindev_dataset_this_seed,
    testing_data=test_dataset if args.test_on_test_set else None,
    ner_tags=ner_tags,
    model_name=args.model_name,
    begin_tag=args.begin_tag,
    end_tag=args.end_tag,
    self_verification=args.self_verification,
    control=args.control,
    n_few_shot=args.n_few_shot,
    criterion=args.criterion,
    keywords=prompt_templates[args.prompt_dict],
    model_kwargs=model_kwargs,
    n_gpus=args.n_gpus,
    random_seed=args.random_seed,
)

logger.info("Evaluating...")
metric_names = {
        "exact": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., add_label_specific_metrics=ner_tags, filter_entities=ner_tags),
        "partial": dict(module="dem", binarize_tag_threshold=1e-5, binarize_label_threshold=1., add_label_specific_metrics=ner_tags, filter_entities=ner_tags),
}
metrics = MetricsCollection({k: get_instance(m) for k, m in metric_names.items()})

s_metrics = ""
for metric_name, metric in metrics.items():
    metric(predicted_dataset, test_dataset if args.test_on_test_set else traindev_dataset_this_seed)
    metric_dict = metric.compute()
    for k,v in metric_dict.items():
        if not isinstance(v, int) and not isinstance(v, float):
            metric_dict[k] = v.item()
        metric_dict[k] = round(metric_dict[k], 3)
    res_dict[metric_name] = metric_dict
    s_metrics+="="*20+metric_name+"="*20+'\n'
    s_metrics+=f'ALL    tp: {metric_dict["tp"]}    precision: {metric_dict["precision"]}    recall: {metric_dict["recall"]}    f1: {metric_dict["f1"]}\n'
    for tag in ner_tags:
        s_metrics+=f'{tag}    tp: {metric_dict[tag+"_tp"]}    precision: {metric_dict[tag+"_precision"]}    recall: {metric_dict[tag+"_recall"]}    f1: {metric_dict[tag+"_f1"]}\n'
print(s_metrics)

full_preds = ""
for i, (o, pred, gold) in enumerate(zip(textual_outputs, predicted_dataset, test_dataset if args.test_on_test_set else traindev_dataset_this_seed)):
        full_preds += '='*50+'\n'
        full_preds += 'input: '+pred['text']+'\n'
        for j,tag in enumerate(ner_tags):
            full_preds += '-'*50+'\n'
            full_preds += tag+' output: '+textual_outputs[j*len(predicted_dataset)+i]+'\n'
            full_preds += 'final: '+str([p['text'] for p in pred['entities'] if p['label']==tag])+'\n'
            full_preds += 'gold: '+str([g['text'] for g in gold['entities'] if g['label']==tag])+'\n'

if args.write_log:
    full_preds_path = os.path.join(script_dir, folder_name)+f'/full_preds_{last_two_dirs}_{model_base_name}_{args.random_seed}_{time_str}.txt'
    res_dict['full_preds_path'] = full_preds_path
    with open(full_preds_path, 'w') as f:
        f.write(full_preds)

    res_dict_path = os.path.join(script_dir, folder_name)+f'/res_dict_{last_two_dirs}_{model_base_name}_{args.random_seed}_{time_str}.json'
    with open(res_dict_path, 'w') as f:
        json.dump(res_dict, f)
