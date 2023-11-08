import datetime
import gc
import json
import os
import numpy as np
import argparse
import logging
import random

from typing import Dict
import string
import torch
from nlstruct import BRATDataset, HuggingfaceNERDataset, get_instance, get_config, InformationExtractor
from nlstruct.metrics import MetricsCollection
from nlstruct.registry import get_instance
from rich_logger import RichTableLogger
from torch.utils.data import DataLoader
from nlstruct.data_utils import sentencize
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from nlstruct.checkpoint import ModelCheckpoint, AlreadyRunningException
import pandas as pd

args = argparse.ArgumentParser()
# args.add_argument("--dataset_name", type=str, default="meczifho/WikiNER/en", help="dataset name")
args.add_argument("--dataset_name", type=str, default="/people/mnaguib/medline", help="dataset name")
args.add_argument('-d', "--load_dataset_from_disk", action="store_true")
args.add_argument("--model_name", type=str, default="camembert-base", help="model name")
args.add_argument('--random_seed', type=int, default=42)
args.add_argument('--partition_seed', type=int, default=1)
args.add_argument('-s', '--training_size', type=int, default=100)
# args.add_argument('-t', '--test_on_test_set', action="store_true")
args = args.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_ner")

shared_cache = {}
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
    "WikiNER" : {0: "O", 1: "LOC", 2: "PER", 3: "FAC", 4: "ORG", }, 
    "conll2003" : {0: "O", 1: "PER", 2: "PER", 3: "ORG", 4: "ORG", 5: "LOC", 6: "LOC", 7: "O", 8: "O", },
    "conll2002" : {0: "O", 1: "PER", 2: "PER", 3: "ORG", 4: "ORG", 5: "LOC", 6: "LOC", 7: "O", 8: "O", },
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
        dataset.test_data = [e for e in dataset.test_data if e['doc_id'].startswith('en')]
    elif args.dataset_name.endswith("WikiNER/fr"):
        dataset.train_data = [e for e in dataset.train_data if e['doc_id'].startswith('fr')]
        dataset.test_data = [e for e in dataset.test_data if e['doc_id'].startswith('fr')]
    elif args.dataset_name.endswith("WikiNER/es"):
        dataset.train_data = [e for e in dataset.train_data if e['doc_id'].startswith('es')]
        dataset.test_data = [e for e in dataset.test_data if e['doc_id'].startswith('es')]
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
    traindev_dataset.extend([s for s in sentences if 5 < len(s['text']) ])
test_dataset = dataset.test_data

folder_name = 'results'
#get script directory
script_dir = os.path.dirname(__file__)
#make the results folder if it doesn't exist
os.makedirs(os.path.join(script_dir, folder_name), exist_ok=True)

#np random deals with choosing the traindev dataset
np.random.seed(args.partition_seed)
#randomy select training_size number of sentences from traindev dataset
traindev_dataset_this_seed = np.random.choice(traindev_dataset, args.training_size, replace=False)
limit=0.8
dataset.train_data = traindev_dataset_this_seed[:int(limit*len(traindev_dataset_this_seed))]
dataset.val_data = traindev_dataset_this_seed[int(limit*len(traindev_dataset_this_seed)):]
dataset.test_data = test_dataset

res_dict = {}
time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
last_two_dirs = '-'.join(args.dataset_name.split('/')[-2:])
model_base_name = os.path.basename(args.model_name)

print(dataset.describe())

res_dict['dataset_name'] = last_two_dirs
res_dict['model_name'] = args.model_name
res_dict['training_size'] = args.training_size
res_dict['time_str'] = time_str
res_dict['last_two_dirs'] = last_two_dirs
res_dict['model_base_name'] = model_base_name
res_dict['test_on_test_set'] = True
res_dict['partition_seed'] = args.partition_seed


word_regex = r'(?:[\w]+(?:[’\'])?)|[!"#$%&\'’\(\)*+,-./:;<=>?@\[\]^_`{|}~]'
sentence_split_regex = r"((?:\s*\n)+\s*|(?:(?<=[\w0-9]{2,}\.|[)]\.)\s+))(?=[[:upper:]]|•|\n)"

gc.collect()
torch.cuda.empty_cache()
gc.collect()

val_check_interval = 400

metric_names = {
        "exact": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., add_label_specific_metrics=ner_tags, filter_entities=ner_tags),
        "partial": dict(module="dem", binarize_tag_threshold=1e-5, binarize_label_threshold=1., add_label_specific_metrics=ner_tags, filter_entities=ner_tags),
}
metrics = MetricsCollection({k: get_instance(m) for k, m in metric_names.items()})


model = InformationExtractor(
    seed=args.random_seed,
    preprocessor=dict(
        module="ner_preprocessor",
        bert_name=args.model_name,  # transformer name
        bert_lower=False,
        split_into_multiple_samples=True,
        sentence_split_regex=sentence_split_regex,  # regex to use to split sentences (must not contain consuming patterns)
        sentence_balance_chars=(),  # try to avoid splitting between parentheses
        sentence_entity_overlap="split",  # raise when an entity spans more than one sentence
        word_regex=word_regex,  # regex to use to extract words (will be aligned with bert tokens), leave to None to use wordpieces as is
        substitutions=(),  # Apply these regex substitutions on sentences before tokenizing
        keep_bert_special_tokens=False,
        min_tokens=0,
        doc_context=False,
        join_small_sentence_rate=0.,
        max_tokens=512,  # split when sentences contain more than 512 tokens
        large_sentences="equal-split",  # for these large sentences, split them in equal sub sentences < 512 tokens
        empty_entities="raise",  # when an entity cannot be mapped to any word, raise
        vocabularies={
            **{  # vocabularies to use, call .train() before initializing to fill/complete them automatically from training data
                "entity_label": dict(module="vocabulary", values=sorted(dataset.labels()), with_unk=True, with_pad=False),
            },
            **({
                    "char": dict(module="vocabulary", values=string.ascii_letters + string.digits + string.punctuation, with_unk=True, with_pad=False),
                })
        },
        fragment_label_is_entity_label=True,
        multi_label=False,
        filter_entities=None,  # "entity_type_score_density", "entity_type_score_lesion"),
    ),
    dynamic_preprocessing=False,

    # Text encoders
    encoder=dict(
        module="concat",
        dropout_p=0.5,
        encoders=[
            dict(
                module="bert",
                path=args.model_name,
                n_layers=4,
                freeze_n_layers=0,  # freeze 0 layer (including the first embedding layer)
                bert_dropout_p=None,
                token_dropout_p=0.,
                proj_size=None,
                output_lm_embeds=False,
                combine_mode="scaled_softmax",
                do_norm=False,
                do_cache=False,
                word_pooler=dict(module="pooler", mode="mean"),
            ),
            *([dict(
                module="char_cnn",
                in_channels=8,
                out_channels=50,
                kernel_sizes=(3, 4, 5),
            )]),
            *([])
        ],
    ),
    decoder=dict(
        module="contiguous_entity_decoder",
        contextualizer=dict(
            module="lstm",
            num_layers=3,
            gate=dict(module="sigmoid_gate", init_value=0., proj=True),
            bidirectional=True,
            hidden_size=400,
            dropout_p=0.4,
            gate_reference="last",
        ),
        span_scorer=dict(
            module="bitag",
            do_biaffine=True,
            do_tagging="full",
            do_length=False,

            threshold=0.5,
            max_fragments_count=200,
            max_length=40,
            hidden_size=150,
            allow_overlap=True,
            dropout_p=0.1,
            tag_loss_weight=1.,
            biaffine_loss_weight=1.,
            eps=1e-14,
        ),
        intermediate_loss_slice=slice(-1, None),
    ),

    _predict_kwargs={},
    batch_size=16,

    # Use learning rate schedules (linearly decay with warmup)
    use_lr_schedules=True,
    warmup_rate=0.1,

    gradient_clip_val=10.,
    _size_factor=5,

    # Learning rates
    main_lr=1e-3,
    fast_lr=1e-3,
    bert_lr=5e-5,

    # Optimizer, can be class or str
    optimizer_cls="transformers.AdamW",
    metrics=metrics,
).train()

model.encoder.encoders[0].cache = shared_cache
os.makedirs("checkpoints", exist_ok=True)

logger = RichTableLogger(key="epoch", fields={
    "epoch": {},
    "step": {},

    "(.*)_?loss": {"goal": "lower_is_better", "format": "{:.2f}"},
    "(.*)_precision": False,  # {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_p"},
    "(.*)_recall": False,  # {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_r"},
    "(.*)_tp": False,
    "val_exact_f1": {"goal": "higher_is_better", "format": "{:.4f}", "name": "val_exact_f1"},
    # "(.*)_f1": {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_f1"},

    ".*_lr|max_grad": {"format": "{:.2e}"},
    # "duration": {"format": "{:.0f}", "name": "dur(s)"},
})
with logger.printer:
    try:
        trainer = pl.Trainer(
            gpus=1,
            progress_bar_refresh_rate=False,
            checkpoint_callback=False,  # do not make checkpoints since it slows down the training a lot
            callbacks=[
                # ModelCheckpoint(path='checkpoints/{hashkey}-{global_step:05d}',),
                EarlyStopping(monitor="val_exact_f1",mode="max", patience=3),
                        ],
            logger=[
                logger,
            ],
            val_check_interval=val_check_interval,
            max_steps=4000,)
        trainer.fit(model, dataset)
        trainer.logger[0].finalize(True)

        result_output_filename = "checkpoints/{}.json".format(trainer.callbacks[0].hashkey)
        model.cuda()
        model.eval()
        model.encoder.encoders[0].cache = shared_cache

        final_metrics = MetricsCollection({k: get_instance(m) for k, m in metric_names.items()})
        
        with torch.no_grad():
            predicted_dataset = model.predict(dataset.test_data)

        s_metrics = ""
        for metric_name, metric in final_metrics.items():
            metric(predicted_dataset, dataset.test_data)
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

        res_dict_path = os.path.join(script_dir, folder_name)+f'/res_dict_{last_two_dirs}_{model_base_name}_{args.random_seed}_{time_str}.json'
        with open(res_dict_path, 'w') as f:
            json.dump(res_dict, f)
    except AlreadyRunningException as e:
        model = None
        print("Experiment was already running")
        print(e)

