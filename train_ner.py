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
from bloom_predict import bloom_predict
from fastchat.model import load_model
from nlstruct import BRATDataset, HuggingfaceNERDataset, get_instance, get_config, InformationExtractor
from nlstruct.metrics import MetricsCollection
from nlstruct.registry import get_instance
from rich_logger import RichTableLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from nlstruct.checkpoint import ModelCheckpoint, AlreadyRunningException
import pandas as pd

args = argparse.ArgumentParser()
args.add_argument("--language", type=str, default="fr", help="language of the dataset")
args.add_argument("--domain", type=str, default="general", help="domain of the dataset")
args.add_argument("--model_name", type=str, default="camembert-base", help="model name")
args.add_argument('--random_seed', type=int, default=42)
args.add_argument('--partition_seed', type=int, default=1)
args.add_argument('-s', '--training_size', type=int, default=100)
args.add_argument('-t', '--test_on_test_set', action="store_true")
args = args.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_ner")

shared_cache = {}
#random deals with choosing the few-shot examples, so we want that fixed
random.seed(args.random_seed)

if args.domain == 'general':
    dataset = HuggingfaceNERDataset(
        dataset_name='meczifho/WikiNER',
        subset=args.language,
        tag_map={
            0: "O",
            1: "LOC",
            2: "PER",
            3: "FAC",
            4: "ORG",
        },
        doc_id_colname="id",
    )
    ner_tags = ['PER', 'LOC', 'ORG']
else :
    dataset = BRATDataset(
        train= "/people/mnaguib/medline/training",
        val= 0, 
        test= "/people/mnaguib/medline/test",
    )
    ner_tags = ["ANAT", "CHEM", "DEVI", "DISO", "GEOG", "LIVB", "OBJC", "PHEN", "PHYS", "PROC"]

dataset.train_data = [e for e in dataset.train_data if len(e['text']) < 512]
dataset.test_data = [e for e in dataset.test_data if len(e['text']) < 512]

folder_name = 'results'
os.makedirs(folder_name, exist_ok=True)

#np random deals with choosing the traindev dataset
np.random.seed(args.partition_seed)
traindev_dataset_this_seed = [dataset.train_data[i] for i in np.random.choice(len(dataset.train_data), size=args.training_size, replace=False)]
dataset.train_data = traindev_dataset_this_seed[:int(0.8*len(traindev_dataset_this_seed))]
dataset.val_data = traindev_dataset_this_seed[int(0.8*len(traindev_dataset_this_seed)):]
time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logfile = open(folder_name+f'/log_{args.domain}_{args.language}_{args.random_seed}_{time_str}.txt', 'w')
logfile.write('language: '+args.language+'\n')
logfile.write('domain: '+args.domain+'\n')
logfile.write('model_name: '+args.model_name+'\n')
logfile.write('training_size: '+str(args.training_size)+'\n')
logfile.write('random_seed: '+str(args.random_seed)+'\n')
logfile.write('partition_seed: '+str(args.partition_seed)+'\n')

print(dataset.describe())


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
                "entity_label": dict(module="vocabulary", values=sorted(dataset.labels()), with_unk=False, with_pad=False),
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

    "(.*)_?loss": {"goal": "lower_is_better", "format": "{:.4f}"},
    "(.*)_precision": False,  # {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_p"},
    "(.*)_recall": False,  # {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_r"},
    "(.*)_tp": False,
    "(.*)_f1": {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_f1"},

    ".*_lr|max_grad": {"format": "{:.2e}"},
    "duration": {"format": "{:.0f}", "name": "dur(s)"},
})
with logger.printer:
    try:
        trainer = pl.Trainer(
            gpus=1,
            progress_bar_refresh_rate=1,
            checkpoint_callback=False,  # do not make checkpoints since it slows down the training a lot
            callbacks=[ModelCheckpoint(path='checkpoints/{hashkey}-{global_step:05d}',),
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
        if not os.path.exists(result_output_filename):
            model.cuda()
            if dataset.test_data:
                print("TEST RESULTS:")
            else:
                print("VALIDATION RESULTS (NO TEST SET):")
            eval_data = dataset.test_data if dataset.test_data else dataset.val_data

            final_metrics = MetricsCollection({
                **{metric_name: get_instance(metric_config) for metric_name, metric_config in metrics.items()},
                **{
                    metric_name: get_instance(metric_config)
                    for label in model.preprocessor.vocabularies['entity_label'].values
                    for metric_name, metric_config in
                    {
                        f"{label}_exact": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., filter_entities=[label], word_regex=word_regex),
                        f"{label}_partial": dict(module="dem", binarize_tag_threshold=1e-5, binarize_label_threshold=1., filter_entities=[label], word_regex=word_regex),
                    }.items()
                }
            })

            results = final_metrics(list(model.predict(eval_data)), eval_data)
            print(pd.DataFrame(results).T)

            def json_default(o):
                if isinstance(o, slice):
                    return str(o)
                raise

            with open(result_output_filename, 'w') as json_file:
                json.dump({
                    "config": {**get_config(model), "max_steps": 400},
                    "results": results,
                }, json_file, default=json_default)
        else:
            with open(result_output_filename, 'r') as json_file:
                results = json.load(json_file)["results"]
                print(pd.DataFrame(results).T)
    except AlreadyRunningException as e:
        model = None
        print("Experiment was already running")
        print(e)


logger.info("Predicting on test set")
model = load_model("checkpoints/{}.ckpt".format(trainer.callbacks[0].hashkey))
model.cuda()
model.eval()
model.encoder.encoders[0].cache = shared_cache
with torch.no_grad():
    predicted_dataset = model.predict(dataset.test_data)

for metric in metrics.values():
        metric(predicted_dataset, dataset.test_data)
        print(metric.compute())
        logfile.write(str(metric.compute())+'\n')
