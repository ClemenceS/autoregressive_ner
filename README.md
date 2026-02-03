# Autoregressive NER

This repo is a fork of the autorgressive_ner repo by Marco Naguib.

This repository is designed to run experiments based on decoder-based approaches.
It allows you to design, test, and optimize prompts, and to evaluate decoder-based methods on your domain.

Multiple corpora can be tested (see `dataset_info.py`).
To use your own  corpus, simply add its information to this file. 

Before running the experiments, you need to update `clm_experiment.py` by specifying the local path to `nlstruct` after downloading it from https://github.com/ClemenceS/nlstruct .

* To run the experiments presented in the paper [Extracting Information in a Low-resource Setting: Case Study on Bioinformatics Workflows](https://arxiv.org/abs/2411.19295) (accepted to IDA 2025) after downloading the corpus : 

```
python clm_experiment.py --dataset_name "where_is_BioToFlow" --dataset_type "article" --model_name meta-llama/Meta-Llama-3-8B-Instruct
```

* To run the experiments presented on the CPL-Article and CPL-Code corpus after downloading the two corpora :

```
python clm_experiment.py --dataset_name "where_is_CPL-Article" --dataset_type "article" --model_name meta-llama/Meta-Llama-3.1-8B-Instruct
```

```
python clm_experiment.py --dataset_name "where_is_CPL-Code" --dataset_type "code" --model_name meta-llama/Meta-Llama-3.1-8B-Instruct
```
