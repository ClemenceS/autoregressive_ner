# autoregressive_ner

This repo is a fork of the autorgressive_ner repo by Marco Naguib.

It is the support for my paper [Extracting Information in a Low-resource Setting: Case Study on Bioinformatics Workflows](https://arxiv.org/abs/2411.19295) (accepted to IDA 2025).

This fork adds the possibility to run the  experiments on the copus [BioToFlow](https://doi.org/10.5281/zenodo.14900544) 

To run the experiments presented in this article after downloading the corpus : 

```
python clm_experiment.py --dataset_name "where_is_BioToFlow" --model_name meta-llama/Meta-Llama-3-8B-Instruct
```