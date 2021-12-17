# bert-ner-cmv
Code for paper: Exploring Cross-sentence Contexts for Named Entity Recognition with BERT
https://aclanthology.org/2020.coling-main.78.pdf
 
## Dependencies:

bert: tokenization.py (added as bert_tokenization.py to this project. FullTokenizer is used instead of keras-bert tokenizer)

keras-bert (https://pypi.org/project/keras-bert/)

Pretrained BERT model, e.g. from:
- https://github.com/TurkuNLP/FinBERT
- https://github.com/google-research/bert

input data e.g. from:
- https://github.com/mpsilfve/finer-data
- https://github.com/TurkuNLP/turku-ner-corpus

Input data is expected to be in CONLL:ish format where Token and Tag are tab separated. 
First string on the line corresponds to Token and second string to Tag
  
## Quickstart

Get pretrained models and data

```
./scripts/get-models.sh
./scripts/get-finer.sh
./scripts/get-turku-ner.sh
```

Experiment on Turku NER corpus data (`run-turku-ner.sh` trains, use different input file and '--use_ner_model' for predicting )

```
./scripts/run-turku-ner.sh

```

Run an experiment on FiNER news data

```
./scripts/run-finer-news.sh

```

If in a Slurm environment, edit `scripts/slurm-run.sh` to match your setup and run

```
sbatch scripts/slurm-run.sh scripts/run-finer-news.sh

```
