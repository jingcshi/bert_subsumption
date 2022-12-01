import numpy as np
import pandas as pd
import json
from sklearn.neighbors import NearestNeighbors
from transformers import TrainingArguments, Trainer
from pybay.bert import EBertModel, EBertForSequenceClassification, EBertTokenizer
import torch
from torch import nn
from datasets import Dataset, load_metric, concatenate_datasets
import data.preprocess as prep
import data.mask_sampling as ms
from functools import partial
import datetime

def tokenize(config, tokenizer, raw):
    raw["text1"] = list(map(partial(prep.normalise_breadcrumb,link=config['data']['breadcrumb_link']), raw["text1"]))
    raw["text2"] = list(map(partial(prep.normalise_breadcrumb,link=config['data']['breadcrumb_link']), raw["text2"]))
    return tokenizer(raw["text1"], raw["text2"], padding='max_length')

def get_extra_negative(config, training, evaluation):
    neg_rate = config['corpus']['negative_ratio']
    pos_count_train = training.filter(lambda x: x['label'] == 0).shape[0]
    target_neg_train = pos_count_train * (neg_rate) / (1-neg_rate)
    extra_neg_train_size = int(target_neg_train-training.shape[0]+pos_count_train)
    pos_count_eval = evaluation.filter(lambda x: x['label'] == 0).shape[0]
    target_neg_eval = pos_count_eval * (neg_rate) / (1-neg_rate)
    extra_neg_eval_size = int(target_neg_eval-evaluation.shape[0]+pos_count_eval)
    tmp_config = config
    tmp_config['corpus']['sample_size_from_masking'] = extra_neg_train_size + extra_neg_eval_size
    tmp_config['corpus']['train_split'] = extra_neg_train_size / (extra_neg_train_size + extra_neg_eval_size)
    tmp_config['corpus']['equivalence_ratio'] = 0
    tmp_config['corpus']['subclass_ratio'] = 0
    tmp_config['corpus']['common_parent_ratio'] = 0
    tmp_config['corpus']['negative_ratio'] = 1
    extra_neg_train, extra_neg_eval = ms.get_samples_from_masking(tmp_config, save=False)
    extra_neg_train.to_csv('/data/ebay/notebooks/jingcshi/Curation/data/train/_extra_neg.csv', index = False)
    extra_neg_eval.to_csv('/data/ebay/notebooks/jingcshi/Curation/data/eval/_extra_neg.csv', index = False)

def fine_tune(config, model, tokenizer, maskready=False, negready=False, resume_from_checkpoint=False):
    train_path = config['corpus']['train_path']
    eval_path = config['corpus']['eval_path']
    sample_neg = config['corpus']['sample_negative']
    use_mask = config['corpus']['mask_taxonomy']
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    training = Dataset.from_pandas(pd.read_csv(train_path))
    evaluation = Dataset.from_pandas(pd.read_csv(eval_path))
    if use_mask:
        if maskready==False:
            print('Sampling data from masking taxonomy')
            ms.get_samples_from_masking(config)
        mask_train_path = './data/train/_from_masking.csv'
        mask_training = Dataset.from_pandas(pd.read_csv(mask_train_path))
        mask_eval_path = './data/eval/_from_masking.csv'
        mask_evaluation = Dataset.from_pandas(pd.read_csv(mask_eval_path))
        training = concatenate_datasets([training, mask_training])
        evaluation = concatenate_datasets([evaluation, mask_evaluation])
    if sample_neg:
        if negready==False:
            print('Sampling random negative data to fulfill negative ratio')
            get_extra_negative(config, training, evaluation)
        sample_neg_train_path = './data/train/_extra_neg.csv'
        sample_neg_training = Dataset.from_pandas(pd.read_csv(sample_neg_train_path))
        sample_neg_eval_path = './data/eval/_extra_neg.csv'
        sample_neg_evaluation = Dataset.from_pandas(pd.read_csv(sample_neg_eval_path))
        training = concatenate_datasets([training, sample_neg_training])
        evaluation = concatenate_datasets([evaluation, sample_neg_evaluation])
    tokenized_training = training.map(lambda x: tokenize(config, tokenizer, x), batched=True)
    tokenized_eval = evaluation.map(lambda x: tokenize(config, tokenizer, x), batched=True)
    training_args = TrainingArguments("test_trainer", per_device_train_batch_size = batch_size, per_device_eval_batch_size = batch_size, num_train_epochs = num_epochs, evaluation_strategy="epoch", learning_rate = learning_rate)
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_training, eval_dataset=tokenized_eval)
    if resume_from_checkpoint == True:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    now = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    model.save_pretrained(f'/data/ebay/notebooks/jingcshi/Curation/utils/bert/{now}')
    tokenizer.save_pretrained(f'/data/ebay/notebooks/jingcshi/Curation/utils/bert/{now}')
    