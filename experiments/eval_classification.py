import numpy as np
import pandas as pd
import json
from pybay.bert import EBertModel, EBertForSequenceClassification, EBertTokenizer
import torch
import data.preprocess as prep
from experiments import classify, generate_candidates
from datasets import Dataset, load_metric

def tp_count(df, pos_label):
    count = 0
    for l, p in zip(df['label'], df['prediction']):
        if l == p and l in pos_label:
            count += 1
    return count

def tn_count(df, pos_label):
    count = 0
    for l, p in zip(df['label'], df['prediction']):
        if l == p and l not in pos_label:
            count += 1
    return count

def fp_count(df, pos_label):
    count = 0
    for l, p in zip(df['label'], df['prediction']):
        if l != p and l not in pos_label:
            count += 1
    return count

def fn_count(df, pos_label):
    count = 0
    for l, p in zip(df['label'], df['prediction']):
        if l != p and l in pos_label:
            count += 1
    return count

def accuracy(table, pos_label):
    return (tp_count(table, pos_label) + tn_count(table, pos_label)) / table.shape[0]

def precision(table, pos_label):
    tp = tp_count(table, pos_label)
    fp = fp_count(table, pos_label)
    if tp + fp == 0:
        return 0
    else:
        return tp / (tp + fp)

def recall(table, pos_label):
    tp = tp_count(table, pos_label)
    fn = fn_count(table, pos_label)
    if tp + fn == 0:
        return 0
    else:
        return tp / (tp + fn)

def f1(table, pos_label):
    p = precision(table, pos_label)
    r = recall(table, pos_label)
    if p + r == 0:
        return 0
    else:
        return 2 * p * r / (p + r)

def evaluate(config, clsmodel, tokenizer, label_names):
    num_labels = len(label_names)
    test_path = config['corpus']['test_path']
    test_set = pd.read_csv(test_path)
    test = Dataset.from_pandas(test_set)
    test_input = test.remove_columns('label')
    inputs = tokenizer(test_input['text1'], test_input['text2'], padding = True, return_tensors = 'pt').to('cuda:0')
    output = generate_candidates.batched_model(clsmodel, inputs, config['embedding']['batch_size'], model_is_cls=True, num_labels = num_labels)
    prediction = np.argmax(output, axis=1)
    test_output = test.add_column('prediction', prediction).to_pandas()
    metric_table = {'label': ['Overall'] + label_names[:-1], 'accuracy': np.zeros(num_labels), 'precision': np.zeros(num_labels), 'recall': np.zeros(num_labels), 'F1': np.zeros(num_labels)}
    pos_label_list = []
    pos_label_list.append(list(range(num_labels - 1)))
    for x in range(num_labels - 1):
        pos_label_list.append([x])
    for x in range(num_labels):
        metric_table['accuracy'][x] = round(accuracy(test_output, pos_label_list[x]),2)
        metric_table['precision'][x] = round(precision(test_output, pos_label_list[x]),2)
        metric_table['recall'][x] = round(recall(test_output, pos_label_list[x]),2)
        metric_table['F1'][x] = round(f1(test_output, pos_label_list[x]),2)
    metric_df = pd.DataFrame(metric_table)
    return test_output, metric_df