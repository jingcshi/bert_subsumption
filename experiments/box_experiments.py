import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.datasets as datasets

from tqdm.notebook import trange, tqdm
import numpy as np
import pandas as pd
from sklearn import metrics

import json
import random
import time

from utils.box_mlp import loss

def train(model, iterator, optimizer, device, t, threshold=0, eps=1e-10):
    epoch_loss = 0
    confusion = {'y': np.array([],dtype=np.int64), 'y_pred': np.array([],dtype=np.int64)}
    model.train()
    d = int(model.output_fc.out_features / 2)
    for sample in tqdm(iterator, desc="Training", leave=False):
        x1 = sample['feature1'].to(device)
        x2 = sample['feature2'].to(device)
        label = sample['label'].to(device)
        n = len(sample['label'])
        confusion['y'] = np.append(confusion['y'], sample['label'].numpy())
        optimizer.zero_grad()
        y1 = model(x1)
        y2 = model(x2)
        batchloss, pred = loss.softbox_loss(y1, y2, n, d, t, label, device, threshold)
        confusion['y_pred'] = np.append(confusion['y_pred'], pred.detach().cpu().numpy().astype(np.int64))
        batchloss.backward()
        optimizer.step()
        epoch_loss += batchloss.detach().item()
    return epoch_loss / len(iterator), confusion

def evaluate(model, iterator, device, t, threshold=0, eps=1e-10):
    epoch_loss = 0
    confusion = {'y': np.array([],dtype=np.int32), 'y_pred': np.array([],dtype=np.int32)}
    model.eval()
    d = int(model.output_fc.out_features / 2)
    with torch.no_grad():
        for sample in tqdm(iterator, desc="Evaluating", leave=False):
            x1 = sample['feature1'].to(device)
            x2 = sample['feature2'].to(device)
            label = sample['label'].to(device)
            confusion['y'] = np.append(confusion['y'], sample['label'].numpy().astype(np.int32))
            y1 = model(x1)
            y2 = model(x2)
            n = len(sample['label'])
            batchloss, pred = loss.softbox_loss(y1, y2, n, d, t, label, device, threshold)
            confusion['y_pred'] = np.append(confusion['y_pred'], pred.detach().cpu().numpy().astype(np.int32))
            epoch_loss += batchloss.detach().item()
    return epoch_loss / len(iterator), confusion

def test(model, iterator, device, t, threshold=0):
    test_results = {'cat1':[],'cat2':[],'label':[],'prediction':[]}
    confusion = {'y': np.array([],dtype=np.int32), 'y_pred': np.array([],dtype=np.int32)}
    model.eval()
    d = int(model.output_fc.out_features / 2)
    with torch.no_grad():
        for sample in tqdm(iterator, desc="Evaluating", leave=False):
            test_results['cat1'] += sample['cat1']
            test_results['cat2'] += sample['cat2']
            test_results['label'] += sample['label'].numpy().astype(np.int32).tolist()
            x1 = sample['feature1'].to(device)
            x2 = sample['feature2'].to(device)
            label = sample['label'].to(device)
            confusion['y'] = np.append(confusion['y'], sample['label'].numpy().astype(np.int32))
            y1 = model(x1)
            y2 = model(x2)
            n = len(sample['label'])
            pred = loss.softbox_loss(y1, y2, n, d, t, label, device, threshold)[1].detach().cpu().numpy().astype(np.int32).tolist()
            test_results['prediction'] += pred
            confusion['y_pred'] = np.append(confusion['y_pred'], pred)
    return pd.DataFrame(test_results),confusion

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def build_confusion_matrix(confusion):
    cm = metrics.confusion_matrix(confusion['y'], confusion['y_pred'],labels=[True,False])
    tp,fn,fp,tn = cm.ravel()
    a = (tp + tn) / (tp + fn + fp + tn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    df = pd.DataFrame(cm,index=['Label subsumption', 'Label negative'], columns = pd.MultiIndex.from_product([[f'F1={f1:.4f}, precision={p:.4f}, recall={r:.4f}'],['Predict subsumption', 'Predict negative']]))
    return df