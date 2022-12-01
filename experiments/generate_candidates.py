import numpy as np
import pandas as pd
import json
import gc
from sklearn.neighbors import NearestNeighbors
from pybay.bert import EBertModel, EBertTokenizer
import torch
from torch import nn
from data import preprocess as prep

def get_lcs(path1, path2):
    if len(path1) > len(path2): return get_lcs(path2, path1)
    lcs = []
    for ind in range(len(path1)):
        if path1[ind] == path2[ind]:
            lcs.append(path1[ind])
        else:
            break
    return lcs

def compute_dist(path1, path2):
    lcs = get_lcs(path1, path2)
    dist = len(path1) + len(path2) - 2*len(lcs)
    return dist

def validate(c1, c2, threshold=3):
    if c1.find(c2) != -1 or c2.find(c1) != -1:
        return False
    elif compute_dist((c1 + ' > ').split(' > ')[:-1], (c2 + ' > ').split(' > ')[:-1]) < threshold:
        return False
    else:
        return True 

def batched_model(model, inputs, batch_size, model_is_cls=False, num_labels=2):
    batched_ids = torch.split(inputs['input_ids'], batch_size)
    batched_types = torch.split(inputs['token_type_ids'], batch_size)
    batched_mask = torch.split(inputs['attention_mask'], batch_size)
    if model_is_cls == True:
        output = np.empty([0,num_labels])
    else:
        output = np.empty([0,768])
    for x in range(len(batched_ids)):
        batch_input = {'input_ids': batched_ids[x],
                 'token_type_ids': batched_types[x],
                 'attention_mask': batched_mask[x]}
        if model_is_cls == True:
            batch_output = (model(**batch_input).logits.detach().cpu().numpy())
        else:
            batch_output = (model(**batch_input)[1].detach().cpu().numpy())
        output = np.concatenate((output, batch_output), axis=0)
    return output

def get_embeddings(config, model, tokenizer):
    classlist = prep.load_ebay_taxonomy(config)
    classlabels = [c[1] for c in classlist]
    batch_size = config['embedding']['batch_size']
    inputs = tokenizer(classlabels, padding = True, return_tensors='pt').to('cuda:0')
    output = batched_model(model, inputs, batch_size)
    return output

def save_candidates(candidates, path):
    cand_dict = {'id1': [], 'text1': [], 'id2': [], 'text2': []}
    for c in candidates:
        cand_dict['id1'].append(c[0][0])
        cand_dict['text1'].append(c[0][1])
        cand_dict['id2'].append(c[1][0])
        cand_dict['text2'].append(c[1][1])
    df = pd.DataFrame(cand_dict)
    df.to_csv(path + '_candidates.csv', index=False)
    
def load_candidates(path):
    df = pd.read_csv(path)
    id1 = list(df['id1'])
    text1 = list(df['text1'])
    id2 = list(df['id2'])
    text2 = list(df['text2'])
    candidates = []
    for idx in range(df.shape[0]):
        candidates.append(((id1[idx],text1[idx]),(id2[idx],text2[idx])))
    return candidates

def generate(config, model, tokenizer, filtering=False, save=False):
    print('Calculating embeddings...')
    embeddings = get_embeddings(config, model, tokenizer)
    print('Completed calculating embeddings.')
    print('Obtaining candidates with k-NN...')
    num_neighbours = config['candidate_generation']['knn_neighbours']
    knn_algorithm = config['candidate_generation']['knn_algorithm']
    nbrs = NearestNeighbors(n_neighbors=num_neighbours+1, algorithm=knn_algorithm).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    print('k-NN computation complete...')
    classlist = prep.load_ebay_taxonomy(config)
    candidates = []
    for row in indices:
        for entry in row[1:]:
            if filtering == False or validate(classlist[row[0]][1],classlist[entry][1]) == True:
                if int(classlist[row[0]][0]) < int(classlist[entry][0]):
                    candidates.append((classlist[row[0]], classlist[entry]))
                else:
                    candidates.append((classlist[entry], classlist[row[0]]))
    candidates = list(set(candidates))
    print('Completed obtaining candidates.')
    if save == True:
        save_candidates(candidates,'/data/ebay/notebooks/jingcshi/Curation/data/')
    return candidates