# bert_subsumption

Subsumption prediction on e-commerce taxonomies

## Abstract

This work studies the characteristics of e-commerce taxonomies, and propose a new subsumption prediction method based on the pre-trained language model BERT that is well adapted to the e-commerce setting. The proposed model utilises textual and structural semantics in a taxonomy, as well as the rich and noisy instance (item) information. Experiments have been conducted on two large-scale e-commerce taxonomies from eBay (proprietary) and AliOpenKG (public), that our method offers substantial improvement over strong baselines.
Click [here](https://drive.google.com/drive/folders/1HBxaNTjNh7JttlwC2i5hdI5bhsbhsCB3?usp=sharing) for the extracted item-augmented taxonomy of AliOpenKG, the subset of AliOpenKG that this work is concerned with.

## Dependencies

- `torch`
- `transformers`
- `sklearn`
- `tqdm`
- `box-embeddings`

## Experiments

### Setup

Download the AliOpenKG taxonomy to the repository folder and set `data[tbox_path]` and `data[abox_path]` to the corresponding paths for TBox and ABox data.
A pretrained BERT model is required to run some experiments. Please set `pretrained[pretrained_path]` and `pretrained[tokenizer_path]` to your BERT model/ tokenizer.

### Running

The experiments with linear classifier, box embedding classifier and k-NN classifier are implemented in `/experiments/run/linear.ipynb`, `/experiments/run/box.ipynb` and `/experiments/run/knn.ipynb` respectively.
