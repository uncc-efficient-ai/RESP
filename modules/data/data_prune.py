import os


import re


import numpy as np


import random


import torch


import tqdm


from datasets import load_dataset, DownloadMode, concatenate_datasets, load_from_disk, Dataset, DatasetDict


class TokenizerWrapper:
    """
    Wrapper class for tokenized input IDs.

    Args:
        input_ids (tensor): The tokenized input IDs from the tokenizer.
    """

    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_c4(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the C4 (Common Crawl) dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation dataset.
    """
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                           split='validation')
    # traindata = load_dataset('json', data_files={'train': 'datasets/c4/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('json', data_files={'validation': 'datasets/c4/c4-validation.00000-of-00008.json.gz'}, split='validation')

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_loaders(name='c4', nsamples=500, seed=0, seqlen=2048, tokenizer=None, **kwargs):
    if name != 'c4':
        raise ValueError(f'Unsupported open-domain calibration dataset: {name}')
    return get_c4(nsamples, seed, seqlen, tokenizer)
