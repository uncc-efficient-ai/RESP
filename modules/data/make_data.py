"""Construct and cache the calibration data used by the paper."""
import os
import pickle
from pathlib import Path

from . import data_prune_ds, data_prune

def make_prune_data(prune_config, model_config, seed, tokenizer, model=None, full_config=None):
    print("Loading calibration data")
    cache = Path(full_config.task.datasets_folder) / 'calibration'
    cache.mkdir(parents=True, exist_ok=True)
    saved_path = prune_config.prune_dataset.extra_config.saved_path
    if saved_path:
        Path(saved_path).parent.mkdir(parents=True, exist_ok=True)
    if prune_config.prune_dataset.pickle_dump:
        if os.path.exists(prune_config.prune_dataset.pickle.prune_path):
            with open(prune_config.prune_dataset.pickle.prune_path, 'rb') as f:
                dataloader = pickle.load(f)
        else:
            raise FileNotFoundError
    else:
        if prune_config.prune_dataset.type in ['downstream']:
            dataloader, _ = data_prune_ds.get_loaders(prune_config.prune_dataset.name,
                                                      seed=seed, tokenizer=tokenizer,
                                                      total_budget=prune_config.prune_dataset.total_budget,
                                                      extra_config=prune_config.prune_dataset.extra_config, model=model, full_config=full_config)
        else:
            dataloader, _ = data_prune.get_loaders(prune_config.prune_dataset.name,
                                                   nsamples=prune_config.prune_dataset.n_samples,
                                                   seed=seed, seqlen=prune_config.prune_dataset.seq_len,
                                                   tokenizer=tokenizer,
                                                   data_path=prune_config.prune_dataset.path,
                                                   base_model=model_config.name)
        with (cache / f'{prune_config.prune_dataset.name}_prune_data.pkl').open('wb') as f:
            pickle.dump(dataloader, f)

    print("dataset loading complete")
    return dataloader
