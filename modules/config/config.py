"""Experiment configuration with repository-relative, portable paths."""

import os
import re
from pathlib import Path

import yaml
from addict import Dict

ROOT = Path(__file__).resolve().parents[2]
PATH_KEYS = {'output_folder', 'output_path', 'log_file_path', 'datasets_folder',
             'saved_path', 'checkpoint_path', 'refreshed_dataloader_path',
             'custom_package_location', 'prune_path'}


class Config:
    class SafeLoaderWithJoin(yaml.SafeLoader):
        pass

    def __init__(self, path, resolve_paths=True):
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        with path.open() as stream:
            values = yaml.load(stream, Loader=self.SafeLoaderWithJoin)
        if not isinstance(values, dict):
            raise ValueError('The configuration must be a YAML mapping.')
        if resolve_paths:
            values = self._resolve(values)
        values['config_path'] = str(path.resolve())
        self.config = Dict(values)
        self.validate()

    @classmethod
    def _resolve(cls, value, key=None):
        if isinstance(value, dict):
            return {k: cls._resolve(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._resolve(v) for v in value]
        if isinstance(value, str) and key in PATH_KEYS:
            expanded = os.path.expandvars(value)
            if re.search(r'\$\{[^}]+\}|\$[A-Za-z_]\w*', expanded):
                raise ValueError(f'Set the environment variable required by {key}: {value}')
            path = Path(expanded).expanduser()
            return str(path if path.is_absolute() else ROOT / path)
        return value

    def validate(self):
        c = self.config
        if c.task.task_mode not in {'prune', 'test'}:
            raise ValueError('Paper experiments support task_mode prune or test.')
        if c.model.name != 'Qwen/Qwen3-8B':
            raise ValueError('The paper configurations target Qwen/Qwen3-8B.')
        if not c.task.output_folder:
            raise ValueError('task.output_folder is required.')
        if c.task.task_mode == 'prune':
            p = c.task.prune
            if not p.prune_dataset.name or not p.prune_metric or not p.func_name:
                raise ValueError('Pruning requires a dataset, prune_metric and func_name.')
            if p.restore and not p.restore_config.checkpoint_path:
                raise ValueError('Restoring requires restore_config.checkpoint_path.')
        allowed = {'gsm8k_cot_sample', 'mathqa_decoding'}
        if not c.evaluation.lm_eval_options.tasks or not set(c.evaluation.lm_eval_options.tasks) <= allowed:
            raise ValueError('Only the paper GSM8K and MathQA evaluation tasks are supported.')

    def save_config(self, folder_path):
        path = Path(folder_path) / 'config.yml'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.config.to_dict(), sort_keys=False))
        return str(path)

    def get_config(self):
        return self.config

    def __getattr__(self, name):
        return getattr(self.config, name)


Config.SafeLoaderWithJoin.add_constructor(
    '!join', lambda loader, node: ''.join(str(v) for v in loader.construct_sequence(node)))
