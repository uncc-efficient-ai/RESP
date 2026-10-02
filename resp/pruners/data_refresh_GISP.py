from modules.model.pruning import hooking_qwen3
import copy
import logging
import math
import os
import random
import sys
import time
import types
import warnings
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import Cache

from modules.eval.setup_eval import eval_lm_eval
from tasks.pruning.pruners import Pruner
from torch import nn
import subprocess
from pathlib import Path
import pathlib
import json, tempfile
from . import non_uniform_pruner
from .layerwrapper import *
from .utils import *
import pickle
import threading

logger = logging.getLogger(__name__)


class data_refresh_GISP(non_uniform_pruner):
    def __init__(self, model, config, data):
        super().__init__(model, config, data)
        self.dense_model = self.back_up_model()
        # self.val_data = [d for d in self.data if
        #                  d["config_name"] == self.config.task.prune.prune_dataset.extra_config.used_config]

        self.before_pruning()
        self.is_gqa = (self.get_model().config.num_key_value_heads < self.get_model().config.num_attention_heads)
        if self.is_gqa:
            self.origin_function = {}
            self.gqa_mask_record = {}

            def monkey_patch_forward():
                layers = self.get_layers()
                for index, l in enumerate(layers):
                    attn_block = getattr(l, self.layer_mapping['attn']['block'])
                    self.origin_function['func'] = type(attn_block).forward
                    self.gqa_mask_record[index] = torch.ones(self.get_model().config.num_attention_heads,
                                                             device=self.get_model().device,
                                                             dtype=getattr(attn_block, self.layer_mapping['attn'][
                                                                 'q_name']).weight.data.dtype)
                    attn_block.forward = types.MethodType(
                        hooking_qwen3(self.gqa_mask_record[index], real_prune=False),
                        attn_block)

            def remove_patch_all():
                layers = self.get_layers()
                for index, l in enumerate(layers):
                    attn_block = getattr(l, self.layer_mapping['attn']['block'])
                    attn_block.forward = types.MethodType(self.origin_function['func'], attn_block)

            monkey_patch_forward()
        self.milestone_ratios = self.config.task.prune.refresh_config.milestone_ratios
        self.milestone_ratios_marker = [False for item in self.milestone_ratios]

    def prune(self):
        func_name = self.config.task.prune.func_name
        if func_name in ['global_grad_sp_v2']:
            self.global_grad_sp_v2()
        elif func_name in ['restore_then_eval']:
            self.restore_then_eval()

    def step(self):
        pass

    def back_up_model(self):
        ori_devices = self.get_model().device
        self.get_model().to('cpu')
        backup_model = copy.deepcopy(self.model)
        self.get_model().to(ori_devices)
        return backup_model

    def restore_model(self, backup_model):
        ori_devices = self.get_model().device
        self.get_model().to('cpu')
        self.model = copy.deepcopy(backup_model)
        self.get_model().to(ori_devices)

    def restore_dense_model(self):
        if self.dense_model is None:
            raise ValueError
        ori_devices = self.get_model().device
        self.get_model().to('cpu')
        self.model = copy.deepcopy(self.dense_model)
        self.get_model().to(ori_devices)

    def global_grad_sp_v2(self):
        head_dim = self.get_model().config.hidden_size // self.get_model().config.num_attention_heads
        head_num = self.get_model().config.num_attention_heads
        hidden_dim = self.get_model().config.hidden_size
        intermediate_size = self.get_model().config.intermediate_size
        layers = self.get_layers()
        metric_target = self.real_metrics_mapping()

        def obtain_info_iterative(layer_idxs):
            layers = self.get_layers()
            avg_loss = self.fill_information()
            grad_norm = {}
            weight_norm = {}
            grad_acc_norm = {}
            W_dicts = {}
            if isinstance(self.data_pos, dict):
                n_samples = sum(x.size(0) for x in self.data_pos.values())
            else:
                if self.config.task.prune.prune_dataset.type in ['open_domain']:
                    n_samples = self.config.task.prune.prune_dataset.n_samples
                else:
                    n_samples = self.data_pos.size(0)

            if isinstance(self.data_neg, dict):
                n_samples_neg = sum(x.size(0) for x in self.data_neg.values())
            else:
                if self.config.task.prune.prune_dataset.type in ['open_domain']:
                    n_samples_neg = self.config.task.prune.prune_dataset.n_samples
                else:
                    n_samples_neg = self.data_neg.size(0)

            for x in tqdm(range(layer_idxs, len(layers)), desc="Processing layers"):
                layer = layers[x]

                if self.config.task.prune.prune_modules in ['mha', 'all']:
                    subset = {}
                    subset.update(
                        {self.layer_mapping['attn']['q']: find_layers(layer)[self.layer_mapping['attn']['q']]})
                    subset.update(
                        {self.layer_mapping['attn']['k']: find_layers(layer)[self.layer_mapping['attn']['k']]})
                    subset.update(
                        {self.layer_mapping['attn']['v']: find_layers(layer)[self.layer_mapping['attn']['v']]})
                    subset.update(
                        {self.layer_mapping['attn']['o']: find_layers(layer)[self.layer_mapping['attn']['o']]})

                    for name in subset:
                        print(f"pruning layer {x} name {name}")
                        current_layer = subset[name]
                        norms = torch.norm(current_layer.weight)
                        weight_norm[f"{x}.{name}"] = norms.detach().clone().cpu()

                        norms = torch.norm(current_layer.weight.grad)
                        grad_norm[f"{x}.{name}"] = norms.detach().clone().cpu()
                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            norms = torch.norm(current_layer.weight.acc_grad)
                            grad_acc_norm[f"{x}.{name}"] = norms.detach().clone().cpu()

                        W_metric_1st = current_layer.weight * (current_layer.weight.grad / n_samples)
                        W_metric_weight = current_layer.weight
                        W_metric_grad = (current_layer.weight.grad / n_samples)

                        if hasattr(current_layer.weight, 'grad_neg') and current_layer.weight.grad_neg is not None:
                            W_metric_1st_neg = current_layer.weight * (
                                    current_layer.weight.grad_neg.to(current_layer.weight.device) / n_samples_neg)
                            W_metric_1st = W_metric_1st - W_metric_1st_neg
                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            W_metric_second = current_layer.weight * current_layer.weight.acc_grad * current_layer.weight
                            W_metric_mix = W_metric_1st - 0.5 * current_layer.weight * current_layer.weight.acc_grad * current_layer.weight

                        W_metric_1st = W_metric_1st.abs()
                        W_metric_weight = W_metric_weight.abs()
                        W_metric_grad = W_metric_grad.abs()

                        # if hasattr(current_layer.weight, 'grad_neg'):
                        #     W_metric_1st_neg = W_metric_1st_neg.abs()

                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            W_metric_second = W_metric_second.abs()
                            W_metric_mix = W_metric_mix.abs()

                        if name in [self.layer_mapping['attn']['o']]:
                            W_metric_1st = W_metric_1st.t()
                            W_metric_weight = W_metric_weight.t()
                            W_metric_grad = W_metric_grad.t()

                            # if hasattr(current_layer.weight, 'grad_neg'):
                            #     W_metric_1st_neg = W_metric_1st_neg.t()

                            if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                                W_metric_second = W_metric_second.t()
                                W_metric_mix = W_metric_mix.t()

                        if self.is_gqa and name in [self.layer_mapping['attn']['k'], self.layer_mapping['attn']['v']]:
                            continue

                        W_metric_1st = W_metric_1st.reshape(W_metric_1st.shape[0] // head_dim, -1)
                        W_metric_1st = W_metric_1st.sum(1)
                        W_metric_weight = W_metric_weight.reshape(W_metric_weight.shape[0] // head_dim, -1)
                        W_metric_weight = W_metric_weight.sum(1)
                        W_metric_grad = W_metric_grad.reshape(W_metric_grad.shape[0] // head_dim, -1)
                        W_metric_grad = W_metric_grad.sum(1)

                        # if hasattr(current_layer.weight, 'grad_neg'):
                        #     W_metric_1st_neg = W_metric_1st_neg.reshape(W_metric_1st_neg.shape[0] // head_dim, -1)
                        #     W_metric_1st_neg = W_metric_1st_neg.sum(1)

                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            W_metric_second = W_metric_second.reshape(W_metric_second.shape[0] // head_dim, -1)
                            W_metric_second = W_metric_second.sum(1)
                            W_metric_mix = W_metric_mix.reshape(W_metric_mix.shape[0] // head_dim, -1)
                            W_metric_mix = W_metric_mix.sum(1)

                        if f"{x}.{self.layer_mapping['attn']['block']}" in W_dicts:
                            W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                '1st'] += W_metric_1st.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                'weight'] += W_metric_weight.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                'grad'] += W_metric_grad.detach().clone().cpu()
                            # if hasattr(current_layer.weight, 'grad_neg'):
                            #     W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                            #         '1st_neg'] += W_metric_1st_neg.detach().clone().cpu()
                            if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                                W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                    '2rd'] += W_metric_second.detach().clone().cpu()
                                W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                    'mix'] += W_metric_mix.detach().clone().cpu()
                        else:
                            W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"] = {}
                            W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                '1st'] = W_metric_1st.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                'weight'] = W_metric_weight.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                'grad'] = W_metric_grad.detach().clone().cpu()
                            # if hasattr(current_layer.weight, 'grad_neg'):
                            #     W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                            #         '1st_neg'] = W_metric_1st_neg.detach().clone().cpu()

                            if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                                W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                    '2rd'] = W_metric_second.detach().clone().cpu()
                                W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                    'mix'] = W_metric_mix.detach().clone().cpu()
                        del W_metric_1st, W_metric_weight, W_metric_grad
                        if hasattr(current_layer.weight, 'grad_neg') and current_layer.weight.grad_neg is not None:
                            del W_metric_1st_neg
                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            del W_metric_second, W_metric_mix

                if self.config.task.prune.prune_modules in ['mlp', 'all']:
                    subset = {}
                    subset.update({self.layer_mapping['mlp']['u']: find_layers(layer)[self.layer_mapping['mlp']['u']]})
                    if 'g' in self.layer_mapping['mlp']:
                        subset.update(
                            {self.layer_mapping['mlp']['g']: find_layers(layer)[self.layer_mapping['mlp']['g']]})
                    subset.update({self.layer_mapping['mlp']['d']: find_layers(layer)[self.layer_mapping['mlp']['d']]})

                    for name in subset:
                        print(f"pruning layer {x} name {name}")
                        current_layer = subset[name]
                        norms = torch.norm(current_layer.weight)
                        weight_norm[f"{x}.{name}"] = norms.detach().clone().cpu()

                        norms = torch.norm(current_layer.weight.grad)
                        grad_norm[f"{x}.{name}"] = norms.detach().clone().cpu()
                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            norms = torch.norm(current_layer.weight.acc_grad)
                            grad_acc_norm[f"{x}.{name}"] = norms.detach().clone().cpu()

                        W_metric_1st = current_layer.weight * (current_layer.weight.grad / n_samples)
                        W_metric_weight = current_layer.weight
                        W_metric_grad = (current_layer.weight.grad / n_samples)

                        if hasattr(current_layer.weight, 'grad_neg') and current_layer.weight.grad_neg is not None:
                            W_metric_1st_neg = current_layer.weight * (
                                    current_layer.weight.grad_neg.to(current_layer.weight.device) / n_samples_neg)
                            W_metric_1st = W_metric_1st - W_metric_1st_neg
                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            W_metric_second = current_layer.weight * current_layer.weight.acc_grad * current_layer.weight
                            W_metric_mix = W_metric_1st - 0.5 * current_layer.weight * current_layer.weight.acc_grad * current_layer.weight

                        W_metric_1st = W_metric_1st.abs()
                        W_metric_weight = W_metric_weight.abs()
                        W_metric_grad = W_metric_grad.abs()

                        # if hasattr(current_layer.weight, 'grad_neg'):
                        #     W_metric_1st_neg = W_metric_1st_neg.abs()

                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            W_metric_second = W_metric_second.abs()
                            W_metric_mix = W_metric_mix.abs()

                        if name in [self.layer_mapping['mlp']['d']]:
                            W_metric_1st = W_metric_1st.t()
                            W_metric_weight = W_metric_weight.t()
                            W_metric_grad = W_metric_grad.t()

                            # if hasattr(current_layer.weight, 'grad_neg'):
                            #     W_metric_1st_neg = W_metric_1st_neg.t()

                            if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                                W_metric_second = W_metric_second.t()
                                W_metric_mix = W_metric_mix.t()

                        W_metric_1st = W_metric_1st.sum(1)
                        W_metric_weight = W_metric_weight.sum(1)
                        W_metric_grad = W_metric_grad.sum(1)

                        # if hasattr(current_layer.weight, 'grad_neg'):
                        #     W_metric_1st_neg = W_metric_1st_neg.sum(1)

                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            W_metric_second = W_metric_second.sum(1)
                            W_metric_mix = W_metric_mix.sum(1)

                        if f"{x}.{self.layer_mapping['mlp']['block']}" in W_dicts:
                            W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                '1st'] += W_metric_1st.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                'weight'] += W_metric_weight.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                'grad'] += W_metric_grad.detach().clone().cpu()
                            # if hasattr(current_layer.weight, 'grad_neg'):
                            #     W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                            #         '1st_neg'] += W_metric_1st_neg.detach().clone().cpu()

                            if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                                W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                    '2rd'] += W_metric_second.detach().clone().cpu()
                                W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                    'mix'] += W_metric_mix.detach().clone().cpu()

                        else:
                            W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"] = {}
                            W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                '1st'] = W_metric_1st.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                'weight'] = W_metric_weight.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                'grad'] = W_metric_grad.detach().clone().cpu()
                            # if hasattr(current_layer.weight, 'grad_neg'):
                            #     W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                            #         '1st_neg'] = W_metric_1st_neg.detach().clone().cpu()
                            if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                                W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                    '2rd'] = W_metric_second.detach().clone().cpu()
                                W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                    'mix'] = W_metric_mix.detach().clone().cpu()
                        del W_metric_1st, W_metric_weight, W_metric_grad
                        if hasattr(current_layer.weight, 'grad_neg') and current_layer.weight.grad_neg is not None:
                            del W_metric_1st_neg
                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            del W_metric_second, W_metric_mix

                torch.cuda.empty_cache()
            if not self.config.task.prune.prune_separate:
                for k, v in W_dicts.items():
                    if f"{self.layer_mapping['mlp']['block']}" in k:
                        structural_size = intermediate_size
                        if 'g' in self.layer_mapping['mlp']:
                            structural_size = structural_size * 3
                        else:
                            structural_size = structural_size * 2
                    elif f"{self.layer_mapping['attn']['block']}" in k:
                        structural_size = 4 * head_dim * hidden_dim
                    else:
                        raise ValueError
                    for v_key, v_value in v.items():
                        v[v_key] = v_value / structural_size
            for k, v in W_dicts.items():
                v['new_1st'] = v['1st']
                # first = v['1st'].float()
                # first_neg = v['1st_neg'].float()
                # v['new_1st'] = first
                # invalid_division = (first_neg == 0) & (first != 0)
                # if invalid_division.any():
                #     raise ValueError
                # result = torch.zeros_like(first)
                # 只在 first_neg 不为零的位置执行除法
                # valid_mask = first_neg != 0
                # result[valid_mask] = first[valid_mask] / first_neg[valid_mask]
                # 赋值给新键
                # v['new_1st_ratio'] = result
                # v['new_1st'] = v['new_1st_ratio'] * first
            return W_dicts, grad_norm, weight_norm, grad_acc_norm, avg_loss

        def prune_oneshot(W_dicts, inter_layer_imp, tr, bypassed_blocks):
            self.before_pruning_step()
            current_mask_record = {}
            layers = self.get_layers()
            # target scope: layer_idxs to len(layers)
            metrics_mha_std = torch.tensor([])
            metrics_mlp_std = torch.tensor([])
            # tr = self.min_max_scope_limit(tr)
            # self.model = self.get_model().to('cpu')
            torch.cuda.empty_cache()
            start_time = time.time()
            global_start = time.time()
            for k, v in W_dicts.items():
                name = k.split('.')
                if int(name[0]) not in range(int(len(layers) * 0.1), len(layers) - 1):
                    if self.config.task.prune.prune_skip:
                        new_v = torch.zeros_like(v[metric_target])
                        new_v = torch.fill(new_v, torch.inf)
                        if name[1] in [self.layer_mapping['attn']['block']]:
                            metrics_mha_std = torch.cat((metrics_mha_std, new_v), dim=0)
                        else:
                            metrics_mlp_std = torch.cat((metrics_mlp_std, new_v), dim=0)
                        continue
                if f"{name[0]}.{name[1]}" in bypassed_blocks:
                    new_v = v[metric_target].clone()
                    non_zero_mask = (new_v != 0)
                    new_v[non_zero_mask] = torch.inf
                    if name[1] in [self.layer_mapping['attn']['block']]:
                        metrics_mha_std = torch.cat((metrics_mha_std, new_v), dim=0)
                    else:
                        metrics_mlp_std = torch.cat((metrics_mlp_std, new_v), dim=0)
                    continue
                if name[1] in [self.layer_mapping['attn']['block']]:
                    if self.config.task.prune.prune_modules in ['mha', 'all']:
                        metrics_mha_std = torch.cat((metrics_mha_std, v[metric_target]), dim=0)
                else:
                    if self.config.task.prune.prune_modules in ['mlp', 'all']:
                        metrics_mlp_std = torch.cat((metrics_mlp_std, v[metric_target]), dim=0)

            pruning_ratio_record_mha = {}
            pruning_ratio_record_mlp = {}
            if self.config.task.prune.prune_separate:
                logger.info(f'cat metrics time cost: {time.time() - start_time}')
                if self.config.task.prune.prune_modules in ['mha', 'all']:
                    start_time = time.time()
                    metrics_mha_std = metrics_mha_std.to('cuda')
                    metrics_mha_std = metrics_mha_std + torch.abs(torch.min(metrics_mha_std))
                    # for z in range(len(layers)):
                    #     metrics_mha_std[head_num * z:(z + 1) * head_num] *= (inter_layer_imp[z])
                    origin_size = metrics_mha_std.size()
                    metrics_mha_std = metrics_mha_std.view(-1)
                    N = metrics_mha_std.numel()
                    k_mha = int(N * (1 - tr))
                    k_mha = N - k_mha + 1
                    metrics_mha_std = metrics_mha_std.to('cpu')
                    threshold, _ = torch.kthvalue(metrics_mha_std, k=k_mha)
                    threshold = threshold.to('cuda')
                    metrics_mha_std = metrics_mha_std.to('cuda')

                    W_mask_mha = (metrics_mha_std <= threshold)
                    W_mask_mha = W_mask_mha.view(origin_size)
                    W_mask_mha = W_mask_mha.to('cpu')
                    logger.info(f'threshold for mha: {threshold}')
                    logger.info(f'total sparsity for mha: {W_mask_mha.sum() / N}')
                    logger.info(f'kth & generate mask for mha time cost: {time.time() - start_time}')

                    mha_size = head_num * head_dim

                    for i in range(len(layers)):
                        if self.is_gqa:
                            submask_layer = W_mask_mha[i * head_num: (i + 1) * head_num]
                            submask_q = submask_layer.repeat_interleave(head_dim)
                            submask_k = submask_layer
                            submask_v = submask_layer
                            submask_o = submask_layer.repeat_interleave(head_dim)
                        else:
                            submask_layer = W_mask_mha[i * head_num: (i + 1) * head_num]
                            submask_q = submask_layer.repeat_interleave(head_dim)
                            submask_k = submask_layer.repeat_interleave(head_dim)
                            submask_v = submask_layer.repeat_interleave(head_dim)
                            submask_o = submask_layer.repeat_interleave(head_dim)
                        for name, vis_name, submask in zip(
                                [self.layer_mapping['attn']['q'], self.layer_mapping['attn']['k'],
                                 self.layer_mapping['attn']['v'], self.layer_mapping['attn']['o']],
                                [self.layer_mapping['attn']['q_name'],
                                 self.layer_mapping['attn']['k_name'],
                                 self.layer_mapping['attn']['v_name'],
                                 self.layer_mapping['attn']['o_name']], [
                                    submask_q, submask_k, submask_v, submask_o]):
                            self.during_pruning_step(find_layers(layers[i])[name], submask, metrics_mha_std, threshold)
                            if name in [self.layer_mapping['attn']['o']]:
                                find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                            else:
                                if self.is_gqa:
                                    if name in [self.layer_mapping['attn']['k'], self.layer_mapping['attn']['v']]:
                                        self.gqa_mask_record[i].data[submask] = 0
                                    else:
                                        find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
                                else:
                                    find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero

                            if vis_name not in pruning_ratio_record_mha:
                                pruning_ratio_record_mha[vis_name] = [submask.sum() / mha_size]
                            else:
                                pruning_ratio_record_mha[vis_name].append(submask.sum() / mha_size)
                    del metrics_mha_std
                    del W_mask_mha

                torch.cuda.empty_cache()

                if self.config.task.prune.prune_modules in ['mlp', 'all']:
                    start_time = time.time()
                    metrics_mlp_std = metrics_mlp_std.to('cuda')
                    metrics_mlp_std = metrics_mlp_std + torch.abs(torch.min(metrics_mlp_std))
                    # for z in range(len(layers)):
                    #     metrics_mlp_std[z * intermediate_size:(z + 1) * intermediate_size] *= (inter_layer_imp[z])
                    origin_size = metrics_mlp_std.size()
                    metrics_mlp_std = metrics_mlp_std.view(-1)
                    N = metrics_mlp_std.numel()
                    k_mlp = int(N * (1 - tr))
                    k_mlp = N - k_mlp + 1
                    metrics_mlp_std = metrics_mlp_std.to('cpu')
                    mlp_threshold, ndx = torch.kthvalue(metrics_mlp_std, k=k_mlp)
                    mlp_threshold = mlp_threshold.to('cuda')
                    metrics_mlp_std = metrics_mlp_std.to('cuda')
                    W_mask_mlp = (metrics_mlp_std <= mlp_threshold)
                    W_mask_mlp = W_mask_mlp.view(origin_size)
                    W_mask_mlp = W_mask_mlp.to('cpu')
                    logger.info(f'threshold for mlp: {mlp_threshold}')
                    logger.info(f'total sparsity for mlp: {W_mask_mlp.sum() / N}')
                    logger.info(f'kth & generate mask for mlp time cost: {time.time() - start_time}')

                    mlp_size = intermediate_size

                    for i in range(len(layers)):
                        if 'g' in self.layer_mapping['mlp']:
                            submask_layer = W_mask_mlp[i * intermediate_size:(i + 1) * intermediate_size]
                            submask_u = submask_layer
                            submask_g = submask_layer
                            submask_d = submask_layer
                            for name, vis_name, submask in zip(
                                    [self.layer_mapping['mlp']['u'], self.layer_mapping['mlp']['g'],
                                     self.layer_mapping['mlp']['d']],
                                    [self.layer_mapping['mlp']['u_name'],
                                     self.layer_mapping['mlp']['g_name'],
                                     self.layer_mapping['mlp']['d_name']], [
                                        submask_u, submask_g, submask_d]):
                                self.during_pruning_step(find_layers(layers[i])[name], submask, metrics_mlp_std,
                                                         mlp_threshold)
                                if name in [self.layer_mapping['mlp']['d']]:
                                    find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                                else:
                                    find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
                                if vis_name not in pruning_ratio_record_mlp:
                                    pruning_ratio_record_mlp[vis_name] = [submask.sum() / mlp_size]
                                else:
                                    pruning_ratio_record_mlp[vis_name].append(submask.sum() / mlp_size)
                        else:
                            submask_layer = W_mask_mlp[i * intermediate_size:(i + 1) * intermediate_size]
                            submask_u = submask_layer
                            submask_d = submask_layer
                            for name, vis_name, submask in zip(
                                    [self.layer_mapping['mlp']['u'],
                                     self.layer_mapping['mlp']['d']],
                                    [self.layer_mapping['mlp']['u_name'],
                                     self.layer_mapping['mlp']['d_name']], [
                                        submask_u, submask_d]):
                                self.during_pruning_step(find_layers(layers[i])[name], submask, metrics_mlp_std,
                                                         mlp_threshold)
                                if name in [self.layer_mapping['mlp']['d']]:
                                    find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                                else:
                                    find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero

                                if vis_name not in pruning_ratio_record_mlp:
                                    pruning_ratio_record_mlp[vis_name] = [submask.sum() / mlp_size]
                                else:
                                    pruning_ratio_record_mlp[vis_name].append(submask.sum() / mlp_size)
                    del metrics_mlp_std
                    del W_mask_mlp
            else:
                if self.config.task.prune.prune_modules in ['all']:
                    size_mha = metrics_mha_std.size()
                    size_mlp = metrics_mlp_std.size()
                    metrics_all_std = torch.cat((metrics_mha_std, metrics_mlp_std), dim=0)
                    logger.info(f'cat metrics time cost: {time.time() - start_time}')

                    start_time = time.time()
                    metrics_all_std = metrics_all_std.to('cuda')
                    metrics_all_std = metrics_all_std + torch.abs(torch.min(metrics_all_std))
                    # for z in range(len(layers)):
                    #     metrics_mlp_std[z * intermediate_size:(z + 1) * intermediate_size] *= (inter_layer_imp[z])
                    origin_size = metrics_all_std.size()
                    metrics_all_std = metrics_all_std.view(-1)
                    N = metrics_all_std.numel()
                    k_all = int(N * (1 - tr))
                    k_all = N - k_all + 1
                    metrics_all_std = metrics_all_std.to('cpu')
                    all_threshold, ndx = torch.kthvalue(metrics_all_std, k=k_all)
                    all_threshold = all_threshold.to('cuda')
                    metrics_all_std = metrics_all_std.to('cuda')
                    W_mask_all = (metrics_all_std <= all_threshold)
                    W_mask_all = W_mask_all.view(origin_size)
                    W_mask_all = W_mask_all.to('cpu')
                    logger.info(f'threshold for all: {all_threshold}')
                    logger.info(f'total sparsity for all: {W_mask_all.sum() / N}')
                    logger.info(f'kth & generate mask for all time cost: {time.time() - start_time}')

                    W_mask_mha = W_mask_all[:size_mha[0]]
                    W_mask_mlp = W_mask_all[size_mha[0]:size_mha[0] + size_mlp[0]]
                    mha_ratio = W_mask_all[:size_mha[0]].sum() / size_mha[0]
                    mlp_ratio = W_mask_all[size_mha[0]:size_mha[0] + size_mlp[0]].sum() / size_mlp[0]
                    del metrics_mha_std
                    del metrics_mlp_std
                    mha_size = head_num * head_dim

                    for i in range(len(layers)):
                        submask_layer = W_mask_mha[i * head_num: (i + 1) * head_num]
                        current_mask_record[f"{i}.{self.layer_mapping['attn']['block']}"] = submask_layer
                        if self.is_gqa:
                            submask_q = submask_layer.repeat_interleave(head_dim)
                            submask_k = submask_layer
                            submask_v = submask_layer
                            submask_o = submask_layer.repeat_interleave(head_dim)
                        else:
                            submask_q = submask_layer.repeat_interleave(head_dim)
                            submask_k = submask_layer.repeat_interleave(head_dim)
                            submask_v = submask_layer.repeat_interleave(head_dim)
                            submask_o = submask_layer.repeat_interleave(head_dim)

                        for name, vis_name, submask in zip(
                                [self.layer_mapping['attn']['q'], self.layer_mapping['attn']['k'],
                                 self.layer_mapping['attn']['v'], self.layer_mapping['attn']['o']],
                                [self.layer_mapping['attn']['q_name'],
                                 self.layer_mapping['attn']['k_name'],
                                 self.layer_mapping['attn']['v_name'],
                                 self.layer_mapping['attn']['o_name']], [
                                    submask_q, submask_k, submask_v, submask_o]):
                            self.during_pruning_step(find_layers(layers[i])[name], submask, metrics_all_std,
                                                     all_threshold)
                            if name in [self.layer_mapping['attn']['o']]:
                                find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                            else:
                                if self.is_gqa:
                                    if name in [self.layer_mapping['attn']['k'], self.layer_mapping['attn']['v']]:
                                        self.gqa_mask_record[i].data[submask] = 0
                                    else:
                                        find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
                                else:
                                    find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero

                            if vis_name not in pruning_ratio_record_mha:
                                pruning_ratio_record_mha[vis_name] = [submask.sum() / mha_size]
                            else:
                                pruning_ratio_record_mha[vis_name].append(submask.sum() / mha_size)
                    del W_mask_mha
                    mlp_size = intermediate_size
                    for i in range(len(layers)):
                        submask_layer = W_mask_mlp[i * intermediate_size:(i + 1) * intermediate_size]
                        current_mask_record[f"{i}.{self.layer_mapping['mlp']['block']}"] = submask_layer
                        if 'g' in self.layer_mapping['mlp']:
                            submask_u = submask_layer
                            submask_g = submask_layer
                            submask_d = submask_layer
                            for name, vis_name, submask in zip(
                                    [self.layer_mapping['mlp']['u'], self.layer_mapping['mlp']['g'],
                                     self.layer_mapping['mlp']['d']],
                                    [self.layer_mapping['mlp']['u_name'],
                                     self.layer_mapping['mlp']['g_name'],
                                     self.layer_mapping['mlp']['d_name']], [
                                        submask_u, submask_g, submask_d]):
                                self.during_pruning_step(find_layers(layers[i])[name], submask, metrics_all_std,
                                                         all_threshold)
                                if name in [self.layer_mapping['mlp']['d']]:
                                    find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                                else:
                                    find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
                                if vis_name not in pruning_ratio_record_mlp:
                                    pruning_ratio_record_mlp[vis_name] = [submask.sum() / mlp_size]
                                else:
                                    pruning_ratio_record_mlp[vis_name].append(submask.sum() / mlp_size)
                        else:
                            submask_u = submask_layer
                            submask_d = submask_layer
                            for name, vis_name, submask in zip(
                                    [self.layer_mapping['mlp']['u'],
                                     self.layer_mapping['mlp']['d']],
                                    [self.layer_mapping['mlp']['u_name'],
                                     self.layer_mapping['mlp']['d_name']], [
                                        submask_u, submask_d]):
                                self.during_pruning_step(find_layers(layers[i])[name], submask, metrics_all_std,
                                                         all_threshold)
                                if name in [self.layer_mapping['mlp']['d']]:
                                    find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                                else:
                                    find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero

                                if vis_name not in pruning_ratio_record_mlp:
                                    pruning_ratio_record_mlp[vis_name] = [submask.sum() / mlp_size]
                                else:
                                    pruning_ratio_record_mlp[vis_name].append(submask.sum() / mlp_size)
                    del W_mask_mlp
                    del W_mask_all
                    del metrics_all_std

                else:
                    raise ValueError()
            logger.info(f'global time cost: {time.time() - global_start}')
            # self.model = self.get_model().to('cuda')
            torch.cuda.empty_cache()
            self.save_helper.append("actual_mask", current_mask_record)
            if self.config.task.prune.prune_separate:
                return pruning_ratio_record_mha, pruning_ratio_record_mlp
            else:
                return pruning_ratio_record_mha, pruning_ratio_record_mlp, mha_ratio, mlp_ratio

        def evaluate_intermediate(eval_intermediate_lists, intermediate_index, prev_sparsity, prev_model, cur_sparsity,
                                  cur_model):
            if intermediate_index >= len(eval_intermediate_lists):
                return False
            next_intermediate_sparsity = eval_intermediate_lists[intermediate_index]
            flag = prev_sparsity <= next_intermediate_sparsity <= cur_sparsity
            if flag:
                distance_prev = abs(prev_sparsity - next_intermediate_sparsity)
                distance_cur = abs(cur_sparsity - next_intermediate_sparsity)
                if distance_prev < distance_cur:
                    cur_model = cur_model.to('cpu')
                    prev_model = prev_model.to('cuda')
                    eval_lm_eval(prev_model, self.tokenizer, self.config, f'sp_{prev_sparsity}_lm_eval',
                                 quick=False)
                    prev_model = prev_model.to('cpu')
                    cur_model = cur_model.to('cuda')
                else:
                    eval_lm_eval(cur_model, self.tokenizer, self.config, f'sp_{cur_sparsity}_lm_eval',
                                 quick=False)
            return flag

        def prune_iterative(iteration, restored=False):
            eval_intermediate = self.config.task.prune.eval_intermediate
            ratios = self.ratio_scheduling(iteration)
            show_diagram_list(ratios,
                              "ratio scheduling",
                              save_location=self.config.task.output_folder)

            iter_index = 0
            ratio_mha, ratio_mlp = None, None
            self.save_helper.instant_save(ratios, "scheduled_ratios")

            validation_window = []

            if restored:
                checkpoint = self.restore_to_prune(
                    target_checkpoint_path=self.config.task.prune.restore_config.checkpoint_path)
                iter_index = checkpoint["iter_index"]

            prev_sparsity = 0
            milestone_ratio_index = 0

            while iter_index < len(ratios):

                r = ratios[iter_index]
                self.save_helper.append("iter_index", iter_index)
                self.save_helper.append("requested_sparsity", r.clone().cpu())

                W_dicts, grad_norm, weight_norm, grad_acc_norm, avg_loss = obtain_info_iterative(0)

                current_sparsity = self.check_sparsity(real_pruning=False)

                cur_checkpoint_path = self.save_files(current_sparsity, W_dicts, grad_norm, weight_norm, grad_acc_norm,
                                                      avg_loss,
                                                      None if not self.is_gqa else self.gqa_mask_record)

                validation_window.append((iter_index, r.clone().cpu(), current_sparsity, cur_checkpoint_path))

                if milestone_ratio_index < len(self.milestone_ratios):
                    current_milestone_ratio = self.milestone_ratios[milestone_ratio_index]
                    if prev_sparsity <= current_milestone_ratio <= current_sparsity:
                        # back first
                        self.restore_to_prune(target_checkpoint_path=validation_window[-1][3])

                        # update data
                        self.data_processed = False
                        self.data = self.update_data(refresh_ratio=1, checkpoint_path=validation_window[-1][3])


                        iter_index = self.save_helper.load(validation_window[-1][3])["iter_index"]
                        # current_sparsity = self.check_sparsity(real_pruning=False)
                        milestone_ratio_index += 1

                        # self.save_helper.delete(validation_window[-1][3])
                        validation_window = []
                        continue

                prev_sparsity = current_sparsity

                iter_index += 1
                self.get_model().zero_grad()
                for param in self.get_model().parameters():
                    param.requires_grad_(False)

                if self.config.task.prune.prune_separate:
                    ratio_mha, ratio_mlp = prune_oneshot(W_dicts, None, r, {})
                else:
                    ratio_mha, ratio_mlp, mha_block_ratio, mlp_block_ratio = prune_oneshot(W_dicts, None, r,
                                                                                           {})
                    self.save_helper.append("block_wise_ratio", {'mha': mha_block_ratio, 'mlp': mlp_block_ratio})
                self.after_pruning_step(r, ratio_mlp, ratio_mha)

        if self.config.task.prune.iterative:
            if self.config.task.prune.restore:
                prune_iterative(self.config.task.prune.iteration, True)
            else:
                prune_iterative(self.config.task.prune.iteration)
        else:
            prune_iterative(1)

        self.finishing_pruning(real_pruning=False)

    def restore_then_eval(self):
        available_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {available_gpus}")
        from modules.eval.answer_extract_gsm8k import main
        from modules.eval.answer_extract_mathqa import main as qa_main
        if available_gpus > 1:
            del self.dense_model
            del self.model
            torch.cuda.empty_cache()
            logger.info(f"Available GPUs > 1, using accelerate to do parallel evaluation.")
            with tempfile.TemporaryDirectory() as td:
                out = pathlib.Path(td) / "result.json"
                subprocess.run([
                    sys.executable, "-m", "accelerate.commands.launch",
                    # sys.executable, "-m", "accelerate.commands.launch",
                    "--num_processes", f"{available_gpus}",
                    "--module", "modules.eval.parallel_lm_eval",
                    "--config_path", self.config.config_path,
                    "--checkpoint_path", self.config.task.prune.restore_config.checkpoint_path,
                    "--result_name", f'sp_{self.config.task.prune.restore_config.current_sparsity}_lm_eval_test',
                    "--out", str(out)
                ], check=True)
                output_path_file_no_refresh = json.loads(out.read_text(encoding="utf-8"))
                if 'qa' in self.config.task.prune.prune_dataset.name:
                    stats_no_refresh_test = qa_main(
                        ['--json_path', str(output_path_file_no_refresh), '--vllm_tp', f"{available_gpus}"])
                else:
                    stats_no_refresh_test = main(
                        ['--json_path', str(output_path_file_no_refresh), '--vllm_tp', f"{available_gpus}"])
                logger.info(f"Eval on test: {stats_no_refresh_test}")
        else:
            self.restore_to_prune(target_checkpoint_path=self.config.task.prune.restore_config.checkpoint_path)
            self.finishing_pruning(real_pruning=False)

    def after_pruning_step(self, r, ratio_mlp, ratio_mha, extra_path=None):
        r = self.check_sparsity(real_pruning=False)
        logger.info(f"ratio mlp: {ratio_mlp}")
        logger.info(f"ratio mha: {ratio_mha}")
        if extra_path is not None:
            show_diagram_dict(ratio_mlp,
                              f"structure-wise grad global={r} (mlp) separate",
                              save_location=self.config.task.output_folder + f'/{extra_path}')
            show_diagram_dict(ratio_mha,
                              f"structure-wise grad global={r} (mha) separate",
                              save_location=self.config.task.output_folder + f'/{extra_path}')
        else:
            show_diagram_dict(ratio_mlp,
                              f"structure-wise grad global={r} (mlp) separate",
                              save_location=self.config.task.output_folder)
            show_diagram_dict(ratio_mha,
                              f"structure-wise grad global={r} (mha) separate",
                              save_location=self.config.task.output_folder)

    def on_rollback_happened(self, milestone_ratio, checkpoint_init, checkpoint_selected):
        logger.info(f"Rollback happened on {milestone_ratio}, from {checkpoint_selected} to {checkpoint_init}")
        self.milestone_ratios_marker[self.milestone_ratios.index(milestone_ratio)] = True
        return

    def get_imps(self):
        return self.W_metrics

    def finishing_pruning(self, real_pruning=True):
        # del self.dense_model
        super().finishing_pruning(real_pruning)

    def restore_to_prune(self, target_sparsity=None, target_checkpoint_path=None, extra_path=None):
        self.restore_dense_model()
        layers = self.get_layers()
        head_dim = self.get_model().config.hidden_size // self.get_model().config.num_attention_heads
        self.save_helper.clean()
        if target_sparsity is not None:
            checkpoint = self.save_helper.load_sparsity(target_sparsity, extra_path=extra_path)
        elif target_checkpoint_path is not None:
            checkpoint = self.save_helper.load(target_checkpoint_path)
        else:
            raise ValueError

        if "actual_mask" not in checkpoint:
            # dense model
            pass
        else:
            mask = checkpoint["actual_mask"]
            for i in range(len(layers)):
                if f"{i}.{self.layer_mapping['attn']['block']}" in mask:
                    submask_layer = mask[f"{i}.{self.layer_mapping['attn']['block']}"]
                    if self.is_gqa:
                        submask_q = submask_layer.repeat_interleave(head_dim)
                        submask_k = submask_layer
                        submask_v = submask_layer
                        submask_o = submask_layer.repeat_interleave(head_dim)
                    else:
                        submask_q = submask_layer.repeat_interleave(head_dim)
                        submask_k = submask_layer.repeat_interleave(head_dim)
                        submask_v = submask_layer.repeat_interleave(head_dim)
                        submask_o = submask_layer.repeat_interleave(head_dim)

                    for name, vis_name, submask in zip(
                            [self.layer_mapping['attn']['q'], self.layer_mapping['attn']['k'],
                             self.layer_mapping['attn']['v'], self.layer_mapping['attn']['o']],
                            [self.layer_mapping['attn']['q_name'],
                             self.layer_mapping['attn']['k_name'],
                             self.layer_mapping['attn']['v_name'],
                             self.layer_mapping['attn']['o_name']], [
                                submask_q, submask_k, submask_v, submask_o]):

                        if name in [self.layer_mapping['attn']['o']]:
                            find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                        else:
                            if self.is_gqa:
                                if name in [self.layer_mapping['attn']['k'], self.layer_mapping['attn']['v']]:
                                    self.gqa_mask_record[i].data[submask] = 0
                                else:
                                    find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
                            else:
                                find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
            for i in range(len(layers)):
                if f"{i}.{self.layer_mapping['mlp']['block']}" in mask:
                    submask_layer = mask[f"{i}.{self.layer_mapping['mlp']['block']}"]
                    if 'g' in self.layer_mapping['mlp']:
                        submask_u = submask_layer
                        submask_g = submask_layer
                        submask_d = submask_layer
                        for name, vis_name, submask in zip(
                                [self.layer_mapping['mlp']['u'], self.layer_mapping['mlp']['g'],
                                 self.layer_mapping['mlp']['d']],
                                [self.layer_mapping['mlp']['u_name'],
                                 self.layer_mapping['mlp']['g_name'],
                                 self.layer_mapping['mlp']['d_name']], [
                                    submask_u, submask_g, submask_d]):
                            if name in [self.layer_mapping['mlp']['d']]:
                                find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                            else:
                                find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
                    else:
                        submask_u = submask_layer
                        submask_d = submask_layer
                        for name, vis_name, submask in zip(
                                [self.layer_mapping['mlp']['u'],
                                 self.layer_mapping['mlp']['d']],
                                [self.layer_mapping['mlp']['u_name'],
                                 self.layer_mapping['mlp']['d_name']], [
                                    submask_u, submask_d]):
                            if name in [self.layer_mapping['mlp']['d']]:
                                find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                            else:
                                find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
            self.save_helper.append("actual_mask", mask)
        return checkpoint

    def save_files(self, current_sparsity, W_dicts, grad_norm, weight_norm, grad_acc_norm, avg_loss,
                   gqa_mask_record=None, extra_path=None):

        self.save_helper.append("w_dicts", W_dicts)
        self.save_helper.append("grad_norm", grad_norm)
        self.save_helper.append("weight_norm", weight_norm)
        self.save_helper.append("grad_acc_norm", grad_acc_norm)
        self.save_helper.append("avg_loss", avg_loss)

        gqa_mask_record_cpu = {}
        for key, value in gqa_mask_record.items():
            if isinstance(value, torch.Tensor):
                gqa_mask_record_cpu[key] = value.cpu()
            else:
                gqa_mask_record_cpu[key] = value
        self.save_helper.append("gqa_mask_record", gqa_mask_record_cpu)
        target_checkpoint_path = self.save_helper.save(current_sparsity, extra_path=extra_path)
        return target_checkpoint_path

    def delete_files(self, current_sparsity):
        self.save_helper.delete(current_sparsity)

    def update_data(self, refresh_ratio=None, checkpoint_path=None):
        self.data_processed = False
        used_config = self.config.task.prune.prune_dataset.extra_config.used_config
        data = [d for d in self.data if d["config_name"] == used_config]
        extra_config = data[0]["gen_params"]
        extra_config['batch_size'] = self.config.task.prune.prune_dataset.extra_config.batch_size

        if refresh_ratio is None:
            sample_size = math.ceil(len(data) * self.config.task.prune.restore_config.refresh_config.refresh_ratio)
        else:
            sample_size = math.ceil(len(data) * refresh_ratio)

        logger.info(f"refresh calibration data points: {sample_size} of {len(data)}")
        if sample_size == 0:
            logger.info(f"No need to refresh.")
        else:
            # 随机选择元素
            selected_elements = random.sample(data, min(sample_size, len(data)))

            # 提取question_index
            question_indices = [element["question_index"] for element in selected_elements]
            logger.info(f"Selected data points index: {question_indices}")

            if 'gsm8k' in self.config.task.prune.prune_dataset.name:
                data = update_gsm8k_model_output_sample(data, question_indices, self.tokenizer,
                                                        self.get_wrapped_model(), extra_config,
                                                        self.check_sparsity(real_pruning=False), self.save_helper)
            elif 'mathqa' in self.config.task.prune.prune_dataset.name:
                data = update_mathqa_model_output_sample(data, question_indices, self.tokenizer, self.get_wrapped_model(), extra_config, self.check_sparsity(real_pruning=False), self.save_helper, config=self.config, checkpoint_path=checkpoint_path)
            else:
                raise ValueError
            return data


from datasets import load_dataset


def gsm8k_replace_dataset(lists):
    ds = load_dataset("gsm8k", "main", split='train', trust_remote_code=True)
    selected_subset = [ds[idx] for idx in lists]
    return selected_subset


def mathqa_replace_dataset(lists):
    ds = load_dataset("math_qa", "main", split='train', trust_remote_code=True)
    selected_subset = [ds[idx] for idx in lists]
    return selected_subset


def update_gsm8k_model_output_sample(prev_datas, lists, tokenizer, model, extra_config, current_sparsity, save_helper):
    example_point = prev_datas[0]
    ds = load_dataset("gsm8k", "main", split='train', trust_remote_code=True)
    max_new_tokens = extra_config.get("max_new_tokens", 2048)
    batch_size = extra_config.get("batch_size", 4)

    enable_thinking = extra_config.get("enable_thinking", True)
    do_sample = extra_config.get("do_sample", True)
    temperature = extra_config.get("temperature", 0.6)

    def build_prompt(question: str, enable_thinking: bool) -> str:
        messages = [{"role": "user", "content": question}]
        # 生成用：加上 generation prompt，先拿字符串再编码更稳妥
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=enable_thinking,
            add_generation_prompt=True
        )

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = eos_id
    pad_id = tokenizer.pad_token_id
    padding_side = getattr(tokenizer, "padding_side", "right")

    # 计算单条样本在 batch 中的真实 prompt 边界 [start, end)
    def prompt_bounds(inputs, b: int) -> Tuple[int, int]:
        attn = inputs["attention_mask"][b]  # [L]
        true_len = int(attn.sum().item())
        L = attn.shape[0] if attn.dim() == 1 else attn.shape[-1]  # 兼容 [L] 或 [B,L]
        if padding_side == "left":
            start = L - true_len
            end = L
        else:  # "right"
            start = 0
            end = true_len
        return start, end

    # 从完整序列中切出生成部分（考虑 eos / pad 早停）
    def cut_gen(full_ids_row: torch.Tensor, gen_start: int) -> torch.Tensor:
        gen_full = full_ids_row[gen_start:]  # 生成 + 可能的 pad/eos
        end = gen_full.shape[0]
        if eos_id is not None:
            pos = (gen_full == eos_id).nonzero(as_tuple=False)
            if pos.numel() > 0:
                end = min(end, int(pos[0].item() + 1))
        if pad_id is not None and pad_id != eos_id:
            pos = (gen_full == pad_id).nonzero(as_tuple=False)
            if pos.numel() > 0:
                end = min(end, int(pos[0].item()))
        return gen_full[:end]

    trainloader = []
    model.eval()

    with torch.no_grad():
        taken = 0
        for i in range(0, len(lists), batch_size):
            batch_idx = lists[i:i + batch_size]
            qs = [ds[idx]["question"] for idx in batch_idx]
            prompts = [build_prompt(q, enable_thinking) for q in qs]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

            gen_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                return_dict_in_generate=True,
                output_scores=False
            )
            full_ids = gen_out.sequences  # [B, L]

            B = inputs["input_ids"].shape[0]
            for b in range(B):
                start, end = prompt_bounds(inputs, b)
                input_ids_b = inputs["input_ids"][b]
                # 逐样本生成切分点 = prompt 的 end
                gen_ids = cut_gen(full_ids[b], gen_start=end)

                output_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

                # 保存去 padding 的输入 ids
                trimmed_input_ids = input_ids_b[start:end].detach().cpu().tolist()
                trainloader.append({
                    "config_idx": example_point["config_idx"],
                    "config_name": example_point["config_name"],
                    "question_index": int(batch_idx[b]),
                    "question": qs[b],
                    "output_text": output_text,
                    "input_ids": trimmed_input_ids,
                    "labels": gen_ids.detach().cpu().tolist(),

                    "gen_params": {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": do_sample,
                        "temperature": temperature,
                        "enable_thinking": enable_thinking,
                    }
                })
                taken += 1
            logger.info(f"Current sampled points: {taken}/{len(lists)}")
        save_helper.instant_save(trainloader, f"refreshed_dataloader_{current_sparsity}_{len(trainloader)}")

        # we have prev_data and new_data
        # aggregate by replace
        for datapoint in trainloader:
            for i in range(len(prev_datas)):
                if prev_datas[i]['question_index'] == datapoint['question_index']:
                    prev_datas[i] = datapoint
                    break
        lengths = [len(r["input_ids"]) + len(r["labels"]) for r in prev_datas]
        L_max = max(lengths) if lengths else 0

        for r in prev_datas:
            p = torch.tensor(r["input_ids"], dtype=torch.long)  # prompt ids (无 pad)
            a = torch.tensor(r["labels"], dtype=torch.long)  # answer ids (生成部分)

            inp = torch.cat([p, a], dim=0)  # [L]
            lab = torch.full_like(inp, -100)  # [L] 先全置 -100
            lab[len(p):] = a  # 答案段 = token ids

            pad_len = L_max - inp.size(0)
            if pad_len > 0:
                inp = torch.cat([inp, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)], dim=0)
                lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)], dim=0)
            r["train_input_ids"] = inp.tolist()
            r["train_labels"] = lab.tolist()
            r["train_attention_mask"] = [1] * (len(r["input_ids"]) + len(r["labels"])) + [0] * pad_len

    return prev_datas


def update_mathqa_model_output_sample(prev_datas, lists, tokenizer, model, extra_config, current_sparsity, save_helper,
                                      config=None, checkpoint_path=None):
    from modules.data.data_prune_ds import format_options_field
    example_point = prev_datas[0]
    ds = load_dataset("math_qa", "main", split='train', trust_remote_code=True)
    max_new_tokens = extra_config.get("max_new_tokens", 2048)
    batch_size = extra_config.get("batch_size", 4)

    enable_thinking = extra_config.get("enable_thinking", True)
    do_sample = extra_config.get("do_sample", True)
    temperature = extra_config.get("temperature", 0.6)

    def build_prompt(question: str, enable_thinking: bool) -> str:
        messages = [{"role": "user", "content": question}]
        # 生成用：加上 generation prompt，先拿字符串再编码更稳妥
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=enable_thinking,
            add_generation_prompt=True
        )

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = eos_id
    pad_id = tokenizer.pad_token_id
    padding_side = getattr(tokenizer, "padding_side", "right")

    # 计算单条样本在 batch 中的真实 prompt 边界 [start, end)
    def prompt_bounds(inputs, b: int) -> Tuple[int, int]:
        attn = inputs["attention_mask"][b]  # [L]
        true_len = int(attn.sum().item())
        L = attn.shape[0] if attn.dim() == 1 else attn.shape[-1]  # 兼容 [L] 或 [B,L]
        if padding_side == "left":
            start = L - true_len
            end = L
        else:  # "right"
            start = 0
            end = true_len
        return start, end

    # 从完整序列中切出生成部分（考虑 eos / pad 早停）
    def cut_gen(full_ids_row: torch.Tensor, gen_start: int) -> torch.Tensor:
        gen_full = full_ids_row[gen_start:]  # 生成 + 可能的 pad/eos
        end = gen_full.shape[0]
        if eos_id is not None:
            pos = (gen_full == eos_id).nonzero(as_tuple=False)
            if pos.numel() > 0:
                end = min(end, int(pos[0].item() + 1))
        if pad_id is not None and pad_id != eos_id:
            pos = (gen_full == pad_id).nonzero(as_tuple=False)
            if pos.numel() > 0:
                end = min(end, int(pos[0].item()))
        return gen_full[:end]

    trainloader = []
    model.eval()

    with torch.no_grad():
        taken = 0
        available_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {available_gpus}")
        tmp_trainloader=[]
        if available_gpus > 1:
            def even_chunks(lst, n):
                n = max(1, min(n, len(lst)))
                q, r = divmod(len(lst), n)
                out, s = [], 0
                for i in range(n):
                    k = q + (1 if i < r else 0)
                    out.append(lst[s:s + k]);
                    s += k
                return out

            logger.info(f"Available GPUs > 1, do parallel generation.")
            device = model.device
            model = model.to('cpu')
            torch.cuda.empty_cache()

            for i in range(0, len(lists), batch_size):
                batch_idx = lists[i:i + batch_size]

                for idx in batch_idx:
                    prob = ds[idx]["Problem"]
                    opt_field = ds[idx].get("options", ds[idx].get("Options", None))

                    options_str = format_options_field(opt_field)
                    tmp_trainloader.append({
                        "config_idx": example_point["config_idx"],
                        "config_name": example_point["config_name"],
                        "question_index": int(idx),
                        "question": f"Question: {prob}\nOptions: {options_str}\nAnswer:",
                        "gen_params": {
                            "max_new_tokens": max_new_tokens,
                            "do_sample": do_sample,
                            "temperature": temperature,
                            "enable_thinking": enable_thinking,
                        }
                    })
                    taken += 1
            # divide trainloader and allocate to subprocess
            logger.info(f"[cfg {example_point['config_idx']}] prepared target={taken}.")

            tmp_dir = tempfile.mkdtemp(prefix="mp_gen_")
            shard_lists = even_chunks(tmp_trainloader, available_gpus)

            procs = []
            shard_out_files = []
            for rank, shard in enumerate(shard_lists):
                if not shard:
                    continue
                shard_tasks = os.path.join(tmp_dir, f"tasks_{rank}.jsonl")
                shard_out = os.path.join(tmp_dir, f"out_{rank}.jsonl")
                shard_out_files.append(shard_out)
                with open(shard_tasks, "w", encoding="utf-8") as f:
                    for t in shard:
                        f.write(json.dumps(t, ensure_ascii=False) + "\n")

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(rank)
                cmd = [
                    sys.executable, "-m", "modules.data.parallel_generate",
                    "--config_path", config.config_path,
                    "--checkpoint_path", checkpoint_path,
                    "--tasks_in", shard_tasks,
                    "--out", shard_out,
                    "--device", "cuda:0"  # 子进程里“相对 0 号卡”
                ]
                procs.append(subprocess.Popen(cmd, env=env))


            # obtain the trainloader from subprocess
            for p in procs:
                p.wait()

            for shard_out in shard_out_files:
                with open(shard_out, "r", encoding="utf-8") as f:
                    for line in f:
                        trainloader.append(json.loads(line))

            model = model.to(device)
        else:
            for i in range(0, len(lists), batch_size):
                batch_idx = lists[i:i + batch_size]
                qs = []
                for idx in batch_idx:
                    prob = ds[idx]["Problem"]
                    opt_field = ds[idx].get("options", ds[idx].get("Options", None))
                    options_str = format_options_field(opt_field)
                    qs.append(f"Question: {prob}\nOptions: {options_str}\nAnswer:")

                prompts = [build_prompt(q, enable_thinking) for q in qs]

                # padding=True 以获得 attention_mask，便于逐样本切分
                inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    return_dict_in_generate=True,
                    output_scores=False
                )
                full_ids = gen_out.sequences  # [B, L]

                B = inputs["input_ids"].shape[0]
                for b in range(B):
                    start, end = prompt_bounds(inputs, b)
                    input_ids_b = inputs["input_ids"][b]
                    # 逐样本生成切分点 = prompt 的 end
                    gen_ids = cut_gen(full_ids[b], gen_start=end)

                    output_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

                    # 保存去 padding 的输入 ids
                    trimmed_input_ids = input_ids_b[start:end].detach().cpu().tolist()
                    trainloader.append({
                        "config_idx": example_point["config_idx"],
                        "config_name": example_point["config_name"],
                        "question_index": int(batch_idx[b]),
                        "question": qs[b],
                        "output_text": output_text,
                        "input_ids": trimmed_input_ids,
                        "labels": gen_ids.detach().cpu().tolist(),

                        "gen_params": {
                            "max_new_tokens": max_new_tokens,
                            "do_sample": do_sample,
                            "temperature": temperature,
                            "enable_thinking": enable_thinking,
                        }
                    })
                    taken += 1
                logger.info(f"Current sampled points: {taken}/{len(lists)}")

        save_helper.instant_save(trainloader, f"refreshed_dataloader_{current_sparsity}_{len(trainloader)}")

        # we have prev_data and new_data
        # aggregate by replace
        for datapoint in trainloader:
            for i in range(len(prev_datas)):
                if prev_datas[i]['question_index'] == datapoint['question_index']:
                    prev_datas[i] = datapoint
                    break
        lengths = [len(r["input_ids"]) + len(r["labels"]) for r in prev_datas]
        L_max = max(lengths) if lengths else 0

        for r in prev_datas:
            p = torch.tensor(r["input_ids"], dtype=torch.long)  # prompt ids (无 pad)
            a = torch.tensor(r["labels"], dtype=torch.long)  # answer ids (生成部分)

            inp = torch.cat([p, a], dim=0)  # [L]
            lab = torch.full_like(inp, -100)  # [L] 先全置 -100
            lab[len(p):] = a  # 答案段 = token ids

            pad_len = L_max - inp.size(0)
            if pad_len > 0:
                inp = torch.cat([inp, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)], dim=0)
                lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)], dim=0)
            r["train_input_ids"] = inp.tolist()
            r["train_labels"] = lab.tolist()
            r["train_attention_mask"] = [1] * (len(r["input_ids"]) + len(r["labels"])) + [0] * pad_len

    return prev_datas
