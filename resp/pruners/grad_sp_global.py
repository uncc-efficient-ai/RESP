from modules.model.pruning import hooking_qwen3
import copy
import logging
import math
import os
import random
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

from . import non_uniform_pruner
from .layerwrapper import *
from .utils import *
import pickle
import threading

logger = logging.getLogger(__name__)


class grad_sp_global(non_uniform_pruner):
    def __init__(self, model, config, data):
        super().__init__(model, config, data)
        self.checkpoint = None

    def prune(self):
        func_name = self.config.task.prune.func_name
        if func_name in ['global_grad_sp_v2']:
            self.global_grad_sp_v2()

    def step(self):
        pass

    def global_grad_sp_v2(self):
        head_dim = self.get_model().config.hidden_size // self.get_model().config.num_attention_heads
        head_num = self.get_model().config.num_attention_heads
        hidden_dim = self.get_model().config.hidden_size
        intermediate_size = self.get_model().config.intermediate_size
        layers = self.get_layers()
        metric_target = self.real_metrics_mapping()
        self.before_pruning()
        self.is_gqa = (self.get_model().config.num_key_value_heads < self.get_model().config.num_attention_heads)
        if self.is_gqa:
            repeat_times = self.get_model().config.num_attention_heads // self.get_model().config.num_key_value_heads
            original_kv_head_count = self.get_model().config.num_key_value_heads
            origin_function = {}
            self.gqa_mask_record = {}

            def monkey_patch_forward():
                for index, l in enumerate(layers):
                    attn_block = getattr(l, self.layer_mapping['attn']['block'])
                    origin_function['func'] = type(attn_block).forward
                    self.gqa_mask_record[index] = torch.ones(self.get_model().config.num_attention_heads,
                                                             device=self.get_model().device,
                                                             dtype=getattr(attn_block, self.layer_mapping['attn'][
                                                                 'q_name']).weight.data.dtype)
                    attn_block.forward = types.MethodType(
                        hooking_qwen3(self.gqa_mask_record[index], real_prune=False),
                        attn_block)

            def remove_patch_all():
                for index, l in enumerate(layers):
                    attn_block = getattr(l, self.layer_mapping['attn']['block'])
                    attn_block.forward = types.MethodType(origin_function['func'], attn_block)

            monkey_patch_forward()

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

        def check_grad_change(cur_W_dicts, historical_info, bypassed_blocks):
            if historical_info == {}:
                return None, False, historical_info
            for k, v in cur_W_dicts.items():
                if k in bypassed_blocks:
                    continue
                prev_step_information = historical_info[k]
                # compare the gradient change with expectation
                prev_metric_target = prev_step_information[metric_target][-1]
                cur_metric_target = v[metric_target].mean()
                if cur_metric_target > self.config.task.prune.boundary_threshold * prev_metric_target:
                    logger.info(
                        f"{metric_target} jump of {k} detected. {metric_target} change from {prev_metric_target} to {cur_metric_target}.")
                    return k, True, historical_info
            return None, False, historical_info

        def fulfill_historical_info(cur_W_dicts, historical_info):
            for k, v in cur_W_dicts.items():
                if k not in historical_info:
                    historical_info[k] = {}
                    for type, score in v.items():
                        historical_info[k][type] = []
                for type, score in v.items():
                    historical_info[k][type].append(score.mean())
            return historical_info

        def remove_historical_info(historical_info):
            for k, v in historical_info.items():
                for type, score in v.items():
                    historical_info[k][type].pop()
            return historical_info

        def fulfill_pruning_ratio_records(ratio_mha, ratio_mlp, pruning_ratio_records, bypassed_blocks):
            if ratio_mha is not None and ratio_mlp is not None:
                for k, v in ratio_mha.items():
                    for layer_index, pruned_ratio in enumerate(v):
                        if f'{str(layer_index)}.{k}' not in pruning_ratio_records:
                            pruning_ratio_records[f'{str(layer_index)}.{k}'] = [torch.tensor(0)]
                        pruning_ratio_records[f'{str(layer_index)}.{k}'].append(pruned_ratio)
                for k, v in ratio_mlp.items():
                    for layer_index, pruned_ratio in enumerate(v):
                        if f'{str(layer_index)}.{k}' not in pruning_ratio_records:
                            pruning_ratio_records[f'{str(layer_index)}.{k}'] = [torch.tensor(0)]
                        pruning_ratio_records[f'{str(layer_index)}.{k}'].append(pruned_ratio)
            return pruning_ratio_records

        def remove_pruning_ratio_records(pruning_ratio_records):
            for k, v in pruning_ratio_records.items():
                if len(v) > 0:
                    pruning_ratio_records[k].pop()
            return pruning_ratio_records

        def back_up_model():
            self.get_model().to('cpu')
            backup_model = copy.deepcopy(self.model)
            self.get_model().to('cuda')
            return backup_model

        def restore_model(backup_model):
            self.get_model().to('cpu')
            self.model = copy.deepcopy(backup_model)
            self.get_model().to('cuda')

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
            rollback_enabled = self.config.task.prune.rollback
            eval_intermediate = self.config.task.prune.eval_intermediate
            rollback_allowed = True
            bypassed_blocks = []
            ratios = self.ratio_scheduling(iteration)
            show_diagram_list(ratios,
                              "ratio scheduling",
                              save_location=self.config.task.output_folder)
            backup_model = None
            backup_sparsity = None
            intermediate_index = 0
            historical_info = {}
            pruning_ratio_records = {}
            iter_index = 0
            ratio_mha, ratio_mlp = None, None
            self.save_helper.instant_save(ratios, "scheduled_ratios")

            if restored:
                self.restore_to_prune()
                iter_index = self.checkpoint["iter_index"]
                if self.config.task.prune.restore_config.refresh_config:
                    self.update_data()
                # self.analysis_current_data()

            buffer_sparsity = self.check_sparsity(real_pruning=False)
            if rollback_enabled or eval_intermediate:
                backup_model = back_up_model()
                backup_sparsity = buffer_sparsity

            while iter_index < len(ratios):
                if len(bypassed_blocks) == 80:
                    break

                r = ratios[iter_index]
                self.save_helper.append("iter_index", iter_index)
                self.save_helper.append("requested_sparsity", r.clone().cpu())

                W_dicts, grad_norm, weight_norm, grad_acc_norm, avg_loss = obtain_info_iterative(0)
                current_sparsity = self.check_sparsity(real_pruning=False)
                self.save_files(current_sparsity, self.is_gqa, W_dicts, grad_norm, weight_norm, grad_acc_norm, avg_loss,
                                pruning_ratio_records, historical_info,
                                None if not self.is_gqa else self.gqa_mask_record)
                if eval_intermediate:
                    flag = evaluate_intermediate(eval_intermediate, intermediate_index, backup_sparsity, backup_model,
                                                 current_sparsity,
                                                 self.get_wrapped_model())
                    if flag:
                        intermediate_index += 1
                bypassed_block, detected, historical_info = check_grad_change(W_dicts, historical_info, bypassed_blocks)
                # if detected and not rollback_allowed:
                #     logger.info(f"Trying to rollback, but we just rollback once!")
                if rollback_enabled and detected and rollback_allowed:
                    # only thing we know is that we can not prune this block anymore.
                    logger.info(f"Rollback from {r} to {ratios[max(0, iter_index - 1)]}.")
                    self.on_rollback_happened()
                    bypassed_blocks.append(bypassed_block)
                    iter_index = max(0, iter_index - 1)
                    self.delete_files(current_sparsity, self.is_gqa)
                    self.evaluation(r, f'{self.config.task.output_folder}/rollback_sp_{current_sparsity}_ppl.pth')
                    restore_model(backup_model)
                    historical_info = remove_historical_info(historical_info)
                    pruning_ratio_records = remove_pruning_ratio_records(pruning_ratio_records)
                    rollback_allowed = False
                    continue
                else:
                    rollback_allowed = True
                    if current_sparsity != buffer_sparsity:
                        historical_info = fulfill_historical_info(W_dicts, historical_info)
                    else:
                        logger.info('No add-on sparsity in this step. Skip updating historical importance records.')
                    iter_index += 1
                self.get_model().zero_grad()
                for param in self.get_model().parameters():
                    param.requires_grad_(False)
                if rollback_enabled or eval_intermediate:
                    backup_model = back_up_model()
                    backup_sparsity = current_sparsity
                if current_sparsity != buffer_sparsity:
                    pruning_ratio_records = fulfill_pruning_ratio_records(ratio_mha, ratio_mlp, pruning_ratio_records,
                                                                          bypassed_blocks)
                else:
                    logger.info('No add-on sparsity in this step. Skip updating pruning ratio records.')

                if self.config.task.prune.prune_separate:
                    ratio_mha, ratio_mlp = prune_oneshot(W_dicts, None, r, bypassed_blocks)
                else:
                    ratio_mha, ratio_mlp, mha_block_ratio, mlp_block_ratio = prune_oneshot(W_dicts, None, r,
                                                                                           bypassed_blocks)
                    self.save_helper.append("block_wise_ratio", {'mha': mha_block_ratio, 'mlp': mlp_block_ratio})
                self.after_pruning_step(r, ratio_mlp, ratio_mha)

        if self.config.task.prune.iterative:
            if self.config.task.prune.restore:
                restore_config = self.config.task.prune.restore_config
                self.checkpoint = self.save_helper.load(restore_config.checkpoint_path)
                prune_iterative(self.config.task.prune.iteration, True)
            else:
                prune_iterative(self.config.task.prune.iteration)
        else:
            prune_iterative(1)

        self.finishing_pruning(real_pruning=False)

    def save_files(self, current_sparsity, is_gqa, W_dicts, grad_norm, weight_norm, grad_acc_norm, avg_loss,
                   pruning_ratio_records, historical_info, gqa_mask_record=None):

        self.save_helper.append("w_dicts", W_dicts)
        self.save_helper.append("grad_norm", grad_norm)
        self.save_helper.append("weight_norm", weight_norm)
        self.save_helper.append("grad_acc_norm", grad_acc_norm)
        self.save_helper.append("avg_loss", avg_loss)
        self.save_helper.append("pruning_ratio_records", pruning_ratio_records)
        self.save_helper.append("historical_info", historical_info)
        if is_gqa:
            gqa_mask_record_cpu = {}
            for key, value in gqa_mask_record.items():
                if isinstance(value, torch.Tensor):
                    gqa_mask_record_cpu[key] = value.cpu()
                else:
                    gqa_mask_record_cpu[key] = value
            self.save_helper.append("gqa_mask_record", gqa_mask_record_cpu)
        self.save_helper.save(current_sparsity)

    def delete_files(self, current_sparsity, is_gqa=False):
        self.save_helper.delete(current_sparsity)

    def after_pruning_step(self, r, ratio_mlp, ratio_mha):
        r = self.check_sparsity(real_pruning=False)
        logger.info(f"ratio mlp: {ratio_mlp}")
        logger.info(f"ratio mha: {ratio_mha}")
        show_diagram_dict(ratio_mlp,
                          f"structure-wise grad global={r} (mlp) separate",
                          save_location=self.config.task.output_folder)
        show_diagram_dict(ratio_mha,
                          f"structure-wise grad global={r} (mha) separate",
                          save_location=self.config.task.output_folder)

    def on_rollback_happened(self):
        return

    def get_imps(self):
        return self.W_metrics

    def finishing_pruning(self, real_pruning=True):
        super().finishing_pruning(real_pruning)

    def restore_to_prune(self):
        layers = self.get_layers()
        head_dim = self.get_model().config.hidden_size // self.get_model().config.num_attention_heads
        head_num = self.get_model().config.num_attention_heads
        hidden_dim = self.get_model().config.hidden_size
        intermediate_size = self.get_model().config.intermediate_size

        mask = self.checkpoint["actual_mask"]
        for i in range(len(layers)):
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

    def analysis_current_data(self):
        # self.update_data()
        used_config = self.config.task.prune.prune_dataset.extra_config.used_config
        data = [d for d in self.data if d["config_name"] == used_config]
        device = self.get_model().device
        all_index = torch.tensor([item["question_index"] for item in data], dtype=torch.long).to(
            device)  # [N, L]
        all_inp = torch.tensor([item["train_input_ids"] for item in data], dtype=torch.long).to(
            device)  # [N, L]
        all_lbl = torch.tensor([item["train_labels"] for item in data], dtype=torch.long).to(
            device)  # [N, L]
        all_label_ids = [item["labels"] for item in data]

        L = all_inp.size(1)
        self.index_pos = {L: all_index}
        self.data_pos = {L: all_inp}
        self.label_pos = {L: all_lbl}
        self.label_ids_pos = {L: all_label_ids}

        buckets0_pos = []
        pos_record = {}
        # 正例部分

        for L, inps in self.data_pos.items():
            labs = self.label_pos[L]  # [N, L]
            index = self.index_pos[L]
            labs_ids = self.label_ids_pos[L]

            # 全部放 GPU0
            buckets0_pos.append((L,
                                 inps.to(device, non_blocking=True),
                                 labs.to(device, non_blocking=True), index, labs_ids))
        with torch.no_grad():
            for L, inps, labs, index, lab_id in tqdm(buckets0_pos, desc=f"Buckets on {device}"):
                N = inps.size(0)
                batch_size = 1
                for j in tqdm(range(0, N, batch_size), desc=f"{device} L={L}"):
                    end = min(j + batch_size, N)
                    batch_in = inps[j:end]  # [B, L]
                    batch_lab = labs[j:end]  # [B, L]
                    i = index[j:end]
                    ids = lab_id[j:end]

                    # if len(ids[0]) == 2048:
                    #     continue
                    if batch_in.dim() == 1:
                        batch_in = batch_in.unsqueeze(0)
                        batch_lab = batch_lab.unsqueeze(0)

                    model_output = self.get_model()(batch_in, labels=batch_lab)
                    entropies = token_entropy_from_logits(model_output.logits)  # [B, L]
                    top1 = token_top1_from_logits(model_output.logits)  # [B, L]
                    gt = token_gt_probs(model_output.logits, batch_lab)
                    # === mask 掉 label==-100 的位置 ===
                    mask = (batch_lab != -100).float()  # [B, L]
                    # 避免 pad 的熵污染
                    entropies = entropies * mask
                    top1 = top1 * mask
                    gt = gt * mask
                    # 你如果还要算平均，可以这样：
                    # avg_entropy = (entropies.sum(dim=1) / mask.sum(dim=1)).cpu()
                    # avg_top1    = (top1.sum(dim=1) / mask.sum(dim=1)).cpu()

                    pos_record[int(i)] = {
                        "entropies": entropies.cpu(),
                        "label_length": int(mask.sum().item()),  # 有效 token 数
                        "top-1": top1.cpu(),
                        "gt": gt.cpu(),
                        "ids": ids[0]
                    }
        self.save_helper.instant_save(pos_record, f'full_new_pos_record_{self.check_sparsity(real_pruning=False)}')
        return

    def update_data(self):
        self.data_processed = False
        used_config = self.config.task.prune.prune_dataset.extra_config.used_config
        data = [d for d in self.data if d["config_name"] == used_config]
        extra_config = data[0]["gen_params"]
        extra_config['batch_size'] = self.config.task.prune.prune_dataset.extra_config.batch_size

        sample_size = math.ceil(len(data) * self.config.task.prune.restore_config.refresh_config.refresh_ratio)

        logger.info(f"refresh calibration data points: {sample_size} of {len(data)}")
        if sample_size == 0:
            logger.info(f"No need to refresh.")
        else:
            # 随机选择元素
            selected_elements = random.sample(data, min(sample_size, len(data)))

            # 提取question_index
            question_indices = [element["question_index"] for element in selected_elements]
            logger.info(f"Selected data points index: {question_indices}")

            self.data = update_gsm8k_model_output_sample(data, question_indices, self.tokenizer,
                                                         self.get_wrapped_model(), extra_config,
                                                         self.check_sparsity(real_pruning=False), self.save_helper)


def update_gsm8k_model_output_sample(prev_datas, lists, tokenizer, model, extra_config, current_sparsity, save_helper):
    example_point = prev_datas[0]
    from datasets import load_dataset
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
