"""Restore Qwen3 structured masks identically for generation and evaluation.

The mask and attention implementations are shared from the original workers.
Only unquantized Qwen3 linear layers are used by the paper configurations.
"""
from __future__ import annotations

import types
from typing import Optional, Tuple
import torch
from torch import nn

LAYER_NAME_MAPPING = {'Qwen': {'attn': {'q': 'self_attn.q_proj', 'k': 'self_attn.k_proj', 'v': 'self_attn.v_proj', 'o': 'self_attn.o_proj', 'q_name': 'q_proj', 'k_name': 'k_proj', 'v_name': 'v_proj', 'o_name': 'o_proj', 'block': 'self_attn'}, 'mlp': {'d': 'mlp.down_proj', 'g': 'mlp.gate_proj', 'u': 'mlp.up_proj', 'd_name': 'down_proj', 'g_name': 'gate_proj', 'u_name': 'up_proj', 'block': 'mlp'}, 'layers': 'model.layers'}}

def find_layers(module, layers=(nn.Linear,), name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def nested_getattr(obj, attr):
    for attr_part in attr.split('.'):
        obj = getattr(obj, attr_part)
    return obj

def get_layers(model):
    return nested_getattr(model, LAYER_NAME_MAPPING['Qwen']['layers'])

def restore_to_prune(model, mask, layer_mapping):
    layers = get_layers(model)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    is_gqa = (model.config.num_key_value_heads < model.config.num_attention_heads)

    if is_gqa:
        origin_function = {}
        gqa_mask_record = {}

        def monkey_patch_forward():
            for index, l in enumerate(layers):
                attn_block = getattr(l, layer_mapping['attn']['block'])
                origin_function['func'] = type(attn_block).forward
                gqa_mask_record[index] = torch.ones(model.config.num_attention_heads,
                                                    device=model.device,
                                                    dtype=getattr(attn_block, layer_mapping['attn'][
                                                        'q_name']).weight.data.dtype)
                attn_block.forward = types.MethodType(
                    hooking_qwen3(gqa_mask_record[index], real_prune=False),
                    attn_block)

        monkey_patch_forward()

    for i in range(len(layers)):
        if f"{i}.{layer_mapping['attn']['block']}" in mask:
            submask_layer = mask[f"{i}.{layer_mapping['attn']['block']}"]
            if is_gqa:
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
                    [layer_mapping['attn']['q'], layer_mapping['attn']['k'],
                     layer_mapping['attn']['v'], layer_mapping['attn']['o']],
                    [layer_mapping['attn']['q_name'],
                     layer_mapping['attn']['k_name'],
                     layer_mapping['attn']['v_name'],
                     layer_mapping['attn']['o_name']], [
                        submask_q, submask_k, submask_v, submask_o]):

                if name in [layer_mapping['attn']['o']]:
                    find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                else:
                    if is_gqa:
                        if name in [layer_mapping['attn']['k'], layer_mapping['attn']['v']]:
                            gqa_mask_record[i].data[submask] = 0
                        else:
                            find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
                    else:
                        find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero

    for i in range(len(layers)):
        if f"{i}.{layer_mapping['mlp']['block']}" in mask:
            submask_layer = mask[f"{i}.{layer_mapping['mlp']['block']}"]
            if 'g' in layer_mapping['mlp']:
                submask_u = submask_layer
                submask_g = submask_layer
                submask_d = submask_layer
                for name, vis_name, submask in zip(
                        [layer_mapping['mlp']['u'], layer_mapping['mlp']['g'],
                         layer_mapping['mlp']['d']],
                        [layer_mapping['mlp']['u_name'],
                         layer_mapping['mlp']['g_name'],
                         layer_mapping['mlp']['d_name']], [
                            submask_u, submask_g, submask_d]):
                    if name in [layer_mapping['mlp']['d']]:
                        find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                    else:
                        find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
            else:
                submask_u = submask_layer
                submask_d = submask_layer
                for name, vis_name, submask in zip(
                        [layer_mapping['mlp']['u'],
                         layer_mapping['mlp']['d']],
                        [layer_mapping['mlp']['u_name'],
                         layer_mapping['mlp']['d_name']], [
                            submask_u, submask_d]):
                    if name in [layer_mapping['mlp']['d']]:
                        find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                    else:
                        find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
    return model

def hooking_qwen3(mask, real_prune=False):
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def eager_attention_forward(
            module: nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            scaling: float,
            dropout: float = 0.0,
            **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        reshaped_mask = mask.view(1, key_states.shape[1], 1, 1)
        reshaped_mask = reshaped_mask.to(key_states.dtype)
        reshaped_mask = reshaped_mask.to(key_states.device)
        if real_prune:
            keep_heads = torch.where(mask)[0]
            key_states = key_states[:, keep_heads]
            value_states = value_states[:, keep_heads]
        else:
            key_states = key_states * reshaped_mask
            value_states = value_states * reshaped_mask

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward
