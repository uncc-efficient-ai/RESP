import math
import types
import warnings
from typing import Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm
from transformers import Cache

from . import non_uniform_pruner
from .layerwrapper import *
from .utils import *

logger = logging.getLogger(__name__)


class owl_sp(non_uniform_pruner):

    def __init__(self, model, config, data):
        super().__init__(model, config, data)
        self.gqa_mask_record = None

    def prune(self):
        func_name = self.config.task.prune.func_name
        if func_name in ['baseline_owl']:
            self.baseline_owl()
        else:
            raise Exception

    def baseline_owl(self):
        """
            Wanda on structured pruning.

            Args:
                args (object): Command line arguments parsed via argparse.
                model (nn.Module): PyTorch model to prune.
                tokenizer (Tokenizer): Tokenizer associated with the model.
                device (torbaseline_uniform_wandach.device, optional): Device to move tensors to. Defaults to CUDA device 0.
            """
        n_samples = self.process_data()
        seq_len = self.config.task.prune.prune_dataset.seq_len
        pruning_ratio = self.config.task.prune.ratio
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        is_gqa = (self.model.config.num_key_value_heads < self.model.config.num_attention_heads)
        layers = self.get_layers()
        if is_gqa:
            repeat_times = self.model.config.num_attention_heads // self.model.config.num_key_value_heads
            original_kv_head_count = self.model.config.num_key_value_heads
            origin_function = {}
            gqa_mask_record = {}

            def monkey_patch_forward():
                for index, l in enumerate(layers):
                    attn_block = getattr(l, self.layer_mapping['attn']['block'])
                    origin_function['func'] = type(attn_block).forward
                    gqa_mask_record[index] = torch.ones(self.get_model().config.num_attention_heads,
                                                        device=self.get_model().device,
                                                        dtype=getattr(attn_block, self.layer_mapping['attn'][
                                                            'q_name']).weight.data.dtype)
                    attn_block.forward = types.MethodType(
                        hooking_qwen3(gqa_mask_record[index], real_prune=False),
                        attn_block)

            def remove_patch_all():
                for index, l in enumerate(layers):
                    attn_block = getattr(l, self.layer_mapping['attn']['block'])
                    attn_block.forward = types.MethodType(origin_function['func'], attn_block)

            monkey_patch_forward()
        self.before_pruning()

        with torch.no_grad():
            inps, outs, attention_mask, position_ids, cache_position, position_embeddings = prepare_calibration_input(
                self.get_model(), self.data, n_samples,
                seq_len)

        inputs = inps.detach().clone()
        all_layer_ratio = []

        for i in range(len(layers)):
            layer = layers[i]

            subset = {}
            subset.update({self.layer_mapping['attn']['k']: find_layers(layer)[self.layer_mapping['attn']['k']]})
            subset.update({self.layer_mapping['attn']['q']: find_layers(layer)[self.layer_mapping['attn']['q']]})
            subset.update({self.layer_mapping['attn']['v']: find_layers(layer)[self.layer_mapping['attn']['v']]})
            subset.update({self.layer_mapping['attn']['o']: find_layers(layer)[self.layer_mapping['attn']['o']]})
            subset.update({self.layer_mapping['mlp']['d']: find_layers(layer)[self.layer_mapping['mlp']['d']]})
            subset.update({self.layer_mapping['mlp']['u']: find_layers(layer)[self.layer_mapping['mlp']['u']]})
            if 'g' in self.layer_mapping['mlp']:
                subset.update({self.layer_mapping['mlp']['g']: find_layers(layer)[self.layer_mapping['mlp']['g']]})

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(n_samples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,
                                    cache_position=cache_position, position_embeddings=position_embeddings)[
                        0]
            for h in handles:
                h.remove()

            layer_wmetric = []

            for name in subset:
                print(f"pruning layer {i} name {name}")
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1)))
                layer_wmetric.append(W_metric)

            for j in range(n_samples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,
                                    cache_position=cache_position, position_embeddings=position_embeddings)[
                        0]
            inps, outs = outs, inps

            layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

            for out_ratio in [self.config.task.prune.Hyper_m]:
                out_ratio_layer = self.check_outlier_mean(layer_wmetric, out_ratio)
                print("layer outlier ratio", out_ratio, out_ratio_layer)

            if not (i not in range(int(len(layers) * 0.1), len(layers) - 1) and self.config.task.prune.prune_skip):
                all_layer_ratio.append(out_ratio_layer)

        logger.info(f"before adjustment: {all_layer_ratio}")

        all_layer_ratio = np.array(all_layer_ratio)

        all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (
                1 / (all_layer_ratio.max() - all_layer_ratio.min()) * self.config.task.prune.Lamda * 2))

        all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio) + (1 - pruning_ratio)

        print(all_layer_ratio, np.mean(all_layer_ratio), np.max(all_layer_ratio), np.min(all_layer_ratio))

        logger.info(f"after adjustment: {all_layer_ratio}")

        torch.cuda.empty_cache()
        ############## prune

        inps = inputs

        all_layer_ratio_counter = 0

        current_mask_record = {}

        for i in tqdm(range(len(layers)), desc="Processing layers"):

            layer = layers[i]
            subset = {}
            subset.update({self.layer_mapping['attn']['o']: find_layers(layer)[self.layer_mapping['attn']['o']]})
            subset.update({self.layer_mapping['mlp']['d']: find_layers(layer)[self.layer_mapping['mlp']['d']]})

            if f"model.layers.{i}" in getattr(self.get_model(), 'hf_device_map',
                                              {}):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = self.get_model().hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids, cache_position, position_embeddings = inps.to(dev), outs.to(
                    dev), attention_mask.to(
                    dev), position_ids.to(dev), cache_position.to(dev), position_embeddings.to(dev)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            if i not in range(int(len(layers) * 0.1), len(layers) - 1) and self.config.task.prune.prune_skip:
                for j in range(n_samples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,
                                        cache_position=cache_position, position_embeddings=position_embeddings)[
                            0]
                inps, outs = outs, inps
                torch.cuda.empty_cache()
            else:
                layer_sparsity_ratio = 1 - all_layer_ratio[all_layer_ratio_counter]
                if layer_sparsity_ratio <= 0:
                    layer_sparsity_ratio = 0.01
                all_layer_ratio_counter += 1
                self.before_pruning_step()
                handles = []
                for name in wrapped_layers:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(n_samples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,
                                        cache_position=cache_position, position_embeddings=position_embeddings)[
                            0]
                for h in handles:
                    h.remove()

                for name in subset:
                    print(f"pruning layer {i} name {name}")
                    W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                        wrapped_layers[name].scaler_row.reshape((1, -1)))

                    if name == self.layer_mapping['attn']['o']:
                        if self.config.task.prune.prune_modules in ['mha', 'all']:
                            W_metric = W_metric.mean(axis=0).reshape(-1, head_dim).sum(
                                dim=1)  # importance score of each head
                            thresh = torch.sort(W_metric.cuda())[0][
                                int(layer_sparsity_ratio * self.model.config.num_attention_heads)].cpu()
                            W_mask = (W_metric >= thresh)

                            self.during_pruning_step(self.substract_attn(layer), W_mask, W_metric, thresh)
                            if is_gqa:
                                gqa_mask_record[i].data[~W_mask] = 0
                            # residue_attn = compress_residue_swift(layer, W_mask, None, None, None, self.model.device,
                            #                                       mapping=self.layer_mapping,
                            #                                       head_dim=head_dim, bias=False, is_gqa=is_gqa)
                            current_mask_record[f"{i}.{self.layer_mapping['attn']['block']}"] = (~W_mask).clone().cpu()
                            residue_attn = compress_residue_swift_zero(layer, W_mask, None, None, None,
                                                                       self.model.device,
                                                                       mapping=self.layer_mapping,
                                                                       head_dim=head_dim, bias=False,
                                                                       is_gqa=is_gqa)
                    else:
                        if self.config.task.prune.prune_modules in ['mlp', 'all']:
                            W_metric = W_metric.mean(axis=0)
                            thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel() * pruning_ratio)].cpu()
                            W_mask = (W_metric >= thresh)
                            self.during_pruning_step(self.substract_mlp(layer), W_mask, W_metric, thresh)
                            current_mask_record[f"{i}.{self.layer_mapping['mlp']['block']}"] = (~W_mask).clone().cpu()
                            residue_mlp = compress_residue_swift_zero(layer, None, W_mask, None, None,
                                                                      self.model.device,
                                                                      mapping=self.layer_mapping,
                                                                      head_dim=head_dim, bias=False,
                                                                      is_gqa=is_gqa)
                    wrapped_layers[name].free()
                    self.W_metrics[f"{i}.{name}"] = W_metric.clone()
                self.after_pruning_step(real_pruning=False, verbose=False)

                if self.config.task.prune.act_type in ['sparse']:
                    for j in range(n_samples):
                        with torch.no_grad():
                            outs[j] = \
                                layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,
                                      cache_position=cache_position, position_embeddings=position_embeddings)[
                                    0]
                inps, outs = outs, inps  # the pruned output as input to the next layer
            torch.cuda.empty_cache()
        if is_gqa:
            self.gqa_mask_record = gqa_mask_record
            gqa_mask_record_cpu = {}
            for key, value in self.gqa_mask_record.items():
                if isinstance(value, torch.Tensor):
                    gqa_mask_record_cpu[key] = value.cpu()
                else:
                    gqa_mask_record_cpu[key] = value
            self.save_helper.append("gqa_mask_record", gqa_mask_record_cpu)
        current_actual_sparsity = self.check_sparsity(real_pruning=False, verbose=False)
        self.save_helper.append("actual_mask", current_mask_record)
        self.save_helper.save(current_actual_sparsity)
        self.finishing_pruning(real_pruning=False)

    def finishing_pruning(self, real_pruning=True):
        super().finishing_pruning(real_pruning)
        if self.is_gqa:
            self.save_helper.instant_save(self.gqa_mask_record, "sp_gqa_mask")

    def process_data(self):
        if not self.data_processed and self.config.task.prune.prune_dataset.type in ['downstream']:
            seq_len = self.config.task.prune.prune_dataset.seq_len
            if "model_output" in self.config.task.prune.prune_dataset.name:
                used_config = self.config.task.prune.prune_dataset.extra_config.used_config
                self.data = [d for d in self.data if d["config_name"] == used_config]

                first_elems = []
                for d in self.data:
                    tup = torch.tensor(d['train_input_ids'], dtype=torch.long)  # 扁平化
                    nonpad = tup[tup != self.tokenizer.pad_token_id]  # 过滤掉所有值为 pad 的元素
                    first_elems.append(nonpad)
                big_tensor = torch.cat(first_elems, dim=0)  # shape [L], L = sum_i len(tup_i[0])
            else:
                first_elems = []
                for tup in self.data:
                    flat = tup[0].view(-1)  # 扁平化
                    nonpad = flat[flat != self.tokenizer.pad_token_id]  # 过滤掉所有值为 pad 的元素
                    first_elems.append(nonpad)
                big_tensor = torch.cat(first_elems, dim=0)  # shape [L], L = sum_i len(tup_i[0])
            # 3b. 切分成 list
            chunked = big_tensor.split(seq_len)  # 返回 tuple，每项 shape [segment_size]
            new_data = []
            for i, chunk in enumerate(chunked):
                if chunk.numel() < seq_len:
                    # auto handle pad in attention mask by GenerationMixin._prepare_attention_mask_for_generation
                    chunk = F.pad(chunk, (0, seq_len - chunk.numel()), value=self.tokenizer.pad_token_id)
                new_data.append((chunk.unsqueeze(0),))
            self.data = new_data
            self.data_processed = True
        return len(self.data)


    def check_outlier_mean(self, mask, threshold):
        W = mask
        count = 0
        total_params = 0

        max_shred = torch.mean(W) * threshold
        count += (W > max_shred).sum().item()
        total_params += W.numel()

        outlier_ratio = float(count) / total_params * 100

        return outlier_ratio

    def get_imps(self):
        return self.W_metrics

    def step(self):
        pass


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

    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    from transformers.processing_utils import Unpack
    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
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
