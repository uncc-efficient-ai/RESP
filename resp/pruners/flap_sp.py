import math
import types
import warnings
from typing import Optional, Tuple

from torch import nn
from tqdm import tqdm
from transformers import Cache

from . import non_uniform_pruner
from .layerwrapper import *
from .utils import *

logger = logging.getLogger(__name__)


class flap_sp(non_uniform_pruner):
    def __init__(self, model, config, data):
        super().__init__(model, config, data)

    def prune(self):
        func_name = self.config.task.prune.func_name
        if func_name in ['baseline_prune_flap']:
            self.baseline_prune_flap()
        elif func_name in ['baseline_prune_flap_patched']:
            self.baseline_prune_flap_patched()
        else:
            raise Exception

    def baseline_prune_flap(self):
        """
        Our FLAP Pruning.

        Args:
            args (object): Command line arguments parsed via argparse.
            model (nn.Module): PyTorch model to prune.
            tokenizer (Tokenizer): Tokenizer associated with the model.
            device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
        """

        self.before_pruning()

        layers = self.get_layers()
        n_samples = self.process_data()

        seq_len = self.config.task.prune.prune_dataset.seq_len
        pruning_ratio = self.config.task.prune.ratio

        structure = self.config.task.prune.structure
        metrics = self.config.task.prune.metrics
        remove_heads = self.config.task.prune.remove_heads
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        bias = self.config.task.prune.bias

        if bias:
            for i in range(len(layers)):
                self.model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(
                    torch.zeros_like(self.model.model.layers[i].self_attn.o_proj.bias, device=self.model.device))
                self.model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(
                    torch.zeros_like(self.model.model.layers[i].mlp.down_proj.bias, device=self.model.device))
                torch.nn.init.zeros_(self.model.model.layers[i].self_attn.o_proj.bias)
                torch.nn.init.zeros_(self.model.model.layers[i].mlp.down_proj.bias)

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(self.model, self.data, n_samples,
                                                                                 seq_len)

        attn_metric_list, mlp_metric_list = [], []
        attn_baseline_inp_list, mlp_baseline_inp_list = [], []
        attn_mask, mlp_mask = [], []

        """
            'IFV': Input Feature Variance
            'WIFV': Weighted Input Feature Variance
            'WIFN': Weighted Input Feature Norm
        """
        metrics_dict = {
            'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
            'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(
                subset[name].weight.data.pow(2), dim=0),
            'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_inp.reshape((1, -1)))).mean(axis=0),
        }

        def cal_remove_neuron(model, structure, pruning_ratio, remove_heads):
            intermediate_size = model.config.intermediate_size
            hidden_size = model.config.hidden_size
            num_layers = model.config.num_hidden_layers
            if structure == "UL-MM":
                remove_params = pruning_ratio * (
                        intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
                remove_head_params = hidden_size * 4 * (remove_heads // num_layers) * 128
                return int((remove_params - remove_head_params) / (hidden_size * 3))
            else:
                remove_params = num_layers * pruning_ratio * (
                        intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
                remove_head_params = hidden_size * 4 * remove_heads * 128
                return int((remove_params - remove_head_params) / (hidden_size * 3))

        # Split into sub-problems, separate statistics for each module
        for i in tqdm(range(len(layers)), desc="Processing layers"):
            layer = layers[i]
            subset = {}
            subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
            subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

            if f"model.layers.{i}" in getattr(self.model, 'hf_device_map',
                                              {}):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = self.model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                    dev), position_ids.to(dev)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = BiasGPT(subset[name], metrics)

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(n_samples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            for h in handles:
                h.remove()

            for name in subset:
                if name == 'self_attn.o_proj':
                    W_metric = metrics_dict[metrics](wrapped_layers, subset, name)
                    if structure == "UL-UM":
                        W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                        thresh = torch.sort(W_metric.cuda())[0][
                            int(pruning_ratio * layer.self_attn.num_heads)].cpu()
                        W_mask = (W_metric >= thresh)
                        attn_mask.append(W_mask)
                    elif structure == "UL-MM":
                        W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                        thresh = torch.sort(W_metric.cuda())[0][
                            remove_heads // len(layers)].cpu()
                        W_mask = (W_metric >= thresh)
                        attn_mask.append(W_mask)
                    else:
                        attn_metric_list.append(W_metric.cpu())
                    attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
                else:
                    W_metric = metrics_dict[metrics](wrapped_layers, subset, name)
                    if structure == "UL-UM":
                        thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel() * pruning_ratio)].cpu()
                        W_mask = (W_metric >= thresh)
                        mlp_mask.append(W_mask)
                    elif structure == "UL-MM":
                        thresh = torch.sort(W_metric.cuda())[0][
                            cal_remove_neuron(self.model, structure, pruning_ratio, remove_heads)].cpu()
                        W_mask = (W_metric >= thresh)
                        mlp_mask.append(W_mask)
                    else:
                        mlp_metric_list.append(W_metric.cpu())
                    mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
                wrapped_layers[name].free()

            inps, outs = outs, inps  # Use the original output as input to the next layer
            torch.cuda.empty_cache()

        standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

        if structure in ["AL-MM", "AL-AM"]:
            attn_metric = torch.stack(attn_metric_list)
            attn_metric = standarlization(attn_metric)
            attn_metric = attn_metric.reshape(len(layers), -1, 128).mean(dim=2)

            mlp_metric = torch.stack(mlp_metric_list)
            mlp_metric = standarlization(mlp_metric)

            if structure == "AL-MM":
                sorted_attn = torch.sort(attn_metric.view(-1), descending=True)[0]
                attn_thres = sorted_attn[-int(remove_heads)]
                attn_mask = (attn_metric > attn_thres)  # 1 means retain

                sorted_mlp = torch.sort(mlp_metric.view(-1), descending=True)[0]
                mlp_thres = sorted_mlp[-cal_remove_neuron(self.model, structure, pruning_ratio, remove_heads)]
                mlp_mask = (mlp_metric > mlp_thres)
            else:
                prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
                sorted_prune, indices = torch.sort(prune_metric, descending=True)
                compression_weight = torch.ones_like(indices)
                compression_weight[indices < attn_metric.numel()] = 512.0 / 3
                threshold = sorted_prune[torch.argmin(torch.abs(
                    torch.cumsum(compression_weight, 0) - torch.sum(compression_weight) * (1 - pruning_ratio)))]
                attn_mask = (attn_metric > threshold)
                mlp_mask = (mlp_metric > threshold)
        else:
            attn_mask = torch.stack(attn_mask)
            mlp_mask = torch.stack(mlp_mask)

        for idx in range(len(layers)):
            if f"model.layers.{i}" in getattr(self.model, 'hf_device_map', {}):
                compress(self.model.model.layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None,
                         self.model.hf_device_map[f"model.layers.{idx}"], mapping=self.layer_mapping, head_dim=head_dim,
                         bias=bias)
            else:
                compress(self.model.model.layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None,
                         self.model.device, mapping=self.layer_mapping, head_dim=head_dim,
                         bias=bias)

            if f"model.layers.{i}" in getattr(self.model, 'hf_device_map', {}):
                compress(self.model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx],
                         self.model.hf_device_map[f"model.layers.{idx}"], mapping=self.layer_mapping, head_dim=head_dim,
                         bias=bias)
            else:
                compress(self.model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx],
                         self.model.device, mapping=self.layer_mapping, head_dim=head_dim,
                         bias=bias)

        self.finishing_pruning(real_pruning=True)

    def baseline_prune_flap_patched(self):
        """
        Our FLAP Pruning.

        Args:
            args (object): Command line arguments parsed via argparse.
            model (nn.Module): PyTorch model to prune.
            tokenizer (Tokenizer): Tokenizer associated with the model.
            device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
        """
        layers = self.get_layers()
        is_gqa = (self.model.config.num_key_value_heads < self.model.config.num_attention_heads)

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

        n_samples = self.process_data()

        seq_len = self.config.task.prune.prune_dataset.seq_len
        pruning_ratio = self.config.task.prune.ratio

        structure = self.config.task.prune.structure
        metrics = self.config.task.prune.metrics

        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        bias = self.config.task.prune.bias

        if bias:
            for i in range(len(layers)):
                self.model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(
                    torch.zeros_like(self.model.model.layers[i].self_attn.o_proj.bias, device=self.model.device))
                self.model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(
                    torch.zeros_like(self.model.model.layers[i].mlp.down_proj.bias, device=self.model.device))
                torch.nn.init.zeros_(self.model.model.layers[i].self_attn.o_proj.bias)
                torch.nn.init.zeros_(self.model.model.layers[i].mlp.down_proj.bias)

        with torch.no_grad():
            inps, outs, attention_mask, position_ids, cache_position, position_embeddings = prepare_calibration_input(
                self.get_model(), self.data, n_samples,
                seq_len)

        attn_metric_list, mlp_metric_list = [], []
        attn_baseline_inp_list, mlp_baseline_inp_list = [], []
        attn_mask, mlp_mask = [], []

        """
            'IFV': Input Feature Variance
            'WIFV': Weighted Input Feature Variance
            'WIFN': Weighted Input Feature Norm
        """
        metrics_dict = {
            'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
            'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(
                subset[name].weight.data.pow(2), dim=0),
            'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_inp.reshape((1, -1)))).mean(axis=0),
        }

        # Split into sub-problems, separate statistics for each module
        for i in tqdm(range(len(layers)), desc="Processing layers"):
            layer = layers[i]
            subset = {}
            subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
            subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

            if f"model.layers.{i}" in getattr(self.model, 'hf_device_map',
                                              {}):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = self.get_model().hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids, cache_position, position_embeddings = inps.to(dev), outs.to(
                    dev), attention_mask.to(
                    dev), position_ids.to(dev), cache_position.to(dev), position_embeddings.to(dev)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = BiasGPT(subset[name], metrics)

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

            for name in subset:
                if name == 'self_attn.o_proj':
                    W_metric = metrics_dict[metrics](wrapped_layers, subset, name)
                    attn_metric_list.append(W_metric.cpu())
                    attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
                else:
                    W_metric = metrics_dict[metrics](wrapped_layers, subset, name)
                    mlp_metric_list.append(W_metric.cpu())
                    mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
                wrapped_layers[name].free()

            inps, outs = outs, inps  # Use the original output as input to the next layer
            torch.cuda.empty_cache()

        standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

        if structure in ["AL-AM"]:
            attn_metric = torch.stack(attn_metric_list)
            attn_metric = standarlization(attn_metric)
            # if is_gqa:
            #     gqa_groups = self.substract_attn(layers[0]).num_key_value_groups
            #     attn_metric = attn_metric.reshape(len(layers), -1, head_dim * gqa_groups).mean(dim=2)
            # else:
            attn_metric = attn_metric.reshape(len(layers), -1, head_dim).mean(dim=2)

            mlp_metric = torch.stack(mlp_metric_list)
            mlp_metric = standarlization(mlp_metric)

            prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
            sorted_prune, indices = torch.sort(prune_metric, descending=True)
            compression_weight = torch.ones_like(indices)
            compression_weight[indices < attn_metric.numel()] = 512.0 / 3
            threshold = sorted_prune[torch.argmin(torch.abs(
                torch.cumsum(compression_weight, 0) - torch.sum(compression_weight) * (1 - pruning_ratio)))]
            attn_mask = (attn_metric > threshold)
            mlp_mask = (mlp_metric > threshold)

        current_mask_record = {}
        layers = self.get_layers()
        for idx in range(len(layers)):
            # if f"model.layers.{i}" in getattr(self.model, 'hf_device_map', {}):
            #     compress(self.model.model.layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None,
            #              self.model.hf_device_map[f"model.layers.{idx}"], mapping=self.layer_mapping, head_dim=head_dim,
            #              bias=bias)
            #
            # else:
            current_layer = layers[idx]
            current_attn_mask = attn_mask[idx]
            current_mlp_mask = mlp_mask[idx]
            self.during_pruning_step(self.substract_attn(current_layer), current_attn_mask, attn_metric, threshold)
            if is_gqa:
                gqa_mask_record[idx].data[~current_attn_mask] = 0
            current_mask_record[f"{idx}.{self.layer_mapping['attn']['block']}"] = (~current_attn_mask).clone().cpu()
            residue_attn = compress_residue_swift_zero(current_layer, attn_mask[idx], None, attn_baseline_inp_list[idx],
                                                       None,
                                                       self.model.device,
                                                       mapping=self.layer_mapping,
                                                       head_dim=head_dim, bias=False,
                                                       is_gqa=is_gqa)

            # if f"model.layers.{i}" in getattr(self.model, 'hf_device_map', {}):
            #     compress(self.model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx],
            #              self.model.hf_device_map[f"model.layers.{idx}"], mapping=self.layer_mapping, head_dim=head_dim,
            #              bias=bias)
            #
            # else:

            self.during_pruning_step(self.substract_mlp(current_layer), current_mlp_mask, mlp_metric, threshold)
            current_mask_record[f"{idx}.{self.layer_mapping['mlp']['block']}"] = (~current_mlp_mask).clone().cpu()
            residue_mlp = compress_residue_swift_zero(current_layer, None, current_mlp_mask, None,
                                                      mlp_baseline_inp_list[idx],
                                                      self.model.device,
                                                      mapping=self.layer_mapping,
                                                      head_dim=head_dim, bias=False,
                                                      is_gqa=is_gqa)
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

    def step(self):
        pass

    def get_imps(self):
        return self.W_metrics

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
