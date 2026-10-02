from modules.model.pruning import LAYER_NAME_MAPPING, find_layers, get_layers, restore_to_prune
import json
import logging
import os
import random
import sys
import types
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import fire
import numpy as np
import yaml
from accelerate import Accelerator
import torch
from datasets import DatasetDict, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Cache
from torch import nn

eval_logger = logging.getLogger(__name__)
eval_logger.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
eval_logger.handlers.clear()
eval_logger.addHandler(h)
eval_logger.propagate = False


from modules.config.config import Config


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(config_path: str = '', checkpoint_path: str = None, tasks_in: str = 'task.json', out: str = 'result.json',
         device='cpu'):
    config = Config(config_path)
    c = config.get_config()
    set_random_seed(c.task.seed)
    tokenizer = AutoTokenizer.from_pretrained(c.model.name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(c.model.name, torch_dtype=torch.bfloat16, device_map=device, )
    layer_mapping = LAYER_NAME_MAPPING['Qwen']
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        if "actual_mask" not in checkpoint:
            # dense model
            pass
        else:
            mask = checkpoint["actual_mask"]
            model = restore_to_prune(model, mask, layer_mapping)
    # 如果有 dataloader/optimizer，都一并丢进 prepare
    check_unstr_sparsity(model)
    # do our works
    model.eval()
    tasks = []
    with open(tasks_in, "r", encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))
    extra_config = c.task.prune.prune_dataset.extra_config

    max_new_tokens = extra_config.get("max_new_tokens", 2048)
    batch_size = extra_config.get("batch_size", 4)
    # batch_size = 1

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
    def prompt_bounds(inputs: Dict[str, torch.Tensor], b: int) -> Tuple[int, int]:
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

    gen_params = tasks[0]['gen_params']
    enable_thinking = gen_params['enable_thinking']
    do_sample = gen_params['do_sample']
    temperature = gen_params['temperature']

    with open(out, "w", encoding="utf-8") as fout, torch.no_grad():
        sampled = 0
        for i in range(0, len(tasks), batch_size):
            shard = tasks[i:i + batch_size]
            prompts = [build_prompt(q['question'], enable_thinking) for q in shard]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
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
                trimmed_input_ids = input_ids_b[start:end].detach().cpu().tolist()

                t = shard[b].copy()
                t.update({
                    "output_text": output_text,
                    "input_ids": trimmed_input_ids,
                    "labels": gen_ids.detach().cpu().tolist(),
                })
                fout.write(json.dumps(t, ensure_ascii=False) + "\n")
                sampled += 1
        eval_logger.info(f"Current sampled points: {sampled}/{len(tasks)}")


def check_unstr_sparsity(model, verbose=True):
    if verbose:
        eval_logger.info("*" * 30)
    count = 0
    total_params = 0
    layers = get_layers(model)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()
        if verbose:
            eval_logger.info(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")
    if verbose:
        eval_logger.info("*" * 30)
    return float(count) / total_params, count, total_params - count


if __name__ == "__main__":
    fire.Fire(main)
