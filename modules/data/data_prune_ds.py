import copy


import json


import logging


import math


import os


import pickle


import re


import string


import subprocess


import sys


import tempfile


import threading


from typing import List, Dict, Any, Tuple


import numpy as np


import random


import torch


import tqdm


from datasets import load_dataset, DownloadMode, concatenate_datasets, load_from_disk, Dataset, DatasetDict


logger = logging.getLogger(__name__)


class TokenizerWrapper:
    """
    Wrapper class for tokenized input IDs.

    Args:
        input_ids (tensor): The tokenized input IDs from the tokenizer.
    """

    def __init__(self, input_ids):
        self.input_ids = input_ids


def estimate_seqlen_gsm8k(ds, tokenizer, percentile=99, sample_size=2000, seed=0, extra_config=None):
    """
    随机采 sample_size 条 GSM8K 样本，测它们 prompt+answer 的 token 长度，返回 percentile 分位数。
    """
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    lengths = []
    contains_cot = extra_config.get("contains_cot", True) if extra_config else True
    zero_shot = extra_config.get("zero_shot", False) if extra_config else False
    for i in idxs[:sample_size or len(idxs)]:
        ex = ds[i]
        # 构 prompt 与 answer
        q = ex['question'].strip()
        if zero_shot:
            prompt = f"Q: {q}\nA: Let's think step by step. Once we can get the first answer, I will provide that in a regex-recognizable format of \"The answer is (\\-?[0-9\\.\\,]+).\" and finish."
        else:
            prompt = f"Q: {q}\nA: "
        ans = ex['answer'].strip()
        if not contains_cot:
            ans = re.sub(r'^#+\s*', '', ans.split("\n")[-1])
        # 合并长度
        pid = tokenizer(prompt, add_special_tokens=False).input_ids
        aid = tokenizer(ans, add_special_tokens=False).input_ids
        lengths.append(len(pid) + len(aid))
    return int(np.percentile(lengths, percentile))


def get_gsm8k(max_tokens: int, seed: int, tokenizer, extra_config=None):
    random.seed(seed)

    ds = load_dataset("gsm8k", "main", split='train', trust_remote_code=True).shuffle(seed)
    seqlen = estimate_seqlen_gsm8k(ds, tokenizer,
                                   percentile=100,
                                   sample_size=0,
                                   seed=seed, extra_config=extra_config)
    logger.info(f"[Auto] set seqlen={seqlen} (GSM8K 100th pct)")
    contains_cot = extra_config.get("contains_cot", True) if extra_config else True
    zero_shot = extra_config.get("zero_shot", False) if extra_config else False
    trainloader = []
    total_tokens = 0

    for ex in ds:
        # prompt
        q = ex['question'].strip()
        if zero_shot:
            prompt = f"Q: {q}\nA: Let's think step by step. Once we can get the first answer, I will provide that in a regex-recognizable format of \"The answer is (\\-?[0-9\\.\\,]+).\" and finish."
        else:
            prompt = f"Q: {q}\nA: "
        p_enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        p_ids = p_enc.input_ids  # [1, Lp]
        Lp = p_ids.size(1)
        # answer
        ans = ex['answer'].strip()
        if not contains_cot:
            ans = re.sub(r'^#+\s*', '', ans.split("\n")[-1])
            if not ans:
                raise ValueError
        a_ids = tokenizer(ans, add_special_tokens=False).input_ids
        La = len(a_ids)

        # skip if answer too long
        if La >= seqlen:
            continue
        # 尾部截断 prompt
        max_p = seqlen - La
        if Lp > max_p:
            p_ids = p_ids[:, -max_p:]
            Lp = max_p

        # 检预算
        L = Lp + La
        if total_tokens + L > max_tokens:
            break

        # 拼接 & mask
        inp = torch.cat([p_ids, torch.tensor([a_ids], dtype=torch.long)], dim=1)
        labels = inp.clone()
        labels[0, :Lp] = -100

        # pad 到 seqlen
        pad_len = seqlen - L
        if pad_len > 0:
            pad_ids = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)
            pad_lbl = torch.full((1, pad_len), -100, dtype=torch.long)
            inp = torch.cat([inp, pad_ids], dim=1)
            labels = torch.cat([labels, pad_lbl], dim=1)

        trainloader.append((inp, labels, True))
        total_tokens += L

    if total_tokens < max_tokens:
        logger.warning(f"GSM8K only generated {total_tokens} tokens, budget was {max_tokens}")

        # 验证集 prompt 流 (使用 test 作为验证)
    val_ds = load_dataset("gsm8k", "main", trust_remote_code=True, split="test")
    val_prompts = [f"Q: {ex['question'].strip()}\nA:" for ex in val_ds]
    val_text = " ".join(val_prompts)
    val_enc = tokenizer(val_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256 * seqlen).input_ids
    valenc = TokenizerWrapper(val_enc)

    return trainloader, valenc, total_tokens


def get_gsm8k_model_output(max_budget: int, seed: int, tokenizer, model, extra_config=None):
    saved_path = extra_config.get("saved_path", None) if extra_config else None

    if saved_path is not None:
        file_path = f'{saved_path}_{seed}_{max_budget}_prune_data.pkl'
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                trainloader = pickle.load(f)
            logger.info(f"Load saved trainloader from {file_path}.")
            return trainloader, None, 0

    budget_format = extra_config.get("budget_format", "token-wise") if extra_config else "token-wise"
    if "sample-wise" in budget_format:
        logger.info(f"Budget format: {budget_format}.")
        trainloader, valenc, total = get_gsm8k_model_output_sample(max_budget, seed, tokenizer, model, extra_config)
    else:
        trainloader, valenc, total = get_gsm8k_model_output_token(max_budget, seed, tokenizer, model, extra_config)
    if saved_path is not None:
        file_path = f'{saved_path}_{seed}_{max_budget}_prune_data.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(trainloader, f)
            logger.info(f"Save trainloader to {file_path}.")
    return trainloader, valenc, total


def get_gsm8k_model_output_token(max_tokens: int, seed: int, tokenizer, model, extra_config):
    random.seed(seed)

    ds = load_dataset("gsm8k", "main", split='train', trust_remote_code=True)
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)

    max_new_tokens = extra_config.get("max_new_tokens", 2048) if extra_config else 2048
    enable_thinking = extra_config.get("enable_thinking", True) if extra_config else True
    do_sample = extra_config.get("do_sample", True) if extra_config else True
    temperature = extra_config.get("temperature", 0.6) if extra_config else 0.6
    batch_size = extra_config.get("batch_size", 4) if extra_config else 4

    total_tokens = 0

    def build_prompt(question: str) -> str:
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

    trainloader = []

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        # ===== 单 GPU / CPU：保持你原来的串行逻辑 =====
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        processed = 0
        with torch.no_grad():
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i + batch_size]
                qs = [ds[idx]["question"] for idx in batch_idx]
                prompts = [build_prompt(q) for q in qs]

                # padding=True 以获得 attention_mask，便于逐样本切分
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

                    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    input_len = end - start
                    output_len = gen_ids.shape[0]
                    to_add = input_len + output_len

                    if total_tokens + to_add > max_tokens:
                        if total_tokens < max_tokens:
                            logger.warning(f"GSM8K only generated {total_tokens} tokens, budget was {max_tokens}")

                        return trainloader, None, total_tokens

                    # 保存去 padding 的输入 ids
                    trimmed_input_ids = input_ids_b[start:end].detach().cpu().tolist()
                    trainloader.append({
                        "question": qs[b],
                        "output_text": output_text,
                        "input_ids": trimmed_input_ids,
                        "labels": gen_ids.detach().cpu().tolist()
                    })
                    total_tokens += to_add
                    processed += 1
    else:
        # ===== 多 GPU 并行（每卡也按 batch 处理）=====
        devices = [f"cuda:{i}" for i in range(num_gpus)]
        models = []
        for d in devices:
            m = copy.deepcopy(model).to(d)
            m.eval()
            models.append(m)

        idx_ptr = {"i": 0}
        lock = threading.Lock()
        stop_flag = {"stop": False}

        def worker(wid: int):
            nonlocal total_tokens, trainloader
            local_model = models[wid]
            device = devices[wid]

            with torch.no_grad():
                while True:
                    with lock:
                        if stop_flag["stop"] or idx_ptr["i"] >= len(indices):
                            return
                        batch_idx = indices[idx_ptr["i"]: idx_ptr["i"] + batch_size]
                        idx_ptr["i"] += batch_size

                    qs = [ds[idx]["question"] for idx in batch_idx]
                    prompts = [build_prompt(q) for q in qs]
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    gen_out = local_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                        return_dict_in_generate=True,
                        output_scores=False
                    )
                    full_ids = gen_out.sequences

                    B = inputs["input_ids"].shape[0]
                    for b in range(B):
                        start, end = prompt_bounds(inputs, b)
                        input_ids_b = inputs["input_ids"][b]
                        gen_ids = cut_gen(full_ids[b], gen_start=end)

                        output_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                        input_len = end - start
                        output_len = gen_ids.shape[0]
                        to_add = input_len + output_len

                        with lock:
                            if stop_flag["stop"]:
                                return
                            if total_tokens + to_add > max_tokens:
                                stop_flag["stop"] = True
                                return

                            trimmed_input_ids = input_ids_b[start:end].detach().cpu().tolist()
                            trainloader.append({
                                "question": qs[b],
                                "output_text": output_text,
                                "input_ids": trimmed_input_ids,
                                "labels": gen_ids.detach().cpu().tolist()
                            })
                            total_tokens += to_add

        threads = []
        for wid in range(num_gpus):
            t = threading.Thread(target=worker, args=(wid,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        if total_tokens < max_tokens:
            logger.warning(f"GSM8K only generated {total_tokens} tokens, budget was {max_tokens}")

        return trainloader, None, total_tokens


def get_gsm8k_model_output_sample(samples: int, seed: int, tokenizer, model, extra_config):
    random.seed(seed)

    ds = load_dataset("gsm8k", "main", split='train', trust_remote_code=True)
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)

    max_new_tokens = extra_config.get("max_new_tokens", 2048)
    batch_size = extra_config.get("batch_size", 4)

    generation_configs = extra_config.generation_config_sets

    # enable_thinking = extra_config.get("enable_thinking", True)
    # do_sample = extra_config.get("do_sample", True)
    # temperature = extra_config.get("temperature", 0.6)

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

    trainloader = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        traindata_c4 = load_dataset('allenai/c4',
                                 data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                                 split='train')

        for cfg_idx, cfg in enumerate(generation_configs):
            name = cfg.get("name", f"cfg_{cfg_idx}")
            if 'gold' in name:
                taken = 0
                enable_thinking = cfg.get("enable_thinking", True)
                for i in range(0, len(indices), batch_size):
                    if taken >= samples:
                        break
                    batch_idx = indices[i:i + batch_size]
                    qs = [ds[idx]["question"] for idx in batch_idx]
                    prompts = [build_prompt(q, enable_thinking) for q in qs]

                    # 构造 prompt，用于得到去 pad 的 input_ids（与其他配置完全一致）
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                    B = inputs["input_ids"].shape[0]
                    for b in range(B):
                        if taken >= samples:
                            break
                        start, end = prompt_bounds(inputs, b)
                        input_ids_b = inputs["input_ids"][b]
                        trimmed_input_ids = input_ids_b[start:end].detach().cpu().tolist()

                        idx_b = int(batch_idx[b])
                        answer_text = ds[idx_b]["answer"]  # gold 的文本答案
                        # gold 的 labels：对齐其它配置的生成输出（通常包含 eos），故在末尾补 eos（若存在）
                        labels_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
                        if eos_id is not None:
                            labels_ids = labels_ids + [eos_id]
                        trainloader.append({
                            "config_idx": cfg_idx,
                            "config_name": name,
                            "question_index": idx_b,
                            "question": qs[b],
                            "output_text": answer_text,  # gold 的原始答案文本
                            "input_ids": trimmed_input_ids,  # prompt 去 pad
                            "labels": labels_ids,  # 仅答案 token（末尾含 eos 时对齐生成配置）

                            "gen_params": {  # 字段不缺失，值对齐/占位
                                "max_new_tokens": max_new_tokens,
                                "do_sample": False,
                                "temperature": 0.0,
                                "enable_thinking": enable_thinking,
                            }
                        })
                        taken += 1
                    logger.info(f"Current sampled points: {taken}/{samples}")
                    # 与其它配置一致：本 config 内部 pack 到最大长度 L_max
                    lengths = [len(r["input_ids"]) + len(r["labels"]) for r in trainloader if
                               r['config_idx'] == cfg_idx]
                    L_max = max(lengths) if lengths else 0

                for r in trainloader:
                    if r['config_idx'] != cfg_idx:
                        continue
                    p = torch.tensor(r["input_ids"], dtype=torch.long)  # prompt ids（无 pad）
                    a = torch.tensor(r["labels"], dtype=torch.long)  # answer ids（含 eos 时与生成一致）

                    inp = torch.cat([p, a], dim=0)  # [L]
                    lab = torch.full_like(inp, -100)  # [L] 先全置 -100
                    lab[len(p):] = a  # 答案段 = token ids

                    pad_len = L_max - inp.size(0)
                    if pad_len > 0:
                        inp = torch.cat([inp, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)],
                                        dim=0)
                        lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)], dim=0)

                    r["train_input_ids"] = inp.tolist()
                    r["train_labels"] = lab.tolist()
                    r["train_attention_mask"] = [1] * (len(r["input_ids"]) + len(r["labels"])) + [0] * pad_len
            elif 'c4' in name:
                taken = 0
                enable_thinking = cfg.get("enable_thinking", True)
                for i in range(0, len(indices), batch_size):
                    if taken >= samples:
                        break
                    batch_idx = indices[i:i + batch_size]
                    qs = [ds[idx]["question"] for idx in batch_idx]
                    prompts = [build_prompt(q, enable_thinking) for q in qs]

                    # 构造 prompt，用于得到去 pad 的 input_ids（与其他配置完全一致）
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                    B = inputs["input_ids"].shape[0]
                    for b in range(B):
                        if taken >= samples:
                            break
                        start, end = prompt_bounds(inputs, b)
                        input_ids_b = inputs["input_ids"][b]
                        trimmed_input_ids = input_ids_b[start:end].detach().cpu().tolist()

                        while True:
                            ii = random.randint(0, len(traindata_c4) - 1)
                            txt = traindata_c4[ii]['text']
                            enc = tokenizer(txt, add_special_tokens=False).input_ids

                            if len(enc) > max_new_tokens:
                                break

                        # 随机切片
                        lo = random.randint(0, len(enc) - max_new_tokens - 1)
                        hi = lo + max_new_tokens
                        c4_answer_ids = enc[lo:hi]

                        labels_ids = c4_answer_ids
                        if eos_id is not None:
                            labels_ids = labels_ids + [eos_id]

                        idx_b = int(batch_idx[b])

                        c4_answer_text = tokenizer.decode(c4_answer_ids, skip_special_tokens=True)

                        trainloader.append({
                            "config_idx": cfg_idx,
                            "config_name": name,
                            "question_index": idx_b,
                            "question": qs[b],
                            "output_text": c4_answer_text,
                            "input_ids": trimmed_input_ids,  # prompt 去 pad
                            "labels": labels_ids,  # 仅答案 token（末尾含 eos 时对齐生成配置）

                            "gen_params": {  # 字段不缺失，值对齐/占位
                                "max_new_tokens": max_new_tokens,
                                "do_sample": False,
                                "temperature": 0.0,
                                "enable_thinking": enable_thinking,
                            }
                        })
                        taken += 1
                    logger.info(f"Current sampled points: {taken}/{samples}")
                    # 与其它配置一致：本 config 内部 pack 到最大长度 L_max
                    lengths = [len(r["input_ids"]) + len(r["labels"]) for r in trainloader if
                               r['config_idx'] == cfg_idx]
                    L_max = max(lengths) if lengths else 0

                for r in trainloader:
                    if r['config_idx'] != cfg_idx:
                        continue
                    p = torch.tensor(r["input_ids"], dtype=torch.long)  # prompt ids（无 pad）
                    a = torch.tensor(r["labels"], dtype=torch.long)  # answer ids（含 eos 时与生成一致）

                    inp = torch.cat([p, a], dim=0)  # [L]
                    lab = torch.full_like(inp, -100)  # [L] 先全置 -100
                    lab[len(p):] = a  # 答案段 = token ids

                    pad_len = L_max - inp.size(0)
                    if pad_len > 0:
                        inp = torch.cat([inp, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)],
                                        dim=0)
                        lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)], dim=0)

                    r["train_input_ids"] = inp.tolist()
                    r["train_labels"] = lab.tolist()
                    r["train_attention_mask"] = [1] * (len(r["input_ids"]) + len(r["labels"])) + [0] * pad_len
                pass
            else:
                enable_thinking = cfg.get("enable_thinking", True)
                do_sample = cfg.get("do_sample", True)
                temperature = cfg.get("temperature", 0.6)

                taken = 0

                for i in range(0, len(indices), batch_size):
                    if taken >= samples:
                        break
                    batch_idx = indices[i:i + batch_size]
                    qs = [ds[idx]["question"] for idx in batch_idx]
                    prompts = [build_prompt(q, enable_thinking) for q in qs]

                    # padding=True 以获得 attention_mask，便于逐样本切分
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
                        if taken >= samples:
                            break
                        start, end = prompt_bounds(inputs, b)
                        input_ids_b = inputs["input_ids"][b]
                        # 逐样本生成切分点 = prompt 的 end
                        gen_ids = cut_gen(full_ids[b], gen_start=end)

                        output_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

                        # 保存去 padding 的输入 ids
                        trimmed_input_ids = input_ids_b[start:end].detach().cpu().tolist()
                        trainloader.append({
                            "config_idx": cfg_idx,
                            "config_name": name,
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
                    logger.info(f"Current sampled points: {taken}/{samples}")

                lengths = [len(r["input_ids"]) + len(r["labels"]) for r in trainloader if r['config_idx'] == cfg_idx]
                L_max = max(lengths) if lengths else 0

                for r in trainloader:
                    if r['config_idx'] != cfg_idx:
                        continue
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

            logger.info(f"Finished for config {cfg_idx + 1}/{len(generation_configs)}.")

        return trainloader, None, len(trainloader)


def get_mathqa_model_output(max_budget: int, seed: int, tokenizer, model, extra_config=None,full_config=None):
    saved_path = extra_config.get("saved_path", None) if extra_config else None

    if saved_path is not None:
        file_path = f'{saved_path}_{seed}_{max_budget}_prune_data.pkl'
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                trainloader = pickle.load(f)
            logger.info(f"Load saved trainloader from {file_path}.")
            return trainloader, None, 0

    budget_format = extra_config.get("budget_format", "token-wise") if extra_config else "token-wise"
    if "sample-wise" in budget_format:
        logger.info(f"Budget format: {budget_format}.")
        trainloader, valenc, total = get_mathqa_model_output_sample(max_budget, seed, tokenizer, model,
                                                                    extra_config,full_config)
    else:
        pass
    if saved_path is not None:
        file_path = f'{saved_path}_{seed}_{max_budget}_prune_data.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(trainloader, f)
            logger.info(f"Save trainloader to {file_path}.")
    return trainloader, valenc, total


def get_mathqa_model_output_sample(samples: int, seed: int, tokenizer, model, extra_config,full_config):
    random.seed(seed)

    ds = load_dataset("math_qa", "main", split='train', trust_remote_code=True)
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)

    max_new_tokens = extra_config.get("max_new_tokens", 2048)
    batch_size = extra_config.get("batch_size", 4)

    generation_configs = extra_config.generation_config_sets

    # enable_thinking = extra_config.get("enable_thinking", True)
    # do_sample = extra_config.get("do_sample", True)
    # temperature = extra_config.get("temperature", 0.6)

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

    trainloader = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for cfg_idx, cfg in enumerate(generation_configs):
            name = cfg.get("name", f"cfg_{cfg_idx}")
            if 'gold' in name:
                taken = 0
                enable_thinking = cfg.get("enable_thinking", True)
                for i in range(0, len(indices), batch_size):
                    if taken >= samples:
                        break
                    batch_idx = indices[i:i + batch_size]
                    # qs = [ds[idx]["Problem"] for idx in batch_idx]

                    qs = []
                    for idx in batch_idx:
                        prob = ds[idx]["Problem"]
                        opt_field = ds[idx].get("options", ds[idx].get("Options", None))
                        options_str = format_options_field(opt_field)
                        qs.append(f"Question: {prob}\nOptions: {options_str}\nAnswer:")

                    prompts = [build_prompt(q, enable_thinking) for q in qs]

                    # 构造 prompt，用于得到去 pad 的 input_ids（与其他配置完全一致）
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                    B = inputs["input_ids"].shape[0]
                    for b in range(B):
                        if taken >= samples:
                            break
                        start, end = prompt_bounds(inputs, b)
                        input_ids_b = inputs["input_ids"][b]
                        trimmed_input_ids = input_ids_b[start:end].detach().cpu().tolist()

                        idx_b = int(batch_idx[b])
                        # answer_text = ds[idx_b]["correct"]  # gold 的文本答案
                        # opt_field = ds[idx_b].get("options", ds[idx_b].get("Options", None))
                        # options_str = format_options_field(opt_field)  # 见前文的 helper；若未添加，请粘贴到文件里
                        # correct_text = str(ds[idx_b].get("correct", "")).strip()
                        # answer_text = f"Options: {options_str}\n Correct: {correct_text}"
                        answer_text = ds[idx_b]['Rationale']
                        # gold 的 labels：对齐其它配置的生成输出（通常包含 eos），故在末尾补 eos（若存在）
                        labels_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
                        if eos_id is not None:
                            labels_ids = labels_ids + [eos_id]
                        trainloader.append({
                            "config_idx": cfg_idx,
                            "config_name": name,
                            "question_index": idx_b,
                            "question": qs[b],
                            "output_text": answer_text,  # gold 的原始答案文本
                            "input_ids": trimmed_input_ids,  # prompt 去 pad
                            "labels": labels_ids,  # 仅答案 token（末尾含 eos 时对齐生成配置）

                            "gen_params": {  # 字段不缺失，值对齐/占位
                                "max_new_tokens": max_new_tokens,
                                "do_sample": False,
                                "temperature": 0.0,
                                "enable_thinking": enable_thinking,
                            }
                        })
                        taken += 1
                    logger.info(f"Current sampled points: {taken}/{samples}")
                    # 与其它配置一致：本 config 内部 pack 到最大长度 L_max
                    lengths = [len(r["input_ids"]) + len(r["labels"]) for r in trainloader if
                               r['config_idx'] == cfg_idx]
                    L_max = max(lengths) if lengths else 0

                for r in trainloader:
                    if r['config_idx'] != cfg_idx:
                        continue
                    p = torch.tensor(r["input_ids"], dtype=torch.long)  # prompt ids（无 pad）
                    a = torch.tensor(r["labels"], dtype=torch.long)  # answer ids（含 eos 时与生成一致）

                    inp = torch.cat([p, a], dim=0)  # [L]
                    lab = torch.full_like(inp, -100)  # [L] 先全置 -100
                    lab[len(p):] = a  # 答案段 = token ids

                    pad_len = L_max - inp.size(0)
                    if pad_len > 0:
                        inp = torch.cat([inp, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)],
                                        dim=0)
                        lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)], dim=0)

                    r["train_input_ids"] = inp.tolist()
                    r["train_labels"] = lab.tolist()
                    r["train_attention_mask"] = [1] * (len(r["input_ids"]) + len(r["labels"])) + [0] * pad_len
            else:
                available_gpus = torch.cuda.device_count()
                logger.info(f"Available GPUs: {available_gpus}")
                enable_thinking = cfg.get("enable_thinking", True)
                do_sample = cfg.get("do_sample", True)
                temperature = cfg.get("temperature", 0.6)

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

                    taken=0
                    temp_trainloader=[]
                    enable_thinking = cfg.get("enable_thinking", True)

                    for i in range(0, len(indices), batch_size):
                        if taken >= samples:
                            break
                        batch_idx = indices[i:i + batch_size]

                        for idx in batch_idx:
                            if taken >= samples:
                                break
                            prob = ds[idx]["Problem"]
                            opt_field = ds[idx].get("options", ds[idx].get("Options", None))
                            options_str = format_options_field(opt_field)
                            temp_trainloader.append({
                                "config_idx": cfg_idx,
                                "config_name": name,
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
                    logger.info(f"[cfg {cfg_idx}] prepared target={taken}.")
                    print()

                    tmp_dir = tempfile.mkdtemp(prefix="mp_gen_")
                    shard_lists = even_chunks(temp_trainloader, available_gpus)

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
                            "--config_path", full_config.config_path,
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
                else:
                    taken = 0
                    for i in range(0, len(indices), batch_size):
                        if taken >= samples:
                            break
                        batch_idx = indices[i:i + batch_size]

                        qs = []
                        for idx in batch_idx:
                            prob = ds[idx]["Problem"]
                            opt_field = ds[idx].get("options", ds[idx].get("Options", None))
                            options_str = format_options_field(opt_field)
                            qs.append(f"Question: {prob}\nOptions: {options_str}\nAnswer:")

                        prompts = [build_prompt(q, enable_thinking) for q in qs]

                        # padding=True 以获得 attention_mask，便于逐样本切分
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
                            if taken >= samples:
                                break
                            start, end = prompt_bounds(inputs, b)
                            input_ids_b = inputs["input_ids"][b]
                            # 逐样本生成切分点 = prompt 的 end
                            gen_ids = cut_gen(full_ids[b], gen_start=end)

                            output_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

                            # 保存去 padding 的输入 ids
                            trimmed_input_ids = input_ids_b[start:end].detach().cpu().tolist()
                            trainloader.append({
                                "config_idx": cfg_idx,
                                "config_name": name,
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
                        logger.info(f"Current sampled points: {taken}/{samples}")

                lengths = [len(r["input_ids"]) + len(r["labels"]) for r in trainloader if r['config_idx'] == cfg_idx]
                L_max = max(lengths) if lengths else 0

                for r in trainloader:
                    if r['config_idx'] != cfg_idx:
                        continue
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

            logger.info(f"Finished for config {cfg_idx + 1}/{len(generation_configs)}.")

        return trainloader, None, len(trainloader)


def format_options_field(opt):
    """将 math_qa 的 options 统一成多行字符串：
    A. xxx
    B. yyy
    ...
    兼容 str / list / dict 三种形态。
    """
    if opt is None:
        return "N/A"
    # 字符串：常见形如 'A) 10 , B) 20 , C) 30' 或用 '####' / ';' 分隔
    if isinstance(opt, str):
        # s = opt.strip()
        # s = re.sub(r'\s*####\s*', '\n', s)
        # s = re.sub(r'\s*[,;]\s*', '\n', s)
        return opt
    # 列表：按 A/B/C... 编号
    if isinstance(opt, (list, tuple)):
        letters = string.ascii_uppercase
        return "\n".join(f"{letters[i]}. {str(x).strip()}" for i, x in enumerate(opt))
    # 字典：按键名排序（不区分大小写），常见 {'A': '...', 'B': '...'}
    if isinstance(opt, dict):
        keys = sorted(opt.keys(), key=lambda k: str(k).lower())
        return "\n".join(f"{str(k).upper()}. {str(opt[k]).strip()}" for k in keys)
    # 其它兜底
    return str(opt)


def get_loaders(name='gsm8k_model_output', seed=0, total_budget=500, tokenizer=None,
                extra_config=None, model=None, full_config=None):
    if name == 'gsm8k_model_output':
        data, validation, _ = get_gsm8k_model_output(total_budget, seed, tokenizer, model, extra_config)
    elif name == 'gsm8k':
        data, validation, _ = get_gsm8k(total_budget, seed, tokenizer, extra_config)
    elif name == 'mathqa_model_output':
        data, validation, _ = get_mathqa_model_output(total_budget, seed, tokenizer, model, extra_config, full_config)
    else:
        raise ValueError(f'Unsupported paper calibration dataset: {name}')
    return data, validation
