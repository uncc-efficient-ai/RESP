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

def main(config_path: str = '', checkpoint_path: str = None,
         substitute_dataset_path: str = None,
         result_name: str = 'lm_eval_result', out: str = 'result.json'):
    config = Config(config_path)
    c = config.get_config()
    set_random_seed(c.task.seed)
    accelerator = Accelerator(mixed_precision="bf16")
    if accelerator.is_main_process:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(c.model.name)
    tok.padding_side = "left"
    model_kwargs = {}
    model_kwargs.update(
        device_map={"": f"{accelerator.device}"}
    )
    model = AutoModelForCausalLM.from_pretrained(c.model.name, torch_dtype=torch.bfloat16, **model_kwargs, )

    layer_mapping = LAYER_NAME_MAPPING['Qwen']
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        mask = checkpoint["actual_mask"]
        model = restore_to_prune(model, mask, layer_mapping)
    # 如果有 dataloader/optimizer，都一并丢进 prepare
    if accelerator.is_main_process:
        check_unstr_sparsity(model)
    # model = accelerator.prepare(model)
    if substitute_dataset_path is not None:
        replace_data = torch.load(substitute_dataset_path)
        result_path = lm_replace_eval(c.evaluation.lm_eval_options, model, tok, accelerator, result_name, replace_data)
        if accelerator.is_main_process:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result_path, f, ensure_ascii=False, indent=2)
    else:
        results_list = lm_simple_eval(c.evaluation.lm_eval_options, model, tok, accelerator, result_name)
        if accelerator.is_main_process:
            with open(out, "w", encoding="utf-8") as f:
                result = results_list[0] if len(results_list) == 1 else results_list
                json.dump(result, f, ensure_ascii=False, indent=2)


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


def lm_simple_eval(lm_eval_options, model, tokenizer, accelerator, result_name):
    import lm_eval
    results_list = []
    if not isinstance(lm_eval_options.num_fewshot, list):
        lm_eval_options.num_fewshot = [lm_eval_options.num_fewshot]
    for num_fewshot in lm_eval_options.num_fewshot:
        wrapped_model = lm_eval.models.huggingface.HFLM_Accelerate(model, tokenizer=tokenizer, accelerator=accelerator,
                                                                   # batch_size=lm_eval_options.batch_size if lm_eval_options.batch_size and quick else 'auto',
                                                                   batch_size=lm_eval_options.batch_size,
                                                                   max_length=lm_eval_options.max_length if lm_eval_options.max_length else None,
                                                                   trust_remote_code=True)

        # monkey patch for qwen3
        if 'Qwen3' in model.config.name_or_path:
            import jinja2
            def apply_chat_template(
                    self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
            ) -> str:
                """
                Method to apply a chat template to a list of chat history between user and model.
                """
                try:
                    chat_templated = self.tokenizer.apply_chat_template(
                        chat_history,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                        continue_final_message=not add_generation_prompt,
                        enable_thinking=lm_eval_options.enable_thinking if lm_eval_options.enable_thinking else False
                    )
                except jinja2.exceptions.TemplateError:
                    eval_logger.warning(
                        "Failed to apply chat template. removing the system role in chat history."
                    )
                    chat_history = [msg for msg in chat_history if msg["role"] != "system"]
                    chat_templated = self.tokenizer.apply_chat_template(
                        chat_history,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                        continue_final_message=not add_generation_prompt,
                        enable_thinking=lm_eval_options.enable_thinking if lm_eval_options.enable_thinking else False
                    )
                return chat_templated

            wrapped_model.apply_chat_template = types.MethodType(apply_chat_template,
                                                                 wrapped_model)

        results = lm_eval.simple_evaluate(  # call simple_evaluate
            model=wrapped_model,
            gen_kwargs=dict(lm_eval_options.gen_kwargs),
            random_seed=lm_eval_options.seed,
            numpy_random_seed=lm_eval_options.seed,
            torch_random_seed=lm_eval_options.seed,
            fewshot_random_seed=lm_eval_options.seed,
            tasks=lm_eval_options.tasks,
            num_fewshot=num_fewshot,
            log_samples=True,
            apply_chat_template=lm_eval_options.apply_chat_template if lm_eval_options.apply_chat_template else False,
            fewshot_as_multiturn=True,
        )
        if not accelerator.is_main_process:
            continue

        def _handle_non_serializable(o):
            if isinstance(o, np.int64) or isinstance(o, np.int32):
                return int(o)
            elif isinstance(o, set):
                return list(o)
            else:
                return str(o)

        path = Path(lm_eval_options.output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file():
            raise FileExistsError(f"File already exists at {path}")
        output_path_file = path.joinpath(f"{str(num_fewshot if num_fewshot else 'None')}_{result_name}.json")
        if path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
        dumped = json.dumps(
            results, indent=2, default=_handle_non_serializable, ensure_ascii=False
        )
        output_path_file.write_text(dumped, encoding="utf-8")
        eval_logger.info(
            f"lm_eval complete. See report in {str(num_fewshot)}_{result_name}.json located in {lm_eval_options.output_path}")
        results_list.append(str(output_path_file))
    return results_list


def lm_replace_eval(lm_eval_options, model, tokenizer, accelerator, result_name, replace_data):
    import lm_eval
    from lm_eval.tasks import TaskManager, get_task_dict
    def my_dataset_factory(*, docs, split_name, **_ignore):
        return DatasetDict({split_name: Dataset.from_list(docs)})

    if not isinstance(lm_eval_options.num_fewshot, list):
        lm_eval_options.num_fewshot = [lm_eval_options.num_fewshot]

    wrapped_model = lm_eval.models.huggingface.HFLM_Accelerate(model, tokenizer=tokenizer, accelerator=accelerator,
                                                               batch_size=lm_eval_options.batch_size,
                                                               max_length=lm_eval_options.max_length if lm_eval_options.max_length else None,
                                                               trust_remote_code=True)
    # monkey patch for qwen3
    if 'Qwen3' in model.config.name_or_path:
        import jinja2
        def apply_chat_template(
                self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
        ) -> str:
            """
            Method to apply a chat template to a list of chat history between user and model.
            """
            try:
                chat_templated = self.tokenizer.apply_chat_template(
                    chat_history,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=not add_generation_prompt,
                    enable_thinking=lm_eval_options.enable_thinking if lm_eval_options.enable_thinking else False
                )
            except jinja2.exceptions.TemplateError:
                eval_logger.warning(
                    "Failed to apply chat template. removing the system role in chat history."
                )
                chat_history = [msg for msg in chat_history if msg["role"] != "system"]
                chat_templated = self.tokenizer.apply_chat_template(
                    chat_history,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=not add_generation_prompt,
                    enable_thinking=lm_eval_options.enable_thinking if lm_eval_options.enable_thinking else False
                )
            return chat_templated

        wrapped_model.apply_chat_template = types.MethodType(apply_chat_template,
                                                             wrapped_model)

    task_manager = TaskManager()
    task_dict = get_task_dict(lm_eval_options.tasks,
                              task_manager=task_manager)
    target_task = task_dict[lm_eval_options.tasks[0]]

    base_cfg = target_task.config.__dict__.copy()

    # 关键：保持原分割名一致（只改数据源）
    split = base_cfg.get("test_split")or base_cfg.get("validation_split")  or "validation"

    # 用 TaskManager.metadata 把 docs 和 split 传给 custom_dataset
    # 构造一个“同壳”任务实例：只注入 custom_dataset；其余配置保持与原任务完全一致
    new_cfg = {**base_cfg}
    new_cfg['generation_kwargs'] = {**base_cfg.get('generation_kwargs', {}), **dict(lm_eval_options.gen_kwargs)}
    new_cfg['num_fewshot'] = lm_eval_options.num_fewshot[0]

    new_cfg['training_split'] = None
    if 'test' in split:
        new_cfg['validation_split'] = None

    new_cfg.update({
        "task": f"{target_task.config.task}_custom",  # 给它起个新名字，防止混淆
        "custom_dataset": my_dataset_factory,  # 只改数据来源
        "metadata": {"docs": replace_data, "split_name": split},
        # 其余诸如 output_type, doc_to_text/target, metric_list, fewshot_config 等全部沿用
        # split 名也沿用 base_cfg 里的（上面 new_cfg 已含有）
    })

    # 用同一个类（通常是 ConfigurableTask 的子类）实例化
    new_task = target_task.__class__(config=new_cfg)

    results = lm_eval.evaluator.evaluate(
        lm=wrapped_model,
        task_dict={new_task.config.task: new_task},
        log_samples=True,
        apply_chat_template=lm_eval_options.apply_chat_template if lm_eval_options.apply_chat_template else False,
        fewshot_as_multiturn=True,
    )
    if not accelerator.is_main_process:
        return None

    def _handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)

    path = Path(lm_eval_options.output_path)
    # check if file or 'dir/results.json' exists
    if path.is_file():
        raise FileExistsError(f"File already exists at {path}")
    output_path_file = path.joinpath(f"{str(new_task.config.num_fewshot)}_{result_name}.json")
    if path.suffix in (".json", ".jsonl"):
        output_path_file = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path = path.parent
    else:
        path.mkdir(parents=True, exist_ok=True)
    dumped = json.dumps(
        results, indent=2, default=_handle_non_serializable, ensure_ascii=False
    )
    output_path_file.write_text(dumped, encoding="utf-8")
    eval_logger.info(
        f"lm_eval complete. See report in {str(new_task.config.num_fewshot)}_{result_name}.json located in {lm_eval_options.output_path}")
    return str(output_path_file)


if __name__ == "__main__":
    fire.Fire(main)
