import json
import logging
import types
from pathlib import Path
from typing import List, Dict

import numpy as np


from datasets import DatasetDict, Dataset


logger = logging.getLogger(__name__)


def lm_simple_eval(lm_eval_options, model, tokenizer, result_name, quick=False):
    import lm_eval
    if not isinstance(lm_eval_options.num_fewshot, list):
        lm_eval_options.num_fewshot = [lm_eval_options.num_fewshot]
    if quick and not lm_eval_options.quick_tasks:
        return
    for num_fewshot in lm_eval_options.num_fewshot:
        wrapped_model = lm_eval.models.huggingface.HFLM(model, tokenizer=tokenizer,
                                                        # batch_size=lm_eval_options.batch_size if lm_eval_options.batch_size and quick else 'auto',
                                                        batch_size=lm_eval_options.batch_size,
                                                        max_length=lm_eval_options.max_length if lm_eval_options.max_length else None)

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
                    logger.warning(
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
            tasks=lm_eval_options.tasks if not quick else lm_eval_options.quick_tasks,
            # tasks=["openbookqa", "arc_easy", "winogrande", "hellaswag", "arc_challenge", "piqa", "boolq"],
            # tasks=["openbookqa"],
            num_fewshot=num_fewshot,
            log_samples=True,
            apply_chat_template=lm_eval_options.apply_chat_template if lm_eval_options.apply_chat_template else False,
            fewshot_as_multiturn=True,
        )

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
        logger.info(
            f"lm_eval complete. See report in {str(num_fewshot)}_{result_name}.json located in {lm_eval_options.output_path}")


def my_dataset_factory(*, docs, split_name, **_ignore):
    return DatasetDict({split_name: Dataset.from_list(docs)})


def lm_replace_eval(lm_eval_options, model, tokenizer, result_name, replace_data, quick=False):
    import lm_eval
    from lm_eval.tasks import TaskManager, get_task_dict
    if not isinstance(lm_eval_options.num_fewshot, list):
        lm_eval_options.num_fewshot = [lm_eval_options.num_fewshot]
    if quick and not lm_eval_options.quick_tasks:
        return

    wrapped_model = lm_eval.models.huggingface.HFLM(model, tokenizer=tokenizer,
                                                    batch_size=lm_eval_options.batch_size if lm_eval_options.batch_size and quick else 'auto',
                                                    max_length=lm_eval_options.max_length if lm_eval_options.max_length else None)

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
                logger.warning(
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
    task_dict = get_task_dict(lm_eval_options.tasks if not quick else lm_eval_options.quick_tasks,
                              task_manager=task_manager)
    target_task = task_dict[lm_eval_options.tasks[0] if not quick else lm_eval_options.quick_tasks[0]]

    base_cfg = target_task.config.__dict__.copy()

    # 关键：保持原分割名一致（只改数据源）
    split = base_cfg.get("test_split") or base_cfg.get("validation_split") or "validation"

    # 用 TaskManager.metadata 把 docs 和 split 传给 custom_dataset
    # 构造一个“同壳”任务实例：只注入 custom_dataset；其余配置保持与原任务完全一致
    new_cfg = {**base_cfg}
    new_cfg['generation_kwargs'] = {**base_cfg.get('generation_kwargs', {}), **dict(lm_eval_options.gen_kwargs)}
    new_cfg['num_fewshot'] = lm_eval_options.num_fewshot[0]
    new_cfg['training_split'] = None
    if split == base_cfg.get('test_split'):
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
    logger.info(
        f"lm_eval complete. See report in {str(new_task.config.num_fewshot)}_{result_name}.json located in {lm_eval_options.output_path}")
    return output_path_file
