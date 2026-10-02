"""Evaluate the in-memory model, including its current pruning masks."""


def eval(c, model, tokenizer):
    import torch
    if c.evaluation.lm_eval:
        with torch.no_grad():
            model.eval()
            eval_lm_eval(model, tokenizer, c, 'lm_eval_result')


def eval_lm_eval(model, tokenizer, c, result_name, quick=False):
    if c.evaluation.lm_eval:
        from .lm_eval_main import lm_simple_eval
        return lm_simple_eval(c.evaluation.lm_eval_options, model, tokenizer, result_name, quick=quick)
