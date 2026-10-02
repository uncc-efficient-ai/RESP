"""Load the Qwen3 model used in the RESP experiments."""

from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from .utils import str_to_torch_dtype
from utils import load_module


def make_hf_model(model_config):
    config = Qwen3Config.from_pretrained(model_config.name, attn_implementation='eager')
    model_class = Qwen3ForCausalLM
    if model_config.custom_modeling:
        model_class = load_module(model_config.custom_config.custom_package_location).Qwen3ForCausalLM
    model = model_class.from_pretrained(
        model_config.name, config=config,
        torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.name)
    tokenizer.padding_side = 'left'
    return model, tokenizer, model.config
