from .hf import make_hf_model


def make_model(model_config):
    return make_hf_model(model_config)
