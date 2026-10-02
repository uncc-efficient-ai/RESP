from abc import ABC, abstractmethod


class Pruner(ABC):
    @abstractmethod
    def __init__(self, model, config, data):
        self.model = model
        self.config = config
        self.data = data

    @abstractmethod
    def prune(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_imps(self):
        pass

    def get_model(self):
        return self.model

    def get_wrapped_model(self):
        return self.model

    def get_model_config(self):
        return self.model.config

    def set_model_config(self, config):
        self.model.config = config

    def before_pruning_step(self):
        pass

    def after_pruning_step(self):
        pass

    def during_pruning_step(self, module, mask, imp_metric, threshold):
        pass

    def before_pruning(self):
        pass

    def finishing_pruning(self):
        pass
