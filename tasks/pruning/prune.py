import logging
import os

from modules.data.make_data import make_prune_data
from modules.eval.setup_eval import eval

from modules.model.make_model import make_model
import modules.system.system as system

from tasks.pruning.make_pruner import make_pruner

logger = logging.getLogger(__name__)


def prune_task(c):
    # load model
    model, tokenizer, config = make_model(c.model)

    model.config.use_cache = False

    if c.task.prune.eval_before:
        logger.info(f"Running evaluation before pruning.")
        eval(c, model, tokenizer)

    prune_data = make_prune_data(c.task.prune, c.model, c.task.seed, tokenizer, model, c)

    pruner_c = make_pruner(c.task.prune)

    model = system.setup_model(model, prune_data, tokenizer, c)

    pruner = pruner_c(model, c, prune_data)

    pruner.tokenizer = tokenizer

    pruner.prune()

    model = pruner.model

    if not system.ddp or system.rank == 0:
        eval(c, model, tokenizer)
