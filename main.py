"""Run a RESP paper experiment from a YAML configuration."""

import argparse
import os
import random
from pathlib import Path

from modules.config.config import Config


def main(config_path, check_config=False):
    config = Config(config_path)
    c = config.get_config()
    if check_config:
        print(f"Valid {c.task.task_mode} configuration: {c.config_path}")
        return

    os.chdir(Path(__file__).resolve().parent)

    import numpy as np
    import torch
    import modules.system.system as system
    from utils import setup_logger

    random.seed(c.task.seed)
    np.random.seed(c.task.seed)
    torch.manual_seed(c.task.seed)
    torch.cuda.manual_seed_all(c.task.seed)
    os.environ['PYTHONHASHSEED'] = str(c.task.seed)
    Path(c.task.output_folder).mkdir(parents=True, exist_ok=True)
    config.save_config(c.task.output_folder)
    logger = setup_logger(c.report)
    logger.info("Loaded config: %s", c.config_path)
    system.init_system(c)

    if c.task.task_mode == 'prune':
        from tasks.pruning.prune import prune_task
        prune_task(c)
    else:
        from tasks.test.test import test_task
        test_task(c)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-path', '--config_path', required=True)
    parser.add_argument('--check-config', action='store_true', help='Validate without loading a model.')
    args = parser.parse_args()
    main(args.config_path, args.check_config)
